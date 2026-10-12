"""Core LangPro model composition.

Top-level model that composes:
  - PrePerceiverConditionEncoder  (multi-modal condition -> ctx tokens)
  - DockingWizardPerceiver        (ctx tokens -> memories)
  - ProGenConditionedDecoder      (memories + protein tokens -> LM logits/loss)

Design goals:
  - End-to-end training with a single LM objective (teacher forcing) + optional aux losses.
  - Strict config/shape/mask assertions to prevent silent segment bugs.
  - Best-effort training stability: frozen backbones stay in eval() even when model.train() is called.
  - High-fidelity generation: optional full-memory injection during generation using a
    custom autoregressive loop (no cache) to ensure injection applies at every step.

This file intentionally avoids "shortcuts" and provides:
  - Robust batch validation
  - Parameter-group construction for optimizers
  - Debug/monitor outputs

The legacy ``DockingWizard*`` class names are retained for checkpoint
compatibility, while the public package is organized under ``langpro``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
import inspect

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


# -----------------------------
# Imports (local modules)
# -----------------------------
from .encoder import PrePerceiverConditionEncoder, PrePerceiverEncoderConfig
from .perceiver import DockingWizardPerceiver, PerceiverModuleConfig
from .decoder import ProGenConditionedDecoder, DecoderOutput
from .fold_planner import ConditionFoldPlanner, FoldPlannerOutput, fold_planner_losses
from .structure_token_planner import (
    StructureTokenPlannerOutput,
    StructureTokenScaffoldPlanner,
    structure_token_planner_losses,
)


# -----------------------------
# Output container
# -----------------------------
@dataclass
class DockingWizardOutput:
    loss_total: Optional[torch.Tensor]
    loss_lm: Optional[torch.Tensor]
    loss_aux: Dict[str, torch.Tensor] = field(default_factory=dict)
    prefix_len: Optional[int] = None
    logits: Optional[torch.Tensor] = None
    protein_hidden_states: Optional[torch.Tensor] = None
    fold_plan: Optional[FoldPlannerOutput] = None
    debug: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Top-level config
# -----------------------------
@dataclass
class DockingWizardModelConfig:
    # Sub-configs
    encoder_cfg: PrePerceiverEncoderConfig = field(default_factory=PrePerceiverEncoderConfig)
    perceiver_cfg: Optional[PerceiverModuleConfig] = None

    # Decoder settings
    decoder_backbone_name_or_path: str = "hugohrban/progen2-base"
    decoder_add_boundary_tokens: bool = True
    decoder_enable_full_injection: bool = True
    decoder_inject_n_head: int = 8
    decoder_inject_dropout: float = 0.0
    decoder_freeze_backbone: bool = True
    decoder_trust_remote_code: bool = True
    decoder_local_files_only: bool = False
    decoder_revision: Optional[str] = None
    decoder_unfreeze_last_n_blocks: int = 0
    decoder_prefix_boundary_scale: float = 1.0

    # Loss weights
    lambda_aux_enc: float = 0.05
    # Optional regularizers (off by default)
    lambda_bridge_gate_l1: float = 0.0
    lambda_condition_awareness: float = 0.0  # penalize if docking-influenced memory collapses to func memory
    cond_awareness_prob: float = 0.15  # only compute condition-awareness loss with this probability
    condition_awareness_max_similarity: float = 0.80
    lambda_condition_contrastive: float = 0.0
    condition_contrastive_temperature: float = 0.07
    lambda_length_prediction: float = 0.0
    protein_length_scale: float = 1024.0
    perceiver_layerscale_init: float = 1e-3
    perceiver_cross_every_block: bool = False
    # Optional sequence and route regularizers.
    eos_loss_weight: float = 1.0  # EOS token upweighting (>1.0 for stronger supervision)
    lambda_diversity: float = 0.0  # Sequence diversity regularization
    lambda_local_composition_unlikelihood: float = 0.0
    local_composition_window: int = 32
    local_composition_min_context: int = 12
    local_composition_dominance_threshold: float = 0.35
    lambda_local_group_unlikelihood: float = 0.0
    local_residue_token_groups: List[List[int]] = field(default_factory=list)
    local_residue_group_thresholds: List[float] = field(default_factory=list)
    lambda_route_comp: float = 0.0
    lambda_route_contrastive: float = 0.0
    route_comp_proj_dim: int = 256
    route_comp_temperature: float = 0.07
    lambda_chem_align: float = 0.0
    lambda_query_diversity: float = 0.0
    n_chem_features: int = 10
    chem_loss_type: Literal["auto", "bce", "mse", "smooth_l1"] = "auto"
    use_hard_negative_contrastive: bool = False
    hard_negative_weight: float = 0.0
    ligand_memory_bank_size: int = 0
    ligand_contrastive_temperature: float = 0.07
    n_interaction_types: int = 6
    lambda_pocket: float = 0.0
    lambda_contact: float = 0.0
    lambda_structure_contacts: float = 0.0
    structure_contact_proj_dim: int = 128
    structure_contact_negative_ratio: float = 4.0
    lambda_interaction: float = 0.0
    lambda_secondary_structure: float = 0.0
    lambda_solvent_accessibility: float = 0.0
    lambda_residue_depth: float = 0.0
    lambda_complementarity: float = 0.0
    complementarity_loss_type: Literal["bce", "mse"] = "mse"
    use_bio_position_weights: bool = False
    alpha_conservation: float = 0.0
    alpha_pocket: float = 0.0
    alpha_contact: float = 0.0
    enable_pocket_aware_injection: bool = False
    pocket_gate_strength: float = 0.5
    enable_ligand_chemical_memory: bool = True
    enable_pocket_interaction_memory: bool = True
    enable_interaction_type_queries: bool = True
    enable_biological_bridge: bool = True
    aux_loss_warmup_steps: int = 0
    aux_loss_warmup_epochs: int = 0
    aux_loss_warmup_mode: Literal["linear", "cosine", "none"] = "none"

    # Training/inference toggles
    enable_full_injection_train: bool = True
    enable_full_injection_generate: bool = True
    full_injection_layer_policy: Literal["all", "last_half", "last_k", "every_other", "none"] = "all"
    full_injection_last_k: int = 0
    share_full_injection_weights: bool = False
    # At training time, choose one value per minibatch. -1 keeps full
    # injection throughout; non-negative values stop after that many protein
    # residues, matching hybrid autoregressive decoding.
    full_injection_train_stop_choices: List[int] = field(default_factory=list)
    # Scale choices are sampled independently of stop choices. Including zero
    # trains the prefix route to preserve the pretrained protein prior.
    full_injection_train_scale_choices: List[float] = field(default_factory=list)
    # Keep validation on the configured low-strength training route instead of
    # silently evaluating at scale 1.0.
    full_injection_eval_scale: float = 1.0
    prefix_condition_train_scale_choices: List[float] = field(default_factory=list)
    pool_condition_memory: bool = False

    # Hierarchical condition -> fold plan -> sequence generation. The planner
    # is condition-only, so structure supervision cannot leak target residues.
    enable_condition_fold_planner: bool = False
    fold_plan_slots: int = 16
    fold_plan_prototypes: int = 64
    fold_plan_archetypes: int = 32
    fold_plan_layers: int = 2
    fold_plan_heads: int = 8
    fold_plan_dropout: float = 0.10
    fold_plan_prototype_temperature: float = 0.50
    fold_plan_condition_dropout: float = 0.10
    fold_plan_local_chemistry_scale: float = 0.15
    fold_plan_archetype_scale: float = 0.20
    fold_plan_sample_during_training: bool = True
    fold_plan_sample_during_generation: bool = True
    fold_plan_hard_during_generation: bool = False
    fold_plan_prefix_scale: float = 0.35
    fold_plan_full_memory_scale: float = 1.0
    fold_plan_injection_warmup_steps: int = 800
    lambda_fold_plan_ss: float = 0.0
    lambda_fold_plan_contacts: float = 0.0
    lambda_fold_plan_global: float = 0.0
    lambda_fold_plan_codebook: float = 0.0
    lambda_fold_plan_chemistry: float = 0.0
    lambda_fold_plan_sequence: float = 0.0
    lambda_fold_plan_pocket: float = 0.0
    lambda_fold_plan_archetype: float = 0.0

    # Residue-aligned discrete scaffold tokens. Training can use a structure
    # posterior, while inference samples only the condition prior.
    enable_structure_token_planner: bool = False
    structure_token_slots: int = 64
    structure_token_codes: int = 128
    structure_token_prefix_slots: int = 16
    structure_token_layers: int = 3
    structure_token_heads: int = 8
    structure_token_dropout: float = 0.10
    structure_token_temperature: float = 0.50
    structure_token_posterior_teacher_forcing: float = 0.75
    structure_token_posterior_final_mix: float = 0.15
    structure_token_posterior_decay_steps: int = 900
    structure_token_geometry_feature_dim: int = 10
    structure_token_sample_during_generation: bool = True
    structure_token_hard_during_generation: bool = False
    structure_token_prefix_scale: float = 0.30
    structure_token_full_memory_scale: float = 1.0
    # One factor is sampled per training batch and applied to both structure
    # token routes. Including zero preserves an unconditional protein-prior
    # route while the condition prior is still learning.
    structure_token_train_scale_choices: List[float] = field(default_factory=list)
    lambda_structure_token_prior: float = 0.0
    lambda_structure_token_posterior: float = 0.0
    lambda_structure_token_kl: float = 0.0
    lambda_structure_token_pocket: float = 0.0
    lambda_structure_token_interaction: float = 0.0
    lambda_structure_token_topology: float = 0.0

    # Encoder aux computation
    # NOTE: Due to current encoder implementation, turning compute_aux_losses=False can
    # inadvertently wrap ligand fusion in no_grad(). For best training, keep True.
    encoder_force_compute_aux: bool = True

    # Frozen backbone behavior
    keep_frozen_backbones_in_eval: bool = True

    # Generation defaults (can be overridden per call)
    gen_max_new_tokens: int = 256
    gen_temperature: float = 1.0
    gen_top_p: float = 0.9
    gen_top_k: int = 0
    gen_do_sample: bool = True
    gen_repetition_penalty: float = 1.0
    gen_eos_token_id: Optional[int] = None
    gen_bos_token_id: Optional[int] = None
    gen_pad_token_id: Optional[int] = None


# -----------------------------
# Helper utilities
# -----------------------------
def _as_long_mask(mask: torch.Tensor) -> torch.Tensor:
    """Ensure mask is int64/long with semantics 1=valid, 0=pad."""
    if mask.dtype == torch.bool:
        return mask.to(dtype=torch.long)
    if not torch.is_floating_point(mask):
        return (mask != 0).to(dtype=torch.long)
    return (mask > 0).to(dtype=torch.long)


def _as_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask
    if torch.is_floating_point(mask):
        return mask > 0
    return mask != 0


def _masked_fill_tokens(labels: torch.Tensor, attention_mask: torch.Tensor, fill: int = -100) -> torch.Tensor:
    """Set pad positions (attention_mask==0) to fill."""
    m = _as_bool_mask(attention_mask)
    out = labels.clone()
    out[~m] = fill
    return out


def _align_residue_values_to_labels(
    residue_values: torch.Tensor,
    labels: torch.Tensor,
    eos_token_id: int,
    *,
    eos_value: float = 0.0,
) -> torch.Tensor:
    """Move residue-indexed values onto ``[BOS] residues [EOS]`` label positions.

    Dataset annotations such as conservation, pocket labels, and user-supplied
    position weights are indexed from residue 0. Protein labels are indexed as
    ``[-100, residue_0, ..., residue_n, EOS, -100, ...]``. Treating the former
    as the latter silently shifts every annotation left and, most importantly,
    assigns the padded zero after the last residue to EOS.
    """
    if residue_values.shape != labels.shape:
        raise ValueError(
            "residue-aligned values must have the same padded shape as labels: "
            f"got {tuple(residue_values.shape)} vs {tuple(labels.shape)}"
        )
    values = torch.nan_to_num(
        residue_values.to(device=labels.device, dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    aligned = torch.zeros_like(values)
    for row in range(labels.size(0)):
        eos_positions = torch.nonzero(
            labels[row].eq(int(eos_token_id)), as_tuple=False
        ).flatten()
        if eos_positions.numel() > 0:
            eos_pos = int(eos_positions[0].item())
            residue_count = max(0, eos_pos - 1)
            if residue_count:
                aligned[row, 1:eos_pos] = values[row, :residue_count]
            aligned[row, eos_pos] = float(eos_value)
        else:
            valid_positions = torch.nonzero(labels[row].ne(-100), as_tuple=False).flatten()
            if valid_positions.numel() > 0:
                start = int(valid_positions[0].item())
                count = int(valid_positions.numel())
                aligned[row, start:start + count] = values[row, :count]
    return aligned


def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering."""
    # logits: [B, V]
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        kth_vals = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_vals, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_mask = cumprobs > top_p
        # Keep at least min_tokens_to_keep
        if min_tokens_to_keep > 1:
            sorted_mask[..., :min_tokens_to_keep] = False
        # Shift mask right to keep first token above threshold
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        # Scatter back to original indexing
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_idx, src=sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

    return logits


def _apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    """Apply repetition penalty (CTRL-style)."""
    if penalty is None or penalty == 1.0:
        return logits
    # logits: [B,V], generated: [B,T]
    for b in range(logits.size(0)):
        tokens = generated[b].tolist()
        if not tokens:
            continue
        uniq = set(tokens)
        for t in uniq:
            if t < 0 or t >= logits.size(-1):
                continue
            val = logits[b, t]
            logits[b, t] = val / penalty if val > 0 else val * penalty
    return logits


def _sequence_diversity_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Entropy-based diversity regularization on valid (non -100) positions."""
    # logits: [B, L, V], labels: [B, L]
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)  # [B,L]
    valid = (labels != -100).to(dtype=entropy.dtype)
    denom = valid.sum().clamp_min(1.0)
    # maximize entropy => minimize negative entropy
    return -((entropy * valid).sum() / denom)


def _local_composition_unlikelihood_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    prefix_len: int,
    window: int,
    min_context: int,
    dominance_threshold: float,
    special_token_ids: Iterable[int],
) -> torch.Tensor:
    """Suppress continuation of a locally dominant residue on natural targets.

    The target at a position is never penalized when it equals the dominant
    context residue. This keeps genuine homorepeats represented in the training
    data, while teaching the model to leave a compositionally collapsed state
    when the natural next residue is different.
    """
    window = max(2, int(window))
    min_context = min(window, max(2, int(min_context)))
    threshold = float(dominance_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("local composition dominance threshold must be in (0, 1)")
    protein_logits = logits[:, int(prefix_len):, :]
    length = min(protein_logits.size(1), labels.size(1))
    if length <= 1:
        return logits.sum() * 0.0
    labels = labels[:, :length]
    next_logits = protein_logits[:, :length - 1, :].float()
    next_targets = labels[:, 1:length]

    padded = F.pad(labels, (window, 0), value=-100)
    contexts = padded.unfold(1, window, 1)[:, 1:length, :]
    valid_context = contexts.ne(-100)
    for token_id in special_token_ids:
        valid_context &= contexts.ne(int(token_id))
    # Count each context position's residue against all other valid positions,
    # then select the residue with the largest local count.
    pair_equal = contexts.unsqueeze(-1).eq(contexts.unsqueeze(-2))
    counts = (pair_equal & valid_context.unsqueeze(-2)).sum(dim=-1)
    counts = counts.masked_fill(~valid_context, 0)
    dominant_count, dominant_position = counts.max(dim=-1)
    dominant_ids = contexts.gather(-1, dominant_position.unsqueeze(-1)).squeeze(-1)
    context_count = valid_context.sum(dim=-1)
    dominance = dominant_count.to(next_logits.dtype) / context_count.clamp_min(1)
    eligible = (
        context_count.ge(min_context)
        & dominance.gt(threshold)
        & next_targets.ne(-100)
        & next_targets.ne(dominant_ids)
        & dominant_ids.ge(0)
        & dominant_ids.lt(next_logits.size(-1))
    )
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    safe_ids = dominant_ids.clamp(0, next_logits.size(-1) - 1)
    dominant_log_probability = F.log_softmax(next_logits, dim=-1).gather(
        -1, safe_ids.unsqueeze(-1)
    ).squeeze(-1)
    dominant_probability = dominant_log_probability.exp().clamp(max=1.0 - 1e-6)
    excess = ((dominance - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
    raw = -torch.log1p(-dominant_probability) * excess
    return raw[eligible].mean()


def _local_group_unlikelihood_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    prefix_len: int,
    window: int,
    min_context: int,
    token_groups: Iterable[Iterable[int]],
    group_thresholds: Iterable[float],
) -> torch.Tensor:
    """Teach the decoder to exit locally collapsed biochemical classes."""
    groups = [sorted({int(token) for token in group}) for group in token_groups]
    thresholds = [float(value) for value in group_thresholds]
    if not groups or len(groups) != len(thresholds):
        return logits.sum() * 0.0
    window = max(2, int(window))
    min_context = min(window, max(2, int(min_context)))
    protein_logits = logits[:, int(prefix_len):, :]
    length = min(protein_logits.size(1), labels.size(1))
    if length <= 1:
        return logits.sum() * 0.0
    labels = labels[:, :length]
    next_logits = protein_logits[:, :length - 1, :].float()
    next_targets = labels[:, 1:length]
    padded = F.pad(labels, (window, 0), value=-100)
    contexts = padded.unfold(1, window, 1)[:, 1:length, :]
    valid_context = contexts.ne(-100)
    context_count = valid_context.sum(dim=-1)
    all_log_normalizer = torch.logsumexp(next_logits, dim=-1)
    losses = []
    for group, threshold in zip(groups, thresholds):
        if not group or not 0.0 < threshold < 1.0:
            continue
        ids = torch.as_tensor(group, device=labels.device, dtype=labels.dtype)
        context_in_group = contexts.unsqueeze(-1).eq(ids).any(dim=-1) & valid_context
        group_fraction = (
            context_in_group.sum(dim=-1).to(next_logits.dtype)
            / context_count.clamp_min(1)
        )
        target_in_group = next_targets.unsqueeze(-1).eq(ids).any(dim=-1)
        eligible = (
            context_count.ge(min_context)
            & group_fraction.gt(threshold)
            & next_targets.ne(-100)
            & ~target_in_group
        )
        if not bool(eligible.any()):
            continue
        group_log_probability = (
            torch.logsumexp(next_logits.index_select(-1, ids), dim=-1)
            - all_log_normalizer
        )
        group_probability = group_log_probability.exp().clamp(max=1.0 - 1e-6)
        excess = (
            (group_fraction - threshold) / (1.0 - threshold)
        ).clamp(0.0, 1.0)
        raw = -torch.log1p(-group_probability) * excess
        losses.append(raw[eligible])
    if not losses:
        return logits.sum() * 0.0
    return torch.cat(losses).mean()


def _clip_infonce(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.07,
    group_ids: Optional[torch.Tensor] = None,
    positive_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Symmetric in-batch InfoNCE with optional multi-positive group labels."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = (z_a @ z_b.t()) / float(temperature)
    if group_ids is None and positive_mask is None:
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    if positive_mask is not None:
        positive = positive_mask.to(device=logits.device, dtype=torch.bool)
        if positive.shape != logits.shape:
            raise ValueError(
                f"positive_mask shape={tuple(positive.shape)} does not match "
                f"logits={tuple(logits.shape)}"
            )
    else:
        ids = group_ids.to(device=logits.device).view(-1)
        if ids.numel() != logits.size(0):
            raise ValueError(
                f"group_ids length={ids.numel()} does not match batch={logits.size(0)}"
            )
        positive = ids[:, None].eq(ids[None, :])
    positive.fill_diagonal_(True)

    def _directional(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        log_prob = x - torch.logsumexp(x, dim=1, keepdim=True)
        return -torch.logsumexp(
            log_prob.masked_fill(~pos, torch.finfo(log_prob.dtype).min),
            dim=1,
        ).mean()

    return 0.5 * (_directional(logits, positive) + _directional(logits.t(), positive.t()))


def _chem_feature_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = "auto") -> torch.Tensor:
    """
    Use BCE for strictly binary targets and MSE for continuous features.
    NaN/Inf targets are masked out so partially populated feature vectors are safe.
    """
    target = target.to(device=pred.device, dtype=pred.dtype)
    valid = torch.isfinite(target)
    if not bool(valid.any()):
        return pred.sum() * 0.0

    target_safe = torch.where(valid, target, torch.zeros_like(target))
    binary = bool(((target_safe[valid] == 0) | (target_safe[valid] == 1)).all())
    loss_type = str(loss_type or "auto").lower()
    if loss_type == "bce" or (loss_type == "auto" and binary):
        loss = F.binary_cross_entropy_with_logits(pred, target_safe, reduction="none")
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(pred, target_safe, reduction="none")
    else:
        loss = F.mse_loss(pred, target_safe, reduction="none")
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    return loss.sum() / valid.to(dtype=loss.dtype).sum().clamp_min(1.0)


def _masked_bce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    valid = torch.isfinite(target) & (mask > 0)
    target = torch.where(valid, target, torch.zeros_like(target))
    positives = ((target > 0.5) & valid).sum().to(logits.dtype)
    negatives = valid.sum().to(logits.dtype) - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight
    )
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    return loss.sum() / valid.to(dtype=loss.dtype).sum().clamp_min(1.0)


def _masked_multilabel_bce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    mask = mask.to(device=logits.device, dtype=logits.dtype).unsqueeze(-1)
    valid = torch.isfinite(target) & (mask > 0)
    target = torch.where(valid, target, torch.zeros_like(target))
    positives = ((target > 0.5) & valid).sum(dim=(0, 1)).to(logits.dtype)
    negatives = valid.sum(dim=(0, 1)).to(logits.dtype) - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
    pos_weight = torch.where(valid.any(dim=(0, 1)), pos_weight, torch.ones_like(pos_weight))
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight
    )
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    return loss.sum() / valid.to(dtype=loss.dtype).sum().clamp_min(1.0)


def _masked_regression_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_type: str = "smooth_l1") -> torch.Tensor:
    target = target.to(device=pred.device, dtype=pred.dtype)
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    valid = torch.isfinite(target) & (mask > 0)
    target = torch.where(valid, target, torch.zeros_like(target))
    if loss_type == "mse":
        loss = F.mse_loss(pred, target, reduction="none")
    else:
        loss = F.smooth_l1_loss(pred, target, reduction="none")
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    return loss.sum() / valid.to(dtype=loss.dtype).sum().clamp_min(1.0)


def _masked_auto_binary_or_regression_loss(logits_or_pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target_f = target.to(device=logits_or_pred.device, dtype=logits_or_pred.dtype)
    valid_values = target_f[torch.isfinite(target_f)]
    if valid_values.numel() > 0 and bool(((valid_values == 0) | (valid_values == 1)).all()):
        return _masked_bce_loss(logits_or_pred, target, mask)
    return _masked_regression_loss(logits_or_pred, target, mask, loss_type="smooth_l1")


def _masked_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=torch.long)
    if not bool(target.ne(-100).any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100)


def _balanced_structure_contact_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pair_mask: torch.Tensor,
    negative_ratio: float,
) -> torch.Tensor:
    """Balanced BCE for sparse, long-range residue-contact maps."""
    target = target.to(device=logits.device, dtype=logits.dtype)
    valid = pair_mask.to(device=logits.device, dtype=torch.bool)
    positive = valid & (target > 0.5)
    negative = valid & ~positive
    positive_count = int(positive.sum().item())
    if positive_count == 0:
        return logits.sum() * 0.0
    max_negative = max(1, int(round(float(negative_ratio) * positive_count)))
    negative_indices = negative.nonzero(as_tuple=False)
    if negative_indices.size(0) > max_negative:
        keep = torch.randperm(negative_indices.size(0), device=logits.device)[:max_negative]
        negative = torch.zeros_like(negative)
        chosen = negative_indices[keep]
        negative[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
    selected = positive | negative
    return F.binary_cross_entropy_with_logits(logits[selected], target[selected])


def _hard_negative_loss(z: torch.Tensor, hard_negative_indices: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if z.size(0) < 2:
        return z.sum() * 0.0
    idx = hard_negative_indices.to(device=z.device, dtype=torch.long)
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    z = F.normalize(z, dim=-1)
    sim = z @ z.t()
    losses = []
    for i in range(idx.size(0)):
        valid = idx[i]
        valid = valid[(valid >= 0) & (valid < z.size(0)) & (valid != i)]
        if valid.numel() > 0:
            losses.append(F.softplus(sim[i, valid] / float(temperature)).mean())
    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()


def _interaction_query_diversity_loss(query_memory: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Penalize off-diagonal query cosine similarity with a tiny optional weight."""
    if query_memory.dim() != 3 or query_memory.size(1) < 2:
        zero = query_memory.sum() * 0.0
        return zero, {
            "query_offdiag_cosine_mean": zero.detach(),
            "query_offdiag_cosine_max": zero.detach(),
        }
    q = F.normalize(query_memory, dim=-1)
    sim = torch.matmul(q, q.transpose(1, 2))
    k = sim.size(1)
    eye = torch.eye(k, device=sim.device, dtype=torch.bool).unsqueeze(0)
    off = sim.masked_select(~eye.expand_as(sim))
    loss = off.pow(2).mean()
    return loss, {
        "query_offdiag_cosine_mean": off.mean().detach(),
        "query_offdiag_cosine_max": off.abs().max().detach(),
    }


# -----------------------------
# Main top-level model
# -----------------------------
class DockingWizardModel(nn.Module):
    def __init__(self, cfg: DockingWizardModelConfig, lig3d_encoder: Optional[nn.Module] = None, lig3d_ckpt: Optional[str] = None):
        super().__init__()
        self.cfg = cfg

        # ---- build perceiver cfg if absent ----
        cfg.encoder_cfg.chem_requirement_dim = int(cfg.n_chem_features)
        cfg.encoder_cfg.pocket_interaction_types = int(cfg.n_interaction_types)
        if cfg.perceiver_cfg is None:
            pcfg = PerceiverModuleConfig(
                d_ctx=cfg.encoder_cfg.d_ctx,
                k_dock=cfg.encoder_cfg.dock_final_tokens,
                k_lig=cfg.encoder_cfg.ligand_view_tokens,
                k_pocket=cfg.encoder_cfg.pocket_final_tokens,
                k_func=cfg.encoder_cfg.func_final_tokens,
            )
        else:
            pcfg = cfg.perceiver_cfg
        pcfg.layerscale_init = float(cfg.perceiver_layerscale_init)
        pcfg.cross_every_block = bool(cfg.perceiver_cross_every_block)
        pcfg.enable_ligand_chemical_memory = bool(cfg.enable_ligand_chemical_memory)
        pcfg.enable_pocket_interaction_memory = bool(cfg.enable_pocket_interaction_memory)
        pcfg.enable_interaction_type_queries = bool(cfg.enable_interaction_type_queries)
        pcfg.enable_biological_bridge = bool(cfg.enable_biological_bridge)
        pcfg.n_interaction_types = int(cfg.n_interaction_types)
        pcfg.n_chem_features = int(cfg.n_chem_features)

        # ---- instantiate submodules ----
        self.encoder = PrePerceiverConditionEncoder(cfg.encoder_cfg, lig3d_encoder=lig3d_encoder, lig3d_ckpt=lig3d_ckpt)
        self.perceiver = DockingWizardPerceiver(pcfg)
        self.decoder = ProGenConditionedDecoder(
            backbone_name_or_path=cfg.decoder_backbone_name_or_path,
            d_ctx=pcfg.d_ctx,
            add_boundary_tokens=cfg.decoder_add_boundary_tokens,
            enable_full_injection=cfg.decoder_enable_full_injection,
            inject_n_head=cfg.decoder_inject_n_head,
            inject_dropout=cfg.decoder_inject_dropout,
            freeze_backbone=cfg.decoder_freeze_backbone,
            trust_remote_code=cfg.decoder_trust_remote_code,
            local_files_only=cfg.decoder_local_files_only,
            revision=cfg.decoder_revision,
            n_interaction_types=cfg.n_interaction_types,
            full_injection_layer_policy=cfg.full_injection_layer_policy,
            full_injection_last_k=cfg.full_injection_last_k,
            share_injection_weights=cfg.share_full_injection_weights,
            unfreeze_last_n_blocks=cfg.decoder_unfreeze_last_n_blocks,
            prefix_boundary_scale=cfg.decoder_prefix_boundary_scale,
        )

        self.perceiver_cfg = pcfg

        self.route_comp_head_dock = nn.Sequential(
            nn.LayerNorm(pcfg.d_ctx),
            nn.Linear(pcfg.d_ctx, cfg.route_comp_proj_dim, bias=False),
        )
        self.route_comp_head_lig = nn.Sequential(
            nn.LayerNorm(pcfg.d_ctx),
            nn.Linear(pcfg.d_ctx, cfg.route_comp_proj_dim, bias=False),
        )
        self.chem_feature_head = nn.Sequential(
            nn.LayerNorm(pcfg.d_ctx),
            nn.Linear(pcfg.d_ctx, cfg.n_chem_features),
        )
        self.complementarity_proj = nn.Linear(cfg.n_interaction_types, cfg.n_chem_features)
        self.condition_contrastive_head = nn.Sequential(
            nn.LayerNorm(pcfg.d_ctx),
            nn.Linear(pcfg.d_ctx, cfg.route_comp_proj_dim, bias=False),
        )
        self.protein_contrastive_head = nn.Sequential(
            nn.LayerNorm(self.decoder.d_model),
            nn.Linear(self.decoder.d_model, cfg.route_comp_proj_dim, bias=False),
        )
        self.protein_length_head = nn.Sequential(
            nn.LayerNorm(pcfg.d_ctx),
            nn.Linear(pcfg.d_ctx, max(32, pcfg.d_ctx // 4)),
            nn.GELU(),
            nn.Linear(max(32, pcfg.d_ctx // 4), 1),
        )
        self.fold_planner = (
            ConditionFoldPlanner(
                d_model=pcfg.d_ctx,
                num_slots=int(cfg.fold_plan_slots),
                num_prototypes=int(cfg.fold_plan_prototypes),
                num_archetypes=int(cfg.fold_plan_archetypes),
                num_layers=int(cfg.fold_plan_layers),
                num_heads=int(cfg.fold_plan_heads),
                prototype_temperature=float(cfg.fold_plan_prototype_temperature),
                local_chemistry_scale=float(cfg.fold_plan_local_chemistry_scale),
                archetype_scale=float(cfg.fold_plan_archetype_scale),
                dropout=float(cfg.fold_plan_dropout),
            )
            if cfg.enable_condition_fold_planner else None
        )
        self.structure_token_planner = (
            StructureTokenScaffoldPlanner(
                d_model=pcfg.d_ctx,
                num_slots=int(cfg.structure_token_slots),
                num_codes=int(cfg.structure_token_codes),
                prefix_slots=int(cfg.structure_token_prefix_slots),
                num_layers=int(cfg.structure_token_layers),
                num_heads=int(cfg.structure_token_heads),
                dropout=float(cfg.structure_token_dropout),
                posterior_temperature=float(cfg.structure_token_temperature),
                posterior_teacher_forcing=float(
                    cfg.structure_token_posterior_teacher_forcing
                ),
                num_interaction_types=int(cfg.n_interaction_types),
                geometry_feature_dim=int(cfg.structure_token_geometry_feature_dim),
            )
            if cfg.enable_structure_token_planner else None
        )
        contact_dim = max(16, int(cfg.structure_contact_proj_dim))
        self.structure_contact_left = nn.Sequential(
            nn.LayerNorm(self.decoder.d_model),
            nn.Linear(self.decoder.d_model, contact_dim),
            nn.GELU(),
            nn.Linear(contact_dim, contact_dim, bias=False),
        )
        self.structure_contact_right = nn.Sequential(
            nn.LayerNorm(self.decoder.d_model),
            nn.Linear(self.decoder.d_model, contact_dim),
            nn.GELU(),
            nn.Linear(contact_dim, contact_dim, bias=False),
        )
        bank_size = max(0, int(cfg.ligand_memory_bank_size))
        self.register_buffer("_ligand_memory_bank", torch.zeros(bank_size, pcfg.d_ctx), persistent=False)
        self.register_buffer("_ligand_memory_bank_count", torch.zeros((), dtype=torch.long), persistent=False)
        self._ligand_memory_bank_ptr = 0

        # ---- strict assertions: prevent silent segment bugs ----
        self._assert_config_alignment()

        # cache token ids defaults (best-effort)
        self._init_special_token_ids()

        # lazy tokenizers for raw inference (InstructPro-like)
        self._tok_pubmed = None
        self._tok_smiles = None
        self._tok_protein = None
        self._hf_cache_root = None 


    # -------------------------
    # Config alignment
    # -------------------------
    def _assert_config_alignment(self) -> None:
        ec = self.cfg.encoder_cfg
        pc = self.perceiver_cfg
        # token budgets
        assert pc.k_dock == ec.dock_final_tokens, f"k_dock mismatch: perceiver={pc.k_dock} vs encoder={ec.dock_final_tokens}"
        assert pc.k_lig == ec.ligand_view_tokens, f"k_lig mismatch: perceiver={pc.k_lig} vs encoder={ec.ligand_view_tokens}"
        assert pc.k_pocket == ec.pocket_final_tokens, f"k_pocket mismatch: perceiver={pc.k_pocket} vs encoder={ec.pocket_final_tokens}"
        assert pc.k_func == ec.func_final_tokens, f"k_func mismatch: perceiver={pc.k_func} vs encoder={ec.func_final_tokens}"
        # d_ctx
        assert pc.d_ctx == ec.d_ctx, f"d_ctx mismatch: perceiver={pc.d_ctx} vs encoder={ec.d_ctx}"
        assert int(self.decoder.d_ctx) == int(pc.d_ctx), f"decoder.d_ctx mismatch: {self.decoder.d_ctx} vs perceiver={pc.d_ctx}"

    def _init_special_token_ids(self) -> None:
        # ProGen models sometimes have missing/odd token ids; we provide robust fallbacks.
        cfg = getattr(self.decoder.backbone, "config", None)
        bos = getattr(cfg, "bos_token_id", None) if cfg is not None else None
        eos = getattr(cfg, "eos_token_id", None) if cfg is not None else None
        pad = getattr(cfg, "pad_token_id", None) if cfg is not None else None

        # User-config overrides
        if self.cfg.gen_bos_token_id is not None:
            bos = self.cfg.gen_bos_token_id
        if self.cfg.gen_eos_token_id is not None:
            eos = self.cfg.gen_eos_token_id
        if self.cfg.gen_pad_token_id is not None:
            pad = self.cfg.gen_pad_token_id

        # ProGen's GPT2 tokenizer/config commonly maps both bos/eos to
        # <|endoftext|>. The protein vocabulary instead uses explicit IDs
        # 1=<|bos|>, 2=<|eos|>, 0=<|pad|>, matching train.encode_protein.
        if bos is not None and eos is not None and int(bos) == int(eos):
            bos, eos = 1, 2
            if pad is None or int(pad) == int(getattr(cfg, "eos_token_id", eos)):
                pad = 0

        # Fallbacks (ProGen2 commonly uses 1 as BOS)
        if bos is None:
            bos = 1
        if eos is None:
            # Some protein LMs use 2 or 0; safest fallback is 2 if vocab allows.
            eos = 2
        if pad is None:
            # If no pad, reuse eos.
            pad = eos

        self.bos_token_id = int(bos)
        self.eos_token_id = int(eos)
        self.pad_token_id = int(pad)

    def _aux_warmup_factor(self, global_step: Optional[int] = None, epoch: Optional[int] = None) -> float:
        mode = str(self.cfg.aux_loss_warmup_mode or "none").lower()
        if mode == "none":
            return 1.0
        progress = 1.0
        if self.cfg.aux_loss_warmup_steps and global_step is not None:
            progress = min(1.0, max(0.0, float(global_step) / float(max(1, self.cfg.aux_loss_warmup_steps))))
        elif self.cfg.aux_loss_warmup_epochs and epoch is not None:
            progress = min(1.0, max(0.0, float(epoch + 1) / float(max(1, self.cfg.aux_loss_warmup_epochs))))
        if mode == "cosine":
            return float(0.5 - 0.5 * math.cos(math.pi * progress))
        return float(progress)

    def _effective_aux_lambdas(self, global_step: Optional[int] = None, epoch: Optional[int] = None) -> Dict[str, float]:
        warm = self._aux_warmup_factor(global_step=global_step, epoch=epoch)
        route_lambda = float(self.cfg.lambda_route_comp or self.cfg.lambda_route_contrastive)
        return {
            "aux_ligand_contrastive": float(self.cfg.lambda_aux_enc),
            "route_contrastive": warm * route_lambda,
            "chem_align": warm * float(self.cfg.lambda_chem_align),
            "interaction_query_diversity": warm * float(self.cfg.lambda_query_diversity),
            "hard_negative_contrastive": warm * float(self.cfg.hard_negative_weight),
            "condition_contrastive": warm * float(self.cfg.lambda_condition_contrastive),
            "length_prediction": warm * float(self.cfg.lambda_length_prediction),
            "loss_pocket": warm * float(self.cfg.lambda_pocket),
            "loss_contact": warm * float(self.cfg.lambda_contact),
            "structure_contacts": warm * float(self.cfg.lambda_structure_contacts),
            "loss_interaction": warm * float(self.cfg.lambda_interaction),
            "loss_secondary_structure": warm * float(self.cfg.lambda_secondary_structure),
            "loss_solvent_accessibility": warm * float(self.cfg.lambda_solvent_accessibility),
            "loss_residue_depth": warm * float(self.cfg.lambda_residue_depth),
            "complementarity": warm * float(self.cfg.lambda_complementarity),
            "bridge_gate_l1": float(self.cfg.lambda_bridge_gate_l1),
            "diversity": float(self.cfg.lambda_diversity),
            "local_composition_unlikelihood": warm * float(
                self.cfg.lambda_local_composition_unlikelihood
            ),
            "local_group_unlikelihood": warm * float(
                self.cfg.lambda_local_group_unlikelihood
            ),
            "fold_plan_ss": warm * float(self.cfg.lambda_fold_plan_ss),
            "fold_plan_contacts": warm * float(self.cfg.lambda_fold_plan_contacts),
            "fold_plan_global": warm * float(self.cfg.lambda_fold_plan_global),
            "fold_plan_codebook": warm * float(self.cfg.lambda_fold_plan_codebook),
            "fold_plan_chemistry": warm * float(self.cfg.lambda_fold_plan_chemistry),
            "fold_plan_sequence": warm * float(self.cfg.lambda_fold_plan_sequence),
            "fold_plan_pocket": warm * float(self.cfg.lambda_fold_plan_pocket),
            "fold_plan_archetype": warm * float(self.cfg.lambda_fold_plan_archetype),
            "structure_token_prior": warm * float(self.cfg.lambda_structure_token_prior),
            "structure_token_posterior": warm * float(self.cfg.lambda_structure_token_posterior),
            "structure_token_kl": warm * float(self.cfg.lambda_structure_token_kl),
            "structure_token_pocket": warm * float(self.cfg.lambda_structure_token_pocket),
            "structure_token_interaction": warm * float(self.cfg.lambda_structure_token_interaction),
            "structure_token_topology": warm * float(self.cfg.lambda_structure_token_topology),
        }

    def plan_fold(
        self,
        per_out: Dict[str, torch.Tensor],
        sample_prototypes: Optional[bool] = None,
        hard_prototypes: Optional[bool] = None,
    ) -> Optional[FoldPlannerOutput]:
        """Predict a fold plan strictly from condition-side memories."""
        if self.fold_planner is None:
            return None
        structure_memories = [
            per_out.get(name, None)
            for name in ("mem_func", "mem_docking")
        ]
        chemistry_memories = [
            per_out.get(name, None)
            for name in ("mem_ligand_chem", "mem_pocket")
        ]
        structure_memories = [memory for memory in structure_memories if memory is not None]
        chemistry_memories = [memory for memory in chemistry_memories if memory is not None]
        if not structure_memories and not chemistry_memories:
            return None
        if not structure_memories:
            structure_memories = list(chemistry_memories)
        if not chemistry_memories:
            chemistry_memories = list(structure_memories)
        if self.training and float(self.cfg.fold_plan_condition_dropout) > 0:
            # Modality dropout prevents the small training set from letting one
            # text/ligand route become a brittle surrogate retrieval key.
            for memories in (structure_memories, chemistry_memories):
                if len(memories) > 1 and random.random() < float(self.cfg.fold_plan_condition_dropout):
                    drop_index = random.randrange(len(memories))
                    memories[drop_index] = torch.zeros_like(memories[drop_index])
        if sample_prototypes is None:
            sample_prototypes = bool(
                self.training and self.cfg.fold_plan_sample_during_training
            )
        if hard_prototypes is None:
            hard_prototypes = bool(
                (not self.training) and self.cfg.fold_plan_hard_during_generation
            )
        return self.fold_planner(
            torch.cat(structure_memories, dim=1),
            chemistry_memory=torch.cat(chemistry_memories, dim=1),
            sample_prototypes=bool(sample_prototypes),
            hard_prototypes=bool(hard_prototypes),
        )

    def _fold_plan_injection_factor(self, global_step: Optional[int]) -> float:
        if not self.training or global_step is None:
            return 1.0
        warmup = max(0, int(self.cfg.fold_plan_injection_warmup_steps))
        if warmup == 0:
            return 1.0
        return min(1.0, max(0.0, float(global_step) / float(warmup)))

    def plan_structure_tokens(
        self,
        per_out: Dict[str, torch.Tensor],
        target_token_ids: Optional[torch.Tensor] = None,
        sample_prior: Optional[bool] = None,
        hard_prior: Optional[bool] = None,
        global_step: Optional[int] = None,
    ) -> Optional[StructureTokenPlannerOutput]:
        """Plan a scaffold from conditions, optionally using a train-only posterior."""
        if self.structure_token_planner is None:
            return None
        memories = [
            per_out.get(name, None)
            for name in (
                "mem_func",
                "mem_docking",
                "mem_ligand_chem",
                "mem_pocket",
            )
        ]
        memories = [memory for memory in memories if memory is not None]
        if not memories:
            return None
        if sample_prior is None:
            sample_prior = bool(
                (not self.training)
                and self.cfg.structure_token_sample_during_generation
            )
        if hard_prior is None:
            hard_prior = bool(
                (not self.training)
                and self.cfg.structure_token_hard_during_generation
            )
        posterior_mix = float(self.cfg.structure_token_posterior_teacher_forcing)
        decay_steps = max(0, int(self.cfg.structure_token_posterior_decay_steps))
        if self.training and global_step is not None and decay_steps > 0:
            progress = min(1.0, max(0.0, float(global_step) / float(decay_steps)))
            initial = float(self.cfg.structure_token_posterior_teacher_forcing)
            final = float(self.cfg.structure_token_posterior_final_mix)
            posterior_mix = initial + progress * (final - initial)
        return self.structure_token_planner(
            torch.cat(memories, dim=1),
            target_token_ids=(target_token_ids if self.training else None),
            sample_prior=bool(sample_prior),
            hard_prior=bool(hard_prior),
            posterior_mix=posterior_mix,
        )

    def predict_protein_length(self, per_out: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Return a per-sample residue-count estimate from Perceiver memories."""
        memories = [
            mem for mem in (
                per_out.get("mem_func", None),
                per_out.get("mem_docking", None),
                per_out.get("mem_ligand_chem", None),
                per_out.get("mem_pocket", None),
            )
            if mem is not None
        ]
        if not memories:
            return None
        condition_repr = torch.cat(memories, dim=1).mean(dim=1)
        normalized = torch.sigmoid(self.protein_length_head(condition_repr).squeeze(-1))
        return normalized * max(1.0, float(self.cfg.protein_length_scale))

    @torch.no_grad()
    def _update_ligand_memory_bank(self, lig_repr: torch.Tensor) -> None:
        if self._ligand_memory_bank.numel() == 0:
            return
        z = F.normalize(lig_repr.detach(), dim=-1)
        bank_size = int(self._ligand_memory_bank.size(0))
        n = min(z.size(0), bank_size)
        ptr = int(self._ligand_memory_bank_ptr)
        end = ptr + n
        if end <= bank_size:
            self._ligand_memory_bank[ptr:end].copy_(z[:n])
        else:
            first = bank_size - ptr
            self._ligand_memory_bank[ptr:].copy_(z[:first])
            self._ligand_memory_bank[:end - bank_size].copy_(z[first:n])
        self._ligand_memory_bank_ptr = (ptr + n) % bank_size
        self._ligand_memory_bank_count.fill_(min(bank_size, int(self._ligand_memory_bank_count.item()) + n))

    # -------------------------
    # Frozen backbone behavior
    # -------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if not self.cfg.keep_frozen_backbones_in_eval:
            return self

        # Encoder backbones
        if getattr(self.cfg.encoder_cfg, "freeze_backbones", False):
            try:
                self.encoder.bert_dock.model.eval()
                self.encoder.bert_func.model.eval()
                self.encoder.bert_ltxt.model.eval()
                self.encoder.smiles_lm.model.eval()
                self.encoder.lig3d.eval()
            except Exception:
                # best-effort
                pass

        # Decoder backbone
        if self.cfg.decoder_freeze_backbone:
            try:
                self.decoder.backbone.eval()
            except Exception:
                pass
        return self

    # -------------------------
    # Batch validation
    # -------------------------
    @staticmethod
    def _require(batch: Dict[str, torch.Tensor], keys: Iterable[str]) -> None:
        missing = [k for k in keys if k not in batch]
        if missing:
            raise KeyError(f"Batch is missing required keys: {missing}")

    def assert_batch_sanity(self, batch: Dict[str, torch.Tensor]) -> None:
        """Validate presence and basic mask correctness to avoid NaNs."""
        self._require(batch, [
            "docking_input_ids", "docking_attention_mask",
            "function_input_ids", "function_attention_mask",
            "ligand_text_input_ids", "ligand_text_attention_mask",
            "smiles_input_ids", "smiles_attention_mask",
            "lig3d_Z", "lig3d_coords", "lig3d_atom_mask",
        ])

        # masks should have at least one valid token/atom per sample (best-effort)
        for mkey in [
            "docking_attention_mask", "function_attention_mask", "ligand_text_attention_mask", "smiles_attention_mask"
        ]:
            m = _as_bool_mask(batch[mkey])
            if not bool(m.any(dim=1).all()):
                # Not a hard error (you might intentionally have empty), but warn via exception message
                # to force the caller to decide.
                raise ValueError(f"Some samples have all-pad for {mkey}. This can cause NaNs in pooling/attention.")

        am = _as_bool_mask(batch["lig3d_atom_mask"])
        if not bool(am.any(dim=1).all()):
            raise ValueError("Some samples have no valid ligand atoms (lig3d_atom_mask all zero).")

        # -------------------------
    # Raw inference helpers (InstructPro-like)
    # -------------------------
    def _set_hf_cache_root(self, hf_cache_root: Optional[str]) -> None:
        """Force HF caches to a user-specified directory (so downloads don't go to ~/.cache)."""
        if not hf_cache_root:
            return
        self._hf_cache_root = hf_cache_root
        os.makedirs(hf_cache_root, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache_root
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_root, "transformers")
        os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache_root, "hub")
        os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_root, "datasets")

    def _ensure_tokenizers(self, hf_cache_root: Optional[str] = None):
        """Lazy-load tokenizers used by encoder-side text and smiles, plus protein decode helper."""
        self._set_hf_cache_root(hf_cache_root)

        # local import to avoid hard dependency at import time
        from transformers import AutoTokenizer

        cache_dir = hf_cache_root or self._hf_cache_root

        if self._tok_pubmed is None:
            PUB_MED_BERT_DIR = os.environ.get(
                "PUB_MED_BERT_DIR", self.cfg.encoder_cfg.pubmedbert_name
            )
            self._tok_pubmed = AutoTokenizer.from_pretrained(
                PUB_MED_BERT_DIR,
                cache_dir=cache_dir,
                local_files_only=self.cfg.decoder_local_files_only,
                use_fast=True,
                trust_remote_code=True,
            )

        if self._tok_smiles is None:
            SMILES_MODEL_DIR = os.environ.get(
                "SMILES_MODEL_DIR", self.cfg.encoder_cfg.smiles_lm_name
            )
            self._tok_smiles = AutoTokenizer.from_pretrained(
                SMILES_MODEL_DIR,
                cache_dir=cache_dir,
                local_files_only=self.cfg.decoder_local_files_only,
                use_fast=True,
                trust_remote_code=True,
            )

        # protein tokenizer is only needed for decoding (generation prompt uses bos_token_id)
        if self._tok_protein is None:
            name = self.cfg.decoder_backbone_name_or_path
            try:
                PROGEN_MODEL_DIR = os.environ.get("PROGEN_MODEL_DIR", name)
                self._tok_protein = AutoTokenizer.from_pretrained(
                    PROGEN_MODEL_DIR,
                    cache_dir=cache_dir,
                    local_files_only=self.cfg.decoder_local_files_only,
                    use_fast=True,
                    revision=self.cfg.decoder_revision,
                    trust_remote_code=True
                )
            except Exception:
                # fallback: allow running without a tokenizer (you can still get token ids)
                self._tok_protein = None

        return self._tok_pubmed, self._tok_smiles, self._tok_protein

    @staticmethod
    def _to_list(x):
        if x is None:
            return None
        return x if isinstance(x, (list, tuple)) else [x]

    @staticmethod
    def _pad_lig3d_batch(Z_list, C_list, device, dtype_coords=torch.float32):
        """Pad variable-length ligand 3D to (B, max_atoms) / (B, max_atoms, 3)."""
        B = len(Z_list)
        max_atoms = max(int(z.shape[0]) for z in Z_list)
        Zb = torch.zeros((B, max_atoms), dtype=torch.long, device=device)
        Cb = torch.zeros((B, max_atoms, 3), dtype=dtype_coords, device=device)
        Mb = torch.zeros((B, max_atoms), dtype=torch.long, device=device)
        for i, (z, c) in enumerate(zip(Z_list, C_list)):
            n = int(z.shape[0])
            Zb[i, :n] = torch.as_tensor(z, dtype=torch.long, device=device)
            Cb[i, :n, :] = torch.as_tensor(c, dtype=dtype_coords, device=device)
            Mb[i, :n] = 1
        return Zb, Cb, Mb

    @staticmethod
    def _try_smiles_to_3d(smiles: str):
        """
        Best-effort RDKit 3D. If RDKit not available or fails, return None.
        Returns (Z: np.int64[n], coords: np.float32[n,3]) or None.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except Exception:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)
            # ETKDG + UFF optimize
            ps = AllChem.ETKDGv3()
            ok = AllChem.EmbedMolecule(mol, ps)
            if ok != 0:
                return None
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            conf = mol.GetConformer()
            zs = []
            coords = []
            for a in mol.GetAtoms():
                zs.append(a.GetAtomicNum())
            for i in range(mol.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                coords.append([p.x, p.y, p.z])
            Z = np.asarray(zs, dtype=np.int64)
            C = np.asarray(coords, dtype=np.float32)
            return Z, C
        except Exception:
            return None

    def prepare_cond_batch_from_raw(
        self,
        instruction: Union[str, List[str]],
        smiles: Union[str, List[str]],
        docking_text: Optional[Union[str, List[str]]] = None,
        function_text: Optional[Union[str, List[str]]] = None,
        ligand_text: Optional[Union[str, List[str]]] = None,
        lig3d: Optional[Dict[str, Union[np.ndarray, torch.Tensor, List]]] = None,
        hf_cache_root: Optional[str] = None,
        device: Optional[torch.device] = None,
        dummy_lig3d_if_missing: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Build a cond_batch that matches assert_batch_sanity() requirements,
        using only raw natural language + SMILES (InstructPro-like).
        """
        device = device or next(self.parameters()).device

        instruction_l = self._to_list(instruction)
        smiles_l = self._to_list(smiles)
        if len(instruction_l) != len(smiles_l):
            raise ValueError(f"Batch size mismatch: instruction={len(instruction_l)} vs smiles={len(smiles_l)}")

        # default mapping: use the same natural language as docking + function (you can override)
        docking_l = self._to_list(docking_text) if docking_text is not None else instruction_l
        function_l = self._to_list(function_text) if function_text is not None else instruction_l
        ligand_text_l = self._to_list(ligand_text) if ligand_text is not None else [""] * len(instruction_l)

        tok_pubmed, tok_smiles, _tok_protein = self._ensure_tokenizers(hf_cache_root)

        # tokenize long text WITHOUT truncation; padding to batch max only
        dock_tok = tok_pubmed(
            docking_l, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True
        )
        func_tok = tok_pubmed(
            function_l, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True
        )
        ligtext_tok = tok_pubmed(
            ligand_text_l, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True
        )
        smi_tok = tok_smiles(
            smiles_l, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True
        )

        # ligand 3D
        Z_list = []
        C_list = []
        if lig3d is not None:
            # accept provided lig3d (already per-sample)
            Z_in = lig3d.get("Z", None)
            C_in = lig3d.get("coords", None)
            if Z_in is None or C_in is None:
                raise ValueError("lig3d provided but missing keys: require {'Z', 'coords'}")
            Z_in = Z_in if isinstance(Z_in, (list, tuple)) else [Z_in]
            C_in = C_in if isinstance(C_in, (list, tuple)) else [C_in]
            if len(Z_in) != len(smiles_l):
                raise ValueError(f"lig3d batch mismatch: {len(Z_in)} vs {len(smiles_l)}")
            for z, c in zip(Z_in, C_in):
                z = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
                c = c.detach().cpu().numpy() if torch.is_tensor(c) else np.asarray(c)
                Z_list.append(z.astype(np.int64))
                C_list.append(c.astype(np.float32))
        else:
            # RDKit best-effort; fallback to dummy 1-atom if missing
            for s in smiles_l:
                out = self._try_smiles_to_3d(s)
                if out is None:
                    if not dummy_lig3d_if_missing:
                        raise RuntimeError("RDKit 3D failed (or RDKit missing) and dummy_lig3d_if_missing=False")
                    Z_list.append(np.asarray([6], dtype=np.int64))          # carbon
                    C_list.append(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
                else:
                    z, c = out
                    Z_list.append(z)
                    C_list.append(c)

        lig3d_Z, lig3d_coords, lig3d_atom_mask = self._pad_lig3d_batch(Z_list, C_list, device=device)

        cond_batch = {
            "docking_input_ids": dock_tok["input_ids"].to(device=device, dtype=torch.long),
            "docking_attention_mask": dock_tok["attention_mask"].to(device=device, dtype=torch.long),
            "function_input_ids": func_tok["input_ids"].to(device=device, dtype=torch.long),
            "function_attention_mask": func_tok["attention_mask"].to(device=device, dtype=torch.long),
            "ligand_text_input_ids": ligtext_tok["input_ids"].to(device=device, dtype=torch.long),
            "ligand_text_attention_mask": ligtext_tok["attention_mask"].to(device=device, dtype=torch.long),
            "smiles_input_ids": smi_tok["input_ids"].to(device=device, dtype=torch.long),
            "smiles_attention_mask": smi_tok["attention_mask"].to(device=device, dtype=torch.long),
            "lig3d_Z": lig3d_Z,
            "lig3d_coords": lig3d_coords,
            "lig3d_atom_mask": lig3d_atom_mask,
        }

        # will raise if missing / all-pad / no atoms
        self.assert_batch_sanity(cond_batch)
        return cond_batch

    def _decoder_input_embedding_layer(self) -> nn.Module:
        """
        ProGen sometimes doesn't implement get_input_embeddings() on base_model;
        use robust fallbacks so generate() never crashes.
        """
        bb = self.decoder.backbone
        try:
            emb = bb.get_input_embeddings()
            if emb is not None:
                return emb
        except NotImplementedError:
            pass
        except Exception:
            pass

        # common fallbacks
        for path in ["transformer.wte", "wte", "model.embed_tokens", "embed_tokens"]:
            obj = bb
            ok = True
            for p in path.split("."):
                if not hasattr(obj, p):
                    ok = False
                    break
                obj = getattr(obj, p)
            if ok and isinstance(obj, nn.Module):
                return obj

        raise RuntimeError("Cannot resolve input embedding layer for decoder backbone.")

    @torch.no_grad()
    def generate_from_raw(
        self,
        instruction: Union[str, List[str]],
        smiles: Union[str, List[str]],
        docking_text: Optional[Union[str, List[str]]] = None,
        function_text: Optional[Union[str, List[str]]] = None,
        ligand_text: Optional[Union[str, List[str]]] = None,
        lig3d: Optional[Dict[str, Union[np.ndarray, torch.Tensor, List]]] = None,
        hf_cache_root: Optional[str] = None,
        return_decoded: bool = True,
        **gen_kwargs,
    ):
        """
        InstructPro-like API:
          (natural language instruction + SMILES) -> protein generation

        It internally builds cond_batch and calls self.generate(...).
        """
        cond_batch = self.prepare_cond_batch_from_raw(
            instruction=instruction,
            smiles=smiles,
            docking_text=docking_text,
            function_text=function_text,
            ligand_text=ligand_text,
            lig3d=lig3d,
            hf_cache_root=hf_cache_root,
        )
        out_ids = self.generate(cond_batch=cond_batch, **gen_kwargs)

        if not return_decoded:
            return out_ids

        # decode best-effort
        _, _, tok_protein = self._ensure_tokenizers(hf_cache_root)
        if tok_protein is None:
            return out_ids  # fallback: return ids if tokenizer unavailable

        if hasattr(tok_protein, "batch_decode"):
            texts = tok_protein.batch_decode(out_ids, skip_special_tokens=True)
        else:
            # very rare fallback
            texts = [tok_protein.decode(x.tolist()) for x in out_ids]

        # ProGen-like tokenizers may insert spaces
        texts = [t.replace(" ", "").replace("\n", "") for t in texts]
        return texts


    # -------------------------
    # Forward
    # -------------------------
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_debug: bool = False,
        compute_aux_losses: Optional[bool] = None,
        global_step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> DockingWizardOutput:

        # Validate required condition fields
        self.assert_batch_sanity(batch)

        # Protein side
        self._require(batch, ["protein_input_ids", "protein_attention_mask"])
        protein_input_ids = batch["protein_input_ids"]
        protein_attention_mask = _as_long_mask(batch["protein_attention_mask"])

        # labels
        if "protein_labels" in batch and batch["protein_labels"] is not None:
            protein_labels = batch["protein_labels"].to(dtype=torch.long)
        else:
            # Default: labels == input_ids with pad=-100
            protein_labels = protein_input_ids.to(dtype=torch.long)
            protein_labels = _masked_fill_tokens(protein_labels, protein_attention_mask, fill=-100)

        position_weights = None
        if "position_weights" in batch and batch["position_weights"] is not None:
            position_weights = _align_residue_values_to_labels(
                batch["position_weights"],
                protein_labels,
                self.eos_token_id,
                eos_value=1.0,
            )
        elif self.cfg.use_bio_position_weights:
            # Base weight 1 for every supervised target, including EOS. Residue
            # annotations are shifted by one to account for the leading BOS.
            bio = protein_labels.ne(-100).to(dtype=torch.float32)
            if "conservation_scores" in batch and batch["conservation_scores"] is not None:
                bio = bio + float(self.cfg.alpha_conservation) * _align_residue_values_to_labels(
                    batch["conservation_scores"], protein_labels, self.eos_token_id
                )
            if "pocket_labels" in batch and batch["pocket_labels"] is not None:
                bio = bio + float(self.cfg.alpha_pocket) * _align_residue_values_to_labels(
                    batch["pocket_labels"], protein_labels, self.eos_token_id
                )
            if "ligand_contact_labels" in batch and batch["ligand_contact_labels"] is not None:
                bio = bio + float(self.cfg.alpha_contact) * _align_residue_values_to_labels(
                    batch["ligand_contact_labels"], protein_labels, self.eos_token_id
                )
            position_weights = bio.clamp_min(0.0)

        # Encoder aux behavior
        if compute_aux_losses is None:
            compute_aux_losses = self.training
        if self.cfg.encoder_force_compute_aux:
            compute_aux_losses_enc = True
        else:
            compute_aux_losses_enc = bool(compute_aux_losses)

        # ---- Encoder ----
        enc_out = self.encoder(
            docking_input_ids=batch["docking_input_ids"],
            docking_attention_mask=_as_long_mask(batch["docking_attention_mask"]),
            function_input_ids=batch["function_input_ids"],
            function_attention_mask=_as_long_mask(batch["function_attention_mask"]),
            ligand_text_input_ids=batch["ligand_text_input_ids"],
            ligand_text_attention_mask=_as_long_mask(batch["ligand_text_attention_mask"]),
            smiles_input_ids=batch["smiles_input_ids"],
            smiles_attention_mask=_as_long_mask(batch["smiles_attention_mask"]),
            lig3d_Z=batch["lig3d_Z"],
            lig3d_coords=batch["lig3d_coords"],
            lig3d_atom_mask=_as_long_mask(batch["lig3d_atom_mask"]),
            compute_aux_losses=compute_aux_losses_enc,
            ligand_group_ids=batch.get("ligand_group_ids", None),
        )
        ctx_tokens = enc_out["ctx_tokens"]
        ctx_mask = _as_long_mask(enc_out["ctx_mask"])

        chem_targets = None
        if "chem_features" in batch and batch["chem_features"] is not None:
            chem_targets = batch["chem_features"]
        elif "ligand_chem_features" in batch and batch["ligand_chem_features"] is not None:
            chem_targets = batch["ligand_chem_features"]

        # ---- Perceiver ----
        need_gate = (self.training and self.cfg.lambda_bridge_gate_l1 > 0)
        per_kwargs = {"return_debug": (return_debug or need_gate)}
        try:
            per_sig = inspect.signature(self.perceiver.forward)
            if chem_targets is not None and "chem_features" in per_sig.parameters:
                per_kwargs["chem_features"] = chem_targets
        except (TypeError, ValueError):
            pass
        per_out = self.perceiver(ctx_tokens, ctx_mask, **per_kwargs)
        mem_func = per_out["mem_func"]
        mem_dl_light = per_out["mem_dl_light"]
        mem_dl_full = per_out["mem_dl_full"]
        mem_pocket = per_out.get("mem_pocket", None)
        mem_ligand_chem_for_dec = per_out.get("mem_ligand_chem", None)
        interaction_embeds = per_out.get("interaction_embeds", None)
        fold_plan_out = self.plan_fold(per_out)
        fold_plan_tokens = fold_plan_out.tokens if fold_plan_out is not None else None
        structure_plan_out = self.plan_structure_tokens(
            per_out,
            target_token_ids=batch.get("structure_token_targets", None),
            sample_prior=(False if not self.training else None),
            hard_prior=(False if not self.training else None),
            global_step=global_step,
        )
        fold_plan_injection_factor = self._fold_plan_injection_factor(global_step)
        prefix_plan_parts = []
        full_plan_parts = []
        if fold_plan_tokens is not None and float(self.cfg.fold_plan_prefix_scale) != 0.0:
            prefix_plan_parts.append(
                fold_plan_tokens
                * float(self.cfg.fold_plan_prefix_scale)
                * fold_plan_injection_factor
            )
        if fold_plan_tokens is not None and float(self.cfg.fold_plan_full_memory_scale) != 0.0:
            full_plan_parts.append(
                fold_plan_tokens
                * float(self.cfg.fold_plan_full_memory_scale)
                * fold_plan_injection_factor
            )
        if structure_plan_out is not None:
            structure_route_factor = 1.0
            if self.training and self.cfg.structure_token_train_scale_choices:
                structure_route_factor = float(
                    random.choice(self.cfg.structure_token_train_scale_choices)
                )
            if float(self.cfg.structure_token_prefix_scale) != 0.0:
                prefix_plan_parts.append(
                    structure_plan_out.prefix_tokens
                    * float(self.cfg.structure_token_prefix_scale)
                    * structure_route_factor
                )
            if float(self.cfg.structure_token_full_memory_scale) != 0.0:
                full_plan_parts.append(
                    structure_plan_out.tokens
                    * float(self.cfg.structure_token_full_memory_scale)
                    * structure_route_factor
                )
        plan_prefix_tokens = (
            torch.cat(prefix_plan_parts, dim=1) if prefix_plan_parts else None
        )
        plan_full_tokens = (
            torch.cat(full_plan_parts, dim=1) if full_plan_parts else None
        )

        # ---- Condition-awareness regularizer ----
        # With small probability during training, run Perceiver again with near-empty
        # docking to check whether docking-influenced memories (mem_pocket) differ from
        # their ablated counterparts. If they are too similar, the model is ignoring
        # docking text and we penalize.
        cond_awareness_loss = None
        if self.training and self.cfg.lambda_condition_awareness > 0 and random.random() < self.cfg.cond_awareness_prob:
            try:
                k_dock = int(self.perceiver.cfg.k_dock)
                ctx_ablated = ctx_tokens.clone()
                k_lig = int(self.perceiver.cfg.k_lig)
                k_pocket = int(self.perceiver.cfg.k_pocket)
                ctx_ablated[:, :k_dock, :] = 0.0
                pocket_start = k_dock + k_lig
                ctx_ablated[:, pocket_start:pocket_start + k_pocket, :] = 0.0
                per_abl_kwargs = {k: v for k, v in per_kwargs.items() if k != "return_debug"}
                with torch.no_grad():
                    per_abl_out = self.perceiver(ctx_ablated, ctx_mask, **per_abl_kwargs)
                mem_pocket_ablated = per_abl_out.get("mem_pocket", None)
                mem_docking_ablated = per_abl_out.get("mem_docking", None)
                similarity_terms = []
                for full_mem, ablated_mem in (
                    (mem_pocket, mem_pocket_ablated),
                    (per_out.get("mem_docking", None), mem_docking_ablated),
                ):
                    if full_mem is None or ablated_mem is None:
                        continue
                    p_full = F.normalize(full_mem.mean(dim=1), dim=-1)
                    p_abl = F.normalize(ablated_mem.detach().mean(dim=1), dim=-1)
                    similarity_terms.append((p_full * p_abl).sum(dim=-1).clamp(-1.0, 1.0).mean())
                if similarity_terms:
                    cos_sim = torch.stack(similarity_terms).mean()
                    cond_awareness_loss = F.relu(
                        cos_sim - float(self.cfg.condition_awareness_max_similarity)
                    )
            except Exception:
                cond_awareness_loss = None

        # ---- Decoder ----
        use_full = bool(self.cfg.enable_full_injection_train) and (mem_dl_full is not None)
        full_injection_stop_after = None
        full_injection_scale = float(self.cfg.full_injection_eval_scale)
        prefix_condition_scale = 1.0
        if use_full and self.training:
            full_injection_scale = 1.0
            if self.cfg.full_injection_train_stop_choices:
                selected_stop = random.choice(self.cfg.full_injection_train_stop_choices)
                full_injection_stop_after = None if int(selected_stop) < 0 else int(selected_stop)
            if self.cfg.full_injection_train_scale_choices:
                full_injection_scale = float(random.choice(self.cfg.full_injection_train_scale_choices))
        if self.training and self.cfg.prefix_condition_train_scale_choices:
            prefix_condition_scale = float(random.choice(self.cfg.prefix_condition_train_scale_choices))
        mem_dl_full_for_dec = mem_dl_full
        if mem_dl_full_for_dec is not None and plan_full_tokens is not None:
            mem_dl_full_for_dec = torch.cat(
                [mem_dl_full_for_dec, plan_full_tokens],
                dim=1,
            )
        dec_out: DecoderOutput = self.decoder(
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
            mem_func=mem_func,
            mem_dl_light=mem_dl_light,
            mem_fold_plan=plan_prefix_tokens,
            mem_dl_full=mem_dl_full_for_dec if use_full else None,
            mem_pocket=mem_pocket if use_full else None,
            mem_ligand_chem=mem_ligand_chem_for_dec if use_full else None,
            interaction_embeds=interaction_embeds if use_full else None,
            labels=protein_labels,
            position_weights=position_weights,
            pocket_gate_signal=batch.get("pocket_labels", None) if self.cfg.enable_pocket_aware_injection else None,
            pocket_gate_strength=float(self.cfg.pocket_gate_strength) if self.cfg.enable_pocket_aware_injection else 0.0,
            use_cache=False,
            eos_weight=float(self.cfg.eos_loss_weight),
            eos_token_id=int(self.eos_token_id),
            full_injection_stop_after=full_injection_stop_after,
            full_injection_scale=full_injection_scale,
            prefix_condition_scale=prefix_condition_scale,
            fold_plan_scale=1.0,
            pool_condition_memory=bool(self.cfg.pool_condition_memory),
        )

        loss_lm = dec_out.loss
        losses_aux: Dict[str, torch.Tensor] = {}

        if (
            fold_plan_out is not None
            and "secondary_structure_labels" in batch
            and "structure_contact_map" in batch
            and "structure_contact_mask" in batch
        ):
            losses_aux.update(
                fold_planner_losses(
                    fold_plan_out,
                    batch["secondary_structure_labels"],
                    batch["structure_contact_map"],
                    batch["structure_contact_mask"],
                    slot_residue_targets=batch.get("slot_residue_composition", None),
                    slot_pocket_targets=batch.get("slot_pocket_targets", None),
                    archetype_targets=batch.get("fold_archetype_id", None),
                    biochemical_prior=batch.get("biochemical_residue_prior", None),
                    biochemical_strength=batch.get("biochemical_prior_strength", None),
                )
            )

        if structure_plan_out is not None and "structure_token_targets" in batch:
            losses_aux.update(
                structure_token_planner_losses(
                    structure_plan_out,
                    batch["structure_token_targets"],
                    token_mask=batch.get("structure_token_mask", None),
                    pocket_targets=batch.get("structure_pocket_slot_targets", None),
                    interaction_targets=batch.get(
                        "structure_interaction_slot_targets", None
                    ),
                    topology_targets=batch.get("structure_slot_contact_map", None),
                    topology_mask=batch.get("structure_slot_contact_mask", None),
                )
            )

        # All values in losses_aux are raw, unweighted components. DockingWizardModel
        # is the single place where lambda_* weights are applied to loss_total.
        if "aux_ligand_contrastive" in enc_out:
            losses_aux["aux_ligand_contrastive"] = enc_out["aux_ligand_contrastive"]
            losses_aux["ligand_contrastive"] = enc_out["aux_ligand_contrastive"]

        route_contrastive_logits = None
        if "mem_docking" in per_out and "mem_ligand_chem" in per_out:
            mem_docking = per_out["mem_docking"]
            mem_ligand_chem = per_out["mem_ligand_chem"]
            dock_repr = mem_docking.mean(dim=1)
            lig_repr = mem_ligand_chem.mean(dim=1)
            if mem_docking.size(0) >= 2 and mem_ligand_chem.size(0) >= 2:
                dock_proj = self.route_comp_head_dock(dock_repr)
                lig_proj = self.route_comp_head_lig(lig_repr)
                losses_aux["route_contrastive"] = _clip_infonce(
                    dock_proj,
                    lig_proj,
                    temperature=float(self.cfg.route_comp_temperature),
                    group_ids=batch.get("ligand_group_ids", None),
                )
                route_contrastive_logits = (
                    F.normalize(dock_proj, dim=-1)
                    @ F.normalize(lig_proj, dim=-1).t()
                ) / float(self.cfg.route_comp_temperature)
            else:
                losses_aux["route_contrastive"] = mem_dl_full.sum() * 0.0

            if self.cfg.use_hard_negative_contrastive and "hard_negative_indices" in batch:
                losses_aux["hard_negative_contrastive"] = _hard_negative_loss(
                    lig_repr,
                    batch["hard_negative_indices"],
                    temperature=float(self.cfg.ligand_contrastive_temperature),
                )
            if self._ligand_memory_bank.numel() > 0:
                count = int(self._ligand_memory_bank_count.item())
                if count > 0:
                    bank = self._ligand_memory_bank[:count].to(device=lig_repr.device, dtype=lig_repr.dtype)
                    sim_bank = F.normalize(lig_repr, dim=-1) @ bank.t()
                    losses_aux["ligand_memory_bank_neg"] = F.softplus(sim_bank / float(self.cfg.ligand_contrastive_temperature)).mean()
                self._update_ligand_memory_bank(lig_repr)

        query_diversity_debug = None
        if self.cfg.enable_interaction_type_queries and "interaction_query_memory" in per_out:
            q_loss, q_debug = _interaction_query_diversity_loss(per_out["interaction_query_memory"])
            losses_aux["interaction_query_diversity"] = q_loss
            query_diversity_debug = q_debug

        condition_memories = [
            mem for mem in (
                per_out.get("mem_func", None),
                per_out.get("mem_docking", None),
                per_out.get("mem_ligand_chem", None),
                per_out.get("mem_pocket", None),
            )
            if mem is not None
        ]
        condition_repr = (
            torch.cat(condition_memories, dim=1).mean(dim=1)
            if condition_memories else None
        )

        # Align the complete multimodal condition with the target protein. The
        # target uses frozen ProGen token embeddings, so this objective cannot
        # be minimized by changing the protein representation itself.
        if (
            self.cfg.lambda_condition_contrastive > 0
            and protein_input_ids.size(0) >= 2
            and condition_repr is not None
        ):
            positive_mask = None
            protein_groups = batch.get("protein_group_ids", None)
            condition_groups = batch.get("condition_group_ids", None)
            if protein_groups is not None or condition_groups is not None:
                positive_mask = torch.eye(
                    protein_input_ids.size(0),
                    dtype=torch.bool,
                    device=protein_input_ids.device,
                )
                for groups in (protein_groups, condition_groups):
                    if groups is None:
                        continue
                    ids = groups.to(device=protein_input_ids.device).view(-1)
                    positive_mask = positive_mask | ids[:, None].eq(ids[None, :])
            with torch.no_grad():
                protein_embeds = self.decoder.input_embeddings(protein_input_ids)
                # ``protein_labels`` already masks the first control/BOS token
                # and padding.  Using it keeps this correct for ProGen's
                # documented "1" control tag (id 3), not only <|bos|> (id 1).
                protein_mask = (
                    protein_labels.ne(-100)
                    & protein_input_ids.ne(int(self.eos_token_id))
                )
                denom = protein_mask.sum(dim=1, keepdim=True).clamp_min(1)
                protein_repr = (
                    protein_embeds * protein_mask.unsqueeze(-1).to(protein_embeds.dtype)
                ).sum(dim=1) / denom.to(protein_embeds.dtype)
            losses_aux["condition_contrastive"] = _clip_infonce(
                self.condition_contrastive_head(condition_repr),
                self.protein_contrastive_head(protein_repr),
                temperature=float(self.cfg.condition_contrastive_temperature),
                positive_mask=positive_mask,
            )

        # Predict target residue count from conditions.  This gives generation
        # a sample-specific EOS prior instead of relying on one fixed minimum
        # length for both 40-aa membrane proteins and 140-aa soluble proteins.
        if self.cfg.lambda_length_prediction > 0 and condition_repr is not None:
            length_scale = max(1.0, float(self.cfg.protein_length_scale))
            predicted_norm = torch.sigmoid(self.protein_length_head(condition_repr).squeeze(-1))
            target_lengths = (
                protein_labels.ne(-100).sum(dim=1).to(predicted_norm.dtype) - 1.0
            ).clamp_min(1.0)
            target_norm = (target_lengths / length_scale).clamp(0.0, 1.0)
            losses_aux["length_prediction"] = F.smooth_l1_loss(
                predicted_norm,
                target_norm,
            )

        if chem_targets is not None and "mem_ligand_chem" in per_out:
            chem_targets = chem_targets.to(dtype=torch.float32, device=mem_dl_full.device)
            chem_pred = enc_out.get("chem_requirements", None)
            if chem_pred is None or chem_pred.shape != chem_targets.shape:
                chem_pred = self.chem_feature_head(per_out["mem_ligand_chem"].mean(dim=1))
            if chem_targets.shape != chem_pred.shape:
                raise ValueError(
                    f"chem feature shape mismatch: target={tuple(chem_targets.shape)} "
                    f"pred={tuple(chem_pred.shape)}. Set n_chem_features to match the dataset."
                )
            losses_aux["chem_align"] = _chem_feature_loss(chem_pred, chem_targets, loss_type=self.cfg.chem_loss_type)

        # Optional bridge gate regularization
        if self.cfg.lambda_bridge_gate_l1 > 0 and "debug" in per_out:
            if "bridge_gate" in per_out["debug"]:
                gate = per_out["debug"]["bridge_gate"]
                losses_aux["bridge_gate_l1"] = gate.mean().abs()
            elif "biological_bridge_gate" in per_out["debug"]:
                gates = per_out["debug"]["biological_bridge_gate"]
                gate_vals = [v.abs() for v in gates.values() if torch.is_tensor(v)]
                if gate_vals:
                    losses_aux["bridge_gate_l1"] = torch.stack(gate_vals).mean()

        if cond_awareness_loss is not None:
            losses_aux["condition_awareness"] = cond_awareness_loss

        # Diversity regularization on protein-token positions
        if self.cfg.lambda_diversity > 0 and dec_out.logits is not None:
            # dec_out.logits includes prefix positions at the left; labels are protein-only
            pfx = int(dec_out.prefix_len)
            prot_logits = dec_out.logits[:, pfx:, :]
            # align length defensively
            min_len = min(prot_logits.size(1), protein_labels.size(1))
            if min_len > 0:
                div_loss = _sequence_diversity_loss(prot_logits[:, :min_len, :], protein_labels[:, :min_len])
                losses_aux["diversity"] = div_loss
        if (
            self.cfg.lambda_local_composition_unlikelihood > 0
            and dec_out.logits is not None
        ):
            losses_aux["local_composition_unlikelihood"] = (
                _local_composition_unlikelihood_loss(
                    dec_out.logits,
                    protein_labels,
                    prefix_len=int(dec_out.prefix_len),
                    window=int(self.cfg.local_composition_window),
                    min_context=int(self.cfg.local_composition_min_context),
                    dominance_threshold=float(
                        self.cfg.local_composition_dominance_threshold
                    ),
                    special_token_ids=(
                        self.pad_token_id,
                        self.bos_token_id,
                        self.eos_token_id,
                    ),
                )
            )
        if self.cfg.lambda_local_group_unlikelihood > 0 and dec_out.logits is not None:
            losses_aux["local_group_unlikelihood"] = _local_group_unlikelihood_loss(
                dec_out.logits,
                protein_labels,
                prefix_len=int(dec_out.prefix_len),
                window=int(self.cfg.local_composition_window),
                min_context=int(self.cfg.local_composition_min_context),
                token_groups=self.cfg.local_residue_token_groups,
                group_thresholds=self.cfg.local_residue_group_thresholds,
            )

        pocket_logits = None
        interaction_logits = None
        prot_hidden = dec_out.protein_hidden_states
        prot_mask = protein_attention_mask.to(dtype=torch.float32)
        if prot_hidden is not None:
            if "pocket_labels" in batch and batch["pocket_labels"] is not None:
                pocket_logits = self.decoder.pocket_head(prot_hidden).squeeze(-1)
                losses_aux["loss_pocket"] = _masked_bce_loss(pocket_logits, batch["pocket_labels"], prot_mask)
            elif self.cfg.enable_pocket_aware_injection:
                pocket_logits = self.decoder.pocket_head(prot_hidden).squeeze(-1)

            if "ligand_contact_labels" in batch and batch["ligand_contact_labels"] is not None:
                contact_logits = self.decoder.contact_head(prot_hidden).squeeze(-1)
                losses_aux["loss_contact"] = _masked_bce_loss(contact_logits, batch["ligand_contact_labels"], prot_mask)

            if "interaction_labels" in batch and batch["interaction_labels"] is not None:
                interaction_logits = self.decoder.interaction_head(prot_hidden)
                losses_aux["loss_interaction"] = _masked_multilabel_bce_loss(
                    interaction_logits,
                    batch["interaction_labels"],
                    prot_mask,
                )
            elif self.cfg.lambda_complementarity > 0:
                interaction_logits = self.decoder.interaction_head(prot_hidden)

            ss_target = None
            if "secondary_structure_labels" in batch and batch["secondary_structure_labels"] is not None:
                ss_target = batch["secondary_structure_labels"]
            elif "dssp" in batch and batch["dssp"] is not None:
                ss_target = batch["dssp"]
            if ss_target is not None:
                ss_logits = self.decoder.secondary_structure_head(prot_hidden)
                losses_aux["loss_secondary_structure"] = _masked_ce_loss(ss_logits, ss_target)

            if (
                "structure_contact_map" in batch
                and "structure_contact_mask" in batch
                and batch["structure_contact_map"] is not None
                and batch["structure_contact_mask"] is not None
            ):
                left = self.structure_contact_left(prot_hidden)
                right = self.structure_contact_right(prot_hidden)
                contact_logits = 0.5 * (
                    torch.matmul(left, right.transpose(1, 2))
                    + torch.matmul(right, left.transpose(1, 2))
                ) / math.sqrt(float(left.size(-1)))
                losses_aux["structure_contacts"] = _balanced_structure_contact_loss(
                    contact_logits,
                    batch["structure_contact_map"],
                    batch["structure_contact_mask"],
                    negative_ratio=float(self.cfg.structure_contact_negative_ratio),
                )

            solvent_target = None
            if "solvent_accessibility" in batch and batch["solvent_accessibility"] is not None:
                solvent_target = batch["solvent_accessibility"]
            elif "rsa" in batch and batch["rsa"] is not None:
                solvent_target = batch["rsa"]
            if solvent_target is not None:
                solvent_pred = self.decoder.solvent_accessibility_head(prot_hidden).squeeze(-1)
                losses_aux["loss_solvent_accessibility"] = _masked_auto_binary_or_regression_loss(
                    solvent_pred,
                    solvent_target,
                    prot_mask,
                )

            depth_target = None
            if "residue_depth" in batch and batch["residue_depth"] is not None:
                depth_target = batch["residue_depth"]
            elif "burial_depth" in batch and batch["burial_depth"] is not None:
                depth_target = batch["burial_depth"]
            if depth_target is not None:
                depth_pred = self.decoder.residue_depth_head(prot_hidden).squeeze(-1)
                losses_aux["loss_residue_depth"] = _masked_regression_loss(
                    depth_pred,
                    depth_target,
                    prot_mask,
                    loss_type="smooth_l1",
                )

            if chem_targets is not None and interaction_logits is not None:
                if pocket_logits is None:
                    pocket_logits = self.decoder.pocket_head(prot_hidden).squeeze(-1)
                pocket_prob = torch.sigmoid(pocket_logits).detach()
                weights = pocket_prob * prot_mask.to(device=pocket_prob.device, dtype=pocket_prob.dtype)
                denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                pocket_response = (interaction_logits * weights.unsqueeze(-1)).sum(dim=1) / denom
                response = self.complementarity_proj(pocket_response)
                target = chem_targets.to(device=response.device, dtype=response.dtype)
                valid = torch.isfinite(target)
                target = torch.where(valid, target, torch.zeros_like(target))
                if self.cfg.complementarity_loss_type == "bce":
                    comp_loss = F.binary_cross_entropy_with_logits(response, target, reduction="none")
                else:
                    comp_loss = F.mse_loss(response, target, reduction="none")
                comp_loss = torch.where(valid, comp_loss, torch.zeros_like(comp_loss))
                losses_aux["complementarity"] = comp_loss.sum() / valid.to(dtype=comp_loss.dtype).sum().clamp_min(1.0)

        # Total loss
        loss_total = None
        effective_lambdas = self._effective_aux_lambdas(global_step=global_step, epoch=epoch)
        if loss_lm is not None:
            loss_total = loss_lm
            if "aux_ligand_contrastive" in losses_aux:
                loss_total = loss_total + effective_lambdas["aux_ligand_contrastive"] * losses_aux["aux_ligand_contrastive"]
            if "bridge_gate_l1" in losses_aux:
                loss_total = loss_total + effective_lambdas["bridge_gate_l1"] * losses_aux["bridge_gate_l1"]
            if "condition_awareness" in losses_aux:
                loss_total = loss_total + self.cfg.lambda_condition_awareness * losses_aux["condition_awareness"]
            if "diversity" in losses_aux:
                loss_total = loss_total + effective_lambdas["diversity"] * losses_aux["diversity"]
            if "local_composition_unlikelihood" in losses_aux:
                loss_total = (
                    loss_total
                    + effective_lambdas["local_composition_unlikelihood"]
                    * losses_aux["local_composition_unlikelihood"]
                )
            if "local_group_unlikelihood" in losses_aux:
                loss_total = (
                    loss_total
                    + effective_lambdas["local_group_unlikelihood"]
                    * losses_aux["local_group_unlikelihood"]
                )
            if "route_contrastive" in losses_aux:
                loss_total = loss_total + effective_lambdas["route_contrastive"] * losses_aux["route_contrastive"]
            if "condition_contrastive" in losses_aux:
                loss_total = loss_total + effective_lambdas["condition_contrastive"] * losses_aux["condition_contrastive"]
            if "length_prediction" in losses_aux:
                loss_total = loss_total + effective_lambdas["length_prediction"] * losses_aux["length_prediction"]
            if "chem_align" in losses_aux:
                loss_total = loss_total + effective_lambdas["chem_align"] * losses_aux["chem_align"]
            if "interaction_query_diversity" in losses_aux:
                loss_total = loss_total + effective_lambdas["interaction_query_diversity"] * losses_aux["interaction_query_diversity"]
            if "hard_negative_contrastive" in losses_aux:
                loss_total = loss_total + effective_lambdas["hard_negative_contrastive"] * losses_aux["hard_negative_contrastive"]
            if "ligand_memory_bank_neg" in losses_aux:
                loss_total = loss_total + effective_lambdas["hard_negative_contrastive"] * losses_aux["ligand_memory_bank_neg"]
            for aux_name in [
                "fold_plan_ss",
                "fold_plan_contacts",
                "fold_plan_global",
                "fold_plan_codebook",
                "fold_plan_chemistry",
                "fold_plan_sequence",
                "fold_plan_pocket",
                "fold_plan_archetype",
                "structure_token_prior",
                "structure_token_posterior",
                "structure_token_kl",
                "structure_token_pocket",
                "structure_token_interaction",
                "structure_token_topology",
                "loss_pocket",
                "loss_contact",
                "structure_contacts",
                "loss_interaction",
                "loss_secondary_structure",
                "loss_solvent_accessibility",
                "loss_residue_depth",
                "complementarity",
            ]:
                if aux_name in losses_aux:
                    weight = float(effective_lambdas[aux_name])
                    if weight != 0.0:
                        loss_total = loss_total + weight * losses_aux[aux_name]

        debug: Dict[str, Any] = {}
        if return_debug:
            debug["enc"] = {
                "ligand_view_weights": enc_out.get("ligand_view_weights", None),
                "ctx_mask_sum": ctx_mask.sum(dim=1),
            }
            debug["per"] = per_out.get("debug", {})
            if route_contrastive_logits is not None:
                debug["route_contrastive_logits"] = route_contrastive_logits
            if query_diversity_debug is not None:
                debug["query_diversity_score"] = losses_aux.get("interaction_query_diversity", None)
                debug["query_offdiag_cosine_mean"] = query_diversity_debug["query_offdiag_cosine_mean"]
                debug["query_offdiag_cosine_max"] = query_diversity_debug["query_offdiag_cosine_max"]
            debug["effective_lambdas"] = effective_lambdas
            if fold_plan_out is not None:
                debug["fold_plan"] = {
                    "tokens_shape": tuple(fold_plan_out.tokens.shape),
                    "prototype_usage": fold_plan_out.prototype_assignments.mean(dim=(0, 1)),
                    "predicted_global_stats": torch.sigmoid(fold_plan_out.global_stats),
                    "pocket_slot_probabilities": torch.softmax(
                        fold_plan_out.pocket_slot_logits, dim=-1
                    ),
                    "archetype_probabilities": torch.softmax(
                        fold_plan_out.archetype_logits, dim=-1
                    ),
                }
            if structure_plan_out is not None:
                debug["structure_token_plan"] = {
                    "tokens_shape": tuple(structure_plan_out.tokens.shape),
                    "posterior_used": bool(structure_plan_out.posterior_used),
                    "prior_entropy": -(
                        torch.softmax(structure_plan_out.prior_logits, dim=-1)
                        * torch.log_softmax(structure_plan_out.prior_logits, dim=-1)
                    ).sum(dim=-1).mean(),
                    "pocket_probabilities": torch.sigmoid(
                        structure_plan_out.pocket_logits
                    ),
                }
            debug["dec"] = {
                "prefix_len": dec_out.prefix_len,
                "protein_hidden_shape": tuple(dec_out.protein_hidden_states.shape) if dec_out.protein_hidden_states is not None else None,
                "full_injection_layer_indices": dec_out.full_injection_layer_indices,
            }
            if pocket_logits is not None:
                debug["pocket_probabilities"] = torch.sigmoid(pocket_logits)
            if interaction_logits is not None:
                debug["interaction_probabilities"] = torch.sigmoid(interaction_logits)

        return DockingWizardOutput(
            loss_total=loss_total,
            loss_lm=loss_lm,
            loss_aux=losses_aux,
            prefix_len=dec_out.prefix_len,
            logits=dec_out.logits,
            protein_hidden_states=dec_out.protein_hidden_states,
            fold_plan=fold_plan_out,
            debug=debug,
        )

    # -------------------------
    # Optimizer param groups
    # -------------------------
    def get_param_groups(
        self,
        lr_encoder: float = 5e-5,
        lr_perceiver: float = 1e-4,
        lr_decoder_new: float = 1e-4,
        lr_decoder_backbone: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """Return parameter groups with different learning rates.

        Best practice:
          - encoder adapter/pooling/calibration layers: smaller lr
          - perceiver: medium lr
          - decoder new layers (proj/boundary/injectors): medium lr
          - decoder backbone (if unfrozen): tiny lr
        """

        def _named_params(module: nn.Module):
            for n, p in module.named_parameters():
                if p.requires_grad:
                    yield n, p

        enc_params: List[nn.Parameter] = []
        per_params: List[nn.Parameter] = []
        dec_new_params: List[nn.Parameter] = []
        dec_backbone_params: List[nn.Parameter] = []

        # Encoder: exclude frozen backbones by name
        for n, p in _named_params(self.encoder):
            if any(n.startswith(prefix) for prefix in ["bert_dock.", "bert_func.", "bert_ltxt.", "smiles_lm.", "lig3d."]):
                # frozen backbone (should not require_grad anyway)
                continue
            enc_params.append(p)

        # Perceiver
        for _, p in _named_params(self.perceiver):
            per_params.append(p)

        # Decoder
        for n, p in _named_params(self.decoder):
            if n.startswith("backbone."):
                dec_backbone_params.append(p)
            else:
                dec_new_params.append(p)

        groups: List[Dict[str, Any]] = []

        if enc_params:
            groups.append({"params": enc_params, "lr": lr_encoder, "weight_decay": weight_decay})
        if per_params:
            groups.append({"params": per_params, "lr": lr_perceiver, "weight_decay": weight_decay})
        if dec_new_params:
            groups.append({"params": dec_new_params, "lr": lr_decoder_new, "weight_decay": weight_decay})
        if dec_backbone_params:
            groups.append({"params": dec_backbone_params, "lr": lr_decoder_backbone, "weight_decay": weight_decay})

        return groups

    # -------------------------
    # Generation
    # -------------------------
    @torch.no_grad()
    def generate(
        self,
        cond_batch: Dict[str, torch.Tensor],
        prompt_input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        return_full_sequence: bool = True,
        enable_full_injection: Optional[bool] = None,
        use_hf_generate_when_possible: bool = False,
        disable_prefix: bool = False,
    ) -> torch.Tensor:
        """Generate protein sequences conditioned on (function+docking+ligand).

        For best fidelity with full-memory injection, this method defaults to a custom
        autoregressive loop (no KV cache) because HF generate often computes only the
        last token per step when use_cache=True, which can bypass position-based injection.

        Args:
          cond_batch: same fields as forward() condition side, tensors on device
          prompt_input_ids: optional initial protein prompt (e.g., BOS). If None, uses bos_token_id.
          return_full_sequence: if True, return prompt+generated ids; else return only newly generated.
          enable_full_injection: override cfg.enable_full_injection_generate.
          use_hf_generate_when_possible: if True and full injection disabled, call HF generate.
        """

        # ============================================================
        # Baseline generation: pure ProGen2 (NO prefix, NO injection)
        # ============================================================
        if disable_prefix:
            self.eval()

            # make sure runtime buffers are clean
            if hasattr(self.decoder, "_rt_prefix_len"):
                self.decoder._rt_prefix_len = None
            if hasattr(self.decoder, "_rt_mem_full"):
                self.decoder._rt_mem_full = None
            if hasattr(self.decoder, "_rt_start_pos"):
                self.decoder._rt_start_pos = None

            device = next(self.parameters()).device

            if prompt_input_ids is None:
                prompt_input_ids = torch.tensor([[self.bos_token_id]], device=device, dtype=torch.long)
            else:
                prompt_input_ids = prompt_input_ids.to(device)

            # ---- patch: make ProGenConfig compatible with transformers DynamicCache ----
            cfg = getattr(self.decoder.backbone, "config", None)
            if cfg is not None and not hasattr(cfg, "num_hidden_layers"):
                # common names in GPT-like configs / remote-code configs
                for k in ["n_layer", "num_layers", "n_layers", "layers", "num_hidden_layer"]:
                    if hasattr(cfg, k):
                        setattr(cfg, "num_hidden_layers", int(getattr(cfg, k)))
                        break
                # final fallback: try to infer from model modules (rarely needed)
                if not hasattr(cfg, "num_hidden_layers"):
                    try:
                        n = len(getattr(self.decoder.backbone, "transformer").h)
                        setattr(cfg, "num_hidden_layers", int(n))
                    except Exception:
                        pass

            # attention mask (avoid warnings / more stable generation)
            attention_mask = torch.ones_like(prompt_input_ids, device=prompt_input_ids.device)

            gen_kwargs = dict(
                input_ids=prompt_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )

            # only pass sampling params when do_sample=True (otherwise transformers warns/ignores)
            if bool(do_sample):
                gen_kwargs.update(dict(
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                ))

            gen_ids = self.decoder.backbone.generate(**gen_kwargs)
            return gen_ids



        self.encoder.eval()
        self.perceiver.eval()
        self.decoder.eval()

        max_new_tokens = int(max_new_tokens or self.cfg.gen_max_new_tokens)
        temperature = float(temperature if temperature is not None else self.cfg.gen_temperature)
        top_p = float(top_p if top_p is not None else self.cfg.gen_top_p)
        top_k = int(top_k if top_k is not None else self.cfg.gen_top_k)
        do_sample = bool(do_sample if do_sample is not None else self.cfg.gen_do_sample)
        repetition_penalty = float(repetition_penalty if repetition_penalty is not None else self.cfg.gen_repetition_penalty)

        eos_token_id = int(eos_token_id if eos_token_id is not None else self.eos_token_id)
        pad_token_id = int(pad_token_id if pad_token_id is not None else self.pad_token_id)

        if enable_full_injection is None:
            enable_full_injection = bool(self.cfg.enable_full_injection_generate)

        # Validate condition batch
        self.assert_batch_sanity(cond_batch)

        # Encode condition -> memories
        enc_out = self.encoder(
            docking_input_ids=cond_batch["docking_input_ids"],
            docking_attention_mask=_as_long_mask(cond_batch["docking_attention_mask"]),
            function_input_ids=cond_batch["function_input_ids"],
            function_attention_mask=_as_long_mask(cond_batch["function_attention_mask"]),
            ligand_text_input_ids=cond_batch["ligand_text_input_ids"],
            ligand_text_attention_mask=_as_long_mask(cond_batch["ligand_text_attention_mask"]),
            smiles_input_ids=cond_batch["smiles_input_ids"],
            smiles_attention_mask=_as_long_mask(cond_batch["smiles_attention_mask"]),
            lig3d_Z=cond_batch["lig3d_Z"],
            lig3d_coords=cond_batch["lig3d_coords"],
            lig3d_atom_mask=_as_long_mask(cond_batch["lig3d_atom_mask"]),
            compute_aux_losses=True,
        )
        ctx_tokens = enc_out["ctx_tokens"]
        ctx_mask = _as_long_mask(enc_out["ctx_mask"])

        chem_targets = None
        if "chem_features" in cond_batch and cond_batch["chem_features"] is not None:
            chem_targets = cond_batch["chem_features"]
        elif "ligand_chem_features" in cond_batch and cond_batch["ligand_chem_features"] is not None:
            chem_targets = cond_batch["ligand_chem_features"]
        per_kwargs = {"return_debug": False}
        try:
            per_sig = inspect.signature(self.perceiver.forward)
            if chem_targets is not None and "chem_features" in per_sig.parameters:
                per_kwargs["chem_features"] = chem_targets
        except (TypeError, ValueError):
            pass
        per_out = self.perceiver(ctx_tokens, ctx_mask, **per_kwargs)
        mem_func = per_out["mem_func"]
        mem_dl_light = per_out["mem_dl_light"]
        mem_dl_full = per_out["mem_dl_full"] if enable_full_injection else None
        mem_pocket = per_out.get("mem_pocket", None) if enable_full_injection else None
        mem_ligand_chem_for_dec = per_out.get("mem_ligand_chem", None) if enable_full_injection else None
        interaction_embeds = per_out.get("interaction_embeds", None) if enable_full_injection else None
        fold_plan_out = self.plan_fold(
            per_out,
            sample_prototypes=bool(self.cfg.fold_plan_sample_during_generation),
            hard_prototypes=bool(self.cfg.fold_plan_hard_during_generation),
        )
        fold_plan_tokens = fold_plan_out.tokens if fold_plan_out is not None else None
        structure_plan_out = self.plan_structure_tokens(
            per_out,
            sample_prior=bool(self.cfg.structure_token_sample_during_generation),
            hard_prior=bool(self.cfg.structure_token_hard_during_generation),
        )
        prefix_plan_parts = []
        full_plan_parts = []
        if fold_plan_tokens is not None and float(self.cfg.fold_plan_prefix_scale) != 0.0:
            prefix_plan_parts.append(
                fold_plan_tokens * float(self.cfg.fold_plan_prefix_scale)
            )
        if fold_plan_tokens is not None and float(self.cfg.fold_plan_full_memory_scale) != 0.0:
            full_plan_parts.append(
                fold_plan_tokens * float(self.cfg.fold_plan_full_memory_scale)
            )
        if structure_plan_out is not None:
            if float(self.cfg.structure_token_prefix_scale) != 0.0:
                prefix_plan_parts.append(
                    structure_plan_out.prefix_tokens
                    * float(self.cfg.structure_token_prefix_scale)
                )
            if float(self.cfg.structure_token_full_memory_scale) != 0.0:
                full_plan_parts.append(
                    structure_plan_out.tokens
                    * float(self.cfg.structure_token_full_memory_scale)
                )
        plan_prefix_tokens = (
            torch.cat(prefix_plan_parts, dim=1) if prefix_plan_parts else None
        )
        plan_full_tokens = (
            torch.cat(full_plan_parts, dim=1) if full_plan_parts else None
        )
        if mem_dl_full is not None and plan_full_tokens is not None:
            mem_dl_full = torch.cat(
                [mem_dl_full, plan_full_tokens],
                dim=1,
            )

        # Prepare prompt
        B = ctx_tokens.size(0)
        device = ctx_tokens.device
        if prompt_input_ids is None:
            prompt_input_ids = torch.full((B, 1), self.bos_token_id, device=device, dtype=torch.long)
        else:
            if prompt_input_ids.dim() == 1:
                prompt_input_ids = prompt_input_ids.unsqueeze(0)
            if prompt_input_ids.size(0) != B:
                raise ValueError(f"prompt_input_ids batch={prompt_input_ids.size(0)} does not match condition batch={B}")
            prompt_input_ids = prompt_input_ids.to(device=device, dtype=torch.long)

        prompt_len = int(prompt_input_ids.size(1))

        # If full injection is disabled and user allows, we can rely on HF generate for speed.
        if (not enable_full_injection) and use_hf_generate_when_possible:
            prefix_embeds, prefix_len = self.decoder.build_prefix(
                mem_func,
                mem_dl_light,
                mem_fold_plan=plan_prefix_tokens,
                fold_plan_scale=1.0,
            )
            emb_layer = self._decoder_input_embedding_layer()
            prompt_embeds = emb_layer(prompt_input_ids)
            inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            attn_mask = torch.ones((B, prefix_len + prompt_len), device=device, dtype=torch.long)
            out_ids = self.decoder.backbone.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=True,
            )
            # HF will include a dummy prefix length; safest is return the tail beyond the initial prompt length.
            # We return the last (prompt_len + new) tokens.
            if return_full_sequence:
                return out_ids[:, - (prompt_len + max_new_tokens):]
            return out_ids[:, -max_new_tokens:]

        # Custom autoregressive loop with KV cache (fast + full-injection-correct)
        prefix_embeds, prefix_len = self.decoder.build_prefix(
            mem_func,
            mem_dl_light,
            mem_fold_plan=plan_prefix_tokens,
            fold_plan_scale=1.0,
        )

        use_full_inj = bool(enable_full_injection) and (mem_dl_full is not None) and bool(self.decoder.enable_full_injection)

        # Prepare runtime injection state (projects mem_dl_full once)
        if use_full_inj:
            self.decoder.enable_runtime_full_injection(
                prefix_len,
                mem_dl_full,
                mem_pocket=mem_pocket,
                mem_ligand_chem=mem_ligand_chem_for_dec,
                interaction_embeds=interaction_embeds,
            )
        else:
            # ensure buffers are clean
            if hasattr(self.decoder, "disable_runtime_full_injection"):
                self.decoder.disable_runtime_full_injection()

        try:
            generated = prompt_input_ids.clone()  # [B, T_prompt]
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            emb = self._decoder_input_embedding_layer()

            # 1) Prime KV cache with (prefix + prompt) in a single forward
            prompt_embeds = emb(generated)  # [B, T_prompt, D]
            inputs_embeds0 = torch.cat([prefix_embeds, prompt_embeds], dim=1)  # [B, L0, D]
            attn_mask0 = torch.ones((B, inputs_embeds0.size(1)), device=device, dtype=torch.long)

            if use_full_inj and hasattr(self.decoder, "update_runtime_start_pos"):
                self.decoder.update_runtime_start_pos(0)

            out0 = self.decoder.backbone(
                inputs_embeds=inputs_embeds0,
                attention_mask=attn_mask0,
                use_cache=True,
                return_dict=True,
            )
            past = getattr(out0, "past_key_values", None)

            # If backbone doesn't return cache for any reason, fall back to the no-cache loop
            if past is None:
                for _ in range(max_new_tokens):
                    gen_embeds = emb(generated)  # [B,T,D]
                    inputs_embeds = torch.cat([prefix_embeds, gen_embeds], dim=1)  # [B,L,D]
                    attn_mask = torch.ones((B, inputs_embeds.size(1)), device=device, dtype=torch.long)

                    out = self.decoder.backbone(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attn_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    logits = out.logits[:, -1, :]  # [B,V]

                    # repetition penalty
                    logits = _apply_repetition_penalty(logits, generated, repetition_penalty)

                    # temperature
                    if temperature is not None and temperature != 1.0:
                        logits = logits / max(float(temperature), 1e-8)

                    # filtering
                    logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

                    if do_sample:
                        probs = torch.softmax(logits, dim=-1)
                        next_id = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        next_id = torch.argmax(logits, dim=-1)

                    next_id = torch.where(finished, torch.full_like(next_id, pad_token_id), next_id)
                    generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
                    finished = finished | (next_id == eos_token_id)
                    if bool(finished.all()):
                        break
            else:
                # 2) Cached step decoding
                logits = out0.logits[:, -1, :]  # logits for last prompt token

                for _ in range(max_new_tokens):
                    # sample next token from current logits
                    step_logits = _apply_repetition_penalty(logits, generated, repetition_penalty)

                    if temperature is not None and temperature != 1.0:
                        step_logits = step_logits / max(float(temperature), 1e-8)

                    step_logits = _top_k_top_p_filtering(step_logits, top_k=top_k, top_p=top_p)

                    if do_sample:
                        probs = torch.softmax(step_logits, dim=-1)
                        next_id = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        next_id = torch.argmax(step_logits, dim=-1)

                    # If finished, force pad
                    next_id = torch.where(finished, torch.full_like(next_id, pad_token_id), next_id)

                    # Append sampled token
                    generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
                    finished = finished | (next_id == eos_token_id)

                    if bool(finished.all()):
                        break

                    # Forward one token with cache to get logits for next step
                    last_embeds = emb(next_id.unsqueeze(1))  # [B,1,D]

                    # Full attention mask length = prefix + (prompt + generated)
                    attn_mask = torch.ones((B, prefix_len + generated.size(1)), device=device, dtype=torch.long)

                    if use_full_inj and hasattr(self.decoder, "update_runtime_start_pos"):
                        # global position of this last token in the full sequence (including prefix)
                        start_pos = prefix_len + generated.size(1) - 1
                        self.decoder.update_runtime_start_pos(start_pos)

                    out = self.decoder.backbone(
                        inputs_embeds=last_embeds,
                        attention_mask=attn_mask,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = getattr(out, "past_key_values", past)
                    logits = out.logits[:, -1, :]
        finally:
            # Clear runtime buffers to avoid accidental reuse
            if hasattr(self.decoder, "disable_runtime_full_injection"):
                self.decoder.disable_runtime_full_injection()

        if return_full_sequence:
            return generated
        return generated[:, -max_new_tokens:]
