"""Generate protein sequences with LangPro using conservative decoding."""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from collections import Counter

from ..architecture.model import (
    DockingWizardModel,
    DockingWizardModelConfig,
    PerceiverModuleConfig,
    _top_k_top_p_filtering,
    _apply_repetition_penalty,
    _as_long_mask,
)


# This ordering is shared with the fold-planner residue heads.  The values are
# the natural-protein background used while training slot-composition targets;
# decoder guidance is expressed relative to it so the head cannot simply add a
# global preference for the most common amino acids.
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_BACKGROUND = (
    0.0825, 0.0137, 0.0545, 0.0675, 0.0386,
    0.0707, 0.0227, 0.0596, 0.0584, 0.0966,
    0.0242, 0.0406, 0.0471, 0.0393, 0.0553,
    0.0656, 0.0534, 0.0687, 0.0109, 0.0292,
)


def build_conditioning_text(
    row: pd.Series,
    use_instruct_format: bool,
    instruction_col: str,
    function_col: str,
    docking_col: str,
    docking_into: str,
) -> str:
    """Build a human-readable condition string for logging and export."""

    def _get(col: str) -> str:
        if not col:
            return ""
        v = row.get(col, "")
        if v is None:
            return ""
        return str(v).strip()

    func = _get(function_col)
    instr = _get(instruction_col)
    dock = _get(docking_col)

    if not use_instruct_format:
        return func

    if dock:
        dock_txt = f"Docking guidance: {dock}"
        if docking_into == "instruction":
            instr = (instr + "\n" + dock_txt).strip() if instr else dock_txt
        elif docking_into == "function":
            func = (func + "\n" + dock_txt).strip() if func else dock_txt
        else:
            instr = (instr + "\n" + dock_txt).strip() if instr else dock_txt

    parts = []
    if instr:
        parts.append("### Instruction\n" + instr)
    if func:
        parts.append("### Function description\n" + func)
    return "\n\n".join(parts).strip()


def smiles_to_ligand_text(smiles: str) -> str:
    """Build a ligand description only when explicitly requested."""
    s = (smiles or "").strip()
    if not s:
        return ""

    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

        m = Chem.MolFromSmiles(s)
        if m is None:
            return f"SMILES: {s}"

        can = Chem.MolToSmiles(m, canonical=True)
        mw = Descriptors.MolWt(m)
        hbd = rdMolDescriptors.CalcNumHBD(m)
        hba = rdMolDescriptors.CalcNumHBA(m)
        tpsa = rdMolDescriptors.CalcTPSA(m)
        clogp = Crippen.MolLogP(m)
        rings = rdMolDescriptors.CalcNumRings(m)
        arom = rdMolDescriptors.CalcNumAromaticRings(m)
        rot = rdMolDescriptors.CalcNumRotatableBonds(m)
        charge = Chem.GetFormalCharge(m)

        return (
            f"Ligand properties: MW {mw:.2f}, HBD {hbd}, HBA {hba}, TPSA {tpsa:.2f}, "
            f"cLogP {clogp:.2f}, rings {rings}, aromatic rings {arom}, rotatable bonds {rot}, "
            f"formal charge {charge}. SMILES: {can}"
        )
    except Exception:
        return f"SMILES: {s}"


def _infer_resampler_layout_from_state_dict(sd: Dict[str, Any]) -> tuple[Optional[int], Optional[bool]]:
    """
    Infer PerceiverResampler(depth, use_self_attn) from checkpoint keys.

    Important:
    - state_dict stores ModuleList block indices, not the constructor's `depth`
    - when use_self_attn=True, each depth contributes:
        CrossAttentionBlock + SelfAttentionBlock
      so total block count = 2 * depth
    """
    prefix = "perceiver.resampler.blocks."
    block_ids = set()
    has_self_attn_block = False
    has_cross_attn_block = False

    for k in sd.keys():
        if not k.startswith(prefix):
            continue

        rest = k[len(prefix):]
        idx_str, sep, suffix = rest.partition(".")
        if not sep or not idx_str.isdigit():
            continue

        block_ids.add(int(idx_str))

        # CrossAttentionBlock has norm_q / norm_kv
        if suffix.startswith("norm_q.") or suffix.startswith("norm_kv."):
            has_cross_attn_block = True

        # SelfAttentionBlock has norm.weight / norm.bias
        if suffix.startswith("norm.weight") or suffix.startswith("norm.bias"):
            has_self_attn_block = True

    if not block_ids:
        return None, None

    total_blocks = max(block_ids) + 1

    if has_self_attn_block:
        inferred_use_self_attn = True
        inferred_depth = total_blocks // 2
    else:
        inferred_use_self_attn = False
        inferred_depth = total_blocks

    return inferred_depth, inferred_use_self_attn

def _infer_resampler_use_self_attn_from_state_dict(sd: Dict[str, Any]) -> Optional[bool]:
    # A shared norm key indicates a self-attention-style resampler block.
    for k in sd.keys():
        if k.startswith("perceiver.resampler.blocks.") and ".norm.weight" in k:
            return True
    return None


def _infer_bridge_use_ffn_from_state_dict(sd: Dict[str, Any]) -> Optional[bool]:
    for k in sd.keys():
        if k.startswith("perceiver.bridge.ff.") or k.startswith("perceiver.bridge.ls_ff."):
            return True
    return None


def _infer_dl_attend_function_from_state_dict(sd: Dict[str, Any]) -> Optional[bool]:
    # Bidirectional cross-branch keys identify the legacy architecture.
    for k in sd.keys():
        if k.startswith("perceiver.func_attends_dl.") or k.startswith("perceiver.dl_attends_func."):
            return True
    return None

def _cfg_from_ckpt_dict(cfg_dict: Dict[str, Any], sd: Optional[Dict[str, Any]] = None) -> DockingWizardModelConfig:
    cfg = DockingWizardModelConfig()

    def _assign(obj: Any, d: Dict[str, Any]) -> None:
        for k, v in d.items():
            if not hasattr(obj, k):
                continue
            cur = getattr(obj, k)
            if (
                obj is cfg
                and k == "perceiver_cfg"
                and cur is None
                and isinstance(v, dict)
            ):
                cur = PerceiverModuleConfig()
                setattr(obj, k, cur)
            if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
                _assign(cur, v)
            else:
                setattr(obj, k, v)

    _assign(cfg, cfg_dict)
    _apply_legacy_cfg_compat(cfg, cfg_dict, sd)
    return cfg


def _infer_bridge_use_ffn_from_state_dict(sd: Dict[str, Any]) -> Optional[bool]:
    for k in sd.keys():
        if k.startswith("perceiver.bridge.ff.") or k.startswith("perceiver.bridge.ls_ff."):
            return True
    return None


def _infer_dl_attend_function_from_state_dict(sd: Dict[str, Any]) -> Optional[bool]:
    for k in sd.keys():
        if k.startswith("perceiver.func_attends_dl.") or k.startswith("perceiver.dl_attends_func."):
            return True
    return None


def _get_recent_window_stats(generated: torch.Tensor, window_size: int = 24):
    """
    Returns:
        dominant_token_id, dominant_fraction, counter
    Assumes batch size = 1.
    Excludes BOS.
    """
    ids = generated[0].tolist()[1:]  # exclude BOS
    if len(ids) == 0:
        return None, 0.0, Counter()

    tail = ids[-int(window_size):]
    if len(tail) == 0:
        return None, 0.0, Counter()

    cnt = Counter(tail)
    dom_id, dom_count = cnt.most_common(1)[0]
    dom_frac = dom_count / float(len(tail))
    return int(dom_id), float(dom_frac), cnt


def _apply_recent_dominance_guard(
    logits: torch.Tensor,
    generated: torch.Tensor,
    eos_id: int,
    pad_id: int,
    bos_id: int,
    window_size: int = 24,
    mask_threshold: float = 0.42,
    penalty_alpha: float = 4.0,
):
    """
    Softly penalize a token if it dominates the recent decoding window.
    We avoid hard-masking here because that can damage local quality.
    """
    dom_id, dom_frac, _ = _get_recent_window_stats(generated, window_size=window_size)
    if dom_id is None:
        return logits, None, 0.0

    if dom_id in {int(eos_id), int(pad_id), int(bos_id)}:
        return logits, dom_id, dom_frac

    if dom_frac >= float(mask_threshold):
        penalty = float(penalty_alpha) * (dom_frac - float(mask_threshold))
        logits[:, int(dom_id)] = logits[:, int(dom_id)] - penalty

    return logits, dom_id, dom_frac

    # soft penalty once dominance becomes noticeable
    if dom_frac >= float(mask_threshold):
        penalty = float(penalty_alpha) * (dom_frac - float(mask_threshold))
        logits[:, int(dom_id)] = logits[:, int(dom_id)] - penalty

    # hard block if the token is overwhelmingly dominant
    if dom_frac >= 0.58:
        logits[:, int(dom_id)] = -1e9

    return logits, dom_id, dom_frac

def _apply_recent_window_frequency_penalty(
    logits: torch.Tensor,
    generated: torch.Tensor,
    window_size: int = 24,
    dominance_threshold: float = 0.35,
    freq_penalty_alpha: float = 2.5,
    exempt_token_ids: Optional[list[int]] = None,
) -> torch.Tensor:
    """
    Penalize tokens that dominate the recent decoding window.

    Example:
    if a token occupies >35% of the last 24 generated tokens,
    subtract an extra penalty from its logit.
    """
    if generated.size(1) <= 1:
        return logits

    exempt = set(exempt_token_ids or [])
    ids = generated[0].tolist()[1:]  # exclude BOS
    if len(ids) == 0:
        return logits

    tail = ids[-int(window_size):]
    total = len(tail)
    if total == 0:
        return logits

    from collections import Counter
    cnt = Counter(tail)

    for tid, c in cnt.items():
        if int(tid) in exempt:
            continue
        frac = c / float(total)
        if frac > float(dominance_threshold):
            penalty = float(freq_penalty_alpha) * (frac - float(dominance_threshold))
            logits[:, int(tid)] = logits[:, int(tid)] - penalty

    return logits


def _apply_margin_antidominance(
    logits: torch.Tensor,
    generated: torch.Tensor,
    eos_id: int,
    pad_id: int,
    bos_id: int,
    window_size: int = 24,
    dominance_threshold: float = 0.42,
    margin_threshold: float = 0.80,
    suppress_logit: float = 2.0,
) -> torch.Tensor:
    """
    If a token already dominates the recent window, and it is still the current
    top-1 choice only by a small margin, suppress it slightly instead of
    hard-stopping or always choosing the second-best token.
    """
    dom_id, dom_frac, _ = _get_recent_window_stats(generated, window_size=window_size)
    if dom_id is None:
        return logits

    if dom_id in {int(eos_id), int(pad_id), int(bos_id)}:
        return logits

    topv, topi = torch.topk(logits, k=min(3, logits.size(-1)), dim=-1)
    top1_id = int(topi[0, 0].item())
    margin = float(topv[0, 0].item() - topv[0, 1].item()) if topv.size(1) > 1 else 999.0

    if dom_frac >= float(dominance_threshold) and top1_id == dom_id and margin <= float(margin_threshold):
        logits[:, top1_id] = logits[:, top1_id] - float(suppress_logit)

    return logits


def _typical_filtering(logits: torch.Tensor, typical_p: float = 0.95, min_tokens_to_keep: int = 1) -> torch.Tensor:
    """
    Typical decoding filter from Meister et al.
    Keeps tokens whose surprisal is close to the sequence entropy.
    """
    typical_p = float(typical_p)
    if typical_p >= 1.0:
        return logits
    typical_p = max(1e-6, min(typical_p, 0.999999))
    min_tokens_to_keep = max(1, int(min_tokens_to_keep))

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(log_probs * probs).sum(dim=-1, keepdim=True)
    shifted_scores = torch.abs((-log_probs) - entropy)

    sorted_shifted, sorted_indices = torch.sort(shifted_scores, descending=False, dim=-1)
    sorted_logits = logits.gather(-1, sorted_indices)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    sorted_indices_to_remove = cumulative_probs > typical_p
    if sorted_indices_to_remove.size(-1) > 1:
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, -1e9)


def _apply_legacy_cfg_compat(
    cfg: DockingWizardModelConfig,
    cfg_dict: Dict[str, Any],
    sd: Optional[Dict[str, Any]] = None,
) -> None:
    perceiver_dict = cfg_dict.get("perceiver_cfg") or {}

    if cfg.perceiver_cfg is None:
        cfg.perceiver_cfg = PerceiverModuleConfig(
            d_ctx=cfg.encoder_cfg.d_ctx,
            k_dock=cfg.encoder_cfg.dock_final_tokens,
            k_lig=cfg.encoder_cfg.ligand_view_tokens,
            k_func=cfg.encoder_cfg.func_final_tokens,
        )

    sd = sd or {}

    inferred_resampler_depth, inferred_resampler_use_self_attn = _infer_resampler_layout_from_state_dict(sd)
    inferred_bridge_use_ffn = _infer_bridge_use_ffn_from_state_dict(sd)
    inferred_dl_attend_function = _infer_dl_attend_function_from_state_dict(sd)

    # Non-structural safe defaults
    if "bridge_gate_init_bias" not in perceiver_dict:
        cfg.perceiver_cfg.bridge_gate_init_bias = -2.0
    if "use_inter_branch_attn" not in perceiver_dict:
        cfg.perceiver_cfg.use_inter_branch_attn = False
    if "inter_branch_num_heads" not in perceiver_dict:
        cfg.perceiver_cfg.inter_branch_num_heads = 8

    # Structural knobs: prefer state_dict evidence over cfg_dict
    if inferred_dl_attend_function is not None:
        cfg.perceiver_cfg.dl_attend_function = inferred_dl_attend_function
    elif "dl_attend_function" not in perceiver_dict:
        cfg.perceiver_cfg.dl_attend_function = False

    if inferred_bridge_use_ffn is not None:
        cfg.perceiver_cfg.bridge_use_ffn = inferred_bridge_use_ffn
    elif "bridge_use_ffn" not in perceiver_dict:
        cfg.perceiver_cfg.bridge_use_ffn = False

    if inferred_resampler_use_self_attn is not None:
        cfg.perceiver_cfg.resampler_use_self_attn = inferred_resampler_use_self_attn
    elif "resampler_use_self_attn" not in perceiver_dict:
        cfg.perceiver_cfg.resampler_use_self_attn = False

    if inferred_resampler_depth is not None:
        cfg.perceiver_cfg.resampler_depth = inferred_resampler_depth
    elif "resampler_depth" not in perceiver_dict:
        cfg.perceiver_cfg.resampler_depth = 2


def _is_legacy_ckpt(cfg_dict: Optional[Dict[str, Any]]) -> bool:
    if not cfg_dict:
        return False
    perceiver_dict = cfg_dict.get("perceiver_cfg") or {}
    legacy_markers = {
        "use_inter_branch_attn",
        "bridge_use_ffn",
        "resampler_use_self_attn",
        "resampler_depth",
    }
    return not legacy_markers.issubset(set(perceiver_dict.keys()))


def _load_state_dict_with_compat(
    model: torch.nn.Module,
    sd: Dict[str, Any],
    legacy_ckpt: bool,
    compact_ckpt: bool = False,
) -> None:
    sd = dict(sd)
    reconstructible_prefixes = (
        "encoder.bert_dock.model.",
        "encoder.bert_func.model.",
        "encoder.bert_ltxt.model.",
        "encoder.smiles_lm.model.",
        "decoder.backbone.",
        "decoder.blocks.",
        "decoder.input_embeddings.",
    ) if compact_ckpt else ()
    if "decoder._runtime_mem_proj.weight" in sd:
        sd.pop("decoder._runtime_mem_proj.weight")
        print("[compat] dropped obsolete decoder._runtime_mem_proj.weight; decoder.full_proj is now registered at init.")

    if not legacy_ckpt:
        incompatible = model.load_state_dict(sd, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        allowed_missing_prefixes = (
            "route_comp_head_dock.",
            "route_comp_head_lig.",
            "chem_feature_head.",
            "complementarity_proj.",
            "perceiver.chem_feature_proj.",
            "perceiver.chem_feature_gate.",
            "decoder.pocket_head.",
            "decoder.contact_head.",
            "decoder.interaction_head.",
            "decoder.secondary_structure_head.",
            "decoder.solvent_accessibility_head.",
            "decoder.residue_depth_head.",
            "structure_contact_left.",
            "structure_contact_right.",
            "fold_planner.",
            "structure_token_planner.",
            "decoder.fold_plan_proj.",
        )
        bad_missing = [
            k for k in missing
            if not any(
                k.startswith(prefix)
                for prefix in allowed_missing_prefixes + reconstructible_prefixes
            )
        ]
        if bad_missing or unexpected:
            msg = ["Checkpoint load found unsupported mismatches."]
            if bad_missing:
                msg.append("Missing keys: " + ", ".join(bad_missing[:20]))
            if unexpected:
                msg.append("Unexpected keys: " + ", ".join(unexpected[:20]))
            raise RuntimeError(" ".join(msg))
        if missing:
            print("[compat] initialized new route/chem heads from current code defaults.")
        return

    incompatible = model.load_state_dict(sd, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    allowed_missing_fragments = (
        ".ln_mem.",
        ".ffn.",
        ".layer_scale",
    )
    allowed_missing_prefixes = (
        "route_comp_head_dock.",
        "route_comp_head_lig.",
        "chem_feature_head.",
        "complementarity_proj.",
        "perceiver.chem_feature_proj.",
        "perceiver.chem_feature_gate.",
        "decoder.pocket_head.",
        "decoder.contact_head.",
        "decoder.interaction_head.",
        "decoder.secondary_structure_head.",
        "decoder.solvent_accessibility_head.",
        "decoder.residue_depth_head.",
        "structure_contact_left.",
        "structure_contact_right.",
        "fold_planner.",
        "structure_token_planner.",
        "decoder.fold_plan_proj.",
    )

    allowed_unexpected_prefixes = (
        "perceiver.func_attends_dl.",
        "perceiver.dl_attends_func.",
    )

    bad_missing = [
        k for k in missing
        if not (
            (k.startswith("decoder.injectors.") and any(frag in k for frag in allowed_missing_fragments))
            or any(
                k.startswith(prefix)
                for prefix in allowed_missing_prefixes + reconstructible_prefixes
            )
        )
    ]

    bad_unexpected = [
        k for k in unexpected
        if not any(k.startswith(prefix) for prefix in allowed_unexpected_prefixes)
    ]

    if bad_missing or bad_unexpected:
        msg = ["Legacy checkpoint compatibility load still found unsupported mismatches."]
        if bad_missing:
            msg.append("Missing keys: " + ", ".join(bad_missing[:20]))
        if bad_unexpected:
            msg.append("Unexpected keys: " + ", ".join(bad_unexpected[:20]))
        raise RuntimeError(" ".join(msg))

    if missing or unexpected:
        print("[compat] legacy checkpoint detected; allowed legacy perceiver / injector mismatches were initialized from current code defaults.")

@torch.no_grad()
def generate_one_langpro(
    model: DockingWizardModel,
    cond_batch: Dict[str, torch.Tensor],
    max_len: int = 256,
    min_gen_len: int = 0,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    enable_full_injection: bool = True,
    run_block_len: int = 5,
    run_force_eos_len: int = 8,
    tail_eos_bonus: float = 3.0,
    eos_bias_start_len: int = 32,
    eos_bias_full_len: int = 96,
    max_eos_bias: float = 3.0,
    disable_full_injection_after: int = -1,
    full_injection_scale: float = 1.0,
    prefix_condition_scale: float = 1.0,
    omit_condition_prefix: bool = False,
    protein_start_token_id: Optional[int] = None,
    pool_condition_memory: bool = False,
    forbidden_token_ids: Optional[list[int]] = None,
    recent_window_size: int = 24,
    dominance_threshold: float = 0.35,
    freq_penalty_alpha: float = 2.5,
    dominance_mask_threshold: float = 0.42,
    dominance_penalty_alpha: float = 4.0,
    dominance_margin_threshold: float = 0.80,
    dominance_suppress_logit: float = 2.0,
    decoding_strategy: str = "typical",
    typical_p: float = 0.95,
    typical_min_tokens_to_keep: int = 2,
    use_predicted_length: bool = False,
    predicted_length_min_ratio: float = 0.80,
    predicted_length_eos_start_ratio: float = 0.90,
    predicted_length_eos_full_ratio: float = 1.05,
    force_predicted_length: bool = False,
    strict_predicted_length: bool = False,
    strict_predicted_length_ratio: float = 1.0,
    slot_residue_bias_scale: float = 0.0,
    ligand_residue_bias_scale: float = 0.0,
    natural_aa_prior_scale: float = 0.0,
    aa_token_ids: Optional[list[int]] = None,
) -> tuple[torch.Tensor, Optional[float]]:
    """
    Conservative no-cache decoding with fixed sampling parameters and no
    heuristic early truncation.

    ``strict_predicted_length`` has deliberately different semantics from the
    older ``force_predicted_length`` flag.  The older flag appends EOS at the
    predicted length but still permits a high-probability EOS substantially
    earlier.  Strict mode blocks premature EOS until the requested fraction of
    the condition-predicted length, then forces EOS at the target.  This avoids
    silently truncating a learned full-chain scaffold into a short peptide.

    The two residue-bias controls are a causal inference path for existing
    neural fold-plan heads.  Until now slot composition and ligand chemistry
    were auxiliary losses only; their predictions never affected emitted amino
    acids.  Biases are relative to the natural-protein background and apply
    only to the canonical 20 amino-acid token IDs.
    """
    model.encoder.eval()
    model.perceiver.eval()
    model.decoder.eval()
    model.assert_batch_sanity(cond_batch)

    device = next(model.parameters()).device
    strict_predicted_length_ratio = float(strict_predicted_length_ratio)
    if not 0.0 < strict_predicted_length_ratio <= 1.0:
        raise ValueError("strict_predicted_length_ratio must be in (0, 1]")
    for name, value in (
        ("slot_residue_bias_scale", slot_residue_bias_scale),
        ("ligand_residue_bias_scale", ligand_residue_bias_scale),
        ("natural_aa_prior_scale", natural_aa_prior_scale),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if (
        float(slot_residue_bias_scale) > 0.0
        or float(ligand_residue_bias_scale) > 0.0
        or float(natural_aa_prior_scale) > 0.0
    ):
        if aa_token_ids is None or len(aa_token_ids) != len(AA20):
            raise ValueError("canonical 20-aa token IDs are required for fold-plan residue guidance")

    enc_out = model.encoder(
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
        compute_aux_losses=False,
    )
    ctx_tokens = enc_out["ctx_tokens"]
    ctx_mask = _as_long_mask(enc_out["ctx_mask"])
    per_out = model.perceiver(ctx_tokens, ctx_mask, return_debug=False)
    predicted_length = None
    predicted_target_residues = None
    if bool(use_predicted_length):
        length_tensor = model.predict_protein_length(per_out)
        if length_tensor is not None and length_tensor.numel() > 0:
            predicted_length = float(length_tensor.reshape(-1)[0].item())
            max_residues = max(1, int(max_len) - 1)
            target = min(max_residues, max(1, int(round(predicted_length))))
            predicted_target_residues = int(target)
            min_gen_len = min(
                max_residues,
                max(
                    int(min_gen_len),
                    int(round(target * float(predicted_length_min_ratio))),
                ),
            )
            eos_bias_start_len = min(
                max_residues,
                max(
                    int(min_gen_len),
                    int(round(target * float(predicted_length_eos_start_ratio))),
                ),
            )
            eos_bias_full_len = min(
                max_residues,
                max(
                    int(eos_bias_start_len) + 1,
                    int(round(target * float(predicted_length_eos_full_ratio))),
                ),
            )
            if bool(strict_predicted_length):
                min_gen_len = max(
                    int(min_gen_len),
                    int(math.ceil(target * strict_predicted_length_ratio)),
                )
                # Do not make EOS attractive before the condition-specific
                # chain has reached its intended length.  At ``target`` the
                # force branch below deterministically terminates decoding.
                eos_bias_start_len = max(int(eos_bias_start_len), int(target))
                eos_bias_full_len = max(int(eos_bias_full_len), int(target) + 1)

    mem_func = per_out["mem_func"]
    mem_dl_light = per_out["mem_dl_light"]
    mem_dl_full = per_out["mem_dl_full"] if enable_full_injection else None
    fold_plan_out = model.plan_fold(
        per_out,
        sample_prototypes=bool(model.cfg.fold_plan_sample_during_generation),
        hard_prototypes=bool(model.cfg.fold_plan_hard_during_generation),
    )
    fold_plan_tokens = fold_plan_out.tokens if fold_plan_out is not None else None
    structure_plan_out = model.plan_structure_tokens(
        per_out,
        sample_prior=bool(model.cfg.structure_token_sample_during_generation),
        hard_prior=bool(model.cfg.structure_token_hard_during_generation),
    )
    prefix_plan_parts = []
    full_plan_parts = []
    if fold_plan_tokens is not None and float(model.cfg.fold_plan_prefix_scale) != 0.0:
        prefix_plan_parts.append(
            fold_plan_tokens * float(model.cfg.fold_plan_prefix_scale)
        )
    if fold_plan_tokens is not None and float(model.cfg.fold_plan_full_memory_scale) != 0.0:
        full_plan_parts.append(
            fold_plan_tokens * float(model.cfg.fold_plan_full_memory_scale)
        )
    if structure_plan_out is not None:
        if float(model.cfg.structure_token_prefix_scale) != 0.0:
            prefix_plan_parts.append(
                structure_plan_out.prefix_tokens
                * float(model.cfg.structure_token_prefix_scale)
            )
        if float(model.cfg.structure_token_full_memory_scale) != 0.0:
            full_plan_parts.append(
                structure_plan_out.tokens
                * float(model.cfg.structure_token_full_memory_scale)
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

    slot_residue_bias = None
    ligand_residue_bias = None
    pocket_slot_weights = None
    natural_aa_prior = None
    if float(natural_aa_prior_scale) > 0.0:
        background = torch.tensor(AA20_BACKGROUND, device=device, dtype=torch.float32)
        background = background / background.sum().clamp_min(1e-8)
        # Centering makes this a composition correction rather than a global
        # boost of the canonical-AA subset.  It is intentionally inference-only
        # and defaults to zero for exact backward compatibility.
        natural_aa_prior = torch.log(background.clamp_min(1e-8))
        natural_aa_prior = natural_aa_prior - natural_aa_prior.mean()
    if fold_plan_out is not None and (
        float(slot_residue_bias_scale) > 0.0
        or float(ligand_residue_bias_scale) > 0.0
    ):
        background = torch.tensor(
            AA20_BACKGROUND,
            device=device,
            dtype=fold_plan_out.slot_residue_logits.dtype,
        )
        background = background / background.sum().clamp_min(1e-8)
        log_background = torch.log(background.clamp_min(1e-8))
        slot_residue_bias = (
            torch.log_softmax(fold_plan_out.slot_residue_logits[0], dim=-1)
            - log_background.unsqueeze(0)
        )
        ligand_residue_bias = (
            torch.log_softmax(fold_plan_out.residue_prior_logits[0], dim=-1)
            - log_background
        )
        pocket_slot_weights = torch.softmax(
            fold_plan_out.pocket_slot_logits[0], dim=-1
        )

    if omit_condition_prefix:
        # A zero condition scale still leaves learned boundary embeddings in
        # the sequence.  This diagnostic path removes the entire prefix so
        # the decoder is evaluated against its actual BOS-only prior.
        prefix_embeds = model._decoder_input_embedding_layer().weight.new_zeros(
            (mem_func.size(0), 0, model.decoder.d_model)
        )
        prefix_len = 0
    else:
        prefix_embeds, prefix_len = model.decoder.build_prefix(
            mem_func,
            mem_dl_light,
            mem_fold_plan=plan_prefix_tokens,
            condition_scale=float(prefix_condition_scale),
            fold_plan_scale=1.0,
            pool_condition_memory=pool_condition_memory,
        )

    use_full = (
        bool(enable_full_injection)
        and (mem_dl_full is not None)
        and bool(getattr(model.decoder, "enable_full_injection", False))
    )

    if use_full and hasattr(model.decoder, "enable_runtime_full_injection"):
        model.decoder.enable_runtime_full_injection(
            prefix_len,
            mem_dl_full,
            start_pos=0,
            global_injection_scale=float(full_injection_scale),
        )
    else:
        if hasattr(model.decoder, "disable_runtime_full_injection"):
            model.decoder.disable_runtime_full_injection()

    cfg = model.decoder.backbone.config
    bos_id = (
        int(protein_start_token_id)
        if protein_start_token_id is not None
        else getattr(cfg, "bos_token_id", 1)
    )
    eos_id = getattr(cfg, "eos_token_id", 2)
    pad_id = getattr(cfg, "pad_token_id", 0)

    generated = torch.full((1, 1), int(bos_id), device=device, dtype=torch.long)
    finished = torch.zeros(1, dtype=torch.bool, device=device)
    emb = model._decoder_input_embedding_layer()
    max_new_tokens = max(0, int(max_len) - generated.size(1))
    temp = 1.0 if temperature is None or temperature <= 0 else float(temperature)

    try:
        for _ in range(max_new_tokens):
            gen_len = int(generated.size(1) - 1)  # exclude BOS

            # The target comes only from the trained condition-specific length
            # head. This optional hard stop prevents a weak EOS logit from
            # extending otherwise healthy generations to the global cap.
            if (
                bool(force_predicted_length or strict_predicted_length)
                and predicted_target_residues is not None
                and gen_len >= int(predicted_target_residues)
            ):
                generated = torch.cat(
                    [generated, torch.full((1, 1), int(eos_id), device=device, dtype=torch.long)],
                    dim=1,
                )
                finished = torch.ones_like(finished)
                break

            # optional: disable full injection in late decoding
            if use_full and hasattr(model.decoder, "enable_runtime_full_injection"):
                if int(disable_full_injection_after) >= 0 and gen_len >= int(disable_full_injection_after):
                    if hasattr(model.decoder, "disable_runtime_full_injection"):
                        model.decoder.disable_runtime_full_injection()
                else:
                    model.decoder.enable_runtime_full_injection(
                        prefix_len,
                        mem_dl_full,
                        start_pos=0,
                        global_injection_scale=float(full_injection_scale),
                    )

            gen_embeds = emb(generated)
            inputs_embeds = torch.cat([prefix_embeds, gen_embeds], dim=1)
            attn_mask = torch.ones((1, inputs_embeds.size(1)), device=device, dtype=torch.long)

            out = model.decoder.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits[:, -1, :]

            # 1) global repetition penalty
            logits = _apply_repetition_penalty(logits, generated, repetition_penalty)

            logits = _apply_recent_window_frequency_penalty(
                logits=logits,
                generated=generated,
                window_size=recent_window_size,
                dominance_threshold=dominance_threshold,
                freq_penalty_alpha=freq_penalty_alpha,
                exempt_token_ids=[bos_id, eos_id, pad_id],
            )

            logits, _, _ = _apply_recent_dominance_guard(
                logits=logits,
                generated=generated,
                eos_id=eos_id,
                pad_id=pad_id,
                bos_id=bos_id,
                window_size=recent_window_size,
                mask_threshold=dominance_mask_threshold,
                penalty_alpha=dominance_penalty_alpha,
            )

            logits = _apply_margin_antidominance(
                logits=logits,
                generated=generated,
                eos_id=eos_id,
                pad_id=pad_id,
                bos_id=bos_id,
                window_size=recent_window_size,
                dominance_threshold=dominance_mask_threshold,
                margin_threshold=dominance_margin_threshold,
                suppress_logit=dominance_suppress_logit,
            )

            if slot_residue_bias is not None and aa_token_ids is not None:
                position_scale = max(
                    1,
                    int(predicted_target_residues or max_new_tokens),
                )
                slot_index = min(
                    int(slot_residue_bias.size(0)) - 1,
                    (gen_len * int(slot_residue_bias.size(0))) // position_scale,
                )
                residue_bias = (
                    float(slot_residue_bias_scale)
                    * slot_residue_bias[slot_index]
                )
                if (
                    ligand_residue_bias is not None
                    and pocket_slot_weights is not None
                    and float(ligand_residue_bias_scale) > 0.0
                ):
                    # Convert the normalized pocket probability into a local
                    # multiplier.  Capping prevents a still-uncertain pocket
                    # position head from overwhelming the pretrained LM.
                    pocket_locality = min(
                        2.0,
                        float(pocket_slot_weights[slot_index].item())
                        * float(pocket_slot_weights.numel()),
                    )
                    residue_bias = residue_bias + (
                        float(ligand_residue_bias_scale)
                        * pocket_locality
                        * ligand_residue_bias
                    )
                logits[:, aa_token_ids] = logits[:, aa_token_ids] + residue_bias.unsqueeze(0)

            if natural_aa_prior is not None and aa_token_ids is not None:
                logits[:, aa_token_ids] = logits[:, aa_token_ids] + (
                    float(natural_aa_prior_scale)
                    * natural_aa_prior.to(dtype=logits.dtype).unsqueeze(0)
                )

            # 2) optional forbidden AA tokens (e.g. X/Z/B/J/U/O)
            logits = _apply_forbidden_token_mask(logits, forbidden_token_ids)

            # 3) temperature
            if temp != 1.0:
                logits = logits / max(temp, 1e-8)

            # 4) local tail collapse guard
            logits = _apply_tail_repeat_guard(
                logits=logits,
                generated=generated,
                eos_id=eos_id,
                pad_id=pad_id,
                min_gen_len=min_gen_len,
                run_block_len=run_block_len,
                run_force_eos_len=run_force_eos_len,
                tail_eos_bonus=tail_eos_bonus,
            )

            # 5) soft EOS preference after sequence becomes long enough
            logits = _apply_length_based_eos_bias(
                logits=logits,
                eos_id=eos_id,
                gen_len=gen_len,
                min_gen_len=min_gen_len,
                eos_bias_start_len=eos_bias_start_len,
                eos_bias_full_len=eos_bias_full_len,
                max_eos_bias=max_eos_bias,
            )

            # 6) hard block EOS only before min_gen_len
            if gen_len < int(min_gen_len):
                logits[:, eos_id] = -1e9

            # Keep a valid constrained distribution in case typical/nucleus
            # filtering removes every finite candidate. This can otherwise
            # surface as NaN probabilities and a CUDA multinomial assert.
            fallback_logits = logits.clone()

            if decoding_strategy == "typical":
                logits = _typical_filtering(
                    logits,
                    typical_p=float(typical_p),
                    min_tokens_to_keep=int(typical_min_tokens_to_keep),
                )
                if int(top_k) > 0:
                    logits = _top_k_top_p_filtering(logits, top_k=int(top_k), top_p=1.0)
            else:
                logits = _top_k_top_p_filtering(logits, top_k=int(top_k), top_p=float(top_p))

            if bool(do_sample):
                probs = torch.softmax(logits, dim=-1)
                valid_probs = torch.isfinite(probs).all(dim=-1) & (probs.sum(dim=-1) > 0)
                if bool(valid_probs.all()):
                    next_id = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    safe_logits = torch.where(
                        torch.isfinite(fallback_logits),
                        fallback_logits,
                        torch.full_like(fallback_logits, -1e9),
                    )
                    sampled = torch.argmax(safe_logits, dim=-1)
                    if bool(valid_probs.any()):
                        sampled[valid_probs] = torch.multinomial(
                            probs[valid_probs], num_samples=1
                        ).squeeze(1)
                    next_id = sampled
            else:
                next_id = torch.argmax(logits, dim=-1)

            next_id = torch.where(finished, torch.full_like(next_id, pad_id), next_id)
            generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
            finished = finished | (next_id == eos_id)

            if bool(finished.all()):
                break
    finally:
        if hasattr(model.decoder, "disable_runtime_full_injection"):
            model.decoder.disable_runtime_full_injection()

    return generated, predicted_length


def _pick_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _suffix_run_length(generated: torch.Tensor) -> int:
    """
    Count how many times the last token repeats consecutively at the tail.
    Assumes batch size = 1.
    """
    if generated.size(1) == 0:
        return 0
    ids = generated[0].tolist()
    last = ids[-1]
    run = 0
    for x in reversed(ids):
        if int(x) == int(last):
            run += 1
        else:
            break
    return run


def _apply_forbidden_token_mask(
    logits: torch.Tensor,
    forbidden_token_ids: Optional[list[int]],
) -> torch.Tensor:
    if not forbidden_token_ids:
        return logits
    for tid in forbidden_token_ids:
        if 0 <= int(tid) < logits.size(-1):
            logits[:, int(tid)] = -1e9
    return logits


def _apply_tail_repeat_guard(
    logits: torch.Tensor,
    generated: torch.Tensor,
    eos_id: int,
    pad_id: int,
    min_gen_len: int,
    run_block_len: int = 5,
    run_force_eos_len: int = 8,
    tail_eos_bonus: float = 3.0,
) -> torch.Tensor:
    """
    1) If the tail already has too many consecutive same tokens, block repeating it again.
    2) If generated length is already sufficient and tail collapse is severe, boost EOS.
    """
    if generated.size(1) <= 1:
        return logits

    last_id = int(generated[0, -1].item())
    if last_id in {int(eos_id), int(pad_id)}:
        return logits

    gen_len = int(generated.size(1) - 1)  # exclude BOS
    run_len = _suffix_run_length(generated)

    if run_len >= int(run_block_len):
        logits[:, last_id] = -1e9

    if gen_len >= int(min_gen_len) and run_len >= int(run_force_eos_len):
        logits[:, int(eos_id)] = logits[:, int(eos_id)] + float(tail_eos_bonus)

    return logits


def _apply_length_based_eos_bias(
    logits: torch.Tensor,
    eos_id: int,
    gen_len: int,
    min_gen_len: int,
    eos_bias_start_len: int = 32,
    eos_bias_full_len: int = 96,
    max_eos_bias: float = 3.0,
) -> torch.Tensor:
    """
    After min_gen_len, gradually increase EOS preference as sequence grows.
    This helps the model stop naturally instead of always drifting to max_len.
    """
    start_len = max(int(min_gen_len), int(eos_bias_start_len))
    full_len = max(start_len + 1, int(eos_bias_full_len))

    if gen_len < start_len:
        return logits

    alpha = (gen_len - start_len) / float(full_len - start_len)
    alpha = max(0.0, min(1.0, alpha))
    logits[:, int(eos_id)] = logits[:, int(eos_id)] + alpha * float(max_eos_bias)
    return logits


def clean_protein_sequence(seq: str) -> str:
    if not seq:
        return ""
    return "".join(c for c in seq if not c.isdigit()).strip()


def _resolve_condition_channels(row: pd.Series, args: argparse.Namespace, columns):
    raw_instr = str(row.get(args.instruction_col, "")).strip() if args.instruction_col in columns else ""
    raw_func = str(row.get(args.function_col, "")).strip() if args.function_col in columns else ""
    raw_dock = str(row.get(args.docking_col, "")).strip() if args.docking_col in columns else ""

    if args.use_instruct_format:
        # Keep instruction, function, and docking semantics in separate channels.
        docking_text = raw_dock
        function_text = raw_func
    else:
        docking_text = raw_dock or raw_instr or raw_func
        function_text = raw_func or raw_instr or raw_dock

    # Backfill empty channels without prompt-style channel mixing.
    if not docking_text:
        docking_text = raw_instr or function_text
    if not function_text:
        function_text = raw_instr or docking_text

    instruction = raw_instr if raw_instr else function_text
    return instruction, function_text, docking_text


def _resolve_ligand_text(row: pd.Series, ligand_smiles: str, mode: str) -> str:
    csv_lig_text = str(row.get("ligand_text", "")).strip() if "ligand_text" in row.index else ""

    if mode == "csv_only":
        return csv_lig_text
    if mode == "csv_or_auto":
        return csv_lig_text if csv_lig_text else smiles_to_ligand_text(ligand_smiles)
    if mode == "empty":
        return ""
    raise ValueError(f"Unsupported ligand text mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-dir", required=True, help="Local dir or HF id for ProGen2 backbone")
    ap.add_argument("--ckpt", required=True, help="Path to ckpts/best.pt from train.py")
    ap.add_argument(
        "--restore-base-decoder-backbone",
        action="store_true",
        help="Diagnostic only: ignore checkpoint decoder-backbone weights and reconstruct the local pretrained ProGen backbone.",
    )
    ap.add_argument("--csv", required=True, help="CSV file")
    ap.add_argument(
        "--allow-empty-ligand",
        action="store_true",
        help=(
            "Ablation-only: permit an empty ligand field. The SMILES/3D "
            "channels then receive tokenizer special tokens and the existing "
            "dummy-atom fallback instead of silently dropping the row."
        ),
    )
    ap.add_argument("--n", type=int, default=5, help="Number of rows to generate")
    ap.add_argument("--seed", type=int, default=20260730, help="Random seed for reproducible sampling")
    ap.add_argument("--max-len", type=int, default=256, help="Max total length including BOS")
    ap.add_argument("--min-gen-len", type=int, default=32, help="Min generated length before EOS allowed")
    ap.add_argument(
        "--protein-control-tag",
        default="",
        help="Optional one-token ProGen control tag used as the sequence start token; local ProGen2-base documentation uses '1'.",
    )

    ap.add_argument("--text-max-len", type=int, default=256, help="Kept for CLI parity")
    ap.add_argument(
        "--text-model",
        type=str,
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        help="Optional override of encoder text backbone",
    )

    ap.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    ap.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
    ap.add_argument("--top-p", type=float, default=0.95, help="Top-p filtering")
    ap.add_argument("--repetition-penalty", type=float, default=1.3, help="Repetition penalty")
    ap.add_argument("--decoding-strategy", type=str, default="typical", choices=["typical", "nucleus"],
                    help="Decoding strategy: typical (paper-based) or nucleus(top-k/top-p)")
    ap.add_argument("--typical-p", type=float, default=0.95,
                    help="Mass for typical decoding (Meister et al.)")
    ap.add_argument("--typical-min-tokens-to-keep", type=int, default=2,
                    help="Minimum tokens kept after typical filtering")
    ap.add_argument(
        "--fold-plan-mode",
        choices=["checkpoint", "stochastic", "soft", "top1"],
        default="checkpoint",
        help="Override checkpoint fold-archetype selection for inference calibration.",
    )
    ap.add_argument("--fold-plan-archetype-scale", type=float, default=None)
    ap.add_argument("--fold-plan-local-chemistry-scale", type=float, default=None)
    ap.add_argument("--fold-plan-prefix-scale", type=float, default=None)
    ap.add_argument(
        "--structure-token-mode",
        choices=["checkpoint", "stochastic", "soft", "top1"],
        default="checkpoint",
        help="Inference selection for the condition-prior scaffold token distribution.",
    )
    ap.add_argument("--structure-token-prefix-scale", type=float, default=None)
    ap.add_argument("--structure-token-full-memory-scale", type=float, default=None)
    ap.add_argument("--run-block-len", type=int, default=5,
                    help="Block repeating the same tail token once its consecutive run reaches this length")
    ap.add_argument("--run-force-eos-len", type=int, default=8,
                    help="If tail collapse is severe after min-gen-len, boost EOS")
    ap.add_argument("--tail-eos-bonus", type=float, default=3.0,
                    help="Extra EOS logit bonus when severe tail collapse is detected")

    ap.add_argument("--eos-bias-start-len", type=int, default=32,
                    help="Start softly boosting EOS after this generated length")
    ap.add_argument("--eos-bias-full-len", type=int, default=96,
                    help="Reach max EOS bias by this generated length")
    ap.add_argument("--max-eos-bias", type=float, default=3.0,
                    help="Maximum soft EOS logit bias at long lengths")
    ap.add_argument(
        "--use-predicted-length",
        action="store_true",
        help="Use the trained condition-specific length head to control EOS.",
    )
    ap.add_argument(
        "--protein-length-scale",
        type=float,
        default=None,
        help=(
            "Inference-only override for the residue-count scale of the trained "
            "length head. This is useful for auditing checkpoints trained on a "
            "truncated length curriculum; the checkpoint value is used when omitted."
        ),
    )
    ap.add_argument("--predicted-length-min-ratio", type=float, default=0.80)
    ap.add_argument("--predicted-length-eos-start-ratio", type=float, default=0.90)
    ap.add_argument("--predicted-length-eos-full-ratio", type=float, default=1.05)
    ap.add_argument(
        "--force-predicted-length",
        action="store_true",
        help="Force EOS at the model-predicted condition-specific length; does not use a reference length.",
    )
    ap.add_argument(
        "--strict-predicted-length",
        action="store_true",
        help=(
            "Block EOS until the condition-predicted chain is nearly complete "
            "and force EOS at that length. Requires --use-predicted-length."
        ),
    )
    ap.add_argument(
        "--strict-predicted-length-ratio",
        type=float,
        default=1.0,
        help="Minimum fraction of the predicted chain emitted before EOS is allowed in strict mode.",
    )
    ap.add_argument(
        "--slot-residue-bias-scale",
        type=float,
        default=0.0,
        help="Causal logit guidance from the neural fold-plan slot-composition head; zero disables it.",
    )
    ap.add_argument(
        "--ligand-residue-bias-scale",
        type=float,
        default=0.0,
        help="Localized causal logit guidance from the neural ligand-chemistry residue head; zero disables it.",
    )
    ap.add_argument(
        "--natural-aa-prior-scale",
        type=float,
        default=0.0,
        help=(
            "Weak zero-centered logit prior from natural-protein amino-acid "
            "frequencies; zero disables it and preserves legacy decoding."
        ),
    )

    ap.add_argument("--disable-full-injection-after", type=int, default=-1,
                    help="Disable full injection after this many generated tokens; -1 means never disable")
    ap.add_argument(
        "--full-injection-scale",
        type=float,
        default=1.0,
        help="Runtime scale for full-memory injection; 0 retains only the learned prefix route.",
    )
    ap.add_argument(
        "--prefix-condition-scale",
        type=float,
        default=1.0,
        help="Runtime scale for function and light-condition prefix embeddings.",
    )
    ap.add_argument(
        "--prefix-boundary-scale",
        type=float,
        default=None,
        help="Optional runtime scale for learned prefix boundary embeddings.",
    )
    ap.add_argument(
        "--omit-condition-prefix",
        action="store_true",
        help="Diagnostic only: omit the entire learned condition prefix, including boundary embeddings.",
    )
    ap.add_argument(
        "--pool-condition-memory",
        action="store_true",
        help="Use one pooled function token and one pooled ligand/docking token in the prefix.",
    )

    ap.add_argument(
        "--mask-ambiguous-aa",
        dest="mask_ambiguous_aa",
        action="store_true",
        default=True,
        help="Mask non-standard amino-acid tokens (enabled by default for foldable protein output).",
    )
    ap.add_argument(
        "--allow-ambiguous-aa",
        dest="mask_ambiguous_aa",
        action="store_false",
        help="Allow X/Z/B/J/U/O tokens; intended only for diagnostic decoding.",
    )
    ap.add_argument(
        "--canonical-aa-only",
        action="store_true",
        help=(
            "Restrict decoding to the canonical 20 amino-acid tokens plus EOS. "
            "This also blocks ProGen control/taxonomy tokens that sequence cleanup "
            "would otherwise silently discard."
        ),
    )

    ap.add_argument("--device", type=str, default=None, help="cuda:0 / cuda:1 / cpu")

    ap.add_argument("--use-instruct-format", action="store_true")
    ap.add_argument("--instruction-col", type=str, default="instruction")
    ap.add_argument("--function-col", type=str, default="protein_function_text")
    ap.add_argument("--docking-col", type=str, default="docking_text")
    ap.add_argument("--docking-into", type=str, default="instruction", choices=["instruction", "function"])

    ap.add_argument("--out-csv", type=str, default=None, help="Save generations to CSV")

    ap.add_argument("--enable-full-injection", "--enable_full_injection", dest="enable_full_injection", action="store_true")
    ap.add_argument("--disable-full-injection", "--disable_full_injection", dest="disable_full_injection", action="store_true")
    ap.add_argument("--enable-pocket-aware-injection", "--enable_pocket_aware_injection", dest="enable_pocket_aware_injection", action="store_true")
    ap.add_argument("--return-debug", "--return_debug", dest="return_debug", action="store_true")
    ap.add_argument("--hf-cache-root", type=str, default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--recent-window-size", type=int, default=24,
                help="Window size for recent token frequency penalty")
    ap.add_argument("--dominance-threshold", type=float, default=0.35,
                    help="If a token occupies more than this fraction in recent window, penalize it")
    ap.add_argument("--freq-penalty-alpha", type=float, default=2.5,
                    help="Strength of recent-window dominance penalty")
    ap.add_argument("--dominance-mask-threshold", type=float, default=0.42,
                    help="If one token exceeds this fraction in recent window, penalize it")
    ap.add_argument("--dominance-penalty-alpha", type=float, default=4.0,
                    help="Penalty strength for dominant token in recent window")
    ap.add_argument("--dominance-margin-threshold", type=float, default=0.80,
                    help="Only suppress dominant top1 when top1-top2 margin is below this value")
    ap.add_argument("--dominance-suppress-logit", type=float, default=2.0,
                    help="Extra logit suppression applied to dominant top1 when margin is small")

    ap.add_argument(
        "--ligand-text-mode",
        type=str,
        default="csv_only",
        choices=["csv_only", "csv_or_auto", "empty"],
        help="Use CSV ligand text by default; automatic descriptions are opt-in.",
    )

    args = ap.parse_args()
    if args.strict_predicted_length and not args.use_predicted_length:
        raise ValueError("--strict-predicted-length requires --use-predicted-length")
    if not 0.0 < float(args.strict_predicted_length_ratio) <= 1.0:
        raise ValueError("--strict-predicted-length-ratio must be in (0, 1]")
    if args.protein_length_scale is not None and (
        not math.isfinite(float(args.protein_length_scale))
        or float(args.protein_length_scale) <= 0.0
    ):
        raise ValueError("--protein-length-scale must be finite and positive")
    for name in (
        "slot_residue_bias_scale",
        "ligand_residue_bias_scale",
        "natural_aa_prior_scale",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _pick_device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt.get("cfg", None)
    sd = ckpt.get("model", ckpt)
    if args.restore_base_decoder_backbone:
        sd = {
            key: value
            for key, value in sd.items()
            if not key.startswith("decoder.backbone.")
        }

    cfg = _cfg_from_ckpt_dict(cfg_dict, sd) if cfg_dict is not None else DockingWizardModelConfig()
    legacy_ckpt = _is_legacy_ckpt(cfg_dict)

    cfg.decoder_backbone_name_or_path = args.pretrained_dir
    cfg.decoder_local_files_only = bool(args.local_files_only)
    if args.protein_length_scale is not None:
        cfg.protein_length_scale = float(args.protein_length_scale)
    if args.prefix_boundary_scale is not None:
        cfg.decoder_prefix_boundary_scale = float(args.prefix_boundary_scale)
    cfg.enable_pocket_aware_injection = bool(args.enable_pocket_aware_injection)
    if args.fold_plan_mode != "checkpoint":
        cfg.fold_plan_sample_during_generation = args.fold_plan_mode == "stochastic"
        cfg.fold_plan_hard_during_generation = args.fold_plan_mode == "top1"
    if args.structure_token_mode != "checkpoint":
        cfg.structure_token_sample_during_generation = (
            args.structure_token_mode == "stochastic"
        )
        cfg.structure_token_hard_during_generation = (
            args.structure_token_mode == "top1"
        )
    for value, attribute in (
        (args.fold_plan_archetype_scale, "fold_plan_archetype_scale"),
        (args.fold_plan_local_chemistry_scale, "fold_plan_local_chemistry_scale"),
        (args.fold_plan_prefix_scale, "fold_plan_prefix_scale"),
    ):
        if value is not None:
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"--{attribute.replace('_', '-')} must be finite and non-negative")
            setattr(cfg, attribute, float(value))
    for value, attribute in (
        (args.structure_token_prefix_scale, "structure_token_prefix_scale"),
        (args.structure_token_full_memory_scale, "structure_token_full_memory_scale"),
    ):
        if value is not None:
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"--{attribute.replace('_', '-')} must be finite and non-negative")
            setattr(cfg, attribute, float(value))

    default_text_model = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    if getattr(cfg, "encoder_cfg", None) is not None and args.text_model and args.text_model != default_text_model:
        cfg.encoder_cfg.pubmedbert_name = args.text_model

    # Match train.py's local model resolution. The frozen encoders resolve
    # these variables during construction, so assigning model.hf_cache_root
    # only after DockingWizardModel(...) is too late.
    if args.hf_cache_root:
        hf_root = Path(args.hf_cache_root).expanduser().resolve()
        os.environ["HF_HOME"] = str(hf_root)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_root)
        os.environ["PUB_MED_BERT_DIR"] = str(hf_root / "PubMedBERT")
        os.environ["SMILES_MODEL_DIR"] = str(
            hf_root / "SMILES_tokenized_PubChem_shard00_160k"
        )
        os.environ["PROGEN_MODEL_DIR"] = str(hf_root / "progen2-base-hf")

    model = DockingWizardModel(cfg)
    if args.hf_cache_root:
        model.hf_cache_root = args.hf_cache_root

    compact_ckpt = str(ckpt.get("checkpoint_type", "")).startswith("compact")
    _load_state_dict_with_compat(
        model,
        sd,
        legacy_ckpt=legacy_ckpt,
        compact_ckpt=compact_ckpt,
    )
    model.to(device)
    model.eval()
    _, _, tok_protein = model._ensure_tokenizers(args.hf_cache_root)
    protein_start_token_id = None
    if args.protein_control_tag:
        start_ids = tok_protein(
            str(args.protein_control_tag), add_special_tokens=False
        )["input_ids"]
        if len(start_ids) != 1:
            raise ValueError(
                f"--protein-control-tag={args.protein_control_tag!r} maps to {start_ids}; expected exactly one token"
            )
        protein_start_token_id = int(start_ids[0])
        print(
            f"[generation] protein control tag {args.protein_control_tag!r} -> "
            f"start_token_id={protein_start_token_id}"
        )

    forbidden_token_ids = []
    if args.mask_ambiguous_aa:
        for aa in ["X", "Z", "B", "J", "U", "O"]:
            try:
                ids = tok_protein.encode(aa, add_special_tokens=False)
                if isinstance(ids, list) and len(ids) == 1:
                    forbidden_token_ids.append(int(ids[0]))
            except Exception:
                pass

    aa_token_ids = []
    for residue in AA20:
        token_ids = tok_protein.encode(residue, add_special_tokens=False)
        if not isinstance(token_ids, list) or len(token_ids) != 1:
            raise ValueError(
                f"Protein tokenizer maps canonical residue {residue!r} to {token_ids}; expected one token"
            )
        aa_token_ids.append(int(token_ids[0]))
    if args.canonical_aa_only:
        eos_token_id = int(tok_protein.eos_token_id)
        allowed = set(aa_token_ids) | {eos_token_id}
        forbidden_token_ids = [
            token_id for token_id in range(len(tok_protein))
            if token_id not in allowed
        ]

    df = pd.read_csv(args.csv)
    if "ligand_smiles" not in df.columns:
        raise ValueError(f"CSV missing required column 'ligand_smiles'. Available: {list(df.columns)[:20]}")

    enable_full_injection = not args.disable_full_injection
    if args.enable_full_injection:
        enable_full_injection = True

    rows_out = []
    n = min(int(args.n), len(df))

    for i in range(n):
        row = df.iloc[i]
        raw_ligand_smiles = row.get("ligand_smiles", "")
        ligand_smiles = (
            "" if pd.isna(raw_ligand_smiles) else str(raw_ligand_smiles).strip()
        )
        if not ligand_smiles and not args.allow_empty_ligand:
            print(f"[skip] row {i}: empty ligand_smiles")
            continue

        cond_text_raw = build_conditioning_text(
            row=row,
            use_instruct_format=bool(args.use_instruct_format),
            instruction_col=args.instruction_col,
            function_col=args.function_col,
            docking_col=args.docking_col,
            docking_into=args.docking_into,
        )

        instruction, function_text, docking_text = _resolve_condition_channels(row, args, df.columns)
        ligand_text = _resolve_ligand_text(row, ligand_smiles, args.ligand_text_mode)

        cond_batch = model.prepare_cond_batch_from_raw(
            instruction=instruction,
            smiles=ligand_smiles,
            docking_text=docking_text,
            function_text=function_text,
            ligand_text=ligand_text,
            hf_cache_root=args.hf_cache_root,
        )

        debug_payload = {}
        if args.return_debug:
            with torch.no_grad():
                enc_dbg = model.encoder(
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
                    compute_aux_losses=False,
                )
                per_dbg = model.perceiver(enc_dbg["ctx_tokens"], _as_long_mask(enc_dbg["ctx_mask"]), return_debug=True)
                debug_payload = {
                    "ligand_view_weights": enc_dbg.get("ligand_view_weights", None).detach().cpu().tolist()
                    if torch.is_tensor(enc_dbg.get("ligand_view_weights", None)) else None,
                    "perceiver_debug_keys": sorted(list((per_dbg.get("debug", {}) or {}).keys())),
                    "memory_shapes": (per_dbg.get("debug", {}) or {}).get("memory_shapes", None),
                }

        ids, predicted_length = generate_one_langpro(
            model=model,
            cond_batch=cond_batch,
            max_len=int(args.max_len),
            min_gen_len=int(args.min_gen_len),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            enable_full_injection=enable_full_injection,
            run_block_len=int(args.run_block_len),
            run_force_eos_len=int(args.run_force_eos_len),
            tail_eos_bonus=float(args.tail_eos_bonus),
            eos_bias_start_len=int(args.eos_bias_start_len),
            eos_bias_full_len=int(args.eos_bias_full_len),
            max_eos_bias=float(args.max_eos_bias),
            disable_full_injection_after=int(args.disable_full_injection_after),
            full_injection_scale=float(args.full_injection_scale),
            prefix_condition_scale=float(args.prefix_condition_scale),
            omit_condition_prefix=bool(args.omit_condition_prefix),
            protein_start_token_id=protein_start_token_id,
            pool_condition_memory=bool(args.pool_condition_memory),
            forbidden_token_ids=forbidden_token_ids,
            recent_window_size=int(args.recent_window_size),
            dominance_threshold=float(args.dominance_threshold),
            freq_penalty_alpha=float(args.freq_penalty_alpha),
            dominance_mask_threshold=float(args.dominance_mask_threshold),
            dominance_penalty_alpha=float(args.dominance_penalty_alpha),
            dominance_margin_threshold=float(args.dominance_margin_threshold),
            dominance_suppress_logit=float(args.dominance_suppress_logit),
            decoding_strategy=str(args.decoding_strategy),
            typical_p=float(args.typical_p),
            typical_min_tokens_to_keep=int(args.typical_min_tokens_to_keep),
            use_predicted_length=bool(args.use_predicted_length),
            predicted_length_min_ratio=float(args.predicted_length_min_ratio),
            predicted_length_eos_start_ratio=float(args.predicted_length_eos_start_ratio),
            predicted_length_eos_full_ratio=float(args.predicted_length_eos_full_ratio),
            force_predicted_length=bool(args.force_predicted_length),
            strict_predicted_length=bool(args.strict_predicted_length),
            strict_predicted_length_ratio=float(args.strict_predicted_length_ratio),
            slot_residue_bias_scale=float(args.slot_residue_bias_scale),
            ligand_residue_bias_scale=float(args.ligand_residue_bias_scale),
            natural_aa_prior_scale=float(args.natural_aa_prior_scale),
            aa_token_ids=aa_token_ids,
        )

        if hasattr(tok_protein, "batch_decode"):
            pred = tok_protein.batch_decode(ids, skip_special_tokens=True)[0]
        else:
            pred = tok_protein.decode(ids[0].tolist())

        pred = pred.replace(" ", "").replace("\n", "").strip()
        pred = clean_protein_sequence(pred)

        print("=" * 80)
        print(f"[row {i}] ligand_smiles: {ligand_smiles}")
        print(f"[conditioning_text] {cond_text_raw[:200]}{'...' if len(cond_text_raw) > 200 else ''}")
        print("pred:", pred[:200] + ("..." if len(pred) > 200 else ""))

        rows_out.append(
            {
                "row_id": i,
                "sample_id": str(row.get("sample_id", f"row_{i}")),
                "condition_id": str(row.get("condition_id", "")),
                "ligand_smiles": ligand_smiles,
                "instruction": instruction,
                "protein_function_text": function_text,
                "docking_text_used": docking_text,
                "ligand_text_used": ligand_text,
                "conditioning_text": cond_text_raw,
                "pred_sequence": pred,
                "pred_sequence_length": len(pred),
                "predicted_target_length": predicted_length,
                "protein_length_scale": float(cfg.protein_length_scale),
                "force_predicted_length": bool(args.force_predicted_length),
                "strict_predicted_length": bool(args.strict_predicted_length),
                "strict_predicted_length_ratio": float(args.strict_predicted_length_ratio),
                "slot_residue_bias_scale": float(args.slot_residue_bias_scale),
                "ligand_residue_bias_scale": float(args.ligand_residue_bias_scale),
                "natural_aa_prior_scale": float(args.natural_aa_prior_scale),
                "omit_condition_prefix": bool(args.omit_condition_prefix),
                "full_injection_scale": float(args.full_injection_scale),
                "prefix_condition_scale": float(args.prefix_condition_scale),
                "protein_control_tag": str(args.protein_control_tag),
                "canonical_aa_only": bool(args.canonical_aa_only),
                "restore_base_decoder_backbone": bool(args.restore_base_decoder_backbone),
                "pool_condition_memory": bool(args.pool_condition_memory),
                "debug": json.dumps(debug_payload, ensure_ascii=False) if args.return_debug else "",
            }
        )
        # Long compute-budgeted panels must survive pre-emption or a shared-GPU
        # OOM.  Atomically checkpoint every completed row; the final write below
        # remains for backward compatibility.
        if args.out_csv:
            out_path = Path(args.out_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            pd.DataFrame(rows_out).to_csv(tmp_path, index=False)
            tmp_path.replace(out_path)
        sys.stdout.flush()

    if args.out_csv:
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(args.out_csv, index=False)
        print("Saved ->", args.out_csv)


if __name__ == "__main__":
    main()
