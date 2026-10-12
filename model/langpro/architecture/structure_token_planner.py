"""Condition-prior / structure-posterior scaffold token planning.

The planner represents a protein scaffold as an ordered sequence of discrete
geometry codes. During training, a posterior sees target structure-token IDs;
at inference only the condition prior is available. No protein sequence,
reference structure, or retrieval index is read at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StructureTokenPlannerOutput:
    tokens: torch.Tensor
    prefix_tokens: torch.Tensor
    prior_logits: torch.Tensor
    posterior_logits: Optional[torch.Tensor]
    token_probabilities: torch.Tensor
    pocket_logits: torch.Tensor
    interaction_logits: torch.Tensor
    topology_logits: torch.Tensor
    posterior_used: bool


class StructureTokenScaffoldPlanner(nn.Module):
    """Generate residue-aligned scaffold codes from condition memories."""

    def __init__(
        self,
        d_model: int,
        num_slots: int = 64,
        num_codes: int = 128,
        prefix_slots: int = 16,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        posterior_temperature: float = 0.5,
        posterior_teacher_forcing: float = 0.75,
        num_interaction_types: int = 6,
        geometry_feature_dim: int = 10,
    ) -> None:
        super().__init__()
        if num_slots < 16:
            raise ValueError("num_slots must be at least 16")
        if num_codes < 16:
            raise ValueError("num_codes must be at least 16")
        if not 1 <= prefix_slots <= num_slots:
            raise ValueError("prefix_slots must be in [1, num_slots]")
        if not 0.0 <= posterior_teacher_forcing <= 1.0:
            raise ValueError("posterior_teacher_forcing must be in [0, 1]")
        self.num_slots = int(num_slots)
        self.num_codes = int(num_codes)
        self.prefix_slots = int(prefix_slots)
        self.posterior_temperature = float(posterior_temperature)
        self.posterior_teacher_forcing = float(posterior_teacher_forcing)

        self.slot_queries = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        self.codebook = nn.Parameter(torch.randn(num_codes, d_model) * 0.02)
        self.register_buffer(
            "geometry_code_features",
            torch.zeros(num_codes, int(geometry_feature_dim)),
            persistent=True,
        )
        self.geometry_code_projection = nn.Linear(
            int(geometry_feature_dim), d_model, bias=False
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prior_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        posterior_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.posterior_encoder = nn.TransformerEncoder(
            posterior_layer, num_layers=max(1, num_layers - 1)
        )
        self.condition_norm = nn.LayerNorm(d_model)
        self.plan_norm = nn.LayerNorm(d_model)
        self.prior_head = nn.Linear(d_model, num_codes)
        self.posterior_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_codes),
        )
        self.code_gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.pocket_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 1)
        )
        self.interaction_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, num_interaction_types)
        )
        topology_dim = max(32, d_model // 4)
        self.topology_left = nn.Linear(d_model, topology_dim, bias=False)
        self.topology_right = nn.Linear(d_model, topology_dim, bias=False)

    def _prefix_pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.prefix_slots == self.num_slots:
            return tokens
        return F.adaptive_avg_pool1d(
            tokens.transpose(1, 2), self.prefix_slots
        ).transpose(1, 2)

    @torch.no_grad()
    def set_geometry_code_features(self, features: torch.Tensor) -> None:
        features = torch.as_tensor(
            features,
            dtype=self.geometry_code_features.dtype,
            device=self.geometry_code_features.device,
        )
        if features.shape != self.geometry_code_features.shape:
            raise ValueError(
                "geometry code features must match planner codebook: "
                f"got={tuple(features.shape)} expected={tuple(self.geometry_code_features.shape)}"
            )
        self.geometry_code_features.copy_(features)

    def forward(
        self,
        condition_memory: torch.Tensor,
        target_token_ids: Optional[torch.Tensor] = None,
        sample_prior: bool = False,
        hard_prior: bool = False,
        posterior_mix: Optional[float] = None,
    ) -> StructureTokenPlannerOutput:
        if condition_memory.ndim != 3:
            raise ValueError("condition_memory must have shape [B,N,D]")
        batch = condition_memory.size(0)
        queries = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        raw = self.prior_decoder(queries, self.condition_norm(condition_memory))
        prior_logits = self.prior_head(raw)

        posterior_logits: Optional[torch.Tensor] = None
        posterior_used = bool(self.training and target_token_ids is not None)
        if posterior_used:
            targets = target_token_ids.to(device=raw.device, dtype=torch.long)
            if targets.shape != prior_logits.shape[:2]:
                raise ValueError(
                    "target structure tokens must match planner slots: "
                    f"targets={tuple(targets.shape)} slots={tuple(prior_logits.shape[:2])}"
                )
            safe_targets = targets.clamp(0, self.num_codes - 1)
            effective_codebook = self.codebook + self.geometry_code_projection(
                self.geometry_code_features
            )
            target_embeddings = F.embedding(safe_targets, effective_codebook)
            posterior_state = self.posterior_encoder(raw + target_embeddings)
            posterior_logits = prior_logits + self.posterior_head(posterior_state)
            posterior_probabilities = F.gumbel_softmax(
                posterior_logits,
                tau=max(self.posterior_temperature, 1e-4),
                hard=False,
                dim=-1,
            )
            prior_probabilities = torch.softmax(prior_logits, dim=-1)
            mix = float(
                self.posterior_teacher_forcing
                if posterior_mix is None else posterior_mix
            )
            mix = min(1.0, max(0.0, mix))
            token_probabilities = (
                mix * posterior_probabilities + (1.0 - mix) * prior_probabilities
            )
        elif hard_prior:
            token_probabilities = F.one_hot(
                prior_logits.argmax(dim=-1), num_classes=self.num_codes
            ).to(prior_logits)
        elif sample_prior:
            token_probabilities = F.gumbel_softmax(
                prior_logits,
                tau=max(self.posterior_temperature, 1e-4),
                hard=False,
                dim=-1,
            )
        else:
            token_probabilities = torch.softmax(prior_logits, dim=-1)

        effective_codebook = self.codebook + self.geometry_code_projection(
            self.geometry_code_features
        )
        code_tokens = token_probabilities @ effective_codebook
        tokens = self.plan_norm(raw + self.code_gate(raw) * code_tokens)
        left = self.topology_left(tokens)
        right = self.topology_right(tokens)
        topology_logits = 0.5 * (
            left @ right.transpose(1, 2) + right @ left.transpose(1, 2)
        ) / max(1.0, float(left.size(-1)) ** 0.5)
        return StructureTokenPlannerOutput(
            tokens=tokens,
            prefix_tokens=self._prefix_pool(tokens),
            prior_logits=prior_logits,
            posterior_logits=posterior_logits,
            token_probabilities=token_probabilities,
            pocket_logits=self.pocket_head(tokens).squeeze(-1),
            interaction_logits=self.interaction_head(tokens),
            topology_logits=topology_logits,
            posterior_used=posterior_used,
        )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def structure_token_planner_losses(
    output: StructureTokenPlannerOutput,
    token_targets: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    pocket_targets: Optional[torch.Tensor] = None,
    interaction_targets: Optional[torch.Tensor] = None,
    topology_targets: Optional[torch.Tensor] = None,
    topology_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    targets = token_targets.to(output.prior_logits.device).long()
    valid = targets.ge(0) & targets.lt(output.prior_logits.size(-1))
    if token_mask is not None:
        valid = valid & token_mask.to(device=valid.device).bool()
    safe_targets = targets.clamp(0, output.prior_logits.size(-1) - 1)
    counts = torch.bincount(
        safe_targets[valid], minlength=output.prior_logits.size(-1)
    ).to(output.prior_logits.dtype)
    observed = counts > 0
    mean_count = counts[observed].mean() if bool(observed.any()) else counts.new_tensor(1.0)
    code_weights = torch.ones_like(counts)
    code_weights[observed] = torch.sqrt(
        mean_count / counts[observed].clamp_min(1.0)
    ).clamp(0.5, 5.0)
    prior_ce = F.cross_entropy(
        output.prior_logits.transpose(1, 2), safe_targets,
        reduction="none", weight=code_weights,
    )
    losses: Dict[str, torch.Tensor] = {
        "structure_token_prior": _masked_mean(prior_ce, valid),
    }
    if output.posterior_logits is not None:
        posterior_ce = F.cross_entropy(
            output.posterior_logits.transpose(1, 2), safe_targets,
            reduction="none", weight=code_weights,
        )
        losses["structure_token_posterior"] = _masked_mean(posterior_ce, valid)
        q_log = F.log_softmax(output.posterior_logits, dim=-1)
        p_log = F.log_softmax(output.prior_logits, dim=-1)
        q = q_log.exp()
        kl = (q * (q_log - p_log)).sum(dim=-1)
        losses["structure_token_kl"] = _masked_mean(kl, valid)

    if pocket_targets is not None:
        pocket = pocket_targets.to(output.pocket_logits).clamp(0.0, 1.0)
        pocket_valid = torch.isfinite(pocket)
        if token_mask is not None:
            pocket_valid = pocket_valid & token_mask.to(pocket_valid.device).bool()
        safe = torch.nan_to_num(pocket, nan=0.0)
        positives = (safe[pocket_valid] > 0.5).sum().to(output.pocket_logits.dtype)
        negatives = pocket_valid.sum().to(output.pocket_logits.dtype) - positives
        pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
        raw = F.binary_cross_entropy_with_logits(
            output.pocket_logits, safe, reduction="none", pos_weight=pos_weight
        )
        losses["structure_token_pocket"] = _masked_mean(raw, pocket_valid)

    if interaction_targets is not None:
        interaction = interaction_targets.to(output.interaction_logits).clamp(0.0, 1.0)
        if interaction.shape != output.interaction_logits.shape:
            raise ValueError(
                "interaction targets must match logits: "
                f"targets={tuple(interaction.shape)} logits={tuple(output.interaction_logits.shape)}"
            )
        interaction_valid = torch.isfinite(interaction)
        if token_mask is not None:
            interaction_valid = interaction_valid & token_mask.to(
                interaction_valid.device
            ).bool().unsqueeze(-1)
        positive = ((interaction > 0.5) & interaction_valid).sum(dim=(0, 1)).to(
            output.interaction_logits.dtype
        )
        negative = interaction_valid.sum(dim=(0, 1)).to(
            output.interaction_logits.dtype
        ) - positive
        pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
        # Completely unobserved classes (for example metal coordination in a
        # metal-free minibatch) stay masked and must not affect the weights.
        pos_weight = torch.where(
            interaction_valid.any(dim=(0, 1)), pos_weight, torch.ones_like(pos_weight)
        )
        raw = F.binary_cross_entropy_with_logits(
            output.interaction_logits,
            torch.nan_to_num(interaction, nan=0.0),
            reduction="none",
            pos_weight=pos_weight,
        )
        losses["structure_token_interaction"] = _masked_mean(raw, interaction_valid)

    if topology_targets is not None:
        topology = topology_targets.to(output.topology_logits).clamp(0.0, 1.0)
        valid_topology = torch.isfinite(topology)
        if topology_mask is not None:
            valid_topology = valid_topology & topology_mask.to(
                valid_topology.device
            ).bool()
        positives = ((topology > 0.5) & valid_topology).sum().to(
            output.topology_logits.dtype
        )
        negatives = valid_topology.sum().to(output.topology_logits.dtype) - positives
        pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
        raw = F.binary_cross_entropy_with_logits(
            output.topology_logits,
            torch.nan_to_num(topology, nan=0.0),
            reduction="none",
            pos_weight=pos_weight,
        )
        losses["structure_token_topology"] = _masked_mean(raw, valid_topology)
    return losses
