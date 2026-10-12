"""Condition-only coarse fold planning for hierarchical protein generation.

The planner never receives target protein tokens. It maps multimodal condition
memories to ordered fold slots, predicts coarse secondary structure and
long-range topology, and exposes the planned slots to the autoregressive
decoder. A learned soft prototype bank parameterizes reusable scaffold motifs
without retrieving a sequence at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FoldPlannerOutput:
    tokens: torch.Tensor
    ss_logits: torch.Tensor
    contact_logits: torch.Tensor
    global_stats: torch.Tensor
    residue_prior_logits: torch.Tensor
    slot_residue_logits: torch.Tensor
    pocket_slot_logits: torch.Tensor
    archetype_logits: torch.Tensor
    prototype_assignments: torch.Tensor


class ConditionFoldPlanner(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_slots: int = 16,
        num_prototypes: int = 64,
        num_archetypes: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        prototype_temperature: float = 0.5,
        local_chemistry_scale: float = 0.15,
        archetype_scale: float = 0.20,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_slots < 4:
            raise ValueError("num_slots must be at least 4")
        if num_prototypes < num_slots:
            raise ValueError("num_prototypes must be >= num_slots")
        if num_archetypes < 4:
            raise ValueError("num_archetypes must be at least 4")
        self.num_slots = int(num_slots)
        self.num_prototypes = int(num_prototypes)
        self.num_archetypes = int(num_archetypes)
        self.prototype_temperature = float(prototype_temperature)
        self.local_chemistry_scale = float(local_chemistry_scale)
        self.archetype_scale = float(archetype_scale)
        self.queries = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model) * 0.02)
        self.archetype_templates = nn.Parameter(
            torch.randn(num_archetypes, num_slots, d_model) * 0.02
        )
        self.chemistry_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.condition_norm = nn.LayerNorm(d_model)
        self.chemistry_norm = nn.LayerNorm(d_model)
        self.plan_norm = nn.LayerNorm(d_model)
        self.pocket_norm = nn.LayerNorm(d_model)
        self.chemistry_cross = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.pocket_slot_query = nn.Linear(d_model, d_model, bias=False)
        self.pocket_slot_key = nn.Linear(d_model, d_model, bias=False)
        self.prototype_logits = nn.Linear(d_model, num_prototypes, bias=False)
        self.prototype_gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.archetype_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_archetypes),
        )
        self.archetype_gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        nn.init.constant_(self.archetype_gate[0].bias, -1.5)
        self.ss_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3)
        )
        contact_dim = max(32, d_model // 4)
        self.contact_left = nn.Linear(d_model, contact_dim, bias=False)
        self.contact_right = nn.Linear(d_model, contact_dim, bias=False)
        self.global_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2)
        )
        self.residue_prior_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 20)
        )
        self.slot_residue_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 20)
        )

    def forward(
        self,
        condition_memory: torch.Tensor,
        chemistry_memory: torch.Tensor | None = None,
        sample_prototypes: bool = False,
        hard_prototypes: bool = False,
    ) -> FoldPlannerOutput:
        if condition_memory.ndim != 3:
            raise ValueError("condition_memory must have shape [B,N,D]")
        batch = condition_memory.size(0)
        query = self.queries.unsqueeze(0).expand(batch, -1, -1)
        raw = self.decoder(query, self.condition_norm(condition_memory))
        if chemistry_memory is None:
            chemistry_memory = condition_memory
        chemistry_query = self.chemistry_query.unsqueeze(0).expand(batch, -1, -1)
        chemistry_memory = self.chemistry_norm(chemistry_memory)
        pocket_context, _ = self.chemistry_cross(
            chemistry_query,
            chemistry_memory,
            chemistry_memory,
            need_weights=False,
        )
        pocket_context = self.pocket_norm(pocket_context)
        pocket_slot_logits = (
            self.pocket_slot_query(raw)
            * self.pocket_slot_key(pocket_context)
        ).sum(dim=-1) / max(1.0, float(raw.size(-1)) ** 0.5)
        pocket_slot_gate = torch.softmax(pocket_slot_logits, dim=-1).unsqueeze(-1)
        localized = raw + (
            self.local_chemistry_scale
            * float(self.num_slots)
            * pocket_slot_gate
            * pocket_context
        )
        archetype_logits = self.archetype_head(raw.mean(dim=1))
        if hard_prototypes:
            archetype_probabilities = F.one_hot(
                archetype_logits.argmax(dim=-1), num_classes=self.num_archetypes
            ).to(archetype_logits)
        elif sample_prototypes:
            archetype_probabilities = F.gumbel_softmax(
                archetype_logits,
                tau=max(self.prototype_temperature, 1e-4),
                hard=False,
                dim=-1,
            )
        else:
            archetype_probabilities = torch.softmax(archetype_logits, dim=-1)
        archetype_tokens = torch.einsum(
            "ba,asd->bsd", archetype_probabilities, self.archetype_templates
        )
        localized = localized + (
            self.archetype_scale
            * self.archetype_gate(localized)
            * archetype_tokens
        )
        assignment_logits = self.prototype_logits(localized) / max(self.prototype_temperature, 1e-4)
        if hard_prototypes:
            assignments = F.one_hot(
                assignment_logits.argmax(dim=-1), num_classes=self.num_prototypes
            ).to(assignment_logits)
        elif sample_prototypes:
            assignments = F.gumbel_softmax(
                assignment_logits,
                tau=max(self.prototype_temperature, 1e-4),
                hard=False,
                dim=-1,
            )
        else:
            assignments = torch.softmax(assignment_logits, dim=-1)
        prototype_tokens = assignments @ self.prototypes
        tokens = self.plan_norm(
            localized + self.prototype_gate(localized) * prototype_tokens
        )
        left = self.contact_left(tokens)
        right = self.contact_right(tokens)
        contact_logits = 0.5 * (
            left @ right.transpose(1, 2) + right @ left.transpose(1, 2)
        ) / max(1.0, float(left.size(-1)) ** 0.5)
        return FoldPlannerOutput(
            tokens=tokens,
            ss_logits=self.ss_head(tokens),
            contact_logits=contact_logits,
            global_stats=self.global_head(tokens.mean(dim=1)),
            residue_prior_logits=self.residue_prior_head(pocket_context.squeeze(1)),
            slot_residue_logits=self.slot_residue_head(tokens),
            pocket_slot_logits=pocket_slot_logits,
            archetype_logits=archetype_logits,
            prototype_assignments=assignments,
        )


def coarse_secondary_structure_targets(
    labels: torch.Tensor,
    num_slots: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool residue classes into normalized sequence-position slots."""
    batch, _ = labels.shape
    target = torch.zeros(batch, num_slots, 3, device=labels.device, dtype=torch.float32)
    mask = torch.zeros(batch, num_slots, device=labels.device, dtype=torch.bool)
    for b in range(batch):
        valid_positions = torch.nonzero(labels[b].ge(0), as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue
        length = int(valid_positions.max().item()) + 1
        for slot in range(num_slots):
            start = (slot * length) // num_slots
            end = max(start + 1, ((slot + 1) * length) // num_slots)
            values = labels[b, start:min(end, length)]
            values = values[(values >= 0) & (values < 3)]
            if values.numel() == 0:
                continue
            target[b, slot] = F.one_hot(values.long(), num_classes=3).float().mean(dim=0)
            mask[b, slot] = True
    return target, mask


def coarse_contact_targets(
    contact_map: torch.Tensor,
    contact_mask: torch.Tensor,
    num_slots: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool long-range residue contacts into a coarse topology graph."""
    batch, length, _ = contact_map.shape
    target = torch.zeros(batch, num_slots, num_slots, device=contact_map.device)
    mask = torch.zeros_like(target, dtype=torch.bool)
    stats = torch.zeros(batch, 2, device=contact_map.device)
    for b in range(batch):
        valid_rows = torch.nonzero(contact_mask[b].any(dim=1), as_tuple=False).flatten()
        n = int(valid_rows.max().item()) + 1 if valid_rows.numel() else length
        positive_separations = []
        valid_pairs = 0.0
        positive_pairs = 0.0
        for i in range(num_slots):
            i0, i1 = (i * n) // num_slots, max((i * n) // num_slots + 1, ((i + 1) * n) // num_slots)
            for j in range(i + 1, num_slots):
                j0, j1 = (j * n) // num_slots, max((j * n) // num_slots + 1, ((j + 1) * n) // num_slots)
                block_mask = contact_mask[b, i0:min(i1, n), j0:min(j1, n)].bool()
                if not block_mask.any():
                    continue
                block_target = contact_map[b, i0:min(i1, n), j0:min(j1, n)]
                value = block_target[block_mask].max()
                target[b, i, j] = target[b, j, i] = value
                mask[b, i, j] = mask[b, j, i] = True
                valid_pairs += 1.0
                if value > 0.5:
                    positive_pairs += 1.0
                    positive_separations.append(float(j - i) / max(1, num_slots - 1))
        stats[b, 0] = positive_pairs / max(1.0, valid_pairs)
        stats[b, 1] = (
            sum(positive_separations) / len(positive_separations)
            if positive_separations else 0.0
        )
    return target, mask, stats


def fold_planner_losses(
    output: FoldPlannerOutput,
    ss_labels: torch.Tensor,
    contact_map: torch.Tensor,
    contact_mask: torch.Tensor,
    slot_residue_targets: torch.Tensor | None = None,
    slot_pocket_targets: torch.Tensor | None = None,
    archetype_targets: torch.Tensor | None = None,
    biochemical_prior: torch.Tensor | None = None,
    biochemical_strength: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    ss_target, ss_mask = coarse_secondary_structure_targets(ss_labels, output.tokens.size(1))
    ss_loss = -(ss_target * F.log_softmax(output.ss_logits, dim=-1)).sum(dim=-1)
    ss_loss = (ss_loss * ss_mask.float()).sum() / ss_mask.float().sum().clamp_min(1.0)
    coarse_contact, coarse_mask, global_target = coarse_contact_targets(
        contact_map, contact_mask, output.tokens.size(1)
    )
    contact_raw = F.binary_cross_entropy_with_logits(
        output.contact_logits, coarse_contact, reduction="none"
    )
    positive_weight = torch.where(coarse_contact > 0.5, 3.0, 1.0)
    contact_loss = (contact_raw * positive_weight * coarse_mask.float()).sum()
    contact_loss = contact_loss / (positive_weight * coarse_mask.float()).sum().clamp_min(1.0)
    global_loss = F.smooth_l1_loss(torch.sigmoid(output.global_stats), global_target)
    assignment_mean = output.prototype_assignments.mean(dim=(0, 1)).clamp_min(1e-8)
    balance_loss = (assignment_mean * torch.log(assignment_mean * assignment_mean.numel())).sum()
    entropy = -(output.prototype_assignments.clamp_min(1e-8).log() * output.prototype_assignments).sum(dim=-1)
    sharpness_loss = entropy.mean() / max(1.0, float(output.prototype_assignments.size(-1)))
    losses = {
        "fold_plan_ss": ss_loss,
        "fold_plan_contacts": contact_loss,
        "fold_plan_global": global_loss,
        "fold_plan_codebook": balance_loss + 0.1 * sharpness_loss,
    }
    if archetype_targets is not None:
        targets = archetype_targets.to(output.archetype_logits.device).long().view(-1)
        valid = targets.ge(0) & targets.lt(output.archetype_logits.size(-1))
        if valid.any():
            losses["fold_plan_archetype"] = F.cross_entropy(
                output.archetype_logits[valid], targets[valid]
            )
    if slot_residue_targets is not None:
        target = slot_residue_targets.to(output.slot_residue_logits).clamp_min(0.0)
        if target.shape != output.slot_residue_logits.shape:
            raise ValueError(
                "slot_residue_targets must match planner logits: "
                f"target={tuple(target.shape)}, logits={tuple(output.slot_residue_logits.shape)}"
            )
        background = target.new_tensor(
            [
                0.0825, 0.0137, 0.0545, 0.0675, 0.0386,
                0.0707, 0.0227, 0.0596, 0.0584, 0.0966,
                0.0242, 0.0406, 0.0471, 0.0393, 0.0553,
                0.0656, 0.0534, 0.0687, 0.0109, 0.0292,
            ]
        )
        background = background / background.sum()
        target = target + 0.05 * background.view(1, 1, -1)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        sequence_loss = -(
            target * F.log_softmax(output.slot_residue_logits, dim=-1)
        ).sum(dim=-1).mean()
        losses["fold_plan_sequence"] = sequence_loss

        if slot_pocket_targets is not None:
            pocket_target = slot_pocket_targets.to(output.pocket_slot_logits).clamp_min(0.0)
            if pocket_target.shape != output.pocket_slot_logits.shape:
                raise ValueError(
                    "slot_pocket_targets must match planner logits: "
                    f"target={tuple(pocket_target.shape)}, logits={tuple(output.pocket_slot_logits.shape)}"
                )
            valid = pocket_target.sum(dim=-1) > 0
            if valid.any():
                pocket_target = pocket_target + 1e-3
                pocket_target = pocket_target / pocket_target.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                losses["fold_plan_pocket"] = -(
                    pocket_target[valid]
                    * F.log_softmax(output.pocket_slot_logits[valid], dim=-1)
                ).sum(dim=-1).mean()
        elif biochemical_prior is not None:
            prior = biochemical_prior.to(target).clamp_min(0.0)
            prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            enrichment = (prior - background.view(1, -1)).clamp_min(0.0)
            pocket_target = (target * enrichment.unsqueeze(1)).sum(dim=-1)
            pocket_target = pocket_target + 1e-3
            pocket_target = pocket_target / pocket_target.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            pocket_loss = -(
                pocket_target
                * F.log_softmax(output.pocket_slot_logits, dim=-1)
            ).sum(dim=-1)
            if biochemical_strength is not None:
                weight = biochemical_strength.to(pocket_loss).view(-1).clamp(0.0, 1.0)
                pocket_loss = (pocket_loss * weight).sum() / weight.sum().clamp_min(1.0)
            else:
                pocket_loss = pocket_loss.mean()
            losses["fold_plan_pocket"] = pocket_loss
    if biochemical_prior is not None:
        target = biochemical_prior.to(output.residue_prior_logits).clamp_min(1e-8)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        chemistry = -(target * F.log_softmax(output.residue_prior_logits, dim=-1)).sum(dim=-1)
        if biochemical_strength is not None:
            weight = biochemical_strength.to(chemistry).view(-1).clamp(0.0, 1.0)
            chemistry = (chemistry * weight).sum() / weight.sum().clamp_min(1.0)
        else:
            chemistry = chemistry.mean()
        losses["fold_plan_chemistry"] = chemistry
    return losses
