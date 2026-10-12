#!/usr/bin/env python3
"""Pairwise Chai preference fine-tuning for the neural generator."""

from __future__ import annotations

import argparse
import json
import math
import os
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers.optimization import Adafactor

from ..generation.generate import _cfg_from_ckpt_dict
from .train import (
    build_model_batch,
    checkpoint_model_state,
    load_checkpoint_model_state,
    set_seed,
    to_device,
)


def cast_batch_floats(
    batch: dict[str, object], dtype: torch.dtype
) -> dict[str, object]:
    """Match floating inputs to a reduced-precision model, preserving IDs."""
    return {
        key: value.to(dtype=dtype)
        if torch.is_tensor(value) and torch.is_floating_point(value)
        else value
        for key, value in batch.items()
    }


def per_sequence_logp(output, labels: torch.Tensor) -> torch.Tensor:
    if output.logits is None or output.prefix_len is None:
        raise RuntimeError("model did not return logits/prefix length")
    prefix = int(output.prefix_len)
    ignored = torch.full(
        (labels.size(0), prefix), -100, dtype=labels.dtype, device=labels.device
    )
    full_labels = torch.cat([ignored, labels], dim=1)
    shift_logits = output.logits[:, :-1].float()
    shift_labels = full_labels[:, 1:]
    mask = shift_labels.ne(-100)
    safe = shift_labels.masked_fill(~mask, 0)
    token_logp = torch.log_softmax(shift_logits, dim=-1).gather(
        -1, safe.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logp * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def interface_residue_nll(
    output,
    labels: torch.Tensor,
    ligand_contact_labels: torch.Tensor,
) -> torch.Tensor:
    """NLL on chosen residues that contact the ligand in a true complex."""
    if output.logits is None or output.prefix_len is None:
        raise RuntimeError("model did not return logits/prefix length")
    prefix = int(output.prefix_len)
    ignored = torch.full(
        (labels.size(0), prefix), -100, dtype=labels.dtype, device=labels.device
    )
    full_labels = torch.cat([ignored, labels], dim=1)
    shift_logits = output.logits[:, :-1].float()
    shift_labels = full_labels[:, 1:]
    valid = shift_labels.ne(-100)
    safe = shift_labels.masked_fill(~valid, 0)
    all_logp = torch.log_softmax(shift_logits, dim=-1).gather(
        -1, safe.unsqueeze(-1)
    ).squeeze(-1)
    start = prefix - 1
    token_logp = all_logp[:, start:start + labels.size(1)]

    contacts = ligand_contact_labels.to(
        device=labels.device, dtype=token_logp.dtype
    )
    aligned = torch.zeros_like(token_logp)
    n = min(labels.size(1) - 1, contacts.size(1))
    if n > 0:
        # protein_labels[0] is ignored BOS; contact_labels[0] is residue 0.
        aligned[:, 1:1 + n] = contacts[:, :n]
    mask = labels.ne(-100) & torch.isfinite(aligned) & aligned.gt(0.5)
    if not mask.any():
        raise RuntimeError("preference batch has no chosen ligand-contact residues")
    return -(token_logp * mask.to(token_logp.dtype)).sum() / mask.sum().clamp_min(1)


def fold_profile_losses(
    fold_plan,
    slot_targets: torch.Tensor,
    slot_pocket_targets: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Supervise condition-only ordered residue profiles and pocket locality."""
    target = slot_targets.to(fold_plan.slot_residue_logits).clamp_min(0.0)
    valid = target.sum(dim=-1).gt(0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    slot_ce = -(target * F.log_softmax(fold_plan.slot_residue_logits, dim=-1)).sum(
        dim=-1
    )
    slot_weights = valid.to(slot_ce.dtype)
    pocket_loss = slot_ce.new_zeros(())
    if slot_pocket_targets is not None:
        pocket = slot_pocket_targets.to(slot_ce).clamp_min(0.0)
        if pocket.shape != slot_ce.shape:
            raise RuntimeError(
                f"slot pocket target shape {tuple(pocket.shape)} does not match "
                f"slot profile shape {tuple(slot_ce.shape)}"
            )
        slot_weights = slot_weights * (1.0 + pocket)
        rows_with_pocket = pocket.sum(dim=-1).gt(0)
        if rows_with_pocket.any():
            normalized = pocket / pocket.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            per_row = -(
                normalized
                * F.log_softmax(fold_plan.pocket_slot_logits, dim=-1)
            ).sum(dim=-1)
            pocket_loss = per_row[rows_with_pocket].mean()
    slot_loss = (slot_ce * slot_weights).sum() / slot_weights.sum().clamp_min(1.0)
    global_target = target.mean(dim=1)
    global_target = global_target / global_target.sum(dim=-1, keepdim=True).clamp_min(
        1e-8
    )
    global_loss = -(
        global_target * F.log_softmax(fold_plan.residue_prior_logits, dim=-1)
    ).sum(dim=-1).mean()
    return slot_loss, global_loss, pocket_loss


def batch_args_from_checkpoint(saved: dict) -> Namespace:
    values = dict(saved)
    defaults = {
        "neural_condition_only": True,
        "random_smiles_aug_prob": 0.0,
        "max_protein_len": 162,
        "protein_start_token_id": None,
        "lig_max_atoms": 64,
        "dock_max_chunks": 1,
        "func_max_chunks": 1,
        "ligtext_max_chunks": 1,
        "smiles_max_chunks": 1,
        "max_docking_len": 128,
        "max_func_len": 128,
        "max_ligtext_len": 128,
        "max_smiles_len": 128,
        "n_chem_features": 10,
        "n_interaction_types": 6,
        "structure_contact_min_sequence_separation": 12,
        "enable_condition_fold_planner": False,
        "enable_structure_token_planner": False,
        "structure_token_slots": 64,
        "structure_token_codes": 128,
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
    values["neural_condition_only"] = True
    # Pairwise ranking is valid only when chosen and rejected see the same
    # ligand condition. Independent randomized SMILES would confound the
    # sequence preference with two different encoder inputs.
    values["random_smiles_aug_prob"] = 0.0
    # Preference is prior-only. Target structure tokens must not act as a
    # posterior shortcut when scoring chosen and rejected sequences.
    values["enable_condition_fold_planner"] = False
    values["enable_structure_token_planner"] = False
    return Namespace(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--condition-dataset", required=True)
    parser.add_argument("--lig3d-npz", required=True)
    parser.add_argument("--hf-cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.20)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--length-weight", type=float, default=1.00)
    parser.add_argument("--interface-anchor-weight", type=float, default=0.0)
    parser.add_argument("--enable-fold-profile-route", action="store_true")
    parser.add_argument("--fold-profile-weight", type=float, default=0.0)
    parser.add_argument("--fold-pocket-weight", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--snapshot-steps",
        default="",
        help="Comma-separated optimizer steps at which compact checkpoints are saved.",
    )
    parser.add_argument("--detach-rejected", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--disable-structure-route", action="store_true")
    parser.add_argument("--lr-planner", type=float, default=3e-5)
    parser.add_argument("--lr-length", type=float, default=3e-5)
    parser.add_argument("--lr-new", type=float, default=2e-6)
    parser.add_argument("--lr-backbone", type=float, default=1e-7)
    parser.add_argument(
        "--optimizer", choices=("adamw", "adafactor"), default="adamw"
    )
    parser.add_argument(
        "--model-dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Parameter dtype used during preference tuning; bf16 substantially reduces GPU memory.",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    snapshot_steps = {
        int(value.strip())
        for value in str(args.snapshot_steps).split(",")
        if value.strip()
    }
    if any(value <= 0 for value in snapshot_steps):
        raise ValueError("--snapshot-steps values must be positive")
    set_seed(args.seed)

    hf_root = Path(args.hf_cache_root).resolve()
    os.environ.update(
        {
            "HF_HOME": str(hf_root),
            "TRANSFORMERS_CACHE": str(hf_root),
            "PUB_MED_BERT_DIR": str(hf_root / "PubMedBERT"),
            "SMILES_MODEL_DIR": str(hf_root / "SMILES_tokenized_PubChem_shard00_160k"),
            "PROGEN_MODEL_DIR": str(hf_root / "progen2-base-hf"),
        }
    )
    try:
        from ..architecture.model import DockingWizardModel
    except ImportError:
        from langpro.architecture.model import DockingWizardModel

    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    preserved_decoder_keys = {
        key for key in state if key.startswith("decoder.backbone.")
    }
    cfg = _cfg_from_ckpt_dict(payload.get("cfg", {}), state)
    cfg.decoder_backbone_name_or_path = str(hf_root / "progen2-base-hf")
    cfg.decoder_local_files_only = True
    if args.enable_fold_profile_route:
        cfg.enable_condition_fold_planner = True
        # The new planner first learns a condition-side profile. It affects
        # generation only through audited causal residue biases, not token
        # injection into the decoder hidden state.
        cfg.fold_plan_prefix_scale = 0.0
        cfg.fold_plan_full_memory_scale = 0.0
        cfg.fold_plan_sample_during_training = False
        cfg.fold_plan_sample_during_generation = False
        cfg.fold_plan_hard_during_generation = False
    if args.disable_structure_route:
        cfg.structure_token_prefix_scale = 0.0
        cfg.structure_token_full_memory_scale = 0.0
    device = torch.device(args.device)
    model_dtype = torch.bfloat16 if args.model_dtype == "bf16" else torch.float32
    model = DockingWizardModel(cfg).to(device=device, dtype=model_dtype)
    load_checkpoint_model_state(
        model, state, compact=str(payload.get("checkpoint_type", "")).startswith("compact")
    )
    saved_train_args = payload.get("args", {})
    del state, payload
    original_structure_sampling = bool(cfg.structure_token_sample_during_generation)
    original_structure_hard = bool(cfg.structure_token_hard_during_generation)
    cfg.structure_token_sample_during_generation = False
    cfg.structure_token_hard_during_generation = False
    # eval() disables dropout/Gumbel generation while retaining autograd. The
    # pairwise objective then scores both sequences on the deterministic soft
    # condition prior that is actually available at inference.
    model.eval()

    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            if name.startswith("decoder.backbone."):
                parameter.requires_grad_(False)

    buckets: dict[str, list[torch.nn.Parameter]] = {
        "planner": [],
        "length": [],
        "new": [],
        "backbone": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("structure_token_planner.", "fold_planner.")):
            buckets["planner"].append(parameter)
        elif name.startswith("protein_length_head."):
            buckets["length"].append(parameter)
        elif name.startswith("decoder.backbone."):
            buckets["backbone"].append(parameter)
        else:
            buckets["new"].append(parameter)
    optimizer_groups = [
        {"params": buckets["planner"], "lr": args.lr_planner},
        {"params": buckets["length"], "lr": args.lr_length},
        {"params": buckets["new"], "lr": args.lr_new},
        {"params": buckets["backbone"], "lr": args.lr_backbone},
    ]
    if args.optimizer == "adafactor":
        optimizer = Adafactor(
            optimizer_groups,
            lr=args.lr_new,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=0.01,
        )
    else:
        optimizer = AdamW(optimizer_groups, weight_decay=0.01)

    pairs = pd.read_csv(args.pairs)
    conditions = pd.read_csv(args.condition_dataset).set_index("pair_key", drop=False)
    unknown = sorted(set(pairs["condition_id"].astype(str)) - set(conditions.index.astype(str)))
    if unknown:
        raise ValueError(f"preference pairs reference unknown training conditions: {unknown[:10]}")
    ligands = np.load(args.lig3d_npz, allow_pickle=True)
    tok_pubmed = AutoTokenizer.from_pretrained(hf_root / "PubMedBERT", local_files_only=True, trust_remote_code=True)
    tok_smiles = AutoTokenizer.from_pretrained(hf_root / "SMILES_tokenized_PubChem_shard00_160k", local_files_only=True, trust_remote_code=True)
    tok_protein = AutoTokenizer.from_pretrained(hf_root / "progen2-base-hf", local_files_only=True, trust_remote_code=True)
    batching = batch_args_from_checkpoint(saved_train_args)
    if args.enable_fold_profile_route:
        batching.enable_condition_fold_planner = True
        batching.fold_plan_slots = int(cfg.fold_plan_slots)
    history = []
    step = 0
    output = Path(args.output_dir)
    (output / "ckpts").mkdir(parents=True, exist_ok=True)

    def save_preference_checkpoint(path: Path, completed_epochs: int) -> None:
        checkpoint_cfg = asdict(cfg)
        checkpoint_cfg["structure_token_sample_during_generation"] = (
            original_structure_sampling
        )
        checkpoint_cfg["structure_token_hard_during_generation"] = (
            original_structure_hard
        )
        saved_state = checkpoint_model_state(model, full_checkpoint=False)
        if preserved_decoder_keys:
            current_state = model.state_dict()
            saved_state.update(
                {
                    key: current_state[key]
                    for key in preserved_decoder_keys
                    if key in current_state
                }
            )
        torch.save(
            {
                "step": step,
                "model": saved_state,
                "cfg": checkpoint_cfg,
                "args": saved_train_args,
                "checkpoint_type": "compact_chai_preference_model_only",
                "preference_training": {
                    "pairs": str(Path(args.pairs).resolve()),
                    "beta": args.beta,
                    "anchor_weight": args.anchor_weight,
                    "length_weight": args.length_weight,
                    "interface_anchor_weight": args.interface_anchor_weight,
                    "fold_profile_weight": args.fold_profile_weight,
                    "fold_pocket_weight": args.fold_pocket_weight,
                    "fold_profile_route_enabled": bool(args.enable_fold_profile_route),
                    "rejected_logp_detached": bool(args.detach_rejected),
                    "preserved_source_decoder_keys": len(preserved_decoder_keys),
                    "backbone_frozen": bool(args.freeze_backbone),
                    "structure_route_disabled": bool(args.disable_structure_route),
                    "learning_rates": {
                        "planner": args.lr_planner,
                        "length": args.lr_length,
                        "new": args.lr_new,
                        "backbone": args.lr_backbone,
                    },
                    "optimizer": args.optimizer,
                    "model_dtype": args.model_dtype,
                    "completed_epochs": completed_epochs,
                    "uses_explicit_preference_weight": (
                        "preference_weight" in pairs.columns
                    ),
                    "fixed_evaluation_scores_used": False,
                },
            },
            path,
        )

    for epoch in range(int(args.epochs)):
        shuffled = pairs.sample(frac=1.0, random_state=args.seed + epoch).reset_index(drop=True)
        for start in range(0, len(shuffled), int(args.batch_size)):
            pair_batch = shuffled.iloc[start:start + int(args.batch_size)]
            source_rows = conditions.loc[pair_batch["condition_id"].astype(str)].reset_index(drop=True)
            chosen_rows = source_rows.copy()
            rejected_rows = source_rows.copy()
            chosen_rows["protein_sequence"] = pair_batch["chosen_sequence"].tolist()
            rejected_rows["protein_sequence"] = pair_batch["rejected_sequence"].tolist()
            target_columns = {
                column for column in source_rows.columns
                if column.startswith("structure_") or column in {
                    "pocket_labels", "ligand_contact_labels", "interaction_labels",
                    "position_weights", "conservation_scores", "burial_depth",
                    "solvent_accessibility", "secondary_structure_labels",
                    "biochemical_residue_prior", "biochemical_prior_strength",
                    "fold_archetype_id", "slot_residue_composition",
                    "slot_pocket_targets", "geometric_pocket_residue_scores",
                    "geometric_pocket_top1_labels", "pocket_asa",
                    "ligand_min_distance",
                }
            }
            chosen_drop = set(target_columns)
            if float(args.interface_anchor_weight) > 0:
                chosen_drop.discard("ligand_contact_labels")
            if args.enable_fold_profile_route:
                chosen_drop.discard("pocket_labels")
            chosen_rows = chosen_rows.drop(
                columns=sorted(chosen_drop & set(chosen_rows.columns))
            )
            rejected_rows = rejected_rows.drop(
                columns=sorted(target_columns & set(rejected_rows.columns))
            )
            chosen = cast_batch_floats(
                to_device(
                    build_model_batch(
                        chosen_rows,
                        ligands,
                        tok_pubmed,
                        tok_smiles,
                        tok_protein,
                        batching,
                    ),
                    device,
                ),
                model_dtype,
            )
            rejected = cast_batch_floats(
                to_device(
                    build_model_batch(
                        rejected_rows,
                        ligands,
                        tok_pubmed,
                        tok_smiles,
                        tok_protein,
                        batching,
                    ),
                    device,
                ),
                model_dtype,
            )
            chosen_contacts = chosen.get("ligand_contact_labels")
            chosen_slot_profiles = chosen.get("slot_residue_composition")
            chosen_slot_pockets = chosen.get("slot_pocket_targets")
            # Remove all target-derived structural annotations. Preference
            # scoring must use exactly the condition-prior inference route.
            for batch in (chosen, rejected):
                for key in list(batch):
                    if key.startswith("structure_") or key in {
                        "pocket_labels", "ligand_contact_labels", "interaction_labels",
                        "position_weights", "conservation_scores", "burial_depth",
                        "solvent_accessibility", "secondary_structure_labels",
                        "biochemical_residue_prior", "biochemical_prior_strength",
                        "fold_archetype_id", "slot_residue_composition",
                        "slot_pocket_targets", "geometric_pocket_residue_scores",
                        "geometric_pocket_top1_labels", "pocket_asa",
                        "ligand_min_distance",
                    }:
                        batch.pop(key, None)
            if args.detach_rejected:
                # A fixed rejected baseline avoids retaining two full ProGen
                # autograd graphs. Ranking still raises the chosen sequence
                # above its same-condition hard negative.
                with torch.no_grad():
                    rejected_out = model(rejected, compute_aux_losses=False)
                    rejected_logp = per_sequence_logp(
                        rejected_out, rejected["protein_labels"]
                    )
                chosen_out = model(chosen, compute_aux_losses=False)
                chosen_logp = per_sequence_logp(
                    chosen_out, chosen["protein_labels"]
                )
            else:
                chosen_out = model(chosen, compute_aux_losses=False)
                rejected_out = model(rejected, compute_aux_losses=False)
                chosen_logp = per_sequence_logp(
                    chosen_out, chosen["protein_labels"]
                )
                rejected_logp = per_sequence_logp(
                    rejected_out, rejected["protein_labels"]
                )
            if "preference_weight" in pair_batch.columns:
                pair_weights = pd.to_numeric(
                    pair_batch["preference_weight"], errors="raise"
                ).clip(0.25, 2.5)
            else:
                pair_weights = pd.to_numeric(
                    pair_batch["score_delta"], errors="raise"
                ).clip(0.05, 0.50) / 0.20
            preference_weight = torch.as_tensor(
                pair_weights.to_numpy(),
                dtype=chosen_logp.dtype,
                device=device,
            )
            ranking = -(
                preference_weight
                * F.logsigmoid(float(args.beta) * (chosen_logp - rejected_logp))
            ).mean()
            anchor = -chosen_logp.mean()
            chosen_length_loss = chosen_out.loss_aux.get("length_prediction")
            if float(args.length_weight) > 0 and chosen_length_loss is None:
                raise RuntimeError(
                    "checkpoint did not expose chosen-sequence length supervision"
                )
            if chosen_length_loss is None:
                chosen_length_loss = torch.zeros((), device=device)
            if float(args.interface_anchor_weight) > 0:
                if chosen_contacts is None:
                    raise RuntimeError(
                        "interface anchoring requires chosen ligand_contact_labels"
                    )
                interface_nll = interface_residue_nll(
                    chosen_out, chosen["protein_labels"], chosen_contacts
                )
            else:
                interface_nll = torch.zeros((), device=device)
            if args.enable_fold_profile_route:
                if chosen_out.fold_plan is None or chosen_slot_profiles is None:
                    raise RuntimeError(
                        "fold-profile training requires planner output and slot targets"
                    )
                fold_slot_loss, fold_global_loss, fold_pocket_loss = (
                    fold_profile_losses(
                        chosen_out.fold_plan,
                        chosen_slot_profiles,
                        chosen_slot_pockets,
                    )
                )
            else:
                fold_slot_loss = torch.zeros((), device=device)
                fold_global_loss = torch.zeros((), device=device)
                fold_pocket_loss = torch.zeros((), device=device)
            loss = (
                ranking
                + float(args.anchor_weight) * anchor
                + float(args.length_weight) * chosen_length_loss
                + float(args.interface_anchor_weight) * interface_nll
                + float(args.fold_profile_weight)
                * (fold_slot_loss + 0.5 * fold_global_loss)
                + float(args.fold_pocket_weight) * fold_pocket_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite Chai preference loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            history.append(
                {
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "ranking_loss": float(ranking.detach().cpu()),
                    "anchor_nll": float(anchor.detach().cpu()),
                    "chosen_length_loss": float(
                        chosen_length_loss.detach().cpu()
                    ),
                    "interface_residue_nll": float(interface_nll.detach().cpu()),
                    "fold_slot_profile_loss": float(fold_slot_loss.detach().cpu()),
                    "fold_global_profile_loss": float(fold_global_loss.detach().cpu()),
                    "fold_pocket_locality_loss": float(fold_pocket_loss.detach().cpu()),
                    "preference_accuracy": float((chosen_logp > rejected_logp).float().mean().cpu()),
                    "mean_logp_margin": float((chosen_logp - rejected_logp).mean().detach().cpu()),
                }
            )
            if int(args.log_every) > 0 and step % int(args.log_every) == 0:
                pd.DataFrame(history).to_csv(output / "train_history.csv", index=False)
                print(json.dumps(history[-1], sort_keys=True), flush=True)
            if step in snapshot_steps:
                pd.DataFrame(history).to_csv(output / "train_history.csv", index=False)
                save_preference_checkpoint(
                    output / "ckpts" / f"snapshot_step_{step:06d}.pt",
                    completed_epochs=epoch,
                )
            if int(args.max_steps) > 0 and step >= int(args.max_steps):
                break
        if int(args.max_steps) > 0 and step >= int(args.max_steps):
            pd.DataFrame(history).to_csv(output / "train_history.csv", index=False)
            break
        pd.DataFrame(history).to_csv(output / "train_history.csv", index=False)
        if epoch + 1 < int(args.epochs):
            save_preference_checkpoint(
                output / "ckpts" / f"snapshot_epoch_{epoch + 1:02d}.pt",
                completed_epochs=epoch + 1,
            )

    pd.DataFrame(history).to_csv(output / "train_history.csv", index=False)
    checkpoint_path = output / "ckpts" / "chai_preference.pt"
    save_preference_checkpoint(checkpoint_path, completed_epochs=int(args.epochs))
    summary = {
        "steps": step,
        "final_loss": history[-1]["loss"] if history else None,
        "final_preference_accuracy": history[-1]["preference_accuracy"] if history else None,
        "checkpoint": str(checkpoint_path),
        "fixed_evaluation_scores_used_for_training": False,
        "inference_input_schema": "SMILES + instruction + protein_function_text",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
