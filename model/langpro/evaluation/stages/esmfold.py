#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible stage-1 ESMFold evaluation for generated proteins.

The historical evaluator only stored ``outputs.plddt.mean()``.  ESMFold's
``plddt`` tensor has 37 atom slots per residue, so this script additionally
reports the standard residue-level C-alpha pLDDT, pTM and PAE.  Results are
written after every sample so an interrupted GPU run can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch


AA20 = set("ACDEFGHIKLMNPQRSTVWY")
CA_INDEX = 1


def set_offline_env(hf_home: Optional[str], offline: bool) -> None:
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_home) / "transformers")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def validate_model_dir(model_dir: Path) -> None:
    required = ["config.json"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    has_weights = any(
        (model_dir / name).is_file()
        for name in ("pytorch_model.bin", "model.safetensors")
    )
    if missing or not has_weights:
        found = sorted(p.name for p in model_dir.glob("*"))
        raise FileNotFoundError(
            f"Invalid ESMFold model directory: {model_dir}; "
            f"missing={missing}, has_weights={has_weights}, found={found}"
        )


def pick_device(device: str) -> str:
    if not device.startswith("cuda"):
        return device
    if not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; falling back to CPU.", file=sys.stderr)
        return "cpu"
    try:
        index = int(device.split(":", 1)[1]) if ":" in device else 0
    except ValueError:
        index = 0
    if not 0 <= index < torch.cuda.device_count():
        print(
            f"[WARN] {device} is invalid for {torch.cuda.device_count()} GPUs; "
            "using cuda:0.",
            file=sys.stderr,
        )
        return "cuda:0"
    return f"cuda:{index}"


def patch_safe_compute_tm() -> None:
    """Compute pTM in fp32 and keep folding usable if only pTM fails."""
    from transformers.models.esm.openfold_utils import loss as of_loss
    # ``modeling_esmfold`` imports ``compute_tm`` into its own module namespace.
    # Patching only ``openfold_utils.loss.compute_tm`` therefore leaves the
    # already-bound reference untouched on several Transformers releases.
    # Patch both locations so fp16 NaNs in the pTM head cannot abort an
    # otherwise valid structure prediction.
    from transformers.models.esm import modeling_esmfold

    if getattr(of_loss, "_langpro_safe_tm_patched", False):
        return
    original = of_loss.compute_tm

    def safe_compute_tm(ptm_logits, max_bin=31, no_bins=64):
        try:
            value = original(ptm_logits.float(), max_bin=max_bin, no_bins=no_bins)
            if torch.isfinite(value).all():
                return value
        except Exception:
            pass
        batch = int(ptm_logits.shape[0]) if hasattr(ptm_logits, "shape") else 1
        device = ptm_logits.device if hasattr(ptm_logits, "device") else None
        return torch.full((batch,), float("nan"), device=device)

    of_loss.compute_tm = safe_compute_tm
    modeling_esmfold.compute_tm = safe_compute_tm
    of_loss._langpro_safe_tm_patched = True


def sanitize_sequence(value: Any) -> str:
    seq = re.sub(r"\s+", "", str(value or "")).upper()
    return "" if seq == "NAN" else seq


def safe_name(value: Any) -> str:
    text = re.sub(r"[^\w.-]+", "_", str(value)).strip("._")
    return text or "sample"


def scalar(value: Any) -> float:
    if value is None:
        return float("nan")
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().float().reshape(-1)[0].cpu().item())
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0]) if arr.size else float("nan")


def finite_or_none(value: Any) -> Optional[float]:
    number = scalar(value)
    return number if math.isfinite(number) else None


def load_model(
    model_dir: Path,
    device: str,
    dtype: torch.dtype,
    chunk_size: Optional[int],
):
    from transformers import AutoTokenizer, EsmForProteinFolding

    print(f"[INFO] Loading ESMFold from {model_dir}")
    print(f"[INFO] device={device}, dtype={dtype}, chunk_size={chunk_size}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    kwargs = {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    try:
        model = EsmForProteinFolding.from_pretrained(
            str(model_dir), dtype=dtype, **kwargs
        )
    except TypeError:
        model = EsmForProteinFolding.from_pretrained(
            str(model_dir), torch_dtype=dtype, **kwargs
        )
    model = model.to(device).eval()
    if chunk_size and hasattr(model, "trunk") and hasattr(model.trunk, "set_chunk_size"):
        model.trunk.set_chunk_size(int(chunk_size))
    return tokenizer, model


def _ca_coordinates(outputs) -> np.ndarray:
    """Return predicted C-alpha coordinates from the final recycle."""
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

    atom37 = atom14_to_atom37(outputs.positions[-1], outputs)
    return atom37[0, :, CA_INDEX].detach().float().cpu().numpy()


def fold_one(
    sequence: str,
    tokenizer,
    model,
    device: str,
    num_recycles: Optional[int],
) -> Tuple[str, Dict[str, Optional[float]]]:
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    if num_recycles is not None:
        inputs["num_recycles"] = int(num_recycles)

    started = time.monotonic()
    with torch.inference_mode():
        outputs = model(**inputs)
    elapsed = time.monotonic() - started
    pdb_text = model.output_to_pdb(outputs)[0]

    plddt = outputs.plddt[0].detach().float()
    ca_plddt = plddt[:, CA_INDEX].cpu().numpy()
    atom_mask = outputs.atom37_atom_exists[0].detach().bool()
    atom_plddt = plddt[atom_mask].cpu().numpy()
    coords = _ca_coordinates(outputs)
    centered = coords - coords.mean(axis=0, keepdims=True)
    radius_gyration = float(np.sqrt(np.square(centered).sum(axis=1).mean()))
    adjacent = np.linalg.norm(coords[1:] - coords[:-1], axis=1)

    pae = getattr(outputs, "predicted_aligned_error", None)
    pae_array = (
        pae[0].detach().float().cpu().numpy()
        if torch.is_tensor(pae) and pae.numel()
        else np.asarray([], dtype=np.float32)
    )
    ptm = finite_or_none(getattr(outputs, "ptm", None))
    mean_ca = float(np.mean(ca_plddt))
    metrics: Dict[str, Optional[float]] = {
        # Legacy value retained only for comparison with old result files.
        "legacy_all_slot_mean_plddt": float(plddt.mean().item()),
        "mean_atom_plddt": float(np.mean(atom_plddt)),
        "mean_ca_plddt": mean_ca,
        "median_ca_plddt": float(np.median(ca_plddt)),
        "min_ca_plddt": float(np.min(ca_plddt)),
        "frac_ca_plddt_ge_0_50": float(np.mean(ca_plddt >= 0.50)),
        "frac_ca_plddt_ge_0_70": float(np.mean(ca_plddt >= 0.70)),
        "frac_ca_plddt_ge_0_80": float(np.mean(ca_plddt >= 0.80)),
        "frac_ca_plddt_ge_0_90": float(np.mean(ca_plddt >= 0.90)),
        "ptm": ptm,
        "mean_pae": float(np.mean(pae_array)) if pae_array.size else None,
        "median_pae": float(np.median(pae_array)) if pae_array.size else None,
        "radius_gyration_angstrom": radius_gyration,
        "mean_adjacent_ca_distance": (
            float(np.mean(adjacent)) if adjacent.size else None
        ),
        "max_adjacent_ca_distance": (
            float(np.max(adjacent)) if adjacent.size else None
        ),
        "fold_seconds": float(elapsed),
    }
    # A transparent ranking score, not a biological binding metric.
    metrics["structure_rank_score"] = (
        0.70 * mean_ca + 0.30 * ptm if ptm is not None else mean_ca
    )
    return pdb_text, metrics


def atomic_write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(list(rows)).to_csv(tmp, index=False)
    tmp.replace(path)


def write_summary(results: pd.DataFrame, out_dir: Path) -> None:
    ok = results.loc[results["status"] == "ok"].copy()
    metrics = [
        "mean_ca_plddt",
        "ptm",
        "mean_pae",
        "frac_ca_plddt_ge_0_70",
        "structure_rank_score",
        "fold_seconds",
    ]
    rows = []
    group_col = "sample_type" if "sample_type" in results.columns else None
    groups = results.groupby(group_col, dropna=False) if group_col else [("all", results)]
    for group, frame in groups:
        ok_group = frame.loc[frame["status"] == "ok"]
        row: Dict[str, Any] = {
            "sample_type": group,
            "requested": int(len(frame)),
            "successful": int(len(ok_group)),
            "success_rate": float(len(ok_group) / len(frame)) if len(frame) else 0.0,
        }
        for metric in metrics:
            raw_values = (
                ok_group[metric]
                if metric in ok_group.columns
                else pd.Series(dtype=np.float64)
            )
            values = pd.to_numeric(raw_values, errors="coerce")
            row[f"mean_{metric}"] = (
                float(values.mean()) if values.notna().any() else None
            )
            row[f"median_{metric}"] = (
                float(values.median()) if values.notna().any() else None
            )
        rows.append(row)
    summary = {
        "metric_notes": {
            "mean_ca_plddt": "Mean residue C-alpha pLDDT; higher is better, range 0-1.",
            "ptm": "ESMFold predicted TM score; higher is better, range 0-1.",
            "mean_pae": "Mean predicted aligned error in Angstrom; lower is better.",
            "structure_rank_score": "0.70 * mean_ca_plddt + 0.30 * ptm.",
            "legacy_all_slot_mean_plddt": (
                "Historical outputs.plddt.mean() over 37 atom slots; retained "
                "for compatibility and not used for ranking."
            ),
        },
        "groups": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-1 ESMFold evaluator")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--seq-col", default="sequence")
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--group-col", default="sample_type")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Local ESMFold model directory.",
    )
    parser.add_argument(
        "--hf-home",
        default="pretrained/huggingface",
        help="Hugging Face cache directory.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--num-recycles", type=int, default=None)
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--no-offline", dest="offline", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    set_offline_env(args.hf_home, args.offline)
    model_dir = Path(args.model_dir).expanduser().resolve()
    validate_model_dir(model_dir)
    out_dir = Path(args.out_dir).expanduser().resolve()
    pdb_dir = out_dir / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"

    source = pd.read_csv(args.csv)
    for col in (args.seq_col, args.id_col):
        if col not in source.columns:
            raise ValueError(f"Missing column {col!r}; available={list(source.columns)}")

    rows = []
    completed = set()
    if args.resume and results_path.is_file():
        previous = pd.read_csv(results_path).replace({np.nan: None})
        rows = previous.to_dict(orient="records")
        completed = {
            str(row["sample_id"])
            for row in rows
            if row.get("status") == "ok"
        }
        print(f"[INFO] Resuming with {len(completed)} completed samples.")

    pending = [
        row
        for _, row in source.iterrows()
        if str(row[args.id_col]) not in completed
    ]
    if not pending:
        results = pd.DataFrame(rows)
        write_summary(results, out_dir)
        print("[DONE] No pending samples.")
        return

    patch_safe_compute_tm()
    device = pick_device(args.device)
    dtype = (
        torch.float16
        if args.dtype == "fp16" and device.startswith("cuda")
        else torch.float32
    )
    tokenizer, model = load_model(
        model_dir, device, dtype, args.chunk_size or None
    )

    total = len(source)
    for source_row in pending:
        sample_id = str(source_row[args.id_col])
        sequence = sanitize_sequence(source_row[args.seq_col])
        sample_type = (
            str(source_row[args.group_col])
            if args.group_col in source.columns
            else "unspecified"
        )
        invalid = sorted(set(sequence) - AA20)
        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "sample_type": sample_type,
            "source_row": source_row.get("source_row", ""),
            "sequence": sequence,
            "sequence_length": len(sequence),
            "pdb_path": str(pdb_dir / f"esmfold_{safe_name(sample_id)}.pdb"),
            "status": "ok",
            "error": "",
        }
        if not sequence:
            record.update(status="error", error="empty_sequence")
        elif invalid:
            record.update(status="error", error=f"invalid_amino_acids:{invalid}")
        else:
            try:
                pdb_text, metrics = fold_one(
                    sequence, tokenizer, model, device, args.num_recycles
                )
                Path(record["pdb_path"]).write_text(pdb_text, encoding="utf-8")
                record.update(metrics)
                print(
                    f"[OK] {sample_id} type={sample_type} len={len(sequence)} "
                    f"CA-pLDDT={metrics['mean_ca_plddt']:.4f} "
                    f"pTM={metrics['ptm']} score={metrics['structure_rank_score']:.4f}"
                )
            except Exception as exc:
                record.update(status="error", error=repr(exc))
                print(f"[ERROR] {sample_id}: {exc!r}", file=sys.stderr)
                if args.fail_fast:
                    rows.append(record)
                    atomic_write_csv(rows, results_path)
                    raise
        rows.append(record)
        atomic_write_csv(rows, results_path)

    results = pd.DataFrame(rows)
    write_summary(results, out_dir)
    print(f"[DONE] {len(results)}/{total} records -> {results_path}")


if __name__ == "__main__":
    main()
