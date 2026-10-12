#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a resumable Chai-1 evaluation with aggregate-score summaries.

The external-runner mode delegates folding to:

    chai-lab fold <chai_input.fasta> <05_chai1/raw2>

The in-process mode uses the Chai Python API and can reuse loaded weights
across samples.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
MODEL_PATTERN = re.compile(r"scores\.model_idx_(\d+)\.npz$")


def safe_name(value: Any) -> str:
    text = re.sub(r"[^\w.-]+", "_", str(value)).strip("._")
    return text or "sample"


def atomic_write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(list(rows)).to_csv(tmp, index=False)
    tmp.replace(path)


def to_scalar(value: Any) -> Optional[float]:
    array = np.asarray(value)
    if not array.size:
        return None
    try:
        scalar = float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def sdf_to_smiles(path: Path) -> str:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    molecule = next((mol for mol in supplier if mol is not None), None)
    if molecule is None:
        raise ValueError(f"Unable to parse ligand SDF: {path}")
    smiles = Chem.MolToSmiles(molecule, canonical=True)
    if not smiles:
        raise ValueError(f"Empty ligand SMILES from: {path}")
    return smiles


def canonical_heavy_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Unable to parse expected ligand SMILES: {smiles}")
    return Chem.MolToSmiles(
        Chem.RemoveHs(molecule),
        canonical=True,
        isomericSmiles=False,
    )


def ligand_matches_expected(observed_smiles: str, expected_smiles: str) -> bool:
    """Compare a 3-D-derived ligand against the dataset's declared chemistry.

    RDKit can infer extra tetrahedral P/S stereochemistry from SDF coordinates
    even when the source SMILES leaves those centres unspecified.  First
    require an exact heavy-atom graph/charge match, then use the expected
    molecule as a chiral substructure query.  Thus every stereocentre declared
    by the dataset must match, while extra SDF-only assignments are accepted.
    """
    observed = Chem.MolFromSmiles(str(observed_smiles))
    expected = Chem.MolFromSmiles(str(expected_smiles))
    if observed is None or expected is None:
        raise ValueError("Unable to parse ligand SMILES for validation")
    observed = Chem.RemoveHs(observed)
    expected = Chem.RemoveHs(expected)
    if canonical_heavy_smiles(observed_smiles) != canonical_heavy_smiles(
        expected_smiles
    ):
        return False
    return observed.HasSubstructMatch(expected, useChirality=True)


def prepare_inputs(
    row: pd.Series, sample_dir: Path, require_pdb: bool = True
) -> Dict[str, Any]:
    sequence = re.sub(r"\s+", "", str(row["sequence"])).upper()
    invalid = sorted(set(sequence) - VALID_AA)
    if not sequence or invalid:
        raise ValueError(
            f"Invalid protein sequence for {row['sample_id']}: {invalid}"
        )
    pdb_value = str(row.get("pdb_path", "")).strip()
    pdb_source = Path(pdb_value).expanduser().resolve() if pdb_value else None
    ligand_source = Path(str(row["ligand_sdf_path"])).expanduser().resolve()
    if require_pdb and (pdb_source is None or not pdb_source.is_file()):
        raise FileNotFoundError(f"Missing PDB: {pdb_source}")
    if not ligand_source.is_file():
        raise FileNotFoundError(f"Missing SDF: {ligand_source}")

    inputs_dir = sample_dir / "00_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    ligand_target = inputs_dir / "ligand_input.sdf"
    pdb_target = None
    if pdb_source is not None and pdb_source.is_file():
        pdb_target = inputs_dir / "protein_input.pdb"
        shutil.copy2(pdb_source, pdb_target)
    shutil.copy2(ligand_source, ligand_target)
    smiles = sdf_to_smiles(ligand_target)
    expected_smiles = str(row.get("ligand_smiles", "")).strip()
    if expected_smiles and not ligand_matches_expected(smiles, expected_smiles):
        raise ValueError(
            f"Ligand SDF/SMILES mismatch for {row['sample_id']}: "
            f"{ligand_source}"
        )

    (inputs_dir / "protein.fasta").write_text(
        f">protein|name=protein\n{sequence}\n", encoding="utf-8"
    )
    (inputs_dir / "ligand.smiles").write_text(
        f"{smiles}\n", encoding="utf-8"
    )
    chai_input = inputs_dir / "chai_input.fasta"
    chai_input.write_text(
        (
            f">protein|name=protein\n{sequence}\n"
            f">ligand|name=ligand\n{smiles}\n"
        ),
        encoding="utf-8",
    )
    metadata = {
        "protein_pdb": str(pdb_target) if pdb_target is not None else None,
        "ligand_sdf": str(ligand_target),
        "protein_sequence_length": len(sequence),
        "ligand_smiles": smiles,
        "chai_input_fasta": str(chai_input),
    }
    (inputs_dir / "input_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def parse_metrics(sample_dir: Path) -> Dict[str, Any]:
    raw_dir = sample_dir / "05_chai1" / "raw2"
    models: List[Dict[str, Any]] = []
    for score_path in sorted(raw_dir.glob("scores.model_idx_*.npz")):
        match = MODEL_PATTERN.fullmatch(score_path.name)
        if not match:
            continue
        model_idx = int(match.group(1))
        with np.load(score_path, allow_pickle=True) as scores:
            item = {
                "model_idx": model_idx,
                "aggregate_score": (
                    to_scalar(scores["aggregate_score"])
                    if "aggregate_score" in scores
                    else None
                ),
                "ptm": to_scalar(scores["ptm"]) if "ptm" in scores else None,
                "iptm": (
                    to_scalar(scores["iptm"]) if "iptm" in scores else None
                ),
                "has_inter_chain_clashes": (
                    bool(
                        np.asarray(
                            scores["has_inter_chain_clashes"]
                        ).reshape(-1)[0]
                    )
                    if "has_inter_chain_clashes" in scores
                    else None
                ),
                "chain_chain_clashes": (
                    np.asarray(scores["chain_chain_clashes"]).tolist()
                    if "chain_chain_clashes" in scores
                    else None
                ),
            }
        models.append(item)

    scored = [item for item in models if item["aggregate_score"] is not None]
    best = max(scored, key=lambda item: item["aggregate_score"]) if scored else {}
    metrics = {
        "num_models": len(models),
        "best_model_idx": best.get("model_idx"),
        "best_aggregate_score": best.get("aggregate_score"),
        "best_ptm": best.get("ptm"),
        "best_iptm": best.get("iptm"),
        "best_has_inter_chain_clashes": best.get(
            "has_inter_chain_clashes"
        ),
        "best_chain_chain_clashes_json": json.dumps(
            best.get("chain_chain_clashes")
        ),
        "all_models_json": json.dumps(models),
    }
    return metrics


def run_one(
    row: pd.Series,
    out_dir: Path,
    runner: Path,
    gpu_id: int,
    downloads_dir: Path,
    timeout_seconds: int,
    expected_models: int,
    require_pdb: bool,
) -> Dict[str, Any]:
    sample_id = str(row["sample_id"])
    sample_dir = out_dir / "outputs" / safe_name(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / f"{safe_name(sample_id)}.log"
    metadata = prepare_inputs(row, sample_dir, require_pdb=require_pdb)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    env["CHAI_DOWNLOADS_DIR"] = str(downloads_dir)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                ["bash", str(runner), str(sample_dir)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=int(timeout_seconds),
                check=False,
                text=True,
            )
            exit_code: Optional[int] = int(completed.returncode)
            timed_out = False
        except subprocess.TimeoutExpired:
            exit_code = None
            timed_out = True
    elapsed = time.monotonic() - started
    metrics = parse_metrics(sample_dir)
    status = (
        "ok"
        if exit_code == 0 and int(metrics["num_models"]) == expected_models
        else "error"
    )
    if timed_out:
        error = f"timeout_after_{timeout_seconds}s"
    elif exit_code != 0:
        error = f"exit_code_{exit_code}"
    elif int(metrics["num_models"]) != expected_models:
        error = (
            f"expected_{expected_models}_models_found_{metrics['num_models']}"
        )
    else:
        error = ""
    return {
        "sample_id": sample_id,
        "sample_type": row.get("sample_type", "unspecified"),
        "source_row": row.get("source_row", ""),
        "sequence": row["sequence"],
        "structure_rank_score": row.get("structure_rank_score", ""),
        "pdb_path": str(Path(str(row.get("pdb_path", ""))).resolve()) if str(row.get("pdb_path", "")).strip() else "",
        "ligand_sdf_path": str(
            Path(str(row["ligand_sdf_path"])).resolve()
        ),
        "ligand_smiles": metadata["ligand_smiles"],
        "sample_dir": str(sample_dir),
        "log_path": str(log_path),
        "status": status,
        "error": error,
        "exit_code": exit_code,
        "elapsed_seconds": float(elapsed),
        **metrics,
    }


def enable_in_process_model_cache() -> Any:
    """Load Chai lazily and retain its read-only exported modules."""
    from chai_lab import chai1 as chai1_module

    if not getattr(chai1_module.load_exported, "_langpro_cached", False):
        original = chai1_module.load_exported
        cached = functools.lru_cache(maxsize=None)(original)
        cached._langpro_cached = True  # type: ignore[attr-defined]
        chai1_module.load_exported = cached
    return chai1_module


def run_one_in_process(
    row: pd.Series,
    out_dir: Path,
    chai1_module: Any,
    expected_models: int,
    num_diffn_timesteps: int,
    num_trunk_recycles: int,
    use_esm_embeddings: bool,
    require_pdb: bool,
) -> Dict[str, Any]:
    """Run one sample without restarting Python or reloading Chai models."""
    sample_id = str(row["sample_id"])
    sample_dir = out_dir / "outputs" / safe_name(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / f"{safe_name(sample_id)}.log"
    metadata = prepare_inputs(row, sample_dir, require_pdb=require_pdb)
    raw_dir = sample_dir / "05_chai1" / "raw2"
    raw_dir.mkdir(parents=True, exist_ok=True)
    recovered_metrics = parse_metrics(sample_dir)
    if int(recovered_metrics["num_models"]) == int(expected_models):
        log_path.write_text(
            "[RECOVERED] Complete Chai outputs existed before result-table "
            "reconstruction.\n",
            encoding="utf-8",
        )
        return {
            "sample_id": sample_id,
            "sample_type": row.get("sample_type", "unspecified"),
            "source_row": row.get("source_row", ""),
            "sequence": row["sequence"],
            "structure_rank_score": row.get("structure_rank_score", ""),
            "pdb_path": str(Path(str(row.get("pdb_path", ""))).resolve()) if str(row.get("pdb_path", "")).strip() else "",
            "ligand_sdf_path": str(
                Path(str(row["ligand_sdf_path"])).resolve()
            ),
            "ligand_smiles": metadata["ligand_smiles"],
            "sample_dir": str(sample_dir),
            "log_path": str(log_path),
            "status": "ok",
            "error": "",
            "exit_code": 0,
            "elapsed_seconds": 0.0,
            "recovered_complete_outputs": True,
            **recovered_metrics,
        }
    started = time.monotonic()
    run_error = ""
    with log_path.open("w", encoding="utf-8") as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                chai1_module.run_inference(
                    fasta_file=Path(metadata["chai_input_fasta"]),
                    output_dir=raw_dir,
                    use_esm_embeddings=bool(use_esm_embeddings),
                    num_trunk_recycles=int(num_trunk_recycles),
                    num_diffn_timesteps=int(num_diffn_timesteps),
                    num_diffn_samples=int(expected_models),
                    device="cuda:0",
                )
            exit_code: Optional[int] = 0
        except Exception as exc:
            exit_code = 1
            run_error = repr(exc)
            log.write(f"\n[ERROR] {run_error}\n")
    elapsed = time.monotonic() - started
    metrics = parse_metrics(sample_dir)
    status = (
        "ok"
        if exit_code == 0 and int(metrics["num_models"]) == expected_models
        else "error"
    )
    if run_error:
        error = run_error
    elif int(metrics["num_models"]) != expected_models:
        error = (
            f"expected_{expected_models}_models_found_{metrics['num_models']}"
        )
    else:
        error = ""
    return {
        "sample_id": sample_id,
        "sample_type": row.get("sample_type", "unspecified"),
        "source_row": row.get("source_row", ""),
        "sequence": row["sequence"],
        "structure_rank_score": row.get("structure_rank_score", ""),
        "pdb_path": str(Path(str(row.get("pdb_path", ""))).resolve()) if str(row.get("pdb_path", "")).strip() else "",
        "ligand_sdf_path": str(
            Path(str(row["ligand_sdf_path"])).resolve()
        ),
        "ligand_smiles": metadata["ligand_smiles"],
        "sample_dir": str(sample_dir),
        "log_path": str(log_path),
        "status": status,
        "error": error,
        "exit_code": exit_code,
        "elapsed_seconds": float(elapsed),
        **metrics,
    }


def write_summary(
    results: pd.DataFrame, out_dir: Path, expected_models: int
) -> None:
    groups: List[Dict[str, Any]] = []
    for sample_type, frame in results.groupby("sample_type", dropna=False):
        ok = frame.loc[frame["status"] == "ok"]
        row: Dict[str, Any] = {
            "sample_type": sample_type,
            "requested": int(len(frame)),
            "successful": int(len(ok)),
            "success_rate": float(len(ok) / len(frame)) if len(frame) else 0.0,
        }
        for metric in (
            "best_aggregate_score",
            "best_ptm",
            "best_iptm",
            "best_has_inter_chain_clashes",
            "elapsed_seconds",
        ):
            values = pd.to_numeric(ok.get(metric), errors="coerce")
            row[f"mean_{metric}"] = (
                float(values.mean()) if values.notna().any() else None
            )
            row[f"median_{metric}"] = (
                float(values.median()) if values.notna().any() else None
            )
        groups.append(row)
    pd.DataFrame(groups).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "protocol": (
                    "Original chai-lab fold defaults."
                    if expected_models == 5
                    else f"Chai-1 screening run with {expected_models} model(s)."
                ),
                "primary_metric": (
                    "Maximum aggregate_score among the five models; "
                    "higher is better."
                ),
                "groups": groups,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-4 Chai-1 evaluator")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--runner",
        default="",
        help="External Chai-1 runner used when --in-process is not selected.",
    )
    parser.add_argument(
        "--downloads-dir",
        default="pretrained/chai1",
        help="Chai-1 model/download cache.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--expected-models",
        type=int,
        default=5,
        help="Required number of Chai score files (default: original five).",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help=(
            "Run Chai through its Python API so sequential samples reuse "
            "ESM and exported model weights."
        ),
    )
    parser.add_argument(
        "--num-diffn-timesteps",
        type=int,
        default=200,
        help="Used only with --in-process (original protocol: 200).",
    )
    parser.add_argument(
        "--num-trunk-recycles",
        type=int,
        default=3,
        help="Used only with --in-process (original protocol: 3).",
    )
    parser.add_argument(
        "--no-esm-embeddings",
        action="store_true",
        help="Disable Chai ESM embeddings for a compute-budgeted screen.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--sequence-only",
        action="store_true",
        help="Do not require/copy a PDB; Chai consumes only sequence and ligand SMILES.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if not args.in_process and not args.runner:
        raise ValueError("--runner is required unless --in-process is selected")
    source = pd.read_csv(args.manifest)
    required = {"sample_id", "sequence", "ligand_sdf_path"}
    if not args.sequence_only:
        required.add("pdb_path")
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"
    runner = Path(args.runner).expanduser().resolve()
    downloads_dir = Path(args.downloads_dir).expanduser().resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"Missing original Chai-1 runner: {runner}")
    if not downloads_dir.is_dir():
        raise FileNotFoundError(f"Missing Chai-1 downloads: {downloads_dir}")
    chai1_module = None
    if args.in_process:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_id))
        os.environ["CHAI_DOWNLOADS_DIR"] = str(downloads_dir)
        chai1_module = enable_in_process_model_cache()

    by_id: Dict[str, Dict[str, Any]] = {}
    if (
        args.resume
        and results_path.is_file()
        and results_path.stat().st_size > 1
    ):
        try:
            previous = pd.read_csv(results_path).replace({np.nan: None})
        except pd.errors.EmptyDataError:
            previous = pd.DataFrame()
        by_id = {
            str(row["sample_id"]): row
            for row in previous.to_dict(orient="records")
        }
        completed = {
            sample_id
            for sample_id, row in by_id.items()
            if row.get("status") == "ok"
        }
        print(f"[INFO] Resuming with {len(completed)} successful samples.")
    else:
        completed = set()

    for _, source_row in source.iterrows():
        sample_id = str(source_row["sample_id"])
        if sample_id in completed:
            print(f"[SKIP] {sample_id}")
            continue
        try:
            if args.in_process:
                result = run_one_in_process(
                    source_row,
                    out_dir,
                    chai1_module,
                    args.expected_models,
                    args.num_diffn_timesteps,
                    args.num_trunk_recycles,
                    not bool(args.no_esm_embeddings),
                    require_pdb=not bool(args.sequence_only),
                )
            else:
                result = run_one(
                    source_row,
                    out_dir,
                    runner,
                    args.gpu_id,
                    downloads_dir,
                    args.timeout_seconds,
                    args.expected_models,
                    require_pdb=not bool(args.sequence_only),
                )
        except Exception as exc:
            result = {
                "sample_id": sample_id,
                "sample_type": source_row.get(
                    "sample_type", "unspecified"
                ),
                "source_row": source_row.get("source_row", ""),
                "status": "error",
                "error": repr(exc),
            }
            if args.fail_fast:
                by_id[sample_id] = result
                atomic_write_csv(by_id.values(), results_path)
                raise
        by_id[sample_id] = result
        atomic_write_csv(by_id.values(), results_path)
        print(
            f"[{result['status'].upper()}] {sample_id} "
            f"aggregate={result.get('best_aggregate_score')} "
            f"ptm={result.get('best_ptm')} "
            f"iptm={result.get('best_iptm')} "
            f"models={result.get('num_models')} "
            f"seconds={result.get('elapsed_seconds')}"
        )

    results = pd.DataFrame(list(by_id.values()))
    write_summary(results, out_dir, args.expected_models)
    print(f"[DONE] {len(results)} records -> {results_path}")


if __name__ == "__main__":
    main()
