#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the project's original DiffDock container protocol with robust metrics.

Defaults match ``stage3_diffdock/03_diffdock_udocker_batch_indexed.sh``:
DiffDock v1.1 config, 10 poses, batch size 1, and the config's 20/19
inference/actual steps.  Each row is checkpointed so long runs can resume.
"""

from __future__ import annotations

import argparse
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


POSE_PATTERN = re.compile(
    r"rank(?P<rank>\d+)_confidence(?P<confidence>-?\d+(?:\.\d+)?)\.sdf$"
)


def safe_name(value: Any) -> str:
    text = re.sub(r"[^\w.-]+", "_", str(value)).strip("._")
    return text or "sample"


def sdf_graph_counts(path: Path) -> tuple[Optional[int], Optional[int]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) >= 4 and "V2000" in lines[3]:
        try:
            return int(lines[3][0:3]), int(lines[3][3:6])
        except ValueError:
            pass
    for line in lines:
        match = re.search(r"M  V30 COUNTS\s+(\d+)\s+(\d+)", line)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def atomic_write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(list(rows)).to_csv(tmp, index=False)
    tmp.replace(path)


def parse_pose_directory(pose_dir: Path) -> Dict[str, Any]:
    rank_to_confidence: Dict[int, float] = {}
    if pose_dir.is_dir():
        for path in pose_dir.glob("rank*_confidence*.sdf"):
            match = POSE_PATTERN.fullmatch(path.name)
            if match:
                rank_to_confidence[int(match.group("rank"))] = float(
                    match.group("confidence")
                )
    ordered = sorted(rank_to_confidence.items())
    top5 = [value for rank, value in ordered if rank <= 5]
    return {
        "top1_confidence": rank_to_confidence.get(1),
        "top5_mean_confidence": (
            float(np.mean(top5)) if top5 else None
        ),
        "best_confidence": (
            max(rank_to_confidence.values()) if rank_to_confidence else None
        ),
        "num_poses_found": len(rank_to_confidence),
        "rank_confidences_json": json.dumps(
            {str(rank): value for rank, value in ordered}
        ),
    }


def parse_pose_metrics(sample_dir: Path) -> Dict[str, Any]:
    return parse_pose_directory(sample_dir / "out" / "complex_0")


def pose_set_is_complete(
    metrics: Dict[str, Any], expected_poses: int
) -> bool:
    """Return whether DiffDock wrote the complete requested pose set.

    A bulk process may time out while working on a later complex.  Complexes
    whose complete pose sets were already written remain valid and should be
    checkpointed instead of being rerun with the unfinished tail.
    """
    try:
        count = float(metrics.get("num_poses_found", 0) or 0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(count) and count >= int(expected_poses)


def write_summary(results: pd.DataFrame, out_dir: Path) -> None:
    groups = []
    for sample_type, frame in results.groupby("sample_type", dropna=False):
        ok = frame.loc[frame["status"] == "ok"]
        applicable = frame.loc[frame["status"] != "not_applicable"]
        row: Dict[str, Any] = {
            "sample_type": sample_type,
            "requested": int(len(frame)),
            "successful": int(len(ok)),
            "success_rate": float(len(ok) / len(frame)) if len(frame) else 0.0,
            "not_applicable": int(
                frame["status"].eq("not_applicable").sum()
            ),
            "applicable_requested": int(len(applicable)),
            "success_rate_among_applicable": (
                float(len(ok) / len(applicable)) if len(applicable) else None
            ),
        }
        for metric in (
            "top1_confidence",
            "top5_mean_confidence",
            "best_confidence",
            "num_poses_found",
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
                "metric_notes": {
                    "top1_confidence": (
                        "DiffDock confidence for rank 1; higher is better."
                    ),
                    "top5_mean_confidence": (
                        "Mean confidence of ranks 1-5; higher is better."
                    ),
                    "success_rate": (
                        "Included to avoid survivor bias from averaging only "
                        "successful complexes."
                    ),
                },
                "groups": groups,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_one(
    row: pd.Series,
    out_dir: Path,
    udocker: str,
    repo: Path,
    container: str,
    gpu_id: int,
    samples_per_complex: int,
    batch_size: int,
    timeout_seconds: int,
) -> Dict[str, Any]:
    sample_id = str(row["sample_id"])
    sample_dir = out_dir / "outputs" / safe_name(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / f"{safe_name(sample_id)}.log"
    protein_source = Path(str(row["pdb_path"])).expanduser().resolve()
    ligand_source = Path(str(row["ligand_sdf_path"])).expanduser().resolve()
    if not protein_source.is_file():
        raise FileNotFoundError(f"Missing PDB: {protein_source}")
    if not ligand_source.is_file():
        raise FileNotFoundError(f"Missing SDF: {ligand_source}")
    shutil.copy2(protein_source, sample_dir / "protein.pdb")
    shutil.copy2(ligand_source, sample_dir / "ligand.sdf")

    # DiffDock loads its YAML *after* argparse and therefore the YAML value
    # silently overrides ``--samples_per_complex``.  Materialize a temporary
    # config with the requested value so screening runs actually use their
    # declared pose count.
    runtime_config = f"/tmp/langpro_inference_gpu_{int(gpu_id)}.yaml"
    inference_command = (
        f"cp default_inference_args.yaml {runtime_config} && "
        f"sed -i 's/^samples_per_complex:.*/samples_per_complex: {int(samples_per_complex)}/' {runtime_config} && "
        "micromamba run -n diffdock python -m inference "
        f"--config {runtime_config} "
        "--protein_path /work/protein.pdb "
        "--ligand_description /work/ligand.sdf "
        "--out_dir /work/out "
        f"--samples_per_complex {int(samples_per_complex)} "
        f"--batch_size {int(batch_size)}"
    )
    command = [
        udocker,
        f"--repo={repo}",
        "run",
        f"--env=CUDA_VISIBLE_DEVICES={int(gpu_id)}",
        "--env=CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "-v",
        f"{sample_dir}:/work",
        container,
        "bash",
        "-lc",
        inference_command,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
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
    metrics = parse_pose_metrics(sample_dir)
    pose_complete = pose_set_is_complete(metrics, samples_per_complex)
    status = "ok" if pose_complete else "error"
    termination_warning = ""
    if timed_out:
        termination_warning = f"timeout_after_{timeout_seconds}s"
    elif exit_code != 0:
        termination_warning = f"exit_code_{exit_code}"
    if pose_complete:
        error = ""
    elif termination_warning:
        error = termination_warning
    else:
        error = (
            f"incomplete_pose_set_expected_{samples_per_complex}_found_"
            f"{metrics['num_poses_found']}"
        )
    return {
        "sample_id": sample_id,
        "sample_type": row.get("sample_type", "unspecified"),
        "source_row": row.get("source_row", ""),
        "sequence": row.get("sequence", ""),
        "structure_rank_score": row.get("structure_rank_score", ""),
        "pdb_path": str(protein_source),
        "ligand_sdf_path": str(ligand_source),
        "sample_dir": str(sample_dir),
        "log_path": str(log_path),
        "status": status,
        "error": error,
        "run_warning": termination_warning if pose_complete else "",
        "exit_code": exit_code,
        "elapsed_seconds": float(elapsed),
        **metrics,
    }


def run_bulk(
    source: pd.DataFrame,
    out_dir: Path,
    udocker: str,
    repo: Path,
    container: str,
    gpu_id: int,
    samples_per_complex: int,
    batch_size: int,
    timeout_seconds: int,
    attempt: int,
) -> List[Dict[str, Any]]:
    """Run several complexes in one official DiffDock CSV invocation.

    This keeps model/configuration parameters unchanged while loading the
    2.5-GB ESM model only once. DiffDock itself still evaluates every complex
    independently and catches per-complex numerical failures.
    """
    bulk_dir = out_dir / "bulk_work"
    inputs_dir = bulk_dir / "inputs"
    pose_root = bulk_dir / f"out_attempt_{int(attempt)}"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    pose_root.mkdir(parents=True, exist_ok=True)
    input_rows = []
    metadata = []
    for _, row in source.iterrows():
        sample_id = str(row["sample_id"])
        complex_name = safe_name(sample_id)
        protein_source = Path(str(row["pdb_path"])).expanduser().resolve()
        ligand_source = Path(
            str(row["ligand_sdf_path"])
        ).expanduser().resolve()
        if not protein_source.is_file():
            raise FileNotFoundError(f"Missing PDB: {protein_source}")
        if not ligand_source.is_file():
            raise FileNotFoundError(f"Missing SDF: {ligand_source}")
        protein_name = f"{complex_name}_protein.pdb"
        ligand_name = f"{complex_name}_ligand.sdf"
        shutil.copy2(protein_source, inputs_dir / protein_name)
        shutil.copy2(ligand_source, inputs_dir / ligand_name)
        input_rows.append(
            {
                "complex_name": complex_name,
                "protein_path": f"/work/inputs/{protein_name}",
                "protein_sequence": "",
                "ligand_description": f"/work/inputs/{ligand_name}",
            }
        )
        metadata.append(
            (
                row,
                complex_name,
                protein_source,
                ligand_source,
            )
        )
    pd.DataFrame(input_rows).to_csv(bulk_dir / "input.csv", index=False)

    # DiffDock applies YAML values after parsing CLI arguments.  Without this
    # temporary override, its default of 10 poses wins even when the wrapper
    # requests a smaller compute-budgeted screen.
    runtime_config = (
        f"/tmp/langpro_inference_gpu_{int(gpu_id)}_attempt_{int(attempt)}.yaml"
    )
    inference_command = (
        f"cp default_inference_args.yaml {runtime_config} && "
        f"sed -i 's/^samples_per_complex:.*/samples_per_complex: {int(samples_per_complex)}/' {runtime_config} && "
        "micromamba run -n diffdock python -m inference "
        f"--config {runtime_config} "
        "--protein_ligand_csv /work/input.csv "
        f"--out_dir /work/{pose_root.name} "
        f"--samples_per_complex {int(samples_per_complex)} "
        f"--batch_size {int(batch_size)}"
    )
    command = [
        udocker,
        f"--repo={repo}",
        "run",
        f"--env=CUDA_VISIBLE_DEVICES={int(gpu_id)}",
        "--env=CUDA_DEVICE_ORDER=PCI_BUS_ID",
        # Concurrent udocker invocations otherwise contend on micromamba's
        # shared process-lock directory and can remain blocked for the whole
        # lifetime of another DiffDock batch.
        f"--env=XDG_CACHE_HOME=/tmp/diffdock_cache_gpu_{int(gpu_id)}",
        # Keep the container's existing model cache even though micromamba
        # receives an isolated process-lock cache.
        "--env=TORCH_HOME=/tmp/torch",
        "-v",
        f"{bulk_dir}:/work",
        container,
        "bash",
        "-lc",
        inference_command,
    ]
    log_path = out_dir / "logs" / f"bulk_attempt_{int(attempt)}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
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

    results = []
    for row, complex_name, protein_source, ligand_source in metadata:
        metrics = parse_pose_directory(pose_root / complex_name)
        pose_complete = pose_set_is_complete(metrics, samples_per_complex)
        status = "ok" if pose_complete else "error"
        termination_warning = ""
        if timed_out:
            termination_warning = f"timeout_after_{timeout_seconds}s"
        elif exit_code != 0:
            termination_warning = f"bulk_exit_code_{exit_code}"
        if pose_complete:
            error = ""
        elif termination_warning:
            error = termination_warning
        else:
            error = (
                f"incomplete_pose_set_expected_{samples_per_complex}_found_"
                f"{metrics['num_poses_found']}"
            )
        results.append(
            {
                "sample_id": str(row["sample_id"]),
                "sample_type": row.get("sample_type", "unspecified"),
                "source_row": row.get("source_row", ""),
                "sequence": row.get("sequence", ""),
                "structure_rank_score": row.get(
                    "structure_rank_score", ""
                ),
                "pdb_path": str(protein_source),
                "ligand_sdf_path": str(ligand_source),
                "sample_dir": str(pose_root / complex_name),
                "log_path": str(log_path),
                "status": status,
                "error": error,
                "run_warning": termination_warning if pose_complete else "",
                "exit_code": exit_code,
                "elapsed_seconds": float(elapsed),
                "bulk_attempt": int(attempt),
                **metrics,
            }
        )
    return results


def recover_complete_bulk_outputs(
    source: pd.DataFrame,
    out_dir: Path,
    by_id: Dict[str, Dict[str, Any]],
    expected_poses: int,
) -> int:
    """Recover completed complexes even if a bulk run never wrote results.csv."""
    attempt_dirs = []
    for path in (out_dir / "bulk_work").glob("out_attempt_*"):
        try:
            attempt = int(path.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        attempt_dirs.append((attempt, path))
    attempt_dirs.sort(reverse=True)
    recovered = 0
    for _, source_row in source.iterrows():
        sample_id = str(source_row["sample_id"])
        if by_id.get(sample_id, {}).get("status") in {
            "ok",
            "not_applicable",
        }:
            continue
        for attempt, attempt_dir in attempt_dirs:
            pose_dir = attempt_dir / safe_name(sample_id)
            metrics = parse_pose_directory(pose_dir)
            if not pose_set_is_complete(metrics, expected_poses):
                continue
            by_id[sample_id] = {
                "sample_id": sample_id,
                "sample_type": source_row.get(
                    "sample_type", "unspecified"
                ),
                "source_row": source_row.get("source_row", ""),
                "sequence": source_row.get("sequence", ""),
                "structure_rank_score": source_row.get(
                    "structure_rank_score", ""
                ),
                "pdb_path": str(
                    Path(str(source_row["pdb_path"]))
                    .expanduser()
                    .resolve()
                ),
                "ligand_sdf_path": str(
                    Path(str(source_row["ligand_sdf_path"]))
                    .expanduser()
                    .resolve()
                ),
                "sample_dir": str(pose_dir),
                "log_path": str(
                    out_dir / "logs" / f"bulk_attempt_{attempt}.log"
                ),
                "status": "ok",
                "error": "",
                "run_warning": (
                    "recovered_from_complete_existing_bulk_output"
                ),
                "exit_code": None,
                "elapsed_seconds": None,
                "bulk_attempt": int(attempt),
                "recovered_complete_pose_set": True,
                **metrics,
            }
            recovered += 1
            break
    return recovered


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-3 DiffDock evaluator")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--repo",
        required=True,
        help="Path to a DiffDock checkout or compatible uDocker repository.",
    )
    parser.add_argument("--container", default="diffdock")
    parser.add_argument("--udocker", default="")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--samples-per-complex", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--bulk",
        action="store_true",
        help=(
            "Use DiffDock's official protein_ligand_csv mode so model "
            "weights are loaded once for all pending complexes."
        ),
    )
    parser.add_argument(
        "--bulk-attempts",
        type=int,
        default=1,
        help="Retry only failed complexes this many bulk invocations.",
    )
    parser.add_argument(
        "--mark-edgeless-not-applicable",
        action="store_true",
        help=(
            "Record SDFs with zero bonds as not_applicable instead of "
            "repeatedly invoking DiffDock, whose ligand graph requires edges."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    source = pd.read_csv(args.manifest)
    required = {"sample_id", "pdb_path", "ligand_sdf_path"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    out_dir = Path(args.out_dir).expanduser().resolve()
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"
    udocker = args.udocker or shutil.which("udocker")
    if not udocker:
        raise FileNotFoundError(
            "udocker not found; run this script in the eval_diffdock environment"
        )
    repo = Path(args.repo).expanduser().resolve()

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
        recovered = 0
        for row in by_id.values():
            if (
                row.get("status") not in {"ok", "not_applicable"}
                and pose_set_is_complete(row, args.samples_per_complex)
            ):
                previous_error = str(row.get("error") or "")
                row["status"] = "ok"
                row["error"] = ""
                row["run_warning"] = previous_error
                row["recovered_complete_pose_set"] = True
                recovered += 1
        if recovered:
            atomic_write_csv(by_id.values(), results_path)
            print(
                f"[INFO] Recovered {recovered} complete pose sets from "
                "an interrupted bulk invocation."
            )
        completed = {
            sample_id
            for sample_id, row in by_id.items()
            if row.get("status") in {"ok", "not_applicable"}
        }
        print(f"[INFO] Resuming with {len(completed)} successful samples.")
    else:
        completed = set()

    if args.resume:
        recovered_from_files = recover_complete_bulk_outputs(
            source,
            out_dir,
            by_id,
            args.samples_per_complex,
        )
        if recovered_from_files:
            atomic_write_csv(by_id.values(), results_path)
            print(
                f"[INFO] Recovered {recovered_from_files} complete pose "
                "sets directly from bulk output directories."
            )
        completed = {
            sample_id
            for sample_id, row in by_id.items()
            if row.get("status") in {"ok", "not_applicable"}
        }

    if args.mark_edgeless_not_applicable:
        for _, source_row in source.iterrows():
            sample_id = str(source_row["sample_id"])
            if sample_id in completed:
                continue
            ligand_path = Path(
                str(source_row["ligand_sdf_path"])
            ).expanduser().resolve()
            atoms, bonds = sdf_graph_counts(ligand_path)
            if bonds == 0:
                by_id[sample_id] = {
                    "sample_id": sample_id,
                    "sample_type": source_row.get(
                        "sample_type", "unspecified"
                    ),
                    "source_row": source_row.get("source_row", ""),
                    "sequence": source_row.get("sequence", ""),
                    "structure_rank_score": source_row.get(
                        "structure_rank_score", ""
                    ),
                    "pdb_path": source_row.get("pdb_path", ""),
                    "ligand_sdf_path": str(ligand_path),
                    "status": "not_applicable",
                    "error": "ligand_sdf_has_zero_bonds",
                    "ligand_num_atoms": atoms,
                    "ligand_num_bonds": bonds,
                    "num_poses_found": 0,
                }
        completed = {
            sample_id
            for sample_id, row in by_id.items()
            if row.get("status") in {"ok", "not_applicable"}
        }
        atomic_write_csv(by_id.values(), results_path)

    if args.bulk:
        if args.bulk_attempts < 1:
            raise ValueError("--bulk-attempts must be at least 1")
        previous_attempts = [
            int(value)
            for value in (
                row.get("bulk_attempt") for row in by_id.values()
            )
            if value is not None and not pd.isna(value)
        ]
        first_attempt = max(previous_attempts, default=0) + 1
        last_attempt = first_attempt + int(args.bulk_attempts)
        for attempt in range(first_attempt, last_attempt):
            completed = {
                sample_id
                for sample_id, row in by_id.items()
                if row.get("status") in {"ok", "not_applicable"}
            }
            pending = source.loc[
                ~source["sample_id"].astype(str).isin(completed)
            ].copy()
            if pending.empty:
                break
            print(
                f"[INFO] Bulk attempt {attempt}: "
                f"{len(pending)} pending complexes."
            )
            try:
                attempt_results = run_bulk(
                    pending,
                    out_dir,
                    udocker,
                    repo,
                    args.container,
                    args.gpu_id,
                    args.samples_per_complex,
                    args.batch_size,
                    args.timeout_seconds,
                    attempt,
                )
            except Exception as exc:
                if args.fail_fast:
                    raise
                attempt_results = [
                    {
                        "sample_id": str(row["sample_id"]),
                        "sample_type": row.get(
                            "sample_type", "unspecified"
                        ),
                        "source_row": row.get("source_row", ""),
                        "status": "error",
                        "error": repr(exc),
                        "bulk_attempt": attempt,
                    }
                    for _, row in pending.iterrows()
                ]
            for result in attempt_results:
                by_id[str(result["sample_id"])] = result
                print(
                    f"[{result['status'].upper()}] {result['sample_id']} "
                    f"top1={result.get('top1_confidence')} "
                    f"top5={result.get('top5_mean_confidence')} "
                    f"poses={result.get('num_poses_found')} "
                    f"seconds={result.get('elapsed_seconds')} "
                    f"attempt={attempt}"
                )
            atomic_write_csv(by_id.values(), results_path)
    else:
        for _, source_row in source.iterrows():
            sample_id = str(source_row["sample_id"])
            if sample_id in completed:
                print(f"[SKIP] {sample_id}")
                continue
            try:
                result = run_one(
                    source_row,
                    out_dir,
                    udocker,
                    repo,
                    args.container,
                    args.gpu_id,
                    args.samples_per_complex,
                    args.batch_size,
                    args.timeout_seconds,
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
                f"top1={result.get('top1_confidence')} "
                f"top5={result.get('top5_mean_confidence')} "
                f"poses={result.get('num_poses_found')} "
                f"seconds={result.get('elapsed_seconds')}"
            )

    results = pd.DataFrame(list(by_id.values()))
    write_summary(results, out_dir)
    print(f"[DONE] {len(results)} records -> {results_path}")


if __name__ == "__main__":
    main()
