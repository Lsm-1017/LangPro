#!/usr/bin/env python3
"""Summarize DiffDock results without conflating unsupported ligand graphs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sdf_counts(path: Path) -> tuple[int | None, int | None]:
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


def ligand_class(smiles: str, atoms: Any, bonds: Any) -> str:
    if atoms is None or bonds is None:
        return "unknown_sdf"
    if int(atoms) < 2 or int(bonds) < 1:
        return "edgeless_not_applicable"
    if "." in str(smiles):
        return "disconnected_cofactor_limited"
    if int(atoms) < 3:
        return "tiny_molecule_limited"
    return "molecular_graph"


def bootstrap_mean_ci(
    values: pd.Series, seed: int = 20260727
) -> list[float | None]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(array):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = rng.choice(
        array, size=(10000, len(array)), replace=True
    ).mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def paired_group_summary(
    group: pd.DataFrame, label: str
) -> dict[str, Any]:
    valid = group.loc[group["paired_success"].eq(True)].copy()
    summary: dict[str, Any] = {
        "group": label,
        "requested_pairs": int(len(group)),
        "successful_pairs": int(len(valid)),
        "paired_success_rate": (
            float(len(valid) / len(group)) if len(group) else None
        ),
    }
    for metric in ("top1_confidence", "top5_mean_confidence"):
        delta = pd.to_numeric(
            valid[f"delta_{metric}"], errors="coerce"
        ).dropna()
        summary.update(
            {
                f"mean_delta_{metric}": (
                    float(delta.mean()) if len(delta) else None
                ),
                f"median_delta_{metric}": (
                    float(delta.median()) if len(delta) else None
                ),
                f"mean_delta_{metric}_bootstrap_95ci": (
                    bootstrap_mean_ci(delta)
                ),
                f"win_rate_{metric}": (
                    float((delta > 0).mean()) if len(delta) else None
                ),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    results = pd.read_csv(args.results)
    stale_columns = {
        "diffdock_status",
        "diffdock_error",
        "diffdock_top1",
        "diffdock_top5",
        "diffdock_pose_count",
        "diffdock_applicable",
        "diffdock_reason",
    }
    manifest = manifest.drop(
        columns=[column for column in stale_columns if column in manifest.columns]
    )
    metadata = []
    for ligand_path, group in manifest.groupby("ligand_sdf_path"):
        atoms, bonds = sdf_counts(Path(str(ligand_path)))
        smiles = str(group.iloc[0].get("ligand_smiles", ""))
        metadata.append(
            {
                "ligand_sdf_path": ligand_path,
                "ligand_smiles": smiles,
                "ligand_num_atoms": atoms,
                "ligand_num_bonds": bonds,
                "diffdock_ligand_class": ligand_class(smiles, atoms, bonds),
            }
        )
    rows = manifest.merge(
        pd.DataFrame(metadata),
        on=["ligand_sdf_path", "ligand_smiles"],
        validate="many_to_one",
    ).merge(
        results[
            [
                "sample_id",
                "status",
                "error",
                "top1_confidence",
                "top5_mean_confidence",
                "num_poses_found",
            ]
        ].rename(
            columns={
                "status": "diffdock_status",
                "error": "diffdock_error",
            }
        ),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    summary_rows = []
    for (ligand_kind, sample_type), group in rows.groupby(
        ["diffdock_ligand_class", "sample_type"], dropna=False
    ):
        ok = group.loc[group["diffdock_status"] == "ok"]
        summary_rows.append(
            {
                "diffdock_ligand_class": ligand_kind,
                "sample_type": sample_type,
                "requested": int(len(group)),
                "successful": int(len(ok)),
                "success_rate": float(len(ok) / len(group)),
                "mean_top1_confidence": (
                    float(ok["top1_confidence"].mean())
                    if len(ok)
                    else None
                ),
                "mean_top5_mean_confidence": (
                    float(ok["top5_mean_confidence"].mean())
                    if len(ok)
                    else None
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    pair_rows = []
    for source_row, group in rows.groupby("source_row"):
        designed = group.loc[group["sample_type"] == "designed"]
        reference = group.loc[group["sample_type"] == "reference"]
        if len(designed) != 1 or len(reference) != 1:
            continue
        design = designed.iloc[0]
        ref = reference.iloc[0]
        paired_success = (
            design["diffdock_status"] == "ok"
            and ref["diffdock_status"] == "ok"
        )
        pair_rows.append(
            {
                "source_row": int(source_row),
                "diffdock_ligand_class": design["diffdock_ligand_class"],
                "ligand_smiles": design["ligand_smiles"],
                "paired_success": paired_success,
                "designed_top1_confidence": design["top1_confidence"],
                "reference_top1_confidence": ref["top1_confidence"],
                "delta_top1_confidence": (
                    design["top1_confidence"] - ref["top1_confidence"]
                    if paired_success
                    else None
                ),
                "designed_top5_mean_confidence": design[
                    "top5_mean_confidence"
                ],
                "reference_top5_mean_confidence": ref[
                    "top5_mean_confidence"
                ],
                "delta_top5_mean_confidence": (
                    design["top5_mean_confidence"]
                    - ref["top5_mean_confidence"]
                    if paired_success
                    else None
                ),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    paired_summaries = [paired_group_summary(pairs, "all")]
    for kind, group in pairs.groupby("diffdock_ligand_class"):
        paired_summaries.append(paired_group_summary(group, str(kind)))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "results_with_applicability.csv", index=False)
    summary.to_csv(out_dir / "summary_by_applicability.csv", index=False)
    pairs.to_csv(out_dir / "paired_results.csv", index=False)
    report = {
        "all_pair_success_rate": float(pairs["paired_success"].mean()),
        "applicable_pair_success_rate": float(
            pairs.loc[
                pairs["diffdock_ligand_class"] == "molecular_graph",
                "paired_success",
            ].mean()
        ),
        "paired_successes": int(pairs["paired_success"].sum()),
        "pairs": int(len(pairs)),
        "paired_summaries": paired_summaries,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
