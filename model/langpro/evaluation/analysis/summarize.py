#!/usr/bin/env python3
"""Join ESMFold, DiffDock and Chai-1 results with matched references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .analyze_chai_pairs import five_model_means
except ImportError:  # Support direct execution from a source checkout.
    from analyze_chai_pairs import five_model_means


METRIC_DIRECTIONS = {
    "esmfold_mean_ca_plddt": 1,
    "esmfold_ptm": 1,
    "esmfold_mean_pae": -1,
    "esmfold_structure_score": 1,
    "diffdock_top1_confidence": 1,
    "diffdock_top5_mean_confidence": 1,
    "chai_best_aggregate_score": 1,
    "chai_best_ptm": 1,
    "chai_best_iptm": 1,
    "chai_five_model_mean_aggregate_score": 1,
    "chai_five_model_mean_ptm": 1,
    "chai_five_model_mean_iptm": 1,
}


def metric_mean(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else None


def ligand_class(smiles: Any) -> str:
    text = str(smiles).strip()
    if text.startswith("[") and text.endswith("]") and "." not in text:
        return "single_atom_ion"
    if "." in text:
        return "disconnected_cofactor"
    if text in {"O=O", "[O][O]"}:
        return "tiny_molecule"
    return "molecular_graph"


def protect_input_columns(
    manifest: pd.DataFrame, evaluator_columns: list[str]
) -> pd.DataFrame:
    """Preserve manifest provenance without shadowing final evaluator fields."""
    collisions = {
        column: f"input_{column}"
        for column in evaluator_columns
        if column in manifest.columns
    }
    return manifest.rename(columns=collisions)


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


def paired_metric_summary(
    paired: pd.DataFrame,
    metric: str,
    direction: int,
    success_column: str | None = None,
) -> dict[str, Any]:
    frame = paired
    if success_column is not None:
        frame = frame.loc[frame[success_column].eq(True)]
    design = pd.to_numeric(
        frame[f"designed_{metric}"], errors="coerce"
    )
    reference = pd.to_numeric(
        frame[f"reference_{metric}"], errors="coerce"
    )
    delta = pd.to_numeric(frame[f"delta_{metric}"], errors="coerce")
    valid = design.notna() & reference.notna() & delta.notna()
    design, reference, delta = design[valid], reference[valid], delta[valid]
    return {
        "paired_observations": int(len(delta)),
        "designed_mean": float(design.mean()) if len(delta) else None,
        "reference_mean": float(reference.mean()) if len(delta) else None,
        "mean_delta": float(delta.mean()) if len(delta) else None,
        "median_delta": float(delta.median()) if len(delta) else None,
        "mean_delta_bootstrap_95ci": bootstrap_mean_ci(delta),
        "win_rate": (
            float((delta * int(direction) > 0).mean())
            if len(delta)
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--diffdock-results", required=True)
    parser.add_argument("--chai-results", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    diffdock = pd.read_csv(args.diffdock_results)
    chai = pd.read_csv(args.chai_results)
    model_json = (
        chai["all_models_json"]
        if "all_models_json" in chai
        else pd.Series([""] * len(chai), index=chai.index)
    )
    ensemble = pd.DataFrame(
        [five_model_means(value) for value in model_json],
        index=chai.index,
    )
    for column in ensemble:
        chai[column] = ensemble[column]
    required_manifest = {
        "sample_id",
        "sample_type",
        "source_row",
        "mean_ca_plddt",
        "mean_pae",
        "ptm",
        "structure_rank_score",
    }
    missing = sorted(required_manifest - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest sample IDs must be unique")
    if diffdock["sample_id"].duplicated().any():
        raise ValueError("DiffDock sample IDs must be unique")
    if chai["sample_id"].duplicated().any():
        raise ValueError("Chai sample IDs must be unique")

    diffdock_columns = [
        "sample_id",
        "status",
        "top1_confidence",
        "top5_mean_confidence",
        "num_poses_found",
        "error",
    ]
    chai_columns = [
        "sample_id",
        "status",
        "best_aggregate_score",
        "best_ptm",
        "best_iptm",
        "five_model_mean_aggregate_score",
        "five_model_mean_ptm",
        "five_model_mean_iptm",
        "best_has_inter_chain_clashes",
        "error",
    ]
    final_evaluator_columns = [
        f"diffdock_{column}"
        for column in diffdock_columns
        if column != "sample_id"
    ] + [
        f"chai_{column}"
        for column in chai_columns
        if column != "sample_id"
    ]
    manifest = protect_input_columns(manifest, final_evaluator_columns)
    rows = manifest.merge(
        diffdock[diffdock_columns].rename(
            columns={
                column: f"diffdock_{column}"
                for column in diffdock_columns
                if column != "sample_id"
            }
        ),
        on="sample_id",
        how="left",
        validate="one_to_one",
    ).merge(
        chai[chai_columns].rename(
            columns={
                column: f"chai_{column}"
                for column in chai_columns
                if column != "sample_id"
            }
        ),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    rows = rows.rename(
        columns={
            "mean_ca_plddt": "esmfold_mean_ca_plddt",
            "ptm": "esmfold_ptm",
            "mean_pae": "esmfold_mean_pae",
            "structure_rank_score": "esmfold_structure_score",
        }
    ).sort_values(["source_row", "sample_type"])

    paired_rows = []
    metrics = list(METRIC_DIRECTIONS)
    for source_row, group in rows.groupby("source_row"):
        designed = group.loc[group["sample_type"] == "designed"]
        reference = group.loc[group["sample_type"] == "reference"]
        if len(designed) != 1 or len(reference) != 1:
            raise ValueError(
                f"source_row={source_row} needs one design and one reference"
            )
        design_row = designed.iloc[0]
        reference_row = reference.iloc[0]
        pair: dict[str, Any] = {
            "source_row": int(source_row),
            "designed_sample_id": design_row["sample_id"],
            "reference_sample_id": reference_row["sample_id"],
            "designed_sequence": design_row.get("sequence", ""),
            "ligand_smiles": design_row.get("ligand_smiles", ""),
            "ligand_class": ligand_class(
                design_row.get("ligand_smiles", "")
            ),
            "diffdock_pair_success": (
                design_row.get("diffdock_status") == "ok"
                and reference_row.get("diffdock_status") == "ok"
            ),
            "diffdock_pair_applicable": (
                design_row.get("diffdock_status") != "not_applicable"
                and reference_row.get("diffdock_status")
                != "not_applicable"
            ),
            "designed_diffdock_status": design_row.get(
                "diffdock_status"
            ),
            "reference_diffdock_status": reference_row.get(
                "diffdock_status"
            ),
            "chai_pair_success": (
                design_row.get("chai_status") == "ok"
                and reference_row.get("chai_status") == "ok"
            ),
            "designed_chai_status": design_row.get("chai_status"),
            "reference_chai_status": reference_row.get("chai_status"),
        }
        for metric in metrics:
            design_value = pd.to_numeric(
                pd.Series([design_row.get(metric)]), errors="coerce"
            ).iloc[0]
            reference_value = pd.to_numeric(
                pd.Series([reference_row.get(metric)]), errors="coerce"
            ).iloc[0]
            pair[f"designed_{metric}"] = design_value
            pair[f"reference_{metric}"] = reference_value
            pair[f"delta_{metric}"] = (
                float(design_value - reference_value)
                if pd.notna(design_value) and pd.notna(reference_value)
                else np.nan
            )
        paired_rows.append(pair)
    paired = pd.DataFrame(paired_rows).sort_values("source_row")

    designed = rows.loc[rows["sample_type"] == "designed"]
    reference = rows.loc[rows["sample_type"] == "reference"]
    summary: dict[str, Any] = {
        "num_pairs": int(len(paired)),
        "chai_pair_success_rate": float(
            paired["chai_pair_success"].mean()
        ),
        "diffdock_pair_success_rate": float(
            paired["diffdock_pair_success"].mean()
        ),
        "diffdock_applicable_pairs": int(
            paired["diffdock_pair_applicable"].sum()
        ),
        "diffdock_not_applicable_pairs": int(
            (~paired["diffdock_pair_applicable"]).sum()
        ),
        "diffdock_pair_success_rate_among_applicable": (
            float(
                paired.loc[
                    paired["diffdock_pair_applicable"],
                    "diffdock_pair_success",
                ].mean()
            )
            if paired["diffdock_pair_applicable"].any()
            else None
        ),
        "diffdock_designed_success_rate": float(
            designed["diffdock_status"].eq("ok").mean()
        ),
        "diffdock_reference_success_rate": float(
            reference["diffdock_status"].eq("ok").mean()
        ),
        "diffdock_designed_status_counts": {
            str(key): int(value)
            for key, value in designed["diffdock_status"]
            .fillna("missing")
            .value_counts()
            .items()
        },
        "diffdock_reference_status_counts": {
            str(key): int(value)
            for key, value in reference["diffdock_status"]
            .fillna("missing")
            .value_counts()
            .items()
        },
    }
    for metric in metrics:
        summary[f"designed_mean_{metric}"] = metric_mean(designed, metric)
        summary[f"reference_mean_{metric}"] = metric_mean(reference, metric)
    metric_summaries = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        if metric.startswith("diffdock_"):
            success_column = "diffdock_pair_success"
        elif metric.startswith("chai_"):
            success_column = "chai_pair_success"
        else:
            success_column = None
        metric_summaries[metric] = paired_metric_summary(
            paired, metric, direction, success_column
        )
    summary["paired_metric_summaries"] = metric_summaries
    class_summaries = {}
    for kind, group in paired.groupby("ligand_class", dropna=False):
        class_summaries[str(kind)] = {
            "pairs": int(len(group)),
            "chai_pair_success_rate": float(
                group["chai_pair_success"].mean()
            ),
            "diffdock_applicable_pairs": int(
                group["diffdock_pair_applicable"].sum()
            ),
            "diffdock_successful_pairs": int(
                group["diffdock_pair_success"].sum()
            ),
            "metrics": {
                metric: paired_metric_summary(
                    group,
                    metric,
                    direction,
                    (
                        "diffdock_pair_success"
                        if metric.startswith("diffdock_")
                        else (
                            "chai_pair_success"
                            if metric.startswith("chai_")
                            else None
                        )
                    ),
                )
                for metric, direction in METRIC_DIRECTIONS.items()
            },
        }
    summary["ligand_class_summaries"] = class_summaries
    summary["chai_any_designed_clash"] = bool(
        designed["chai_best_has_inter_chain_clashes"]
        .fillna(False)
        .astype(bool)
        .any()
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "results.csv", index=False)
    paired.to_csv(out_dir / "paired_results.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"[DONE] joint results -> {out_dir}")


if __name__ == "__main__":
    main()
