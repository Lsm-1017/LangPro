#!/usr/bin/env python3
"""Produce paired Chai-1 design/reference summaries from stage-4 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRICS = (
    "aggregate_score",
    "ptm",
    "iptm",
    "five_model_mean_aggregate_score",
    "five_model_mean_ptm",
    "five_model_mean_iptm",
)


def ligand_class(smiles: Any) -> str:
    text = str(smiles).strip()
    if text.startswith("[") and text.endswith("]") and "." not in text:
        return "single_atom_ion"
    if "." in text:
        return "disconnected_cofactor"
    if text in {"O=O", "[O][O]"}:
        return "tiny_molecule"
    return "molecular_graph"


def bootstrap_mean_ci(values: pd.Series, seed: int = 20260727) -> list[float | None]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(array):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def five_model_means(value: Any) -> dict[str, float | None]:
    """Extract robust five-model means from the serialized Chai metrics."""
    try:
        models = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        models = []
    output: dict[str, float | None] = {}
    for metric in ("aggregate_score", "ptm", "iptm"):
        values = pd.to_numeric(
            pd.Series(
                [
                    model.get(metric)
                    for model in models
                    if isinstance(model, dict)
                ],
                dtype=object,
            ),
            errors="coerce",
        ).dropna()
        output[f"five_model_mean_{metric}"] = (
            float(values.mean()) if len(values) == 5 else None
        )
    return output


def paired_rows(manifest: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    stale_columns = {
        "chai_status",
        "chai_error",
        "chai_aggregate",
        "chai_ptm",
        "chai_iptm",
        "chai_clashes",
    }
    manifest = manifest.drop(
        columns=[column for column in stale_columns if column in manifest.columns]
    )
    manifest = manifest.rename(
        columns={metric: f"input_{metric}" for metric in METRICS if metric in manifest}
    )
    results = results.rename(
        columns={
            "best_aggregate_score": "aggregate_score",
            "best_ptm": "ptm",
            "best_iptm": "iptm",
            "best_has_inter_chain_clashes": "has_inter_chain_clashes",
        }
    )
    model_json = (
        results["all_models_json"]
        if "all_models_json" in results
        else pd.Series([""] * len(results), index=results.index)
    )
    ensemble = pd.DataFrame(
        [five_model_means(value) for value in model_json],
        index=results.index,
    )
    for column in ensemble:
        results[column] = ensemble[column]
    keep = ["sample_id", "status", "error", *METRICS, "has_inter_chain_clashes"]
    merged = manifest.merge(
        results[keep].rename(
            columns={
                "status": "chai_status",
                "error": "chai_error",
            }
        ),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for source_row, group in merged.groupby("source_row"):
        designed = group.loc[group["sample_type"] == "designed"]
        reference = group.loc[group["sample_type"] == "reference"]
        if len(designed) != 1 or len(reference) != 1:
            continue
        design, ref = designed.iloc[0], reference.iloc[0]
        success = design["chai_status"] == "ok" and ref["chai_status"] == "ok"
        row: dict[str, Any] = {
            "source_row": int(source_row),
            "ligand_smiles": design.get("ligand_smiles", ""),
            "ligand_class": ligand_class(design.get("ligand_smiles", "")),
            "paired_success": bool(success),
            "designed_status": design["chai_status"],
            "reference_status": ref["chai_status"],
            "designed_has_clashes": design.get("has_inter_chain_clashes"),
            "reference_has_clashes": ref.get("has_inter_chain_clashes"),
        }
        for metric in METRICS:
            d_value, r_value = design.get(metric), ref.get(metric)
            row[f"designed_{metric}"] = d_value
            row[f"reference_{metric}"] = r_value
            row[f"delta_{metric}"] = d_value - r_value if success else None
            row[f"win_{metric}"] = bool(d_value > r_value) if success else None
        rows.append(row)
    return pd.DataFrame(rows)


def group_summary(group: pd.DataFrame, label: str) -> dict[str, Any]:
    valid = group.loc[group["paired_success"]].copy()
    summary: dict[str, Any] = {
        "group": label,
        "requested_pairs": int(len(group)),
        "successful_pairs": int(len(valid)),
        "paired_success_rate": float(len(valid) / len(group)) if len(group) else None,
    }
    for metric in METRICS:
        design = valid[f"designed_{metric}"]
        reference = valid[f"reference_{metric}"]
        delta = valid[f"delta_{metric}"]
        summary.update(
            {
                f"designed_mean_{metric}": float(design.mean()) if len(valid) else None,
                f"reference_mean_{metric}": float(reference.mean()) if len(valid) else None,
                f"mean_delta_{metric}": float(delta.mean()) if len(valid) else None,
                f"median_delta_{metric}": float(delta.median()) if len(valid) else None,
                f"mean_delta_{metric}_bootstrap_95ci": bootstrap_mean_ci(delta),
                f"win_rate_{metric}": float((delta > 0).mean()) if len(valid) else None,
            }
        )
    summary["any_designed_clashes"] = bool(
        valid["designed_has_clashes"].eq(True).any()
    )
    summary["any_reference_clashes"] = bool(
        valid["reference_has_clashes"].eq(True).any()
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
    pairs = paired_rows(manifest, results)
    summaries = [group_summary(pairs, "all")]
    for kind, group in pairs.groupby("ligand_class"):
        summaries.append(group_summary(group, str(kind)))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_dir / "paired_results.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summaries[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
