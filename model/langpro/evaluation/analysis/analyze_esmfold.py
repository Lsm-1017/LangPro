#!/usr/bin/env python3
"""Summarize paired ESMFold design/reference results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC_DIRECTIONS = {
    "structure_rank_score": 1,
    "mean_ca_plddt": 1,
    "ptm": 1,
    "mean_pae": -1,
}


def bootstrap_ci(values: pd.Series, seed: int = 20260727) -> list[float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)

    rows: list[dict[str, Any]] = []
    for source_row, group in manifest.groupby("source_row"):
        design = group.loc[group["sample_type"] == "designed"]
        reference = group.loc[group["sample_type"] == "reference"]
        if len(design) != 1 or len(reference) != 1:
            continue
        design, reference = design.iloc[0], reference.iloc[0]
        row: dict[str, Any] = {
            "source_row": int(source_row),
            "designed_sample_id": design["sample_id"],
            "reference_sample_id": reference["sample_id"],
            "ligand_smiles": design.get("ligand_smiles", ""),
        }
        for metric, direction in METRIC_DIRECTIONS.items():
            d_value = float(design[metric])
            r_value = float(reference[metric])
            delta = d_value - r_value
            row[f"designed_{metric}"] = d_value
            row[f"reference_{metric}"] = r_value
            row[f"delta_{metric}"] = delta
            row[f"win_{metric}"] = bool(delta * direction > 0)
        rows.append(row)
    pairs = pd.DataFrame(rows)

    summary: dict[str, Any] = {"pairs": int(len(pairs))}
    for metric, direction in METRIC_DIRECTIONS.items():
        delta = pairs[f"delta_{metric}"]
        summary.update(
            {
                f"designed_mean_{metric}": float(
                    pairs[f"designed_{metric}"].mean()
                ),
                f"reference_mean_{metric}": float(
                    pairs[f"reference_{metric}"].mean()
                ),
                f"mean_delta_{metric}": float(delta.mean()),
                f"median_delta_{metric}": float(delta.median()),
                f"mean_delta_{metric}_bootstrap_95ci": bootstrap_ci(delta),
                f"win_rate_{metric}": float((delta * direction > 0).mean()),
            }
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_dir / "paired_results.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
