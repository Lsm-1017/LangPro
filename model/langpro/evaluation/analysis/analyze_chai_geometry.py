#!/usr/bin/env python3
"""Audit protein-ligand geometry in the best Chai-1 prediction per sample.

The confidence scores remain the primary Chai metrics.  These geometry
descriptors add interpretable checks for interface presence, severe atomic
overlap, ligand burial, and metal-donor coordination without pretending that
predicted structures are experimental binding measurements.

Requires the optional structure parser dependency ``gemmi``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pandas as pd


AA3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
DONOR_ELEMENTS = {"N", "O", "S"}
METAL_ELEMENTS = {
    "CA",
    "CO",
    "CU",
    "FE",
    "MG",
    "MN",
    "MO",
    "NI",
    "ZN",
}


def ligand_class(smiles: Any) -> str:
    text = str(smiles).strip()
    if text.startswith("[") and text.endswith("]") and "." not in text:
        return "single_atom_ion"
    if "." in text:
        return "disconnected_cofactor"
    if text in {"O=O", "[O][O]"}:
        return "tiny_molecule"
    return "molecular_graph"


def heavy_atoms(chain: gemmi.Chain) -> list[tuple[gemmi.Residue, gemmi.Atom]]:
    atoms = []
    for residue in chain:
        for atom in residue:
            if atom.element.name.upper() != "H":
                atoms.append((residue, atom))
    return atoms


def split_protein_ligand_atoms(
    structure: gemmi.Structure,
) -> tuple[
    list[tuple[gemmi.Residue, gemmi.Atom]],
    list[tuple[gemmi.Residue, gemmi.Atom]],
]:
    if not len(structure):
        raise ValueError("Chai CIF contains no models")
    chains = list(structure[0])
    if len(chains) < 2:
        raise ValueError("Chai CIF needs protein and ligand chains")

    def amino_acid_residues(chain: gemmi.Chain) -> int:
        return sum(residue.name.upper() in AA3 for residue in chain)

    protein_chain = max(chains, key=amino_acid_residues)
    if amino_acid_residues(protein_chain) < 1:
        raise ValueError("Unable to identify protein chain in Chai CIF")
    protein_atoms = heavy_atoms(protein_chain)
    ligand_atoms = [
        item
        for chain in chains
        if chain.name != protein_chain.name
        for item in heavy_atoms(chain)
    ]
    if not protein_atoms or not ligand_atoms:
        raise ValueError("Empty protein or ligand atom set in Chai CIF")
    return protein_atoms, ligand_atoms


def geometry_metrics(cif_path: Path) -> dict[str, Any]:
    structure = gemmi.read_structure(str(cif_path))
    protein, ligand = split_protein_ligand_atoms(structure)
    distances = np.empty((len(protein), len(ligand)), dtype=np.float32)
    for i, (_, protein_atom) in enumerate(protein):
        for j, (_, ligand_atom) in enumerate(ligand):
            distances[i, j] = protein_atom.pos.dist(ligand_atom.pos)

    ligand_minimum = distances.min(axis=0)
    protein_minimum = distances.min(axis=1)
    residue_contacts: dict[float, int] = {}
    for cutoff in (3.0, 4.0, 5.0):
        contacted = {
            (
                int(protein[i][0].seqid.num),
                protein[i][0].name,
            )
            for i in np.flatnonzero(protein_minimum <= cutoff)
        }
        residue_contacts[cutoff] = len(contacted)

    metal_columns = [
        index
        for index, (_, atom) in enumerate(ligand)
        if atom.element.name.upper() in METAL_ELEMENTS
    ]
    donor_rows = [
        index
        for index, (_, atom) in enumerate(protein)
        if atom.element.name.upper() in DONOR_ELEMENTS
    ]
    metal_donor_distances = (
        distances[np.ix_(donor_rows, metal_columns)].reshape(-1)
        if donor_rows and metal_columns
        else np.asarray([], dtype=np.float32)
    )
    return {
        "cif_path": str(cif_path),
        "protein_heavy_atoms": int(len(protein)),
        "ligand_heavy_atoms": int(len(ligand)),
        "minimum_protein_ligand_distance": float(distances.min()),
        "severe_clash_atom_pairs_lt_1_5a": int((distances < 1.5).sum()),
        "protein_contact_atoms_3a": int((protein_minimum <= 3.0).sum()),
        "protein_contact_atoms_4a": int((protein_minimum <= 4.0).sum()),
        "protein_contact_atoms_5a": int((protein_minimum <= 5.0).sum()),
        "interface_residues_3a": int(residue_contacts[3.0]),
        "interface_residues_4a": int(residue_contacts[4.0]),
        "interface_residues_5a": int(residue_contacts[5.0]),
        "ligand_contact_fraction_4a": float(
            (ligand_minimum <= 4.0).mean()
        ),
        "ligand_contact_fraction_5a": float(
            (ligand_minimum <= 5.0).mean()
        ),
        "interface_present_within_5a": bool(distances.min() <= 5.0),
        "metal_atoms": int(len(metal_columns)),
        "metal_donor_contacts_3a": int(
            (metal_donor_distances <= 3.0).sum()
        ),
        "minimum_metal_donor_distance": (
            float(metal_donor_distances.min())
            if metal_donor_distances.size
            else None
        ),
    }


def mean_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else None


def bootstrap_mean_ci(
    values: pd.Series, seed: int = 20260730
) -> list[float | None]:
    """Return a deterministic paired-bootstrap interval for a mean delta."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(numeric):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = rng.choice(
        numeric, size=(10000, len(numeric)), replace=True
    ).mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def geometry_group_summary(
    group: pd.DataFrame, descriptors: tuple[str, ...]
) -> dict[str, Any]:
    valid = group.loc[group["paired_geometry_success"]].copy()
    summary: dict[str, Any] = {
        "pairs": int(len(group)),
        "successful_pairs": int(len(valid)),
        "paired_success_rate": (
            float(len(valid) / len(group)) if len(group) else None
        ),
    }
    for descriptor in descriptors:
        summary[f"mean_designed_{descriptor}"] = mean_or_none(
            valid[f"designed_{descriptor}"]
        )
        summary[f"mean_reference_{descriptor}"] = mean_or_none(
            valid[f"reference_{descriptor}"]
        )
        summary[f"mean_delta_{descriptor}"] = mean_or_none(
            valid[f"delta_{descriptor}"]
        )
        summary[f"mean_delta_{descriptor}_bootstrap_95ci"] = (
            bootstrap_mean_ci(valid[f"delta_{descriptor}"])
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
        "status",
        "best_model_idx",
        "sample_dir",
        "best_aggregate_score",
        "best_ptm",
        "best_iptm",
    }
    manifest = manifest.drop(
        columns=[column for column in stale_columns if column in manifest.columns]
    )
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest sample IDs must be unique")
    if results["sample_id"].duplicated().any():
        raise ValueError("Chai result sample IDs must be unique")
    merged = manifest.merge(
        results[
            [
                "sample_id",
                "status",
                "best_model_idx",
                "sample_dir",
                "best_aggregate_score",
                "best_ptm",
                "best_iptm",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    rows = []
    for _, row in merged.iterrows():
        item = {
            "sample_id": row["sample_id"],
            "sample_type": row["sample_type"],
            "source_row": int(row["source_row"]),
            "ligand_smiles": row.get("ligand_smiles", ""),
            "ligand_class": ligand_class(row.get("ligand_smiles", "")),
            "chai_status": row.get("status"),
            "best_aggregate_score": row.get("best_aggregate_score"),
            "best_ptm": row.get("best_ptm"),
            "best_iptm": row.get("best_iptm"),
            "geometry_status": "error",
            "geometry_error": "",
        }
        try:
            if row.get("status") != "ok":
                raise ValueError(f"Chai status is {row.get('status')}")
            model_idx = int(row["best_model_idx"])
            cif_path = (
                Path(str(row["sample_dir"]))
                / "05_chai1"
                / "raw2"
                / f"pred.model_idx_{model_idx}.cif"
            )
            if not cif_path.is_file():
                raise FileNotFoundError(cif_path)
            item.update(geometry_metrics(cif_path))
            item["geometry_status"] = "ok"
        except Exception as exc:
            item["geometry_error"] = repr(exc)
        rows.append(item)
    geometry = pd.DataFrame(rows).sort_values(
        ["source_row", "sample_type"]
    )

    pair_rows = []
    descriptors = (
        "minimum_protein_ligand_distance",
        "severe_clash_atom_pairs_lt_1_5a",
        "interface_residues_4a",
        "ligand_contact_fraction_4a",
        "metal_donor_contacts_3a",
        "minimum_metal_donor_distance",
    )
    for source_row, group in geometry.groupby("source_row"):
        design = group.loc[group["sample_type"] == "designed"]
        reference = group.loc[group["sample_type"] == "reference"]
        if len(design) != 1 or len(reference) != 1:
            continue
        design, reference = design.iloc[0], reference.iloc[0]
        success = (
            design["geometry_status"] == "ok"
            and reference["geometry_status"] == "ok"
        )
        pair = {
            "source_row": int(source_row),
            "ligand_smiles": design["ligand_smiles"],
            "ligand_class": design["ligand_class"],
            "paired_geometry_success": bool(success),
        }
        for descriptor in descriptors:
            d_value = pd.to_numeric(
                pd.Series([design.get(descriptor)]), errors="coerce"
            ).iloc[0]
            r_value = pd.to_numeric(
                pd.Series([reference.get(descriptor)]), errors="coerce"
            ).iloc[0]
            pair[f"designed_{descriptor}"] = d_value
            pair[f"reference_{descriptor}"] = r_value
            pair[f"delta_{descriptor}"] = (
                float(d_value - r_value)
                if success and pd.notna(d_value) and pd.notna(r_value)
                else None
            )
        pair_rows.append(pair)
    pairs = pd.DataFrame(pair_rows).sort_values("source_row")

    summary: dict[str, Any] = {
        "samples": int(len(geometry)),
        "successful_samples": int(geometry["geometry_status"].eq("ok").sum()),
        "pairs": int(len(pairs)),
        "successful_pairs": int(
            pairs["paired_geometry_success"].eq(True).sum()
        ),
        "designed_severe_clash_samples": int(
            (
                pd.to_numeric(
                    geometry.loc[
                        geometry["sample_type"] == "designed",
                        "severe_clash_atom_pairs_lt_1_5a",
                    ],
                    errors="coerce",
                )
                > 0
            ).sum()
        ),
        "designed_without_interface_within_5a": int(
            geometry.loc[
                geometry["sample_type"] == "designed",
                "interface_present_within_5a",
            ]
            .eq(False)
            .sum()
        ),
        "descriptor_notes": {
            "severe_clash_atom_pairs_lt_1_5a": (
                "Heavy-atom pairs below 1.5 Å; zero is preferred."
            ),
            "interface_residues_4a": (
                "Protein residues with any heavy atom within 4 Å of ligand."
            ),
            "ligand_contact_fraction_4a": (
                "Fraction of ligand heavy atoms within 4 Å of protein."
            ),
            "metal_donor_contacts_3a": (
                "Protein N/O/S atoms within 3 Å of predicted ligand metals; "
                "a descriptor, not a validated coordination score."
            ),
        },
    }
    group_summaries = {
        "all": geometry_group_summary(pairs, descriptors)
    }
    for kind, group in pairs.groupby("ligand_class"):
        group_summaries[str(kind)] = geometry_group_summary(
            group, descriptors
        )
    summary["ligand_class_summaries"] = group_summaries

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry.to_csv(out_dir / "sample_geometry.csv", index=False)
    pairs.to_csv(out_dir / "paired_geometry.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
