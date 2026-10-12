"""Compute RDKit ligand descriptors used by LangPro.

The stage adds a Bemis-Murcko scaffold and molecular weight, logP,
hydrogen-bond, polar-surface-area, charge, aromaticity, rotatable-bond,
and ring-count descriptors. Prefer the public pipeline entry point:
python -m dataset.expansion.run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Single-molecule computation (lazy import for safety)
# -----------------------------
def compute_scaffold_and_props(smiles: str) -> dict:
    """Return a dict with all computed properties for one SMILES."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski, MolSurf
    from rdkit.Chem.Scaffolds import MurckoScaffold

    result = {
        "scaffold_smiles":     "",
        "chem_MW":             0.0,
        "chem_logP":           0.0,
        "chem_HBD":            0,
        "chem_HBA":            0,
        "chem_TPSA":           0.0,
        "chem_charge":         0,
        "chem_aromatic_ratio": 0.0,
        "chem_rotatable_bonds": 0,
        "chem_num_rings":      0,
        "chem_num_heavy_atoms": 0,
    }

    s = (smiles or "").strip()
    if not s:
        return result

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return result

    # ---- Scaffold ----
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is not None:
            scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(mol=scaffold)
            result["scaffold_smiles"] = scaffold_smi or ""
    except Exception:
        pass

    # ---- Chem properties ----
    try:
        mol = Chem.AddHs(mol)
    except Exception:
        pass

    result["chem_MW"]          = round(Descriptors.MolWt(mol), 2)
    result["chem_logP"]        = round(Crippen.MolLogP(mol), 4)
    result["chem_HBD"]         = Lipinski.NumHDonors(mol)
    result["chem_HBA"]         = Lipinski.NumHAcceptors(mol)
    result["chem_TPSA"]        = round(MolSurf.TPSA(mol), 2)
    result["chem_charge"]      = Chem.GetFormalCharge(mol)
    result["chem_rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    result["chem_num_rings"]       = rdMolDescriptors.CalcNumRings(mol)

    # aromatic ratio
    heavy = mol.GetNumHeavyAtoms()
    result["chem_num_heavy_atoms"] = heavy
    if heavy > 0:
        arom = 0
        for a in mol.GetAtoms():
            if a.GetIsAromatic():
                arom += 1
        result["chem_aromatic_ratio"] = round(arom / heavy, 4)

    return result


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="原始 dataset.csv 路径")
    ap.add_argument("--output", required=True, help="输出 CSV 路径")
    ap.add_argument("--smiles_col", type=str, default="ligand_smiles")
    ap.add_argument("--log_interval", type=int, default=5000)
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ---- Read ----
    print(f"[phase1] reading {input_path}")
    df = pd.read_csv(input_path)
    n_total = len(df)
    print(f"[phase1] total rows = {n_total}")

    smiles_col = args.smiles_col
    if smiles_col not in df.columns:
        print(f"[phase1] ERROR: column '{smiles_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    # ---- Compute ----
    scaffold_list: List[str] = []
    props_keys = [
        "chem_MW", "chem_logP", "chem_HBD", "chem_HBA",
        "chem_TPSA", "chem_charge", "chem_aromatic_ratio",
        "chem_rotatable_bonds", "chem_num_rings", "chem_num_heavy_atoms",
    ]
    props_accum = {k: [] for k in props_keys}

    print(f"[phase1] computing scaffold + chem properties for {n_total} ligands ...")
    t_last = time.time()

    for i, smi in enumerate(df[smiles_col]):
        r = compute_scaffold_and_props(str(smi) if pd.notna(smi) else "")

        scaffold_list.append(r["scaffold_smiles"])
        for k in props_keys:
            props_accum[k].append(r[k])

        if (i + 1) % args.log_interval == 0:
            elapsed = time.time() - t_last
            t_last = time.time()
            eta = elapsed / args.log_interval * (n_total - i - 1)
            print(f"[phase1] {i+1}/{n_total} ({100*(i+1)/n_total:.1f}%)  batch_time={elapsed:.1f}s  eta={eta/60:.1f}min")

    # ---- Merge into df ----
    df["scaffold_smiles"] = scaffold_list
    for k in props_keys:
        df[k] = props_accum[k]

    # ---- Quick stats ----
    non_empty_scaffold = sum(1 for x in scaffold_list if x)
    print(f"[phase1] non-empty scaffolds: {non_empty_scaffold}/{n_total} ({100*non_empty_scaffold/n_total:.1f}%)")
    for k in props_keys:
        vals = props_accum[k]
        print(f"[phase1] {k}: min={np.min(vals):.4f}  max={np.max(vals):.4f}  mean={np.mean(vals):.4f}")

    # ---- Save ----
    df.to_csv(output_path, index=False)
    elapsed_total = time.time() - t_start
    print(f"[phase1] saved to {output_path}  ({n_total} rows, {len(df.columns)} cols)")
    print(f"[phase1] total time: {elapsed_total/60:.1f} min")
    print("[phase1] DONE.")


if __name__ == "__main__":
    main()
