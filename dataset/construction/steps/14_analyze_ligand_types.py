import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
TRIPLE_PATH = Path('../data_intermediate/triples_with_ligand_text_full.csv')
OUT_DETAIL = Path('../data_intermediate/ligand_type_detail.csv')
OUT_COUNT = Path('../data_intermediate/ligand_type_counts.csv')
METAL_SYMBOLS = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Al', 'Ga', 'In', 'Sn', 'Pb', 'Bi', 'As', 'Sb', 'Se', 'Te', 'B', 'Si', 'Ge'}

def classify_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return {'parse_ok': False, 'type': 'empty', 'num_atoms': None, 'heavy_atoms': None, 'num_c': None, 'num_metals': None, 'total_charge': None, 'mw': None}
    smi = smiles.strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {'parse_ok': False, 'type': 'parse_fail', 'num_atoms': None, 'heavy_atoms': None, 'num_c': None, 'num_metals': None, 'total_charge': None, 'mw': None}
    atoms = list(mol.GetAtoms())
    num_atoms = len(atoms)
    heavy_atoms = mol.GetNumHeavyAtoms()
    num_c = sum((1 for a in atoms if a.GetAtomicNum() == 6))
    num_metals = sum((1 for a in atoms if a.GetSymbol() in METAL_SYMBOLS))
    total_charge = sum((a.GetFormalCharge() for a in atoms))
    try:
        mw = float(Descriptors.MolWt(mol))
    except Exception:
        mw = None
    has_carbon = num_c > 0
    has_metal = num_metals > 0
    if heavy_atoms <= 2:
        lig_type = 'simple_ion_or_diatomic'
    elif not has_carbon:
        lig_type = 'inorganic_with_metal' if has_metal else 'inorganic_no_carbon'
    elif has_metal:
        lig_type = 'metal_organic_or_cofactor'
    else:
        lig_type = 'organic'
    return {'parse_ok': True, 'type': lig_type, 'num_atoms': num_atoms, 'heavy_atoms': heavy_atoms, 'num_c': num_c, 'num_metals': num_metals, 'total_charge': total_charge, 'mw': mw}

def main():
    df = pd.read_csv(TRIPLE_PATH)
    smiles = df['ligand_smiles_canon'].dropna().astype(str).unique()
    records = []
    for smi in smiles:
        info = classify_smiles(smi)
        info['ligand_smiles_canon'] = smi
        records.append(info)
    out = pd.DataFrame(records)
    out.to_csv(OUT_DETAIL, index=False)
    counts = out.groupby('type').size().reset_index(name='count').sort_values('count', ascending=False)
    counts.to_csv(OUT_COUNT, index=False)
    total = len(out)
    ok = int(out['parse_ok'].sum())
    print('=== Ligand Type Summary ===')
    print('unique ligands:', total)
    print('parse_ok:', ok, 'parse_fail:', total - ok)
    print(counts.to_string(index=False))
    print('\n=== Examples (top 5 each type) ===')
    for t in counts['type'].tolist():
        ex = out[out['type'] == t].head(5)['ligand_smiles_canon'].tolist()
        print(f"[{t}] n={int((out['type'] == t).sum())}:")
        for s in ex:
            print('  ', s)
    print('\nSaved detail to:', OUT_DETAIL)
    print('Saved counts to:', OUT_COUNT)
if __name__ == '__main__':
    main()
