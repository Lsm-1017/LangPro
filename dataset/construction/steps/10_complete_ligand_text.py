import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
IN_PATH = Path('../data_intermediate/triples_with_ligand_text_partial.csv')
OUT_PATH = Path('../data_intermediate/triples_with_ligand_text_full.csv')

def describe_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    heavy = mol.GetNumHeavyAtoms()
    rings = Lipinski.RingCount(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    text = f'This ligand is a small molecule with a molecular weight of about {mw:.1f} Da, containing {heavy} heavy atoms and {rings} ring(s). It has approximately {hbd} hydrogen bond donor(s) and {hba} hydrogen bond acceptor(s), which may enable specific interactions in the binding pocket.'
    return text

def main():
    df = pd.read_csv(IN_PATH)
    print('Status:', len(df))
    mask_no_text = df['ligand_text_moltextnet'].isna()
    print('MolTextNet', mask_no_text.sum())
    unique_missing = df.loc[mask_no_text, 'ligand_smiles_canon'].dropna().unique()
    print('MolTextNet', len(unique_missing))
    fallback_map = {}
    for smi in unique_missing:
        fallback_map[smi] = describe_smiles(smi)
    df['ligand_text_fallback'] = df.apply(lambda row: fallback_map.get(row['ligand_smiles_canon']) if pd.isna(row['ligand_text_moltextnet']) else None, axis=1)

    def pick_text(row):
        if isinstance(row['ligand_text_moltextnet'], str) and row['ligand_text_moltextnet'].strip():
            return row['ligand_text_moltextnet']
        return row['ligand_text_fallback']
    df['ligand_text'] = df.apply(pick_text, axis=1)

    def pick_src(row):
        if isinstance(row['ligand_text_moltextnet'], str) and row['ligand_text_moltextnet'].strip():
            return 'moltextnet'
        elif isinstance(row['ligand_text_fallback'], str) and row['ligand_text_fallback'].strip():
            return 'rdkit_auto'
        else:
            return None
    df['ligand_text_src'] = df.apply(pick_src, axis=1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print('Status:', OUT_PATH)
    print(df[['ligand_name', 'ligand_smiles_canon', 'ligand_text_src']].head())
if __name__ == '__main__':
    main()
