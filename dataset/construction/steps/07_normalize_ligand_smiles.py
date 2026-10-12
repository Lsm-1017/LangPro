import pandas as pd
from pathlib import Path
from rdkit import Chem
IN_PATH = Path('../data_intermediate/triples_clean.csv')
OUT_PATH = Path('../data_intermediate/triples_clean_norm.csv')

def canon(smiles: str):
    if not isinstance(smiles, str):
        return None
    s = smiles.strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def main():
    df = pd.read_csv(IN_PATH)
    df['ligand_smiles_canon'] = df['ligand_smiles'].apply(canon)
    before = len(df)
    df = df[df['ligand_smiles_canon'].notna()]
    after = len(df)
    print(f'SMILES{before} -> {after}')
    print('Status:')
    print(df[['ligand_name', 'ligand_smiles', 'ligand_smiles_canon']].head())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print('Status:', OUT_PATH)
if __name__ == '__main__':
    main()
