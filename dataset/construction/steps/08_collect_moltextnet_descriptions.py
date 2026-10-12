from pathlib import Path
import pandas as pd
from rdkit import Chem
from datasets import load_from_disk
IN_PATH = Path('../data_intermediate/triples_clean_norm.csv')
OUT_PATH = Path('../data_intermediate/moltext_smiles.csv')
MOLTEXT_DIR = Path('../moltextnet_hf')

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
    target_smiles = df['ligand_smiles_canon'].dropna().astype(str).str.strip().unique()
    target_smiles_set = set(target_smiles)
    print('SMILES', len(target_smiles_set))
    print(f'MolTextNet{MOLTEXT_DIR} ...')
    ds_all = load_from_disk(str(MOLTEXT_DIR))
    print(ds_all)
    try:
        ds = ds_all['train']
    except TypeError:
        ds = ds_all
    rows = []
    hit_count = 0
    for i, row in enumerate(ds):
        smi = row['canonical_smiles']
        desc = row['description']
        cano = canon(smi)
        if cano in target_smiles_set:
            hit_count += 1
            rows.append({'ligand_smiles_canon': cano, 'ligand_text_moltextnet': desc})
        if (i + 1) % 100000 == 0:
            print(f'MolTextNet:{i + 1}Status:{hit_count}Status:')
    out_df = pd.DataFrame(rows).drop_duplicates(subset=['ligand_smiles_canon'])
    print('MolTextNet', len(out_df))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print('Status:', OUT_PATH)
    print(out_df.head())
if __name__ == '__main__':
    main()
