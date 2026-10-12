import pandas as pd
from pathlib import Path
from rdkit import Chem
from bioservices import ChEBI
IN_PATH = Path('../data_intermediate/ligand_ids.csv')
OUT_PATH = Path('../data_intermediate/ligand_smiles.csv')

def fetch_smiles_for_chebi(chebi_id: str):
    ch = fetch_smiles_for_chebi.ch
    try:
        res = ch.getCompleteEntity(chebi_id)
    except Exception as e:
        print(f'[WARN] ChEBI{chebi_id}: {e}')
        return None
    smiles = getattr(res, 'smiles', None)
    if not smiles:
        print(f'[WARN] {chebi_id}ChEBI SMILES')
        return None
    return smiles
fetch_smiles_for_chebi.ch = ChEBI()

def main():
    df_ids = pd.read_csv(IN_PATH)
    rows = []
    total = len(df_ids)
    print(f'Status:{total}ID ChEBI WebService')
    for i, lig_id in enumerate(df_ids['ligand_id'], start=1):
        print(f'[{i}/{total}] fetching {lig_id} ...')
        raw_smiles = fetch_smiles_for_chebi(lig_id)
        if not raw_smiles:
            print(f'-> SMILES')
            continue
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            print(f'-> RDKit SMILES:{raw_smiles}')
            continue
        cano_smiles = Chem.MolToSmiles(mol, canonical=True)
        rows.append({'ligand_id': lig_id, 'smiles': cano_smiles})
    lig_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lig_df.to_csv(OUT_PATH, index=False)
    print('SMILES', len(lig_df))
    print('Status:', OUT_PATH)
    print('Status:')
    print(lig_df.head())
if __name__ == '__main__':
    main()
