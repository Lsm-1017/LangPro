import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import inchi
TRIPLES_PATH = Path('../data_intermediate/triples_clean_norm.csv')
OUT_MY_LIGANDS = Path('../data_intermediate/my_ligands_inchikey.csv')

def main():
    df = pd.read_csv(TRIPLES_PATH)
    smiles_list = df['ligand_smiles_canon'].dropna().unique()
    records = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f'[WARN] SMILES :{smi}')
            continue
        try:
            ikey = inchi.MolToInchiKey(mol)
        except Exception as e:
            print(f'[WARN] InChIKey :{smi}, {e}')
            continue
        records.append({'ligand_smiles_canon': smi, 'inchikey': ikey})
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_MY_LIGANDS, index=False)
    print(f'ligand :{len(smiles_list)}')
    print(f'InChIKey :{len(out_df)}')
    print(':', OUT_MY_LIGANDS)
if __name__ == '__main__':
    main()
