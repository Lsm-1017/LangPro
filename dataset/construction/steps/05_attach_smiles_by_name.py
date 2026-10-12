import pandas as pd
from pathlib import Path
PAIR_PATH = Path('../data_intermediate/uniprot_with_ligands.csv')
LIG_PATH = Path('../data_intermediate/ligand_smiles.csv')
OUT_PATH = Path('../data_intermediate/triples_raw.csv')

def main():
    df_pair = pd.read_csv(PAIR_PATH)
    df_lig = pd.read_csv(LIG_PATH)
    lig_map = df_lig.dropna(subset=['ligand_name', 'smiles']).set_index('ligand_name')['smiles'].to_dict()
    print('ligand_name -> SMILES', len(lig_map))
    df_pair['ligand_smiles'] = df_pair['ligand_name'].map(lig_map)
    before = len(df_pair)
    df_pair = df_pair[df_pair['ligand_smiles'].notna()]
    after = len(df_pair)
    print(f'( , ){before}')
    print(f'SMILES{after}')
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_pair.to_csv(OUT_PATH, index=False)
    print('Status:', OUT_PATH)
    print('Status:')
    print(df_pair.head())
if __name__ == '__main__':
    main()
