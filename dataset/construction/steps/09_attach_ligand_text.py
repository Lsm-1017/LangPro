import pandas as pd
from pathlib import Path
TRIPLE_PATH = Path('../data_intermediate/triples_clean_norm.csv')
MOLTEXT_PATH = Path('../data_intermediate/moltext_smiles.csv')
OUT_PATH = Path('../data_intermediate/triples_with_ligand_text_partial.csv')

def main():
    df = pd.read_csv(TRIPLE_PATH)
    mt = pd.read_csv(MOLTEXT_PATH)
    print('Status:', len(df))
    print('MolTextNet', len(mt))
    df = df.merge(mt, on='ligand_smiles_canon', how='left')
    df['ligand_text_src'] = df['ligand_text_moltextnet'].apply(lambda x: 'moltextnet' if isinstance(x, str) and len(x.strip()) > 0 else None)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print('Status:', OUT_PATH)
    print(df[['ligand_name', 'ligand_smiles_canon', 'ligand_text_src']].head())
if __name__ == '__main__':
    main()
