import pandas as pd
from pathlib import Path
IN_PATH = Path('../data_intermediate/uniprot_with_ligands.csv')
OUT_PATH = Path('../data_intermediate/ligand_ids.csv')

def main():
    df = pd.read_csv(IN_PATH)
    ligids = sorted(df['ligand_chebi'].dropna().unique())
    print('CHEBI', len(ligids))
    print('10', ligids[:10])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'ligand_id': ligids}).to_csv(OUT_PATH, index=False)
    print('ID', OUT_PATH)
if __name__ == '__main__':
    main()
