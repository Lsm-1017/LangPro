import pandas as pd
from pathlib import Path
IN_PATH = Path('../data_intermediate/uniprot_with_ligands.csv')
OUT_PATH = Path('../data_intermediate/ligand_names.csv')

def main():
    df = pd.read_csv(IN_PATH)
    names = df['ligand_name'].dropna().astype(str).str.strip()
    unique_names = sorted(names.unique())
    print('ligand_name', len(unique_names))
    print('20', unique_names[:20])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'ligand_name': unique_names}).to_csv(OUT_PATH, index=False)
    print('Status:', OUT_PATH)
if __name__ == '__main__':
    main()
