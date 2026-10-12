import pandas as pd
from pathlib import Path
IN_TSV = Path('../data_intermediate/uniprot_index.tsv')
OUT_IDX = Path('../data_intermediate/protein_uniprot_index.csv')
OUT_LONG = Path('../data_intermediate/uniprot2pdb_long.csv')

def split_pdb_cell(x: str):
    if not isinstance(x, str):
        return []
    parts = [p.strip().upper() for p in x.split(';')]
    return [p for p in parts if p]

def main():
    df = pd.read_csv(IN_TSV, sep='\t')
    rename = {'Entry': 'uniprot_id', 'Entry Name': 'entry_name', 'Protein names': 'protein_names', 'Gene Names': 'gene_names', 'EC number': 'ec_number', 'Organism (ID)': 'organism_id', 'PDB': 'pdb_raw', 'Reviewed': 'reviewed'}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    OUT_IDX.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_IDX, index=False)
    print('Saved:', OUT_IDX, 'n=', len(df))
    rows = []
    has_pdb = 0
    for _, r in df.iterrows():
        pdbs = split_pdb_cell(r.get('pdb_raw', None))
        if pdbs:
            has_pdb += 1
        for pdb in pdbs:
            rows.append({'uniprot_id': r['uniprot_id'], 'pdb_id': pdb})
    long_df = pd.DataFrame(rows).drop_duplicates()
    long_df.to_csv(OUT_LONG, index=False)
    print('Saved:', OUT_LONG, 'pairs=', len(long_df))
    print(f'proteins_with_pdb: {has_pdb}/{len(df)} ({has_pdb / len(df):.3f})')
    print('unique pdb:', long_df['pdb_id'].nunique() if len(long_df) else 0)
if __name__ == '__main__':
    main()
