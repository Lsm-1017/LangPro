import pandas as pd
import re
from pathlib import Path
RAW_PATH = Path('uniprot_raw.tsv')
OUT_PATH = Path('uniprot_with_ligands.csv')

def main():
    df = pd.read_csv(RAW_PATH, sep='\t')
    func_col = 'Function [CC]'
    lig_col = 'Binding site'
    df = df[df[func_col].notna() & df[lig_col].notna()]
    rows = []
    for idx, row in df.iterrows():
        uniprot_id = row['Entry']
        seq = row['Sequence']
        func = row[func_col]
        lig_info = str(row[lig_col])
        parts = re.split('[;\\n]', lig_info)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            name_match = re.search('/ligand="([^"]+)"', p)
            id_match = re.search('/ligand_id="([^"]+)"', p)
            ligand_name = name_match.group(1) if name_match else None
            ligand_id_raw = id_match.group(1) if id_match else None
            ligand_chebi = None
            if ligand_id_raw:
                m2 = re.search('CHEBI:(\\d+)', ligand_id_raw)
                if m2:
                    ligand_chebi = f'CHEBI:{m2.group(1)}'
            else:
                m3 = re.search('CHEBI:(\\d+)', p)
                if m3:
                    ligand_chebi = f'CHEBI:{m3.group(1)}'
            rows.append({'uniprot_id': uniprot_id, 'sequence': seq, 'function_raw': func, 'ligand_annotation': p, 'ligand_name': ligand_name, 'ligand_id_raw': ligand_id_raw, 'ligand_chebi': ligand_chebi})
    df_out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)
    print('save', OUT_PATH)
    print(df_out.head(5))
if __name__ == '__main__':
    main()
