import pandas as pd
import re
from pathlib import Path
IN_PATH = Path('../data_intermediate/triples_raw.csv')
OUT_PATH = Path('../data_intermediate/triples_clean.csv')

def main():
    df = pd.read_csv(IN_PATH)
    aa_regex = re.compile('^[ACDEFGHIKLMNPQRSTVWY]+$')

    def clean_seq(seq: str):
        if not isinstance(seq, str):
            return None
        s = seq.strip().upper()
        if not aa_regex.match(s):
            return None
        if len(s) > 1024:
            return None
        return s
    df['sequence_clean'] = df['sequence'].apply(clean_seq)
    before = len(df)
    df = df[df['sequence_clean'].notna()]
    after = len(df)
    print(f'Status:{before} -> {after}')

    def build_function_text(fun: str):
        if not isinstance(fun, str):
            return None
        f = fun.strip()
        if not f:
            return None
        return 'Please design a protein with the following function:\n<FUNCTION> ' + f + ' </FUNCTION>'
    df['function_text'] = df['function_raw'].apply(build_function_text)
    before2 = len(df)
    df = df[df['function_text'].notna()]
    after2 = len(df)
    print(f'Status:{before2} -> {after2}')
    before3 = len(df)
    df = df.drop_duplicates(subset=['uniprot_id', 'ligand_smiles', 'sequence_clean'])
    after3 = len(df)
    print(f'( ){before3} -> {after3}')
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print('Status:', OUT_PATH)
    print('Status:')
    print(df[['uniprot_id', 'sequence_clean', 'ligand_name', 'ligand_smiles', 'function_text']].head())
if __name__ == '__main__':
    main()
