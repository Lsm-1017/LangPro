import argparse
import json
from pathlib import Path
import pandas as pd

def norm_str(x) -> str:
    if x is None:
        return ''
    if isinstance(x, float) and pd.isna(x):
        return ''
    s = str(x).strip()
    if s.lower() == 'nan':
        return ''
    return s

def detect_sentence_col(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for key in ['docking_sentence', 'sentence', 'sent_text', 'sent', 'text']:
        if key in lower:
            return lower[key]
    for c in df.columns:
        if 'sentence' in c.lower():
            return c
    raise ValueError(f'Cannot find sentence column in file. cols={df.columns.tolist()}')

def build_pair_sentence_agg(sent_df: pd.DataFrame, max_sent_per_pair: int=80, max_sent_per_paper: int=10, sep: str='\n', dedup_sentence: bool=True, include_paper_key: bool=False) -> pd.DataFrame:
    if 'pair_key' not in sent_df.columns:
        if 'uniprot_id' in sent_df.columns and 'inchikey' in sent_df.columns:
            sent_df['pair_key'] = sent_df['uniprot_id'].astype(str).str.strip() + '__' + sent_df['inchikey'].astype(str).str.strip().str.upper()
        else:
            raise ValueError('sentence file missing pair_key (and cannot derive from uniprot_id+inchikey).')
    sent_col = detect_sentence_col(sent_df)
    sort_cols = []
    if 'rank' in sent_df.columns:
        sort_cols.append('rank')
    if 'paper_rank' in sent_df.columns:
        sort_cols.append('paper_rank')
    if 'sent_rank' in sent_df.columns:
        sort_cols.append('sent_rank')
    if sort_cols:
        sent_df = sent_df.sort_values(['pair_key'] + sort_cols, kind='mergesort')
    else:
        sent_df = sent_df.sort_values(['pair_key'], kind='mergesort')
    sent_df['pair_key'] = sent_df['pair_key'].astype(str).str.strip()
    sent_df[sent_col] = sent_df[sent_col].apply(norm_str)
    sent_df = sent_df[sent_df[sent_col] != ''].copy()
    if max_sent_per_paper > 0 and 'paper_key' in sent_df.columns:
        sent_df['paper_key'] = sent_df['paper_key'].apply(norm_str)
        sent_df = sent_df.groupby(['pair_key', 'paper_key'], sort=False, as_index=False).head(max_sent_per_paper).copy()
    if max_sent_per_pair > 0:
        sent_df = sent_df.groupby('pair_key', sort=False, as_index=False).head(max_sent_per_pair).copy()
    if dedup_sentence:
        sent_df['_sent_norm'] = sent_df[sent_col].str.lower().str.strip()
        sent_df = sent_df.drop_duplicates(subset=['pair_key', '_sent_norm'], keep='first').copy()
        sent_df = sent_df.drop(columns=['_sent_norm'])

    def agg_one(g: pd.DataFrame):
        sents = g[sent_col].tolist()
        if include_paper_key and 'paper_key' in g.columns:
            pks = g['paper_key'].tolist()
            sents = [f'[{pk}] {st}' if pk else st for pk, st in zip(pks, sents)]
        docking_text = sep.join(sents)
        paper_cnt = g['paper_key'].nunique() if 'paper_key' in g.columns else 0
        return pd.Series({'docking_text': docking_text, 'docking_sentence_count': len(sents), 'docking_paper_count': int(paper_cnt)})
    out = sent_df.groupby('pair_key', sort=False).apply(agg_one).reset_index()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main_csv', default='../data_intermediate/triples_with_ligand_text_full.csv')
    ap.add_argument('--ligand_core', default='../data_intermediate/ligand_core.csv')
    ap.add_argument('--sentence_csv', default='../data_intermediate/pair_docking_sentence_candidates.csv')
    ap.add_argument('--out_csv', default='../data_intermediate/triples_with_ligand_text_full_with_docking.csv')
    ap.add_argument('--report_json', default='../data_intermediate/22c_backfill_report.json')
    ap.add_argument('--max_sent_per_pair', type=int, default=80)
    ap.add_argument('--max_sent_per_paper', type=int, default=10)
    ap.add_argument('--sep', default='\n')
    ap.add_argument('--dedup_sentence', action='store_true', help='--dedup_sentence input path or value.')
    ap.add_argument('--include_paper_key', action='store_true', help='--include_paper_key input path or value.')
    args = ap.parse_args()
    main_path = Path(args.main_csv)
    lig_path = Path(args.ligand_core)
    sent_path = Path(args.sentence_csv)
    if not main_path.exists():
        raise FileNotFoundError(main_path)
    if not lig_path.exists():
        raise FileNotFoundError(lig_path)
    if not sent_path.exists():
        raise FileNotFoundError(sent_path)
    df = pd.read_csv(main_path)
    lig = pd.read_csv(lig_path)
    sent = pd.read_csv(sent_path)
    if 'uniprot_id' not in df.columns:
        raise ValueError('main_csv must contain uniprot_id')
    if 'ligand_smiles_canon' not in df.columns:
        raise ValueError('main_csv must contain ligand_smiles_canon (for mapping to inchikey)')
    if 'ligand_smiles_canon' not in lig.columns or 'inchikey' not in lig.columns:
        raise ValueError(f'ligand_core.csv must contain ligand_smiles_canon + inchikey. cols={lig.columns.tolist()}')
    lig['ligand_smiles_canon_n'] = lig['ligand_smiles_canon'].apply(norm_str)
    lig['inchikey_n'] = lig['inchikey'].apply(norm_str).str.upper()
    smi2ik = dict(zip(lig['ligand_smiles_canon_n'], lig['inchikey_n']))
    df['uniprot_id_n'] = df['uniprot_id'].apply(norm_str)
    df['ligand_smiles_canon_n'] = df['ligand_smiles_canon'].apply(norm_str)
    df['inchikey'] = df['ligand_smiles_canon_n'].map(smi2ik).fillna('')
    df['pair_key'] = df['uniprot_id_n'] + '__' + df['inchikey']
    miss_ik = (df['inchikey'] == '').sum()
    agg = build_pair_sentence_agg(sent_df=sent, max_sent_per_pair=args.max_sent_per_pair, max_sent_per_paper=args.max_sent_per_paper, sep=args.sep, dedup_sentence=args.dedup_sentence, include_paper_key=args.include_paper_key)
    out = df.merge(agg, on='pair_key', how='left')
    for c in ['docking_text', 'docking_sentence_count', 'docking_paper_count']:
        if c not in out.columns:
            out[c] = '' if c == 'docking_text' else 0
    out['docking_text'] = out['docking_text'].fillna('')
    out['docking_sentence_count'] = out['docking_sentence_count'].fillna(0).astype(int)
    out['docking_paper_count'] = out['docking_paper_count'].fillna(0).astype(int)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    pairs_total = out['pair_key'].nunique()
    pairs_with_text = out.groupby('pair_key')['docking_text'].apply(lambda x: (x.astype(str).str.len() > 0).any()).sum()
    report = {'main_csv': str(main_path), 'sentence_csv': str(sent_path), 'out_csv': str(out_path), 'rows_main': int(len(df)), 'pairs_total_in_main': int(pairs_total), 'pairs_with_nonempty_docking_text': int(pairs_with_text), 'pair_hit_ratio': float(pairs_with_text / max(pairs_total, 1)), 'rows_with_nonempty_docking_text': int((out['docking_text'].astype(str).str.len() > 0).sum()), 'miss_inchikey_rows_in_main': int(miss_ik), 'args': {'max_sent_per_pair': args.max_sent_per_pair, 'max_sent_per_paper': args.max_sent_per_paper, 'dedup_sentence': bool(args.dedup_sentence), 'include_paper_key': bool(args.include_paper_key), 'sep': args.sep}}
    rep_path = Path(args.report_json)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', out_path)
    print('Saved report:', rep_path)
    print('pairs_total_in_main:', report['pairs_total_in_main'])
    print('pairs_with_nonempty_docking_text:', report['pairs_with_nonempty_docking_text'], 'ratio:', report['pair_hit_ratio'])
    print('miss_inchikey_rows_in_main:', report['miss_inchikey_rows_in_main'])
if __name__ == '__main__':
    main()
