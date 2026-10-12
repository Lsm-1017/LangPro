import json
import argparse
from pathlib import Path
import pandas as pd

def norm_series(s: pd.Series) -> pd.Series:
    return s.fillna('').astype(str).str.strip()

def pick_first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--triples', default='../data_intermediate/triples_with_ligand_text_full.csv')
    ap.add_argument('--ligand_core', default='../data_intermediate/ligand_core.csv')
    ap.add_argument('--sentences', default='../data_intermediate/pair_docking_sentence_candidates.csv')
    ap.add_argument('--out_csv', default='../data_intermediate/triples_expanded_docking_sentences.csv')
    ap.add_argument('--out_report', default='../data_intermediate/triples_expanded_docking_sentences_report.json')
    ap.add_argument('--out_audit_lost', default='../data_intermediate/22b_audit_lost_pairs.csv')
    ap.add_argument('--max_sent_per_pair', type=int, default=80)
    ap.add_argument('--max_sent_per_paper', type=int, default=10)
    ap.add_argument('--dedup_triples_per_pair', action='store_true', help='--dedup_triples_per_pair option.')
    args = ap.parse_args()
    tri = pd.read_csv(args.triples)
    lig = pd.read_csv(args.ligand_core)
    snt = pd.read_csv(args.sentences)
    sent_col = pick_first_col(snt, ['docking_sentence', 'sentence', 'sent_text', 'sent', 'text'])
    paper_col = pick_first_col(snt, ['paper_key', 'pmcid', 'pmid', 'doi'])
    rank_col = pick_first_col(snt, ['sent_rank', 'rank', 'sentence_rank', 'hit_rank', 'score'])
    if sent_col is None:
        raise ValueError(f'Cannot find sentence column in {args.sentences}. cols={snt.columns.tolist()}')
    if paper_col is None:
        paper_col = '__paper__'
        snt[paper_col] = ''
    snt['pair_key_n'] = norm_series(snt['pair_key'])
    snt[paper_col] = norm_series(snt[paper_col])
    if rank_col is None:
        snt['_rank'] = range(1, len(snt) + 1)
        rank_col = '_rank'
    else:
        snt[rank_col] = pd.to_numeric(snt[rank_col], errors='coerce').fillna(10 ** 9).astype(int)
    snt[sent_col] = norm_series(snt[sent_col])
    snt = snt[snt[sent_col] != ''].copy()
    tri['uniprot_id_n'] = norm_series(tri['uniprot_id'])
    tri['lig_smiles_n'] = norm_series(tri['ligand_smiles_canon'])
    lig['lig_smiles_n'] = norm_series(lig['ligand_smiles_canon'])
    lig['inchikey_n'] = norm_series(lig['inchikey']).str.upper()
    tri2 = tri.merge(lig[['lig_smiles_n', 'inchikey_n']], on='lig_smiles_n', how='left')
    tri2['inchikey_n'] = tri2['inchikey_n'].fillna('').astype(str).str.strip().str.upper()
    tri2['pair_key_n'] = tri2['uniprot_id_n'] + '__' + tri2['inchikey_n']
    miss_inchikey_rows = int((tri2['inchikey_n'] == '').sum())
    if args.dedup_triples_per_pair:
        tri2 = tri2.sort_values(['pair_key_n']).drop_duplicates('pair_key_n', keep='first')
    merged = tri2.merge(snt[['pair_key_n', paper_col, rank_col, sent_col] + [c for c in ['source', 'source_tag', 'text_type'] if c in snt.columns]], on='pair_key_n', how='inner')
    merged = merged.sort_values(['pair_key_n', paper_col, rank_col], ascending=[True, True, True])
    merged = merged.groupby(['pair_key_n', paper_col], as_index=False, group_keys=False).head(args.max_sent_per_paper)
    merged = merged.groupby('pair_key_n', as_index=False, group_keys=False).head(args.max_sent_per_pair)
    merged = merged.rename(columns={'pair_key_n': 'pair_key', paper_col: 'paper_key', rank_col: 'sent_rank', sent_col: 'docking_sentence'})
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    pairs_total_in_triples = tri2['pair_key_n'].nunique()
    pairs_with_any_sentence = merged['pair_key'].nunique()
    report = {'pairs_total_in_triples': int(pairs_total_in_triples), 'pairs_with_any_sentence': int(pairs_with_any_sentence), 'pair_hit_ratio': float(pairs_with_any_sentence / max(1, pairs_total_in_triples)), 'output_rows': int(len(merged)), 'mean_rows_per_hit_pair': float(len(merged) / max(1, pairs_with_any_sentence)), 'max_sent_per_pair': args.max_sent_per_pair, 'max_sent_per_paper': args.max_sent_per_paper, 'miss_inchikey_rows_in_triples': miss_inchikey_rows}
    Path(args.out_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pairs_sent = set(snt['pair_key_n'].unique().tolist())
    pairs_out = set(merged['pair_key'].unique().tolist())
    lost = sorted(list(pairs_sent - pairs_out))
    pd.DataFrame({'pair_key': lost}).to_csv(args.out_audit_lost, index=False)
    print('Saved:', out_path, 'rows=', len(merged))
    print('Saved:', args.out_report)
    print('Saved:', args.out_audit_lost, 'lost_pairs=', len(lost))
    print('Report:', report)
if __name__ == '__main__':
    main()
