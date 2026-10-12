import json
import argparse
import pandas as pd
from pathlib import Path

def norm(s):
    return s.fillna('').astype(str).str.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', default='../data_intermediate/triples_with_ligand_text_full_with_docking_plus_24b_tight_v3.csv')
    ap.add_argument('--main_docking_col', default='docking_text')
    ap.add_argument('--sent_csv', default='../data_intermediate/pair_docking_sentence_candidates_25a_backfill.csv')
    ap.add_argument('--out_main', default='../data_intermediate/triples_with_ligand_text_full_with_docking_plus_25a.csv')
    ap.add_argument('--report', default='../data_intermediate/25b_backfill_report.json')
    ap.add_argument('--max_sent_per_pair', type=int, default=80)
    ap.add_argument('--dedup_sentence', action='store_true', default=True)
    ap.add_argument('--only_fill_missing', action='store_true', default=True)
    args = ap.parse_args()
    main_df = pd.read_csv(args.main, low_memory=False)
    if 'pair_key' not in main_df.columns:
        raise ValueError('main pair_key')
    if args.main_docking_col not in main_df.columns:
        raise ValueError(f'main{args.main_docking_col}')
    dock = norm(main_df[args.main_docking_col])
    before_nonempty = int((dock != '').sum())
    total_pairs = int(main_df['pair_key'].nunique())
    sent = pd.read_csv(args.sent_csv, low_memory=False)
    need = {'pair_key', 'sentence'}
    miss = [c for c in need if c not in sent.columns]
    if miss:
        raise ValueError(f'sent_csv :{miss}. cols={sent.columns.tolist()}')
    sent['pair_key'] = sent['pair_key'].astype(str)
    sent['sentence'] = sent['sentence'].fillna('').astype(str).str.strip()
    sent = sent[sent['sentence'] != ''].copy()
    if 'score' in sent.columns:
        sent = sent.sort_values(['pair_key', 'score'], ascending=[True, False])
    else:
        sent = sent.sort_values(['pair_key'])

    def agg(g: pd.DataFrame) -> str:
        sents = g['sentence'].tolist()
        if args.dedup_sentence:
            seen = set()
            uniq = []
            for x in sents:
                k = ' '.join(x.lower().split())
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(x)
            sents = uniq
        sents = sents[:args.max_sent_per_pair]
        return '\n'.join(sents)
    agg_df = sent.groupby('pair_key', sort=False).apply(agg).reset_index()
    agg_df.columns = ['pair_key', 'docking_text_new']
    out = main_df.merge(agg_df, on='pair_key', how='left')
    old = norm(out[args.main_docking_col])
    new = norm(out['docking_text_new'])
    if args.only_fill_missing:
        fill_mask = (old == '') & (new != '')
        out.loc[fill_mask, args.main_docking_col] = out.loc[fill_mask, 'docking_text_new']
    else:
        fill_mask = new != ''
        out.loc[fill_mask, args.main_docking_col] = out.loc[fill_mask, 'docking_text_new']
    out = out.drop(columns=['docking_text_new'])
    after_nonempty = int((norm(out[args.main_docking_col]) != '').sum())
    filled_now = int(after_nonempty - before_nonempty)
    Path(args.out_main).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_main, index=False)
    report = {'pairs_total': total_pairs, 'nonempty docking before': before_nonempty, 'nonempty docking after': after_nonempty, 'filled_pairs_now_effective': filled_now, 'sent_pairs_in_csv': int(sent['pair_key'].nunique()), 'args': vars(args)}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('Saved main:', args.out_main)
    print('Saved report:', args.report)
    print('pairs_total:', total_pairs)
    print('nonempty docking before:', before_nonempty, 'after:', after_nonempty)
    print('filled_pairs_now_effective:', filled_now)
if __name__ == '__main__':
    main()
