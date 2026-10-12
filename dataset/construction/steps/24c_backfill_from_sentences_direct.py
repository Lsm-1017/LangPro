import argparse
import json
from pathlib import Path
import pandas as pd
SENT_COL_CANDIDATES = ['sentence', 'sent_text', 'docking_sentence', 'text', 'sent', 'candidate_sentence']

def norm_str(x):
    if x is None:
        return ''
    if isinstance(x, float) and pd.isna(x):
        return ''
    s = str(x).strip()
    if s.lower() == 'nan':
        return ''
    return s

def find_sentence_col(df: pd.DataFrame) -> str:
    for c in SENT_COL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if 'sentence' in c.lower():
            return c
    raise ValueError(f'Cannot find sentence column. cols={df.columns.tolist()}')

def uniq_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', default='../data_intermediate/triples_with_ligand_text_full_with_docking.csv')
    ap.add_argument('--sent_csv', default='../data_intermediate/pair_docking_sentence_candidates_paironly_backfill_tight_v3.csv')
    ap.add_argument('--out_main', default='../data_intermediate/triples_with_ligand_text_full_with_docking_backfilled_24c.csv')
    ap.add_argument('--report', default='../data_intermediate/24c_backfill_report.json')
    ap.add_argument('--pair_col', default='pair_key')
    ap.add_argument('--docking_col', default='docking_text')
    ap.add_argument('--only_fill_missing', action='store_true', default=True, help='--only_fill_missing option.')
    ap.add_argument('--append_to_existing', action='store_true', default=False, help='--append_to_existing option.')
    ap.add_argument('--max_sent_per_pair', type=int, default=80)
    ap.add_argument('--dedup_sentence', action='store_true', default=True)
    ap.add_argument('--joiner', default='\n')
    ap.add_argument('--min_sent_len', type=int, default=15)
    args = ap.parse_args()
    main_path = Path(args.main)
    sent_path = Path(args.sent_csv)
    out_path = Path(args.out_main)
    rep_path = Path(args.report)
    if not main_path.exists():
        raise FileNotFoundError(main_path)
    if not sent_path.exists():
        raise FileNotFoundError(sent_path)
    dfm = pd.read_csv(main_path)
    if args.pair_col not in dfm.columns:
        raise ValueError(f'main missing {args.pair_col}. cols={dfm.columns.tolist()}')
    if args.docking_col not in dfm.columns:
        dfm[args.docking_col] = ''
    dfs = pd.read_csv(sent_path)
    if args.pair_col not in dfs.columns:
        raise ValueError(f'sent_csv missing {args.pair_col}. cols={dfs.columns.tolist()}')
    s_col = find_sentence_col(dfs)
    dfs[args.pair_col] = dfs[args.pair_col].astype(str).map(norm_str)
    dfs[s_col] = dfs[s_col].map(norm_str)
    dfs = dfs[(dfs[args.pair_col] != '') & (dfs[s_col] != '')]
    dfs = dfs[dfs[s_col].str.len() >= args.min_sent_len].copy()
    sort_cols = []
    if 'rank' in dfs.columns:
        sort_cols.append('rank')
    if 'paper_key' in dfs.columns:
        sort_cols.append('paper_key')
    if sort_cols:
        dfs = dfs.sort_values([args.pair_col] + sort_cols)
    pair2text = {}
    pair2n = {}
    for pk, g in dfs.groupby(args.pair_col, sort=False):
        sents = g[s_col].astype(str).tolist()
        if args.dedup_sentence:
            low = [x.strip().lower() for x in sents]
            keep_idx = []
            seen = set()
            for i, lx in enumerate(low):
                if lx in seen:
                    continue
                seen.add(lx)
                keep_idx.append(i)
            sents = [sents[i].strip() for i in keep_idx]
        else:
            sents = [x.strip() for x in sents]
        if args.max_sent_per_pair > 0:
            sents = sents[:args.max_sent_per_pair]
        txt = args.joiner.join([x for x in sents if x])
        if txt:
            pair2text[pk] = txt
            pair2n[pk] = len(sents)
    dfm[args.pair_col] = dfm[args.pair_col].astype(str).map(norm_str)
    before_nonempty = (dfm[args.docking_col].fillna('').astype(str).map(norm_str) != '').sum()
    pairs_total = dfm[args.pair_col].nunique()

    def merge_one(row):
        pk = row[args.pair_col]
        old = norm_str(row[args.docking_col])
        new = pair2text.get(pk, '')
        if not new:
            return old
        if args.append_to_existing:
            old_s = [x.strip() for x in old.split(args.joiner) if x.strip()]
            new_s = [x.strip() for x in new.split(args.joiner) if x.strip()]
            merged = old_s + new_s
            if args.dedup_sentence:
                low = [x.lower() for x in merged]
                keep = []
                seen = set()
                for i, lx in enumerate(low):
                    if lx in seen:
                        continue
                    seen.add(lx)
                    keep.append(i)
                merged = [merged[i] for i in keep]
            merged = merged[:args.max_sent_per_pair] if args.max_sent_per_pair > 0 else merged
            return args.joiner.join(merged)
        if args.only_fill_missing and old != '':
            return old
        return new
    dfm[args.docking_col] = dfm.apply(merge_one, axis=1)
    after_nonempty = (dfm[args.docking_col].fillna('').astype(str).map(norm_str) != '').sum()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfm.to_csv(out_path, index=False)
    filled_pairs_now = sum((1 for pk in pair2text.keys() if norm_str(dfm.loc[dfm[args.pair_col] == pk, args.docking_col].iloc[0]) != ''))
    rep = {'main': str(main_path), 'sent_csv': str(sent_path), 'pairs_total_in_main': int(pairs_total), 'pairs_with_nonempty_docking_before': int(before_nonempty), 'pairs_with_sentence_in_sent_csv': int(len(pair2text)), 'pairs_with_nonempty_docking_after': int(after_nonempty), 'filled_pairs_now_effective': int(filled_pairs_now), 'max_sent_per_pair': args.max_sent_per_pair, 'min_sent_len': args.min_sent_len, 'dedup_sentence': bool(args.dedup_sentence), 'only_fill_missing': bool(args.only_fill_missing), 'append_to_existing': bool(args.append_to_existing)}
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved main:', out_path)
    print('Saved report:', rep_path)
    print('pairs_total:', pairs_total)
    print('nonempty docking before:', before_nonempty, 'after:', after_nonempty)
    print('pairs_with_sentence_in_sent_csv:', len(pair2text))
    print('filled_pairs_now_effective:', filled_pairs_now)
if __name__ == '__main__':
    main()
