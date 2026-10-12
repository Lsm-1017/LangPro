import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', default='../data_intermediate/pair_paper_candidates.csv')
    ap.add_argument('--out', default='../data_intermediate/pair_paper_candidates_paironly.csv')
    ap.add_argument('--include_intersection', action='store_true', help='--include_intersection option.')
    args = ap.parse_args()
    inp = Path(args.inp)
    if not inp.exists():
        raise FileNotFoundError(inp)
    df = pd.read_csv(inp)
    need_cols = {'pair_key', 'paper_key', 'rank', 'source_tag'}
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f'missing columns: {miss} in {inp}')
    tags = ['pair']
    if args.include_intersection:
        tags.append('intersection')
    out = df[df['source_tag'].astype(str).isin(tags)].copy()
    out['pair_key'] = out['pair_key'].astype(str)
    out['paper_key'] = out['paper_key'].astype(str)
    out = out.sort_values(['pair_key', 'rank', 'paper_key'], kind='mergesort')
    out['rank'] = out.groupby('pair_key').cumcount() + 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print('Saved:', args.out)
    print('rows:', len(out))
    print('pairs:', out['pair_key'].nunique())
    print('unique papers:', out['paper_key'].nunique())
    print('avg papers per pair:', round(len(out) / max(out['pair_key'].nunique(), 1), 2))
if __name__ == '__main__':
    main()
