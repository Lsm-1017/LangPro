import re
import pandas as pd
from pathlib import Path
POOL = '../data_intermediate/paper_pool.csv'
CAND = '../data_intermediate/pair_paper_candidates.csv'
FTI = '../data_intermediate/paper_fulltext_index.csv'

def ft_ok(path: str) -> bool:
    if not path:
        return False
    p = Path(path)
    return p.exists() and p.is_file() and (p.stat().st_size > 200)

def norm_series(s):
    return s.fillna('').astype(str).str.strip()
BAD_AB_PATTERNS = ['no abstract', 'abstract not available', 'not available', 'n/?a\\b', '\u6682\u65e0\u6458\u8981', '\u65e0\u6458\u8981']
BAD_AB_RE = re.compile('|'.join(BAD_AB_PATTERNS), re.I)

def is_good_abstract(text: str, min_len: int=30) -> bool:
    t = (text or '').strip()
    if not t:
        return False
    t_low = t.lower()
    if BAD_AB_RE.search(t_low):
        return False
    if len(t) < min_len:
        return False
    return True

def main():
    pool = pd.read_csv(POOL)
    cand = pd.read_csv(CAND)
    fti = pd.read_csv(FTI) if Path(FTI).exists() else pd.DataFrame(columns=['paper_key', 'fulltext_path'])
    pool['paper_key_n'] = norm_series(pool.get('paper_key'))
    cand['paper_key_n'] = norm_series(cand.get('paper_key'))
    fti['paper_key_n'] = norm_series(fti.get('paper_key'))
    pool = pool.drop_duplicates('paper_key_n', keep='first')
    fti = fti.drop_duplicates('paper_key_n', keep='first')
    m = cand.merge(pool[['paper_key_n', 'abstract']], on='paper_key_n', how='left')
    m = m.merge(fti[['paper_key_n', 'fulltext_path']], on='paper_key_n', how='left')
    ft_path = norm_series(m.get('fulltext_path'))
    ft_nonempty = ft_path != ''
    ft_exists = [False] * len(m)
    idx = m.index[ft_nonempty].tolist()
    paths = ft_path[ft_nonempty].tolist()
    for i, p in zip(idx, paths):
        ft_exists[i] = ft_ok(p)
    ft_exists = pd.Series(ft_exists, index=m.index)
    ab = norm_series(m.get('abstract'))
    ab_good = ab.apply(lambda x: is_good_abstract(x, min_len=30))
    row_has_text = ft_exists | ab_good
    pair_has_text = m.assign(row_has_text=row_has_text).groupby('pair_key')['row_has_text'].any()
    pairs_total = int(pair_has_text.shape[0])
    pairs_with_text = int(pair_has_text.sum())
    pairs_no_text = pairs_total - pairs_with_text
    print('=== Pair Text Coverage ===')
    print('pairs_total:', pairs_total)
    print('pairs_with_text:', pairs_with_text, 'ratio:', pairs_with_text / pairs_total)
    print('pairs_no_text:', pairs_no_text, 'ratio:', pairs_no_text / pairs_total)
    per_pair_text_rows = m.assign(row_has_text=row_has_text).groupby('pair_key')['row_has_text'].sum()
    print('\n=== Per-pair text rows stats (row-level) ===')
    print('mean:', float(per_pair_text_rows.mean()))
    print('median:', float(per_pair_text_rows.median()))
    print('p10:', float(per_pair_text_rows.quantile(0.1)))
    print('p90:', float(per_pair_text_rows.quantile(0.9)))
    out = pd.DataFrame({'pair_key': per_pair_text_rows.index, 'text_rows': per_pair_text_rows.values, 'has_text': pair_has_text.values})
    out_path = Path('../data_intermediate/pair_text_coverage_report.csv')
    out.to_csv(out_path, index=False)
    print('\nSaved:', out_path)
if __name__ == '__main__':
    main()
