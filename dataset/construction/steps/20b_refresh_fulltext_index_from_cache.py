import re
from pathlib import Path
import pandas as pd
PAPER_POOL = Path('../data_intermediate/paper_pool.csv')
CAND_UNIQ = Path('../data_intermediate/paper_candidates_unique.csv')
CACHE_DIR = Path('../data_fulltext/epmc_txt')
PMCID_IDX = Path('../data_intermediate/pmcid_fulltext_txt_index.csv')
PAPER_IDX = Path('../data_intermediate/paper_fulltext_index.csv')
EPMC_URL_TPL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'

def norm_str(x):
    if x is None:
        return ''
    if isinstance(x, float) and pd.isna(x):
        return ''
    s = str(x).strip()
    if s.lower() == 'nan':
        return ''
    return s

def norm_pmcid(x):
    s = norm_str(x)
    if not s:
        return ''
    if s.lower().startswith('pmc'):
        s = 'PMC' + s[3:]
    elif s.isdigit():
        s = 'PMC' + s
    if not s.startswith('PMC'):
        return ''
    return s

def main():
    if not PAPER_POOL.exists():
        raise FileNotFoundError(PAPER_POOL)
    if not CAND_UNIQ.exists():
        raise FileNotFoundError(CAND_UNIQ)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PMCID_IDX.parent.mkdir(parents=True, exist_ok=True)
    PAPER_IDX.parent.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(PAPER_POOL)
    cand = pd.read_csv(CAND_UNIQ)
    if 'paper_key' not in pool.columns or 'pmcid' not in pool.columns:
        raise ValueError('paper_pool.csv paper_key, pmcid')
    if 'paper_key' not in cand.columns:
        raise ValueError('paper_candidates_unique.csv paper_key')
    cand_keys = set(cand['paper_key'].astype(str).tolist())
    pool = pool[pool['paper_key'].astype(str).isin(cand_keys)].copy()
    pool['paper_key'] = pool['paper_key'].astype(str)
    pool['pmcid'] = pool['pmcid'].apply(norm_pmcid)
    pmcid2keys = {}
    for pk, pmc in zip(pool['paper_key'], pool['pmcid']):
        if not pmc:
            continue
        pmcid2keys.setdefault(pmc, []).append(pk)
    if PMCID_IDX.exists():
        old = pd.read_csv(PMCID_IDX)
    else:
        old = pd.DataFrame(columns=['pmcid', 'status', 'txt_gz_path', 'bytes', 'source', 'url'])
    for c in ['pmcid', 'status', 'txt_gz_path', 'bytes', 'source', 'url']:
        if c not in old.columns:
            old[c] = ''
    old['pmcid'] = old['pmcid'].astype(str)
    old_records = old.to_dict(orient='records')
    old_map = {}
    for rec in old_records:
        pmc = norm_pmcid(rec.get('pmcid'))
        if not pmc:
            continue
        rec['pmcid'] = pmc
        old_map[pmc] = dict(rec)
    cached = {}
    for p in CACHE_DIR.glob('PMC*.txt.gz'):
        m = re.match('(PMC\\d+)\\.txt\\.gz$', p.name)
        if not m:
            continue
        pmc = m.group(1)
        sz = p.stat().st_size
        if sz > 200:
            cached[pmc] = (str(p), int(sz))
    updated = 0
    for pmc, (path, sz) in cached.items():
        rec = old_map.get(pmc, {})
        old_path = norm_str(rec.get('txt_gz_path'))
        old_status = norm_str(rec.get('status'))
        if old_path != path or old_status not in ('ok', 'ok_cached'):
            old_map[pmc] = {'pmcid': pmc, 'status': 'ok_cached', 'txt_gz_path': path, 'bytes': sz, 'source': 'epmc_fullTextXML', 'url': EPMC_URL_TPL.format(pmcid=pmc)}
            updated += 1
    new_idx = pd.DataFrame.from_records(list(old_map.values()))
    if not new_idx.empty:
        new_idx = new_idx.drop_duplicates(subset=['pmcid'], keep='last')
    want_cols = ['pmcid', 'status', 'txt_gz_path', 'bytes', 'source', 'url']
    for c in want_cols:
        if c not in new_idx.columns:
            new_idx[c] = ''
    new_idx = new_idx[want_cols]
    new_idx.to_csv(PMCID_IDX, index=False)
    pmc2row = {str(r['pmcid']): (norm_str(r.get('status')), norm_str(r.get('txt_gz_path'))) for _, r in new_idx.iterrows()} if not new_idx.empty else {}
    paper_rows = []
    for pmc, keys in pmcid2keys.items():
        status, path = pmc2row.get(pmc, ('', ''))
        for pk in keys:
            paper_rows.append({'paper_key': pk, 'pmcid': pmc, 'fulltext': path, 'fulltext_path': path, 'fulltext_status': status, 'fulltext_source': 'epmc_fullTextXML'})
    paper_df = pd.DataFrame(paper_rows).drop_duplicates(subset=['paper_key'], keep='first')
    paper_df.to_csv(PAPER_IDX, index=False)
    print('cache_files:', len(cached))
    print('pmcid_index_rows:', len(new_idx), 'updated_from_cache:', updated)
    print('Saved:', PMCID_IDX)
    print('Saved:', PAPER_IDX)
if __name__ == '__main__':
    main()
