import re
import gzip
import time
import random
import argparse
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
EPMC_FULLTEXT_XML = 'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'

def norm_str(x):
    if x is None:
        return ''
    try:
        if isinstance(x, float) and pd.isna(x):
            return ''
    except Exception:
        pass
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

def extract_pmcid_from_paper_key(paper_key: str) -> str:
    s = norm_str(paper_key)
    if not s:
        return ''
    m = re.search('(PMC\\d+)', s)
    return m.group(1) if m else ''

def extract_plaintext_from_jats(xml_bytes: bytes) -> str:
    txt = xml_bytes.decode('utf-8', errors='ignore')
    txt = re.sub('<script.*?</script>', ' ', txt, flags=re.S | re.I)
    txt = re.sub('<style.*?</style>', ' ', txt, flags=re.S | re.I)
    txt = re.sub('<[^>]+>', ' ', txt)
    txt = txt.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    txt = txt.replace('&quot;', '"').replace('&apos;', "'")
    txt = re.sub('\\s+', ' ', txt).strip()
    return txt

def safe_get(session: requests.Session, url: str, timeout=120, max_retry=6):
    for k in range(max_retry):
        try:
            r = session.get(url, timeout=timeout, headers={'Accept': 'application/xml', 'User-Agent': 'langpro-epmc-fulltext/1.0'})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** k + random.random())
                continue
            return r
        except requests.RequestException:
            time.sleep(2 ** k + random.random())
    return None

def build_targets_topk(pair_candidates_csv: str, topk: int):
    cand = pd.read_csv(pair_candidates_csv)
    need_cols = {'pair_key', 'paper_key', 'rank'}
    miss = need_cols - set(cand.columns)
    if miss:
        raise ValueError(f'{pair_candidates_csv} missing cols: {miss}, got={cand.columns.tolist()}')
    cand['pair_key'] = cand['pair_key'].astype(str)
    cand['paper_key'] = cand['paper_key'].astype(str)
    cand_top = cand.sort_values(['pair_key', 'rank']).groupby('pair_key', as_index=False).head(topk).copy()
    cand_top['pmcid'] = cand_top['paper_key'].apply(extract_pmcid_from_paper_key)
    cand_top = cand_top[cand_top['pmcid'] != ''].copy()
    pmcid2keys = {}
    for pmc, pk in zip(cand_top['pmcid'], cand_top['paper_key']):
        pmcid2keys.setdefault(pmc, []).append(pk)
    pmcids = sorted(pmcid2keys.keys())
    return (pmcid2keys, pmcids)

def build_targets_cand(paper_pool_csv: str, candidates_unique_csv: str):
    pool = pd.read_csv(paper_pool_csv)
    if 'paper_key' not in pool.columns or 'pmcid' not in pool.columns:
        raise ValueError('paper_pool.csv paper_key pmcid')
    pool['paper_key'] = pool['paper_key'].astype(str)
    pool['pmcid'] = pool['pmcid'].apply(norm_pmcid)
    cand = pd.read_csv(candidates_unique_csv)
    if 'paper_key' not in cand.columns:
        raise ValueError('paper_candidates_unique.csv paper_key')
    cand_keys = set(cand['paper_key'].astype(str).tolist())
    pool = pool[pool['paper_key'].isin(cand_keys)].copy()
    pmcid2keys = {}
    for pk, pmc in zip(pool['paper_key'], pool['pmcid']):
        if not pmc:
            continue
        pmcid2keys.setdefault(pmc, []).append(pk)
    pmcids = sorted(pmcid2keys.keys())
    return (pmcid2keys, pmcids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', choices=['topk', 'cand'], default='topk', help='--scope option.')
    ap.add_argument('--topk', type=int, default=20, help='--topk numeric setting.')
    ap.add_argument('--pair_candidates', default='../data_intermediate/pair_paper_candidates.csv', help='--pair_candidates option.')
    ap.add_argument('--paper_pool', default='../data_intermediate/paper_pool.csv')
    ap.add_argument('--candidates_unique', default='../data_intermediate/paper_candidates_unique.csv')
    ap.add_argument('--out_dir', default='../data_fulltext/epmc_txt')
    ap.add_argument('--pmcid_index', default='../data_intermediate/pmcid_fulltext_txt_index.csv')
    ap.add_argument('--paper_index', default='../data_intermediate/paper_fulltext_index.csv')
    ap.add_argument('--sleep', type=float, default=0.06)
    ap.add_argument('--timeout', type=int, default=120)
    ap.add_argument('--max_retry', type=int, default=6)
    ap.add_argument('--max_pmcid', type=int, default=0, help='--max_pmcid numeric setting.')
    args = ap.parse_args()
    if args.scope == 'topk':
        pmcid2keys, pmcids = build_targets_topk(args.pair_candidates, args.topk)
        print(f'scope=topk topk={args.topk}')
    else:
        pmcid2keys, pmcids = build_targets_cand(args.paper_pool, args.candidates_unique)
        print('scope=cand (paper_candidates_unique)')
    print('PMCID targets:', len(pmcids))
    if args.max_pmcid and args.max_pmcid > 0:
        pmcids = pmcids[:args.max_pmcid]
        print('PMCID targets after limit:', len(pmcids))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pmcid_index = Path(args.pmcid_index)
    paper_index = Path(args.paper_index)
    pmcid_index.parent.mkdir(parents=True, exist_ok=True)
    paper_index.parent.mkdir(parents=True, exist_ok=True)
    done_pmcid = set()
    if pmcid_index.exists():
        old = pd.read_csv(pmcid_index)
        if 'pmcid' in old.columns:
            done_pmcid = set(old['pmcid'].astype(str).tolist())
    print('already_done_pmcid:', len(done_pmcid))
    session = requests.Session()
    pmcid_rows = []
    ok, miss, fail = (0, 0, 0)
    to_fetch = [p for p in pmcids if p not in done_pmcid]
    print('pmcid_to_fetch:', len(to_fetch))
    for pmc in tqdm(to_fetch, desc='EPMC fulltext (PMCID)'):
        out_txt = out_dir / f'{pmc}.txt.gz'
        if out_txt.exists() and out_txt.stat().st_size > 200:
            pmcid_rows.append({'pmcid': pmc, 'status': 'ok_cached', 'txt_gz_path': str(out_txt), 'bytes': int(out_txt.stat().st_size), 'source': 'epmc_fullTextXML', 'url': EPMC_FULLTEXT_XML.format(pmcid=pmc)})
            ok += 1
            continue
        url = EPMC_FULLTEXT_XML.format(pmcid=pmc)
        r = safe_get(session, url, timeout=args.timeout, max_retry=args.max_retry)
        if r is None:
            pmcid_rows.append({'pmcid': pmc, 'status': 'request_fail', 'txt_gz_path': '', 'bytes': 0, 'source': 'epmc_fullTextXML', 'url': url})
            fail += 1
            time.sleep(args.sleep)
            continue
        if r.status_code == 200 and r.content and (r.content[:1] == b'<'):
            txt = extract_plaintext_from_jats(r.content)
            if len(txt) < 500:
                pmcid_rows.append({'pmcid': pmc, 'status': 'txt_too_short', 'txt_gz_path': '', 'bytes': 0, 'source': 'epmc_fullTextXML', 'url': url})
                miss += 1
            else:
                with gzip.open(out_txt, 'wb') as f:
                    f.write(txt.encode('utf-8'))
                pmcid_rows.append({'pmcid': pmc, 'status': 'ok', 'txt_gz_path': str(out_txt), 'bytes': int(out_txt.stat().st_size), 'source': 'epmc_fullTextXML', 'url': url})
                ok += 1
        else:
            pmcid_rows.append({'pmcid': pmc, 'status': f'http_{r.status_code}', 'txt_gz_path': '', 'bytes': 0, 'source': 'epmc_fullTextXML', 'url': url})
            miss += 1
        time.sleep(args.sleep)
        if len(pmcid_rows) >= 2000:
            new_df = pd.DataFrame(pmcid_rows)
            if pmcid_index.exists():
                old = pd.read_csv(pmcid_index)
                out = pd.concat([old, new_df], ignore_index=True)
                out = out.drop_duplicates(subset=['pmcid'], keep='last')
            else:
                out = new_df
            out.to_csv(pmcid_index, index=False)
            pmcid_rows = []
    if pmcid_rows:
        new_df = pd.DataFrame(pmcid_rows)
        if pmcid_index.exists():
            old = pd.read_csv(pmcid_index)
            out = pd.concat([old, new_df], ignore_index=True)
            out = out.drop_duplicates(subset=['pmcid'], keep='last')
        else:
            out = new_df
        out.to_csv(pmcid_index, index=False)
    idx = pd.read_csv(pmcid_index) if pmcid_index.exists() else pd.DataFrame(columns=['pmcid', 'status', 'txt_gz_path'])
    pmc2row = {str(r['pmcid']): (str(r.get('status', '')), str(r.get('txt_gz_path', ''))) for _, r in idx.iterrows()}
    paper_rows = []
    for pmc, keys in pmcid2keys.items():
        status, path = pmc2row.get(pmc, ('', ''))
        for pk in keys:
            paper_rows.append({'paper_key': pk, 'pmcid': pmc, 'fulltext': path, 'fulltext_path': path, 'fulltext_status': status, 'fulltext_source': 'epmc_fullTextXML'})
    paper_df = pd.DataFrame(paper_rows).drop_duplicates(subset=['paper_key'], keep='first')
    paper_df.to_csv(paper_index, index=False)
    print('Saved PMCID index:', pmcid_index)
    print('Saved paper_fulltext_index:', paper_index)
    print('download_ok:', ok, 'missing_no_fulltext:', miss, 'fail:', fail)
if __name__ == '__main__':
    main()
