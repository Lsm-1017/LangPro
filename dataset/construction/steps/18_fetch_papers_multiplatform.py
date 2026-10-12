import os
import re
import json
import time
import math
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import requests
import pandas as pd
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
PROTEIN_INDEX = Path('../data_intermediate/protein_search_index.csv')
LIGAND_INDEX = Path('../data_intermediate/ligand_search_index.csv')
PAIR_INDEX = Path('../data_intermediate/pair_search_index.csv')
OUT_POOL = Path('../data_intermediate/paper_pool.csv')
OUT_SOURCES = Path('../data_intermediate/paper_pool_sources.jsonl')
CACHE_DIR = Path('../data_intermediate/.cache_papers')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EUPMC_SEARCH = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
OPENALEX_WORKS = 'https://api.openalex.org/works'
DOCKING_HINT = '(docking OR "molecular docking" OR docked OR "binding mode" OR "binding site" OR "in complex with" OR complex OR "crystal structure" OR "co-crystal" OR "binding affinity")'

def is_nan(x) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False

def is_nullish(x) -> bool:
    if x is None:
        return True
    if is_nan(x):
        return True
    if isinstance(x, str) and x.strip() == '':
        return True
    return False

def to_text(x) -> str:
    if x is None or is_nan(x):
        return ''
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()

def to_text_lower(x) -> str:
    return to_text(x).lower()

def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=6, connect=6, read=6, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=['GET', 'POST'], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    s.headers.update({'User-Agent': 'instructpro_bench/step18 paper fetcher (contact: you)', 'Accept': 'application/json,text/plain,*/*'})
    return s
SESSION = build_session()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def safe_json_load(x):
    if x is None or is_nan(x):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        t = x.strip()
        if not t:
            return []
        try:
            y = json.loads(t)
            return y if isinstance(y, list) else []
        except Exception:
            return []
    return []

def normalize_doi(doi: str) -> str:
    doi = to_text(doi)
    if not doi:
        return ''
    doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '')
    doi = doi.replace('doi:', '').strip()
    return doi.lower()

def eu_quote(term: str) -> str:
    term = to_text(term)
    if not term:
        return ''
    term = term.replace('"', '\\"')
    return f'"{term}"' if ' ' in term or len(term) >= 4 or re.search('[A-Za-z]\\d', term) else term

def build_or_group(terms: List[str], max_terms: int) -> str:
    terms = [to_text(t) for t in terms if not is_nullish(t)]
    seen = set()
    uniq = []
    for t in terms:
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        uniq.append(t)
    uniq = uniq[:max_terms]
    if not uniq:
        return ''
    parts = [eu_quote(t) for t in uniq if eu_quote(t)]
    if not parts:
        return ''
    if len(parts) == 1:
        return parts[0]
    return '(' + ' OR '.join(parts) + ')'

def eupmc_search(query: str, page_size: int=25, max_hits: int=100, timeout: int=60, sleep: float=0.12) -> List[Dict[str, Any]]:
    query = to_text(query)
    if not query:
        return []
    cache_key = sha1(f'eupmc::{query}::{page_size}::{max_hits}')
    cache_path = CACHE_DIR / f'{cache_key}.json'
    if cache_path.exists() and cache_path.stat().st_size > 50:
        try:
            return json.loads(cache_path.read_text(encoding='utf-8'))
        except Exception:
            pass
    all_res = []
    cursor = '*'
    while len(all_res) < max_hits:
        params = {'query': query, 'format': 'json', 'pageSize': int(page_size), 'cursorMark': cursor, 'resultType': 'core'}
        try:
            r = SESSION.get(EUPMC_SEARCH, params=params, timeout=timeout)
        except Exception:
            break
        if r.status_code != 200:
            break
        try:
            js = r.json()
        except Exception:
            break
        result_list = js.get('resultList', {}).get('result', []) or []
        if not result_list:
            break
        all_res.extend(result_list)
        next_cursor = js.get('nextCursorMark')
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(sleep)
    try:
        cache_path.write_text(json.dumps(all_res, ensure_ascii=False), encoding='utf-8')
    except Exception:
        pass
    return all_res[:max_hits]

def openalex_search(search_text: str, per_page: int=25, max_hits: int=50, timeout: int=60, sleep: float=0.12) -> List[Dict[str, Any]]:
    search_text = to_text(search_text)
    if not search_text:
        return []
    cache_key = sha1(f'openalex::{search_text}::{per_page}::{max_hits}')
    cache_path = CACHE_DIR / f'{cache_key}.json'
    if cache_path.exists() and cache_path.stat().st_size > 50:
        try:
            return json.loads(cache_path.read_text(encoding='utf-8'))
        except Exception:
            pass
    out = []
    page = 1
    while len(out) < max_hits:
        params = {'search': search_text, 'per-page': int(per_page), 'page': int(page)}
        try:
            r = SESSION.get(OPENALEX_WORKS, params=params, timeout=timeout)
        except Exception:
            break
        if r.status_code != 200:
            break
        try:
            js = r.json()
        except Exception:
            break
        results = js.get('results', []) or []
        if not results:
            break
        out.extend(results)
        if len(results) < per_page:
            break
        page += 1
        time.sleep(sleep)
    try:
        cache_path.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    except Exception:
        pass
    return out[:max_hits]

def openalex_abstract(inv_idx: Optional[Dict[str, List[int]]]) -> str:
    if not isinstance(inv_idx, dict) or not inv_idx:
        return ''
    pairs = []
    for w, poss in inv_idx.items():
        if not isinstance(poss, list):
            continue
        for p in poss:
            if isinstance(p, int):
                pairs.append((p, w))
    if not pairs:
        return ''
    pairs.sort(key=lambda x: x[0])
    return ' '.join([w for _, w in pairs])

def eupmc_min_fields(x: Dict[str, Any]) -> Dict[str, Any]:
    return {'source': 'europepmc', 'pmid': to_text(x.get('pmid')), 'pmcid': to_text(x.get('pmcid')), 'doi': normalize_doi(x.get('doi')), 'title': to_text(x.get('title')), 'journal': to_text(x.get('journalTitle')), 'year': to_text(x.get('pubYear')), 'author': to_text(x.get('authorString')), 'is_oa': 1 if to_text_lower(x.get('isOpenAccess')) == 'y' else 0, 'has_fulltext': 1 if to_text_lower(x.get('hasFullText')) == 'y' else 0, 'abstract': to_text(x.get('abstractText'))}

def openalex_min_fields(x: Dict[str, Any]) -> Dict[str, Any]:
    doi = normalize_doi(x.get('doi'))
    oa = x.get('open_access') or {}
    primary = x.get('primary_location') or {}
    abs_text = openalex_abstract(x.get('abstract_inverted_index'))
    journal = ''
    if isinstance(primary, dict):
        src = primary.get('source')
        if isinstance(src, dict):
            journal = to_text(src.get('display_name'))
    return {'source': 'openalex', 'pmid': '', 'pmcid': '', 'doi': doi, 'title': to_text(x.get('display_name')), 'journal': journal, 'year': to_text(x.get('publication_year')), 'author': '', 'is_oa': 1 if bool(oa.get('is_oa')) else 0, 'has_fulltext': 1 if bool(oa.get('oa_url')) else 0, 'oa_url': to_text(oa.get('oa_url')), 'landing_page': to_text(primary.get('landing_page_url')) if isinstance(primary, dict) else '', 'abstract': to_text(abs_text)}

def paper_key(rec: Dict[str, Any]) -> str:
    pmcid = to_text(rec.get('pmcid'))
    pmid = to_text(rec.get('pmid'))
    doi = to_text(rec.get('doi'))
    title = to_text_lower(rec.get('title'))
    if pmcid:
        return f'PMCID:{pmcid}'
    if pmid:
        return f'PMID:{pmid}'
    if doi:
        return f'DOI:{doi}'
    return 'TITLE:' + sha1(title)[:16]

def build_entity_queries(entity_terms: List[str], max_terms: int, add_docking_hint: bool) -> List[Tuple[str, str]]:
    g = build_or_group(entity_terms, max_terms=max_terms)
    if not g:
        return []
    q0 = g
    q1 = f'{g} AND {DOCKING_HINT}' if add_docking_hint else g
    return [('base', q0), ('dockhint', q1)]

def read_csv_robust(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ['pmid', 'pmcid', 'doi', 'title', 'year', 'journal', 'author', 'uniprot_id', 'ligand_smiles_canon', 'inchikey']:
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['protein', 'ligand', 'pair', 'all'], default='all', help='--mode option.')
    ap.add_argument('--max_protein_terms', type=int, default=12)
    ap.add_argument('--max_ligand_terms', type=int, default=12)
    ap.add_argument('--max_hits_per_query', type=int, default=80)
    ap.add_argument('--page_size', type=int, default=25)
    ap.add_argument('--sleep', type=float, default=0.12, help='--sleep numeric setting.')
    ap.add_argument('--max_entities', type=int, default=0, help='--max_entities numeric setting.')
    ap.add_argument('--no_openalex', action='store_true', help='--no_openalex option.')
    args = ap.parse_args()
    prot_df = read_csv_robust(PROTEIN_INDEX)
    lig_df = read_csv_robust(LIGAND_INDEX)
    pair_df = read_csv_robust(PAIR_INDEX)
    seen_pairs = set()
    if OUT_SOURCES.exists():
        for line in OUT_SOURCES.read_text(encoding='utf-8', errors='ignore').splitlines():
            try:
                js = json.loads(line)
                key = (to_text(js.get('mode')), to_text(js.get('entity_id')))
                if key[0] and key[1]:
                    seen_pairs.add(key)
            except Exception:
                continue
    pool: Dict[str, Dict[str, Any]] = {}
    if OUT_POOL.exists() and OUT_POOL.stat().st_size > 100:
        old = read_csv_robust(OUT_POOL)
        for _, r in old.iterrows():
            d = r.to_dict()
            d = {k: '' if is_nan(v) else v for k, v in d.items()}
            pool[paper_key(d)] = d
    OUT_SOURCES.parent.mkdir(parents=True, exist_ok=True)

    def merge_into_pool(hit: Dict[str, Any]):
        k = paper_key(hit)
        if k not in pool:
            pool[k] = hit
            return
        base = pool[k]
        for kk, vv in hit.items():
            if is_nullish(base.get(kk)) and (not is_nullish(vv)):
                base[kk] = vv

    def process_one(mode: str, entity_id: str, entity_terms: List[str]):
        entity_id = to_text(entity_id)
        if not entity_id:
            return
        if (mode, entity_id) in seen_pairs:
            return
        entity_terms = [to_text(t) for t in entity_terms or [] if not is_nullish(t)]
        if not entity_terms:
            return
        queries = build_entity_queries(entity_terms, max_terms=args.max_protein_terms if mode == 'protein' else args.max_ligand_terms, add_docking_hint=True)
        all_hits: List[Dict[str, Any]] = []
        for qname, q in queries:
            ep_hits = eupmc_search(q, page_size=args.page_size, max_hits=args.max_hits_per_query, sleep=args.sleep)
            ep_min = [eupmc_min_fields(x) for x in ep_hits]
            all_hits.extend(ep_min)
            if not args.no_openalex:
                search_txt = ' '.join([t for t in entity_terms[:6] if t])
                if qname == 'dockhint':
                    search_txt = f'{search_txt} docking binding complex'
                oa_hits = openalex_search(search_txt, per_page=25, max_hits=min(50, args.max_hits_per_query), sleep=args.sleep)
                oa_min = [openalex_min_fields(x) for x in oa_hits]
                all_hits.extend(oa_min)
        rec = {'mode': mode, 'entity_id': entity_id, 'n_terms': len(entity_terms), 'terms_preview': entity_terms[:10], 'hits': [{'key': paper_key(h), 'doi': to_text(h.get('doi')), 'pmid': to_text(h.get('pmid')), 'pmcid': to_text(h.get('pmcid')), 'year': to_text(h.get('year')), 'title': to_text(h.get('title'))[:200], 'source': to_text(h.get('source')), 'is_oa': int(h.get('is_oa', 0) or 0), 'has_fulltext': int(h.get('has_fulltext', 0) or 0)} for h in all_hits]}
        with OUT_SOURCES.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        for h in all_hits:
            merge_into_pool(h)
        seen_pairs.add((mode, entity_id))
    if args.mode in ['protein', 'all']:
        it = list(prot_df.itertuples(index=False))
        if args.max_entities and args.max_entities > 0:
            it = it[:args.max_entities]
        for r in tqdm(it, desc='fetch papers by protein'):
            row = pd.Series(r._asdict())
            uni = to_text(row.get('uniprot_id', '') or row.get('Entry', ''))
            if not uni:
                continue
            terms = safe_json_load(row.get('protein_terms_json', '')) or safe_json_load(row.get('terms_json', ''))
            if not terms:
                terms = [uni]
                for k in ['protein_name', 'Protein names', 'Gene Names', 'gene_names']:
                    v = to_text(row.get(k, ''))
                    if v:
                        terms.extend([x for x in re.split('[;,\\s]+', v) if x])
            process_one('protein', uni, terms)
    if args.mode in ['ligand', 'all']:
        it = list(lig_df.itertuples(index=False))
        if args.max_entities and args.max_entities > 0:
            it = it[:args.max_entities]
        for r in tqdm(it, desc='fetch papers by ligand'):
            row = pd.Series(r._asdict())
            inchikey = to_text(row.get('inchikey'))
            smi = to_text(row.get('ligand_smiles_canon'))
            lid = inchikey or smi
            if not lid:
                continue
            terms = safe_json_load(row.get('ligand_terms_json', '')) or safe_json_load(row.get('synonyms_json', ''))
            if not terms:
                terms = []
                name = to_text(row.get('ligand_name'))
                if name:
                    terms.append(name)
            name = to_text(row.get('ligand_name'))
            if name and name not in terms:
                terms.insert(0, name)
            process_one('ligand', lid, terms)
    if args.mode in ['pair', 'all']:
        it = list(pair_df.itertuples(index=False))
        if args.max_entities and args.max_entities > 0:
            it = it[:args.max_entities]
        for r in tqdm(it, desc='fetch papers by pair'):
            row = pd.Series(r._asdict())
            uni = to_text(row.get('uniprot_id'))
            inchikey = to_text(row.get('inchikey'))
            smi = to_text(row.get('ligand_smiles_canon'))
            lid = inchikey or smi
            if not uni or not lid:
                continue
            pid = f'{uni}__{lid}'
            if ('pair', pid) in seen_pairs:
                continue
            p_terms = safe_json_load(row.get('protein_terms_json', ''))
            l_terms = safe_json_load(row.get('ligand_terms_json', ''))
            if not p_terms:
                p_terms = [uni]
            if not l_terms:
                ln = to_text(row.get('ligand_name'))
                l_terms = [ln] if ln else []
            pq = build_or_group(p_terms, max_terms=min(args.max_protein_terms, 10))
            lq = build_or_group(l_terms, max_terms=min(args.max_ligand_terms, 10))
            if not pq or not lq:
                continue
            queries = [('pair_base', f'{pq} AND {lq}'), ('pair_dockhint', f'{pq} AND {lq} AND {DOCKING_HINT}')]
            all_hits: List[Dict[str, Any]] = []
            for qname, q in queries:
                ep_hits = eupmc_search(q, page_size=args.page_size, max_hits=args.max_hits_per_query, sleep=args.sleep)
                all_hits.extend([eupmc_min_fields(x) for x in ep_hits])
                if not args.no_openalex:
                    search_txt = ' '.join([t for t in p_terms[:3] + l_terms[:3] if to_text(t)])
                    if qname.endswith('dockhint'):
                        search_txt += ' docking binding complex'
                    oa_hits = openalex_search(search_txt, per_page=25, max_hits=min(50, args.max_hits_per_query), sleep=args.sleep)
                    all_hits.extend([openalex_min_fields(x) for x in oa_hits])
            rec = {'mode': 'pair', 'entity_id': pid, 'protein': uni, 'ligand': lid, 'hits': [{'key': paper_key(h), 'doi': to_text(h.get('doi')), 'pmid': to_text(h.get('pmid')), 'pmcid': to_text(h.get('pmcid')), 'year': to_text(h.get('year')), 'title': to_text(h.get('title'))[:200], 'source': to_text(h.get('source')), 'is_oa': int(h.get('is_oa', 0) or 0), 'has_fulltext': int(h.get('has_fulltext', 0) or 0)} for h in all_hits]}
            with OUT_SOURCES.open('a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            for h in all_hits:
                merge_into_pool(h)
            seen_pairs.add(('pair', pid))
    out_df = pd.DataFrame(list(pool.values()))
    out_df['paper_key'] = out_df.apply(lambda r: paper_key(r.to_dict()), axis=1)
    if 'year' in out_df.columns:
        out_df['year'] = out_df['year'].astype(str).str.replace('.0', '', regex=False)
    OUT_POOL.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_POOL, index=False)
    print('Saved:', OUT_POOL, 'n=', len(out_df))
    print('Saved:', OUT_SOURCES, 'lines=', sum((1 for _ in OUT_SOURCES.open('r', encoding='utf-8'))))
if __name__ == '__main__':
    main()
