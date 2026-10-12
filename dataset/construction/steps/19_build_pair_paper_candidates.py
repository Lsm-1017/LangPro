import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
PAIR_INDEX = Path('../data_intermediate/pair_search_index.csv')
PAPER_POOL = Path('../data_intermediate/paper_pool.csv')
SOURCES_JSONL = Path('../data_intermediate/paper_pool_sources.jsonl')
OUT_PAIR_CAND = Path('../data_intermediate/pair_paper_candidates.csv')
OUT_PAPER_UNI = Path('../data_intermediate/paper_candidates_unique.csv')

def norm_str(x):
    if x is None:
        return ''
    try:
        if pd.isna(x):
            return ''
    except Exception:
        pass
    return str(x).strip()

def norm_uniprot(x):
    return norm_str(x).upper()

def norm_inchikey(x):
    s = norm_str(x).upper()
    if s.startswith('INCHIKEY='):
        s = s[len('INCHIKEY='):]
    return s

def make_pair_key(uniprot_id, inchikey):
    u = norm_uniprot(uniprot_id)
    k = norm_inchikey(inchikey)
    if not u or not k:
        return ''
    return f'{u}__{k}'

def safe_get(rec, keys, default=None):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default

def build_id2paperkey_maps(pool_df: pd.DataFrame):
    doi2k = {}
    pmid2k = {}
    pmcid2k = {}
    oa2k = {}
    cols = set(pool_df.columns)
    if 'paper_key' not in cols:
        raise ValueError('paper_pool.csv paper_key sources')
    for _, r in pool_df.iterrows():
        pk = norm_str(r.get('paper_key'))
        if not pk:
            continue
        doi = norm_str(r.get('doi')).lower()
        if doi:
            doi2k[doi] = pk
        pmid = norm_str(r.get('pmid'))
        if pmid:
            pmid2k[pmid] = pk
        pmcid = norm_str(r.get('pmcid')).upper()
        if pmcid:
            pmcid2k[pmcid] = pk
        oa = norm_str(r.get('openalex_id'))
        if oa:
            oa2k[oa] = pk
    return (doi2k, pmid2k, pmcid2k, oa2k)

def paper_obj_to_key(obj, doi2k, pmid2k, pmcid2k, oa2k):
    if obj is None:
        return ''
    if isinstance(obj, str):
        s = obj.strip()
        return s
    if isinstance(obj, dict):
        pk = norm_str(obj.get('paper_key'))
        if pk:
            return pk
        doi = norm_str(obj.get('doi')).lower()
        if doi and doi in doi2k:
            return doi2k[doi]
        pmcid = norm_str(obj.get('pmcid')).upper()
        if pmcid and pmcid in pmcid2k:
            return pmcid2k[pmcid]
        pmid = norm_str(obj.get('pmid'))
        if pmid and pmid in pmid2k:
            return pmid2k[pmid]
        oa = norm_str(obj.get('openalex_id'))
        if oa and oa in oa2k:
            return oa2k[oa]
    return ''

def extract_entity_info(src_rec: dict):
    etype = safe_get(src_rec, ['entity_type', 'mode', 'query_mode', 'type'], '')
    etype = norm_str(etype).lower()
    if etype in ['by_protein', 'protein']:
        etype = 'protein'
    elif etype in ['by_ligand', 'ligand']:
        etype = 'ligand'
    elif etype in ['by_pair', 'pair']:
        etype = 'pair'
    uniprot_id = safe_get(src_rec, ['uniprot_id', 'protein_id', 'protein', 'entry'], '')
    inchikey = safe_get(src_rec, ['inchikey', 'ligand_inchikey', 'ligand_key'], '')
    pair_key = safe_get(src_rec, ['pair_key'], '')
    entity_id = safe_get(src_rec, ['entity_id', 'entity', 'id'], '')
    entity_id = norm_str(entity_id)
    if not uniprot_id and etype == 'protein' and entity_id:
        uniprot_id = entity_id.split(':')[-1]
    if not inchikey and etype == 'ligand' and entity_id:
        inchikey = entity_id.split(':')[-1]
    if not pair_key and etype == 'pair' and entity_id:
        pair_key = entity_id.split(':')[-1]
    uniprot_id = norm_uniprot(uniprot_id)
    inchikey = norm_inchikey(inchikey)
    pair_key = norm_str(pair_key)
    if etype == 'pair':
        if not pair_key and uniprot_id and inchikey:
            pair_key = make_pair_key(uniprot_id, inchikey)
    return (etype, uniprot_id, inchikey, pair_key)

def extract_paper_list(src_rec: dict):
    for k in ['paper_keys', 'paper_key_list', 'papers', 'results', 'hits', 'result_set', 'top_papers']:
        if k in src_rec and src_rec[k] is not None:
            return src_rec[k]
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top_pair', type=int, default=50)
    ap.add_argument('--top_intersection', type=int, default=50)
    ap.add_argument('--top_protein', type=int, default=50, help='--top_protein input path or value.')
    ap.add_argument('--top_ligand', type=int, default=50, help='--top_ligand input path or value.')
    ap.add_argument('--max_per_pair', type=int, default=120)
    args = ap.parse_args()
    pair_df = pd.read_csv(PAIR_INDEX)
    cols = set(pair_df.columns)
    need_cols = []
    if 'uniprot_id' not in cols:
        need_cols.append('uniprot_id')
    if 'inchikey' not in cols:
        need_cols.append('inchikey')
    if need_cols:
        raise ValueError(f'pair_search_index.csv :{need_cols}17_build_search_indices.py')
    pair_df['uniprot_id'] = pair_df['uniprot_id'].apply(norm_uniprot)
    pair_df['inchikey'] = pair_df['inchikey'].apply(norm_inchikey)
    if 'pair_key' not in cols:
        pair_df['pair_key'] = pair_df.apply(lambda r: make_pair_key(r['uniprot_id'], r['inchikey']), axis=1)
    else:
        pair_df['pair_key'] = pair_df['pair_key'].apply(norm_str)
    print('pairs_total:', len(pair_df))
    print('pair_key empty:', int((pair_df['pair_key'] == '').sum()))
    pool = pd.read_csv(PAPER_POOL)
    doi2k, pmid2k, pmcid2k, oa2k = build_id2paperkey_maps(pool)
    prot2papers = {}
    lig2papers = {}
    pair2papers = {}
    type_count = {'protein': 0, 'ligand': 0, 'pair': 0, 'other': 0}
    hit_count = {'protein': 0, 'ligand': 0, 'pair': 0}
    with SOURCES_JSONL.open('r', encoding='utf-8') as f:
        for line in tqdm(f, desc='parse sources'):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            etype, uni, ik, pk = extract_entity_info(rec)
            if etype not in ['protein', 'ligand', 'pair']:
                type_count['other'] += 1
                continue
            type_count[etype] += 1
            papers_raw = extract_paper_list(rec)
            keys = []
            if isinstance(papers_raw, list):
                for obj in papers_raw:
                    k = paper_obj_to_key(obj, doi2k, pmid2k, pmcid2k, oa2k)
                    if k:
                        keys.append(k)
            else:
                keys = []
            seen = set()
            dedup = []
            for k in keys:
                if k not in seen:
                    dedup.append(k)
                    seen.add(k)
            if not dedup:
                continue
            if etype == 'protein' and uni:
                prot2papers[uni] = dedup
                hit_count['protein'] += 1
            elif etype == 'ligand' and ik:
                lig2papers[ik] = dedup
                hit_count['ligand'] += 1
            elif etype == 'pair' and pk:
                pair2papers[pk] = dedup
                hit_count['pair'] += 1
    print('source record counts:', type_count)
    print('entities with non-empty papers:', hit_count)
    print('prot2papers:', len(prot2papers), 'lig2papers:', len(lig2papers), 'pair2papers:', len(pair2papers))
    out_rows = []
    for _, r in tqdm(pair_df.iterrows(), total=len(pair_df), desc='build candidates'):
        uni = r['uniprot_id']
        ik = r['inchikey']
        pk = r['pair_key']
        pair_list = pair2papers.get(pk, [])
        prot_list = prot2papers.get(uni, [])
        lig_list = lig2papers.get(ik, [])
        inter = []
        if prot_list and lig_list:
            lig_set = set(lig_list)
            common = [p for p in prot_list if p in lig_set]
            if common:
                rank_p = {p: i for i, p in enumerate(prot_list)}
                rank_l = {p: i for i, p in enumerate(lig_list)}
                common.sort(key=lambda p: rank_p.get(p, 10 ** 9) + rank_l.get(p, 10 ** 9))
                inter = common
        cand = []
        src_tag = {}

        def add_list(lst, tag, topn):
            nonlocal cand
            for p in lst[:topn]:
                if p not in src_tag:
                    src_tag[p] = tag
                    cand.append(p)
        add_list(pair_list, 'pair', args.top_pair)
        add_list(inter, 'intersection', args.top_intersection)
        add_list(prot_list, 'protein', args.top_protein)
        add_list(lig_list, 'ligand', args.top_ligand)
        cand = cand[:args.max_per_pair]
        for rank, pkey in enumerate(cand, start=1):
            out_rows.append({'pair_key': pk, 'uniprot_id': uni, 'inchikey': ik, 'paper_key': pkey, 'rank': rank, 'source_tag': src_tag.get(pkey, 'unknown')})
    out_df = pd.DataFrame(out_rows)
    OUT_PAIR_CAND.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PAIR_CAND, index=False)
    if len(out_df) > 0:
        wanted = set(out_df['paper_key'].astype(str).tolist())
        pool_sub = pool[pool['paper_key'].astype(str).isin(wanted)].copy()
    else:
        pool_sub = pool.iloc[0:0].copy()
    pool_sub.to_csv(OUT_PAPER_UNI, index=False)
    pairs_with_any = out_df['pair_key'].nunique() if len(out_df) else 0
    print('Saved:', OUT_PAIR_CAND, 'rows=', len(out_df))
    print('Saved:', OUT_PAPER_UNI, 'unique_papers=', len(pool_sub))
    print('pairs_total:', len(pair_df), 'pairs_with_any_candidate:', pairs_with_any)
if __name__ == '__main__':
    main()
