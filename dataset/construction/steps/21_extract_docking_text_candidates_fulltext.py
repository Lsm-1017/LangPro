import re
import os
import json
import gzip
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from functools import lru_cache
import pandas as pd
from tqdm import tqdm

def norm_str(x) -> str:
    if x is None:
        return ''
    if isinstance(x, float) and pd.isna(x):
        return ''
    s = str(x).strip()
    if s.lower() == 'nan':
        return ''
    return s

def safe_read_gz_text(path: str) -> str:
    if not path:
        return ''
    p = Path(path)
    if not p.exists() or p.stat().st_size < 50:
        return ''
    try:
        with gzip.open(p, 'rb') as f:
            b = f.read()
        return b.decode('utf-8', errors='ignore')
    except Exception:
        return ''

@lru_cache(maxsize=256)
def cached_gz_text(path: str) -> str:
    return safe_read_gz_text(path)

def normalize_spaces(s: str) -> str:
    s = s.replace('\xa0', ' ')
    s = re.sub('\\s+', ' ', s).strip()
    return s
_SENT_SPLIT = re.compile('(?<=[\\.\\?\\!\u3002\uff01\uff1f])\\s+')

def split_sentences(text: str, max_sent: int=2000):
    text = normalize_spaces(text)
    if not text:
        return []
    if len(text) > 2000000:
        text = text[:2000000]
    parts = _SENT_SPLIT.split(text)
    sents = []
    for x in parts:
        x = x.strip()
        if len(x) < 20:
            continue
        if x.lower().startswith('references') or x.lower().startswith('bibliography'):
            continue
        sents.append(x)
        if len(sents) >= max_sent:
            break
    return sents

def parse_term_field(val):
    s = norm_str(val)
    if not s:
        return []
    if s.startswith('[') and s.endswith(']') or (s.startswith('{') and s.endswith('}')):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                out = []
                for t in obj:
                    t = norm_str(t)
                    if t:
                        out.append(t)
                return out
        except Exception:
            pass
    if ';' in s:
        return [t.strip() for t in s.split(';') if t.strip()]
    if '|' in s:
        return [t.strip() for t in s.split('|') if t.strip()]
    if ',' in s:
        arr = [t.strip() for t in s.split(',') if t.strip()]
        if len(arr) <= 1:
            return [s]
        return arr
    return [s]

def build_terms_from_protein_row(row: dict):
    terms = []
    uni = norm_str(row.get('uniprot_id') or row.get('Entry') or row.get('entry') or row.get('accession'))
    if uni:
        terms.append(uni)
    for k in ['Gene Names', 'gene_names', 'Gene Names (primary)', 'gene', 'genes']:
        if k in row and norm_str(row.get(k)):
            for t in re.split('[\\s;/,]+', norm_str(row.get(k))):
                t = t.strip()
                if t and len(t) >= 2:
                    terms.append(t)
    for k in ['Protein names', 'protein_name', 'protein_names', 'name', 'Entry Name', 'entry_name']:
        if k in row and norm_str(row.get(k)):
            s = norm_str(row.get(k))
            terms.append(s)
            for m in re.findall('\\(([^)]+)\\)', s):
                for t in re.split('[;/,]+', m):
                    t = t.strip()
                    if t and len(t) >= 3:
                        terms.append(t)
    out = []
    seen = set()
    for t in terms:
        t = t.strip()
        if not t:
            continue
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return out

def build_terms_from_ligand_row(row: dict):
    terms = []
    inchikey = norm_str(row.get('inchikey') or row.get('InChIKey') or row.get('inchikey14'))
    if inchikey:
        terms.append(inchikey)
    for k in ['ligand_name', 'name', 'preferred_name', 'canonical_name']:
        if k in row and norm_str(row.get(k)):
            terms.append(norm_str(row.get(k)))
    syn_cols = [c for c in row.keys() if 'syn' in c.lower() or 'alias' in c.lower() or 'names' in c.lower()]
    for c in syn_cols:
        arr = parse_term_field(row.get(c))
        for t in arr:
            if t and len(t) >= 2:
                terms.append(t)
    out = []
    seen = set()
    for t in terms:
        t = t.strip()
        if not t:
            continue
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return out

def pick_wordish_terms(terms, max_terms=40):
    clean = []
    for t in terms:
        tl = t.lower().strip()
        if len(tl) < 2:
            continue
        if tl in {'protein', 'enzyme', 'binding', 'ligand', 'substrate', 'inhibitor'}:
            continue
        clean.append(t)
    clean = sorted(clean, key=lambda x: len(x), reverse=True)
    wordish = []
    nonword = []
    for t in clean:
        if re.fullmatch('[A-Za-z0-9_:-]+', t):
            wordish.append(t)
        else:
            nonword.append(t)
    wordish = wordish[:max_terms]
    nonword = nonword[:max_terms]
    return (wordish, nonword)

def compile_entity_matcher(terms):
    wordish, nonword = pick_wordish_terms(terms)
    if wordish:
        pat = '(?i)\\b(?:%s)\\b' % '|'.join([re.escape(t) for t in wordish])
        reg = re.compile(pat)
    else:
        reg = None
    nonword_l = [t.lower() for t in nonword if t]

    def matcher(text_lower: str):
        hits = []
        if reg is not None:
            for m in reg.finditer(text_lower):
                hits.append(m.group(0))
                if len(hits) >= 6:
                    break
        if nonword_l:
            for t in nonword_l:
                if t and t in text_lower:
                    hits.append(t)
                    if len(hits) >= 6:
                        break
        out = []
        seen = set()
        for h in hits:
            hl = h.lower()
            if hl not in seen:
                seen.add(hl)
                out.append(h)
        return out
    return matcher
KW = ['\\bdock(?:ed|ing)?\\b', 'molecular docking', 'autodock', 'vina', 'glide', 'gold\\b', 'flexx\\b', '\\bbind(?:s|ing)?\\b', '\\bbinding\\b', 'affinity', '\\bKd\\b', '\\bKi\\b', '\\bIC50\\b', '\\bEC50\\b', 'active site', 'binding site', 'binding pocket', 'pocket', 'interaction(?:s)?', 'hydrogen bond', 'hydrophobic', 'salt bridge', 'complex(?:es)?', 'co-crystal', 'crystal structure', 'cofactor', 'substrate', 'inhibitor']
KW_RE = re.compile('(?i)(' + '|'.join(KW) + ')')

def keyword_hits(text: str, max_hits=6):
    hs = []
    for m in KW_RE.finditer(text):
        hs.append(m.group(0))
        if len(hs) >= max_hits:
            break
    out = []
    seen = set()
    for x in hs:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl)
            out.append(x)
    return out

def looks_like_candidate_sentence(sent: str, prot_matcher, lig_matcher):
    s = sent.strip()
    if len(s) < 25:
        return None
    if s.count('|') >= 6:
        return None
    if re.search('^\\(?\\d+\\)?\\s', s) and len(s) < 40:
        return None
    kw = keyword_hits(s)
    if not kw:
        return None
    sl = s.lower()
    ph = prot_matcher(sl)
    if not ph:
        return None
    lh = lig_matcher(sl)
    if not lh:
        return None
    score = 0
    score += 3 * len(kw)
    score += 2 * min(len(ph), 3)
    score += 2 * min(len(lh), 3)
    if 60 <= len(s) <= 280:
        score += 2
    return {'sentence': s, 'kw_hits': '|'.join(kw), 'prot_hits': '|'.join(ph[:6]), 'lig_hits': '|'.join(lh[:6]), 'score': score}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', default='../data_intermediate/pair_paper_candidates.csv')
    ap.add_argument('--paper_pool', default='../data_intermediate/paper_pool.csv')
    ap.add_argument('--paper_fulltext_index', default='../data_intermediate/paper_fulltext_index.csv')
    ap.add_argument('--protein_index', default='../data_intermediate/protein_search_index.csv')
    ap.add_argument('--ligand_index', default='../data_intermediate/ligand_search_index.csv')
    ap.add_argument('--max_rank', type=int, default=20, help='--max_rank numeric setting.')
    ap.add_argument('--max_sent_per_paper', type=int, default=20, help='--max_sent_per_paper input path or value.')
    ap.add_argument('--max_sent_total_per_pair', type=int, default=80, help='--max_sent_total_per_pair input path or value.')
    ap.add_argument('--min_text_len', type=int, default=200, help='--min_text_len input path or value.')
    ap.add_argument('--flush_every', type=int, default=50, help='--flush_every option.')
    ap.add_argument('--out_csv', default='../data_intermediate/pair_docking_sentence_candidates.csv')
    ap.add_argument('--stats_json', default='../data_intermediate/pair_docking_sentence_candidates_stats.json')
    ap.add_argument('--resume', action='store_true', help='--resume option.')
    ap.add_argument('--reprocess_no_hits', action='store_true', help='--reprocess_no_hits option.')
    args = ap.parse_args()
    CAND = Path(args.candidates)
    POOL = Path(args.paper_pool)
    FTIDX = Path(args.paper_fulltext_index)
    PROT = Path(args.protein_index)
    LIG = Path(args.ligand_index)
    out_csv = Path(args.out_csv)
    stats_json = Path(args.stats_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(CAND)
    need_cols = {'pair_key', 'uniprot_id', 'inchikey', 'paper_key', 'rank'}
    if not need_cols.issubset(set(cand.columns)):
        raise ValueError(f'pair_paper_candidates.csv{need_cols - set(cand.columns)}')
    cand['pair_key'] = cand['pair_key'].astype(str)
    cand['paper_key'] = cand['paper_key'].astype(str)
    cand['uniprot_id'] = cand['uniprot_id'].astype(str)
    cand['inchikey'] = cand['inchikey'].astype(str)
    cand = cand[cand['rank'] <= args.max_rank].copy()
    cand = cand.sort_values(['pair_key', 'rank'])
    pairs = cand['pair_key'].unique().tolist()
    print('pairs_total:', len(pairs))
    print('cand_rows (rank<=max_rank):', len(cand))
    done_pairs = set()
    pair_has_any_hit = set()
    if out_csv.exists() and (args.resume or args.reprocess_no_hits):
        old = pd.read_csv(out_csv)
        if 'pair_key' in old.columns and len(old) > 0:
            pair_has_any_hit = set(old['pair_key'].astype(str).unique().tolist())
            if args.resume:
                done_pairs = pair_has_any_hit
            elif args.reprocess_no_hits:
                done_pairs = set()
        print('existing_output_pairs_with_hits:', len(pair_has_any_hit))
    pool = pd.read_csv(POOL)
    if 'paper_key' not in pool.columns:
        raise ValueError('paper_pool.csv paper_key')
    pool['paper_key'] = pool['paper_key'].astype(str)
    abs_col = 'abstract' if 'abstract' in pool.columns else None
    title_col = 'title' if 'title' in pool.columns else None
    pmcid_col = 'pmcid' if 'pmcid' in pool.columns else None
    doi_col = 'doi' if 'doi' in pool.columns else None
    year_col = 'year' if 'year' in pool.columns else None
    paper_keys = set(cand['paper_key'].unique().tolist())
    pool = pool[pool['paper_key'].isin(paper_keys)].copy()
    abstract_map = {}
    title_map = {}
    meta_map = {}
    for _, r in pool.iterrows():
        pk = norm_str(r.get('paper_key'))
        if not pk:
            continue
        abstract_map[pk] = norm_str(r.get(abs_col)) if abs_col else ''
        title_map[pk] = norm_str(r.get(title_col)) if title_col else ''
        meta_map[pk] = {'pmcid': norm_str(r.get(pmcid_col)) if pmcid_col else '', 'doi': norm_str(r.get(doi_col)) if doi_col else '', 'year': norm_str(r.get(year_col)) if year_col else '', 'source': norm_str(r.get('source')), 'journal': norm_str(r.get('journal'))}
    ft_path_map = {}
    ft_status_map = {}
    if FTIDX.exists():
        fti = pd.read_csv(FTIDX)
        if 'paper_key' not in fti.columns:
            raise ValueError('paper_fulltext_index.csv paper_key')
        fti['paper_key'] = fti['paper_key'].astype(str)
        path_col = None
        for c in ['fulltext_path', 'fulltext', 'txt_gz_path']:
            if c in fti.columns:
                path_col = c
                break
        if path_col is None:
            raise ValueError(f'paper_fulltext_index.csv ={fti.columns.tolist()}')
        status_col = None
        for c in ['fulltext_status', 'status']:
            if c in fti.columns:
                status_col = c
                break
        for _, r in fti.iterrows():
            pk = norm_str(r.get('paper_key'))
            if not pk:
                continue
            ft_path_map[pk] = norm_str(r.get(path_col))
            if status_col:
                ft_status_map[pk] = norm_str(r.get(status_col))
            else:
                ft_status_map[pk] = ''
    prot_df = pd.read_csv(PROT)
    lig_df = pd.read_csv(LIG)
    if 'uniprot_id' not in prot_df.columns:
        if 'Entry' in prot_df.columns:
            prot_df = prot_df.rename(columns={'Entry': 'uniprot_id'})
        else:
            raise ValueError(f'protein_search_index.csv uniprot_id/Entry{prot_df.columns.tolist()}')
    if 'inchikey' not in lig_df.columns:
        if 'InChIKey' in lig_df.columns:
            lig_df = lig_df.rename(columns={'InChIKey': 'inchikey'})
        else:
            raise ValueError(f'ligand_search_index.csv inchikey/InChIKey{lig_df.columns.tolist()}')
    prot_terms = {}
    for _, r in prot_df.iterrows():
        row = r.to_dict()
        uni = norm_str(row.get('uniprot_id'))
        if not uni:
            continue
        terms = build_terms_from_protein_row(row)
        prot_terms[uni] = terms
    lig_terms = {}
    for _, r in lig_df.iterrows():
        row = r.to_dict()
        ik = norm_str(row.get('inchikey'))
        if not ik:
            continue
        terms = build_terms_from_ligand_row(row)
        lig_terms[ik] = terms
    prot_matcher = {}
    lig_matcher = {}
    for uni, terms in prot_terms.items():
        prot_matcher[uni] = compile_entity_matcher(terms)
    for ik, terms in lig_terms.items():
        lig_matcher[ik] = compile_entity_matcher(terms)
    pair2rows = defaultdict(list)
    for _, r in cand.iterrows():
        pair2rows[r['pair_key']].append({'pair_key': r['pair_key'], 'uniprot_id': r['uniprot_id'], 'inchikey': r['inchikey'], 'paper_key': r['paper_key'], 'rank': int(r['rank']), 'source_tag': norm_str(r.get('source_tag'))})
    out_rows = []
    pairs_with_hits = 0
    total_sent = 0
    used_fulltext_papers = 0
    used_abstract_papers = 0
    paper_scanned = 0
    miss_reason = Counter()

    def flush():
        nonlocal out_rows
        if not out_rows:
            return
        new_df = pd.DataFrame(out_rows)
        if out_csv.exists():
            old = pd.read_csv(out_csv)
            out = pd.concat([old, new_df], ignore_index=True)
            if {'pair_key', 'paper_key', 'sentence'}.issubset(out.columns):
                out = out.drop_duplicates(subset=['pair_key', 'paper_key', 'sentence'], keep='first')
        else:
            out = new_df
        out.to_csv(out_csv, index=False)
        out_rows = []
    pbar = tqdm(pairs, desc='extract docking sentences')
    t0 = time.time()
    for idx, pair_key in enumerate(pbar, start=1):
        if args.resume and pair_key in done_pairs:
            continue
        if args.reprocess_no_hits and pair_key in pair_has_any_hit:
            continue
        rows = pair2rows.get(pair_key, [])
        if not rows:
            continue
        uni = rows[0]['uniprot_id']
        ik = rows[0]['inchikey']
        pm = prot_matcher.get(uni, None)
        lm = lig_matcher.get(ik, None)
        if pm is None or lm is None:
            miss_reason['no_entity_matcher'] += 1
            continue
        pair_hits = []
        for pr in rows:
            pk = pr['paper_key']
            rank = pr['rank']
            paper_scanned += 1
            ft_path = norm_str(ft_path_map.get(pk, ''))
            text = ''
            text_source = ''
            if ft_path:
                text = cached_gz_text(ft_path)
                if text and len(text) >= args.min_text_len:
                    text_source = 'fulltext'
                    used_fulltext_papers += 1
                else:
                    text = ''
            if not text:
                ab = norm_str(abstract_map.get(pk, ''))
                if ab and len(ab) >= 80:
                    text = ab
                    text_source = 'abstract'
                    used_abstract_papers += 1
            if not text:
                miss_reason['no_text'] += 1
                continue
            sents = split_sentences(text)
            if not sents:
                miss_reason['no_sentences'] += 1
                continue
            paper_cands = []
            for s in sents:
                hit = looks_like_candidate_sentence(s, pm, lm)
                if hit is None:
                    continue
                hit['pair_key'] = pair_key
                hit['uniprot_id'] = uni
                hit['inchikey'] = ik
                hit['paper_key'] = pk
                hit['rank'] = rank
                hit['source_tag'] = pr.get('source_tag', '')
                hit['text_source'] = text_source
                meta = meta_map.get(pk, {})
                hit['pmcid'] = meta.get('pmcid', '')
                hit['doi'] = meta.get('doi', '')
                hit['year'] = meta.get('year', '')
                hit['title'] = title_map.get(pk, '')
                paper_cands.append(hit)
                if len(paper_cands) >= args.max_sent_per_paper * 3:
                    break
            if not paper_cands:
                continue
            paper_cands.sort(key=lambda x: x['score'], reverse=True)
            paper_cands = paper_cands[:args.max_sent_per_paper]
            for j, d in enumerate(paper_cands, start=1):
                d['sent_rank_in_paper'] = j
            pair_hits.extend(paper_cands)
            if len(pair_hits) >= args.max_sent_total_per_pair:
                break
        if pair_hits:
            pairs_with_hits += 1
            pair_hits.sort(key=lambda x: (x['rank'], -x['score']))
            pair_hits = pair_hits[:args.max_sent_total_per_pair]
            for k, d in enumerate(pair_hits, start=1):
                d['sent_rank_in_pair'] = k
            out_rows.extend(pair_hits)
            total_sent += len(pair_hits)
        else:
            miss_reason['pair_no_hit'] += 1
        if idx % args.flush_every == 0:
            flush()
        if idx % 20 == 0:
            elapsed = time.time() - t0
            pbar.set_postfix({'pairs_with_hits': pairs_with_hits, 'hit_ratio': round(pairs_with_hits / max(1, idx), 3), 'sent': total_sent, 'ft_papers': used_fulltext_papers, 'ab_papers': used_abstract_papers, 'elapsed_min': round(elapsed / 60, 1)})
    flush()
    pairs_total = len(pairs)
    hit_ratio = pairs_with_hits / max(1, pairs_total)
    stats = {'pairs_total': pairs_total, 'pairs_with_hits': pairs_with_hits, 'pair_hit_ratio': hit_ratio, 'total_sentences': total_sent, 'avg_sentences_per_hit_pair': total_sent / max(1, pairs_with_hits), 'papers_scanned': paper_scanned, 'papers_used_fulltext': used_fulltext_papers, 'papers_used_abstract': used_abstract_papers, 'paper_text_source_ratio_fulltext': used_fulltext_papers / max(1, used_fulltext_papers + used_abstract_papers), 'paper_text_source_ratio_abstract': used_abstract_papers / max(1, used_fulltext_papers + used_abstract_papers), 'miss_reason_top10': miss_reason.most_common(10), 'args': vars(args)}
    stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n=== Extraction Summary ===')
    print(f'pairs_total: {pairs_total}')
    print(f'pairs_with_hits: {pairs_with_hits}')
    print(f'pair_hit_ratio: {hit_ratio:.4f}')
    print(f'total_sentences: {total_sent}')
    print(f"avg_sentences_per_hit_pair: {stats['avg_sentences_per_hit_pair']:.2f}")
    print(f'papers_scanned: {paper_scanned}')
    print(f'papers_used_fulltext: {used_fulltext_papers}')
    print(f'papers_used_abstract: {used_abstract_papers}')
    print(f"fulltext_ratio_among_used_papers: {stats['paper_text_source_ratio_fulltext']:.4f}")
    print('miss_reason_top10:', stats['miss_reason_top10'])
    print('Saved:', str(out_csv))
    print('Saved:', str(stats_json))
if __name__ == '__main__':
    main()
