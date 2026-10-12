import re
import json
import gzip
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm

def norm_str(x) -> str:
    if x is None:
        return ''
    try:
        if pd.isna(x):
            return ''
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() == 'nan':
        return ''
    return s

def is_nonempty(x) -> bool:
    return norm_str(x) != ''

def safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower().replace(' ', '_'): c for c in df.columns}
    for k in candidates:
        kk = k.lower().replace(' ', '_')
        if kk in cols:
            return cols[kk]
    return None

def split_alias_field(val: str) -> List[str]:
    s = norm_str(val)
    if not s:
        return []
    parts = re.split('[;\\|\\t]', s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        sub = [x.strip() for x in p.split(',') if x.strip()]
        out.extend(sub if sub else [p])
    return out

def clean_alias(a: str) -> str:
    a = norm_str(a)
    if not a:
        return ''
    a = re.sub('\\([^)]*\\)', ' ', a).strip()
    a = re.sub('\\s+', ' ', a).strip()
    return a

def alias_filter_basic(a: str, min_len: int, max_len: int) -> bool:
    if not a:
        return False
    if len(a) < min_len or len(a) > max_len:
        return False
    if re.fullmatch('[\\W_]+', a):
        return False
    if re.fullmatch('\\d+', a):
        return False
    bad = {'protein', 'human', 'mouse', 'cell', 'cells', 'binding', 'ligand'}
    if a.lower() in bad:
        return False
    return True

def build_protein_alias_map(protein_index_csv: str) -> Tuple[Dict[str, List[str]], str]:
    df = safe_read_csv(protein_index_csv)
    id_col = detect_col(df, ['uniprot_id', 'entry', 'uniprot', 'Entry'])
    if not id_col:
        raise ValueError(f'protein_index has no uniprot id col. cols={df.columns.tolist()}')
    cand_cols = []
    for k in ['entry_name', 'Entry_Name', 'Entry Name', 'protein_names', 'Protein_names', 'Protein names', 'protein_name', 'gene_names', 'Gene Names', 'gene_name', 'recommended_name', 'name']:
        c = detect_col(df, [k])
        if c and c not in cand_cols:
            cand_cols.append(c)
    mp: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        uid = norm_str(r.get(id_col))
        if not uid:
            continue
        aliases = set()
        aliases.add(uid)
        for c in cand_cols:
            v = clean_alias(str(r.get(c, '')))
            for item in split_alias_field(v):
                item = clean_alias(item)
                if alias_filter_basic(item, min_len=4, max_len=80):
                    aliases.add(item)
        extra = set()
        for a in list(aliases):
            for t in re.split('\\s+', a):
                t = clean_alias(t)
                if alias_filter_basic(t, min_len=4, max_len=30):
                    extra.add(t)
        aliases |= extra
        mp[uid] = sorted(list(aliases), key=lambda x: (-len(x), x.lower()))
    return (mp, id_col)

def build_ligand_alias_map(ligand_index_csv: str) -> Tuple[Dict[str, List[str]], str]:
    df = safe_read_csv(ligand_index_csv)
    key_col = detect_col(df, ['inchikey', 'InChIKey', 'inchi_key'])
    if not key_col:
        raise ValueError(f'ligand_index has no inchikey col. cols={df.columns.tolist()}')
    cand_cols = []
    for k in ['synonyms', 'syns', 'name', 'ligand_name', 'canonical_name', 'preferred_name']:
        c = detect_col(df, [k])
        if c and c not in cand_cols:
            cand_cols.append(c)
    mp: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        ik = norm_str(r.get(key_col))
        if not ik:
            continue
        aliases = set()
        for c in cand_cols:
            v = clean_alias(str(r.get(c, '')))
            for item in split_alias_field(v):
                item = clean_alias(item)
                if alias_filter_basic(item, min_len=3, max_len=80):
                    aliases.add(item)
        extra = set()
        for a in list(aliases):
            for t in re.split('\\s+', a):
                t = clean_alias(t)
                if alias_filter_basic(t, min_len=4, max_len=40):
                    extra.add(t)
        aliases |= extra
        mp[ik] = sorted(list(aliases), key=lambda x: (-len(x), x.lower()))
    return (mp, key_col)

def make_alt_pattern(aliases: List[str], max_aliases: int=40) -> Optional[re.Pattern]:
    if not aliases:
        return None
    uniq = []
    seen = set()
    for a in aliases:
        a = clean_alias(a)
        if not a:
            continue
        if a.lower() in seen:
            continue
        seen.add(a.lower())
        uniq.append(a)
    uniq = sorted(uniq, key=lambda x: (-len(x), x.lower()))[:max_aliases]
    escaped = [re.escape(x) for x in uniq]
    if not escaped:
        return None
    pat = '(?<![A-Za-z0-9])(?:%s)(?![A-Za-z0-9])' % '|'.join(escaped)
    return re.compile(pat, flags=re.I)
STRONG_BIND = re.compile('(?i)\\b(bind|binding|bound|affinity|kd|ki|ic50|dock|docking|inhibit|inhibitor|substrate)\\b')
WEAK_BIND = re.compile('(?i)\\b(interact|interaction|complex|active site|pocket|pose|screening|assay)\\b')

def split_sentences(text: str) -> List[str]:
    t = norm_str(text)
    if not t:
        return []
    sents = re.split('(?<=[\\.\\?\\!])\\s+|\\n+', t)
    sents = [re.sub('\\s+', ' ', s).strip() for s in sents]
    sents = [s for s in sents if len(s) >= 20]
    return sents

def load_fulltext_gz(path: str, max_chars: int=200000) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ''
    try:
        with gzip.open(p, 'rb') as f:
            data = f.read()
        txt = data.decode('utf-8', errors='ignore')
        txt = re.sub('\\s+', ' ', txt).strip()
        if max_chars and len(txt) > max_chars:
            txt = txt[:max_chars]
        return txt
    except Exception:
        return ''

def extract_best_spans(text: str, prot_pat: Optional[re.Pattern], lig_pat: Optional[re.Pattern], source_bonus: float, window: int, min_score: float, max_sent_per_paper: int, require_bind: bool, require_cooccur: bool) -> List[Tuple[str, float, int, int, int]]:
    sents = split_sentences(text)
    if not sents or not prot_pat or (not lig_pat):
        return []
    prot_hit = [bool(prot_pat.search(s)) for s in sents]
    lig_hit = [bool(lig_pat.search(s)) for s in sents]
    strong = [len(STRONG_BIND.findall(s)) for s in sents]
    weak = [len(WEAK_BIND.findall(s)) for s in sents]
    results = []
    for i, s in enumerate(sents):
        b = strong[i] + weak[i]
        if require_bind and b == 0:
            continue
        lo = max(0, i - window)
        hi = min(len(sents) - 1, i + window)
        span_idx = list(range(lo, hi + 1))
        prot_span = any((prot_hit[j] for j in span_idx))
        lig_span = any((lig_hit[j] for j in span_idx))
        if require_cooccur and (not (prot_span and lig_span)):
            continue
        both_same = int(prot_hit[i] and lig_hit[i])
        span_text = ' '.join((sents[j] for j in span_idx))
        score = 2.0 * strong[i] + 0.8 * weak[i] + 1.0 * int(prot_span) + 1.0 * int(lig_span) + 1.5 * both_same + source_bonus
        if score < min_score:
            continue
        results.append((span_text, score, strong[i], weak[i], both_same))
    seen = set()
    uniq = []
    for span_text, score, sc, wc, both_same in sorted(results, key=lambda x: -x[1]):
        key = re.sub('\\s+', ' ', span_text.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((span_text, score, sc, wc, both_same))
        if len(uniq) >= max_sent_per_paper:
            break
    return uniq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', default='../data_intermediate/triples_with_ligand_text_full_with_docking_plus_24b_tight_v3.csv')
    ap.add_argument('--main_docking_col', default='docking_text')
    ap.add_argument('--candidates', default='../data_intermediate/pair_paper_candidates.csv')
    ap.add_argument('--paper_pool', default='../data_intermediate/paper_pool.csv')
    ap.add_argument('--fulltext_index', default='../data_intermediate/paper_fulltext_index.csv')
    ap.add_argument('--protein_index', default='../data_intermediate/protein_search_index.csv')
    ap.add_argument('--ligand_index', default='../data_intermediate/ligand_search_index.csv')
    ap.add_argument('--source_tags', default='pair,intersection,protein,ligand', help='--source_tags option.')
    ap.add_argument('--max_rank', type=int, default=120)
    ap.add_argument('--window', type=int, default=2, help='--window numeric setting.')
    ap.add_argument('--min_score', type=float, default=1.5)
    ap.add_argument('--max_sent_per_paper', type=int, default=12)
    ap.add_argument('--max_sent_total_per_pair', type=int, default=120)
    ap.add_argument('--max_aliases_protein', type=int, default=30)
    ap.add_argument('--max_aliases_ligand', type=int, default=30)
    ap.add_argument('--require_bind', action='store_true', default=True)
    ap.add_argument('--no_require_bind', action='store_true', default=False)
    ap.add_argument('--require_cooccur', action='store_true', default=True)
    ap.add_argument('--no_require_cooccur', action='store_true', default=False)
    ap.add_argument('--sleep', type=float, default=0.0)
    ap.add_argument('--flush_every', type=int, default=50)
    ap.add_argument('--out_csv', default='../data_intermediate/pair_docking_sentence_candidates_25a_backfill.csv')
    ap.add_argument('--out_stats', default='../data_intermediate/pair_docking_sentence_candidates_25a_backfill_stats.json')
    ap.add_argument('--max_pairs', type=int, default=0, help='--max_pairs numeric setting.')
    args = ap.parse_args()
    if args.no_require_bind:
        args.require_bind = False
    if args.no_require_cooccur:
        args.require_cooccur = False
    allow_tags = set([x.strip() for x in args.source_tags.split(',') if x.strip()])
    main_df = safe_read_csv(args.main)
    if 'pair_key' not in main_df.columns:
        raise ValueError('main pair_key')
    if args.main_docking_col not in main_df.columns:
        raise ValueError(f'main{args.main_docking_col}Status:')
    docking = main_df[args.main_docking_col].fillna('').astype(str).str.strip()
    missing_pairs = sorted(main_df.loc[docking == '', 'pair_key'].astype(str).unique().tolist())
    print(f"main pairs_total: {main_df['pair_key'].nunique()} pairs_with_docking: {(docking != '').sum()} pairs_missing_docking: {len(missing_pairs)}")
    cand = safe_read_csv(args.candidates)
    need_cols = {'pair_key', 'uniprot_id', 'inchikey', 'paper_key', 'rank', 'source_tag'}
    miss = [c for c in need_cols if c not in cand.columns]
    if miss:
        raise ValueError(f'candidates :{miss}. cols={cand.columns.tolist()}')
    cand['pair_key'] = cand['pair_key'].astype(str)
    cand['paper_key'] = cand['paper_key'].astype(str)
    cand['source_tag'] = cand['source_tag'].fillna('').astype(str).str.strip()
    cand = cand[cand['pair_key'].isin(set(missing_pairs))].copy()
    cand = cand[cand['source_tag'].isin(allow_tags)].copy()
    cand = cand[cand['rank'] <= args.max_rank].copy()
    cand = cand.sort_values(['pair_key', 'rank'], ascending=[True, True])
    print('candidate rows (missing pairs only):', len(cand), 'pairs:', cand['pair_key'].nunique())
    pool = safe_read_csv(args.paper_pool)
    if 'paper_key' not in pool.columns:
        raise ValueError('paper_pool.csv paper_key')
    pool['paper_key'] = pool['paper_key'].astype(str)
    abs_col = detect_col(pool, ['abstract'])
    if not abs_col:
        raise ValueError('paper_pool.csv abstract')
    pool_map = dict(zip(pool['paper_key'], pool[abs_col].fillna('').astype(str).tolist()))
    fti = safe_read_csv(args.fulltext_index) if Path(args.fulltext_index).exists() else pd.DataFrame(columns=['paper_key', 'fulltext_path'])
    if 'paper_key' in fti.columns:
        fti['paper_key'] = fti['paper_key'].astype(str)
    ft_path_col = detect_col(fti, ['fulltext_path', 'fulltext', 'path', 'txt_gz_path'])
    if not ft_path_col:
        ft_path_col = None
    ft_map = {}
    if ft_path_col and len(fti) > 0:
        ft_map = dict(zip(fti['paper_key'].astype(str).tolist(), fti[ft_path_col].fillna('').astype(str).tolist()))
    prot_alias_map, _ = build_protein_alias_map(args.protein_index)
    lig_alias_map, _ = build_ligand_alias_map(args.ligand_index)
    out_path = Path(args.out_csv)
    done_pairs = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            old = safe_read_csv(str(out_path))
            if 'pair_key' in old.columns:
                done_pairs = set(old['pair_key'].astype(str).unique().tolist())
        except Exception:
            done_pairs = set()
    print('already_done_pairs:', len(done_pairs))
    prot_pat_cache: Dict[str, Optional[re.Pattern]] = {}
    lig_pat_cache: Dict[str, Optional[re.Pattern]] = {}

    def get_prot_pat(uid: str) -> Optional[re.Pattern]:
        if uid in prot_pat_cache:
            return prot_pat_cache[uid]
        aliases = prot_alias_map.get(uid, [uid])
        pat = make_alt_pattern(aliases, max_aliases=args.max_aliases_protein)
        prot_pat_cache[uid] = pat
        return pat

    def get_lig_pat(ik: str) -> Optional[re.Pattern]:
        if ik in lig_pat_cache:
            return lig_pat_cache[ik]
        aliases = lig_alias_map.get(ik, [])
        pat = make_alt_pattern(aliases, max_aliases=args.max_aliases_ligand)
        lig_pat_cache[ik] = pat
        return pat
    BONUS = {'pair': 0.6, 'intersection': 0.4, 'protein': 0.0, 'ligand': 0.0}
    rows_buf = []
    stats = {'pairs_total_in_main': int(main_df['pair_key'].nunique()), 'pairs_missing_docking': int(len(missing_pairs)), 'pairs_processed_now': 0, 'pairs_with_hits_now': 0, 'total_sentences_now': 0, 'miss_reason': {}, 'args': vars(args), 'allow_source_tags': sorted(list(allow_tags))}

    def flush():
        nonlocal rows_buf
        if not rows_buf:
            return
        df_out = pd.DataFrame(rows_buf)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            df_out.to_csv(out_path, index=False, mode='a', header=False)
        else:
            df_out.to_csv(out_path, index=False)
        rows_buf = []
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    pairs = cand['pair_key'].unique().tolist()
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
        print('Pair targets after limit:', len(pairs))
    pbar = tqdm(pairs, desc='25a multisource backfill', ncols=110)
    for pk in pbar:
        if pk in done_pairs:
            continue
        sub = cand[cand['pair_key'] == pk]
        if sub.empty:
            continue
        uid = norm_str(sub['uniprot_id'].iloc[0])
        ik = norm_str(sub['inchikey'].iloc[0])
        prot_pat = get_prot_pat(uid)
        lig_pat = get_lig_pat(ik)
        kept = []
        total_kept = 0
        for _, rr in sub.iterrows():
            if total_kept >= args.max_sent_total_per_pair:
                break
            paper_key = norm_str(rr['paper_key'])
            rank = int(rr['rank'])
            src = norm_str(rr['source_tag'])
            bonus = BONUS.get(src, 0.0)
            txt = ''
            text_type = ''
            ft_path = norm_str(ft_map.get(paper_key, ''))
            if ft_path:
                txt = load_fulltext_gz(ft_path)
                if txt:
                    text_type = 'fulltext'
            if not txt:
                txt = norm_str(pool_map.get(paper_key, ''))
                text_type = 'abstract' if txt else ''
            if not txt:
                continue
            spans = extract_best_spans(txt, prot_pat=prot_pat, lig_pat=lig_pat, source_bonus=bonus, window=args.window, min_score=args.min_score, max_sent_per_paper=args.max_sent_per_paper, require_bind=args.require_bind, require_cooccur=args.require_cooccur)
            if not spans:
                continue
            for span_text, score, strong_cnt, weak_cnt, both_same in spans:
                if total_kept >= args.max_sent_total_per_pair:
                    break
                kept.append({'pair_key': pk, 'uniprot_id': uid, 'inchikey': ik, 'paper_key': paper_key, 'rank': rank, 'source_tag': src, 'text_type': text_type, 'score': float(score), 'strong_bind_cnt': int(strong_cnt), 'weak_bind_cnt': int(weak_cnt), 'both_same_sentence': int(both_same), 'sentence': span_text})
                total_kept += 1
        stats['pairs_processed_now'] += 1
        if kept:
            stats['pairs_with_hits_now'] += 1
            stats['total_sentences_now'] += len(kept)
            rows_buf.extend(kept)
        else:
            stats['miss_reason']['pair_no_hit'] = stats['miss_reason'].get('pair_no_hit', 0) + 1
        hit_ratio = stats['pairs_with_hits_now'] / max(1, stats['pairs_processed_now'])
        pbar.set_postfix(hit_ratio=round(hit_ratio, 3), pairs_done=stats['pairs_processed_now'], pairs_hit=stats['pairs_with_hits_now'], sent=stats['total_sentences_now'])
        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)
        if args.flush_every and stats['pairs_processed_now'] % args.flush_every == 0:
            flush()
    flush()
    print('Saved:', args.out_csv)
    print('Saved:', args.out_stats)
    print('Summary:', {'pairs_missing_docking': stats['pairs_missing_docking'], 'pairs_processed_now': stats['pairs_processed_now'], 'pairs_with_hits_now': stats['pairs_with_hits_now'], 'pair_hit_ratio_now': stats['pairs_with_hits_now'] / max(1, stats['pairs_processed_now']), 'total_sentences_now': stats['total_sentences_now'], 'miss_reason': stats['miss_reason']})
if __name__ == '__main__':
    main()
