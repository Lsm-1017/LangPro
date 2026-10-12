import re
import json
import time
import gzip
import argparse
from pathlib import Path
from collections import defaultdict, OrderedDict
import pandas as pd
from tqdm import tqdm

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

def safe_read_text(path: Path, max_bytes: int=3000000) -> str:
    if not path.exists() or not path.is_file():
        return ''
    try:
        if path.suffix == '.gz':
            with gzip.open(path, 'rb') as f:
                b = f.read(max_bytes)
            return b.decode('utf-8', errors='ignore')
        else:
            with open(path, 'rb') as f:
                b = f.read(max_bytes)
            return b.decode('utf-8', errors='ignore')
    except Exception:
        return ''
_SENT_SPLIT = re.compile('(?<=[\\.\\?!\u3002\uff1f\uff01])\\s+|\\n+')

def split_sentences(text: str):
    text = re.sub('\\s+', ' ', text or '').strip()
    if not text:
        return []
    sents = _SENT_SPLIT.split(text)
    out = []
    for s in sents:
        s = s.strip()
        if len(s) < 30:
            continue
        if len(s) > 400:
            continue
        out.append(s)
    return out

def normalize_token(t: str) -> str:
    t = norm_str(t).lower()
    t = re.sub('\\s+', ' ', t).strip()
    return t

def build_phrase_list(raw_list, max_keep=20, min_len=3):
    stop = {'protein', 'proteins', 'ligand', 'ligands', 'binding', 'bind', 'bound', 'interaction', 'interactions', 'human', 'homo sapiens', 'cell', 'cells', 'study', 'results', 'method', 'methods', 'data', 'analysis'}
    phrases = []
    seen = set()
    for x in raw_list:
        s = normalize_token(x)
        if not s:
            continue
        if s in stop:
            continue
        if len(s) < min_len:
            continue
        if s in seen:
            continue
        seen.add(s)
        phrases.append(s)
        if len(phrases) >= max_keep:
            break
    return phrases

def compile_matchers(phrases):
    substr = []
    regs = []
    for p in phrases:
        if ' ' in p or '-' in p or '/' in p:
            substr.append(p)
            continue
        if len(p) <= 4:
            regs.append(re.compile(f'\\b{re.escape(p)}\\b', re.I))
        else:
            substr.append(p)
    return (substr, regs)

def any_phrase_hit(text_lc: str, substr_list, regex_list):
    for p in substr_list:
        if p in text_lc:
            return p
    for rgx in regex_list:
        m = rgx.search(text_lc)
        if m:
            return m.group(0)
    return ''
TRIG_BIND = re.compile('\\b(bind|binding|bound|affinity|interact|interaction|inhibit|inhibition|docking|dock)\\b', re.I)
TRIG_SITE = re.compile('\\b(active site|binding site|binding pocket|pocket|cavity)\\b', re.I)
TRIG_MEASURE = re.compile('\\b(Kd|Ki|IC50|EC50|pIC50|pKi|pKd|\u0394G|dG|free energy)\\b', re.I)
TRIG_NUMUNIT = re.compile('(\\d+(\\.\\d+)?\\s*(pM|nM|uM|\u00b5M|mM|M))', re.I)
NEG_JUNK = re.compile('\\b(references?|bibliography|copyright|supplementary|supplemental|table\\s*\\d+|figure\\s*\\d+)\\b', re.I)

def score_sentence(s: str):
    score = 0.0
    reasons = []
    if NEG_JUNK.search(s):
        score -= 2.0
        reasons.append('neg_junk')
    if TRIG_BIND.search(s):
        score += 1.0
        reasons.append('bind')
    if TRIG_SITE.search(s):
        score += 1.0
        reasons.append('site')
    if TRIG_MEASURE.search(s):
        score += 2.0
        reasons.append('measure')
    if TRIG_NUMUNIT.search(s):
        score += 1.0
        reasons.append('numunit')
    if TRIG_MEASURE.search(s) and TRIG_NUMUNIT.search(s):
        score += 2.0
        reasons.append('measure+num')
    return (score, reasons)

def load_protein_mentions(path: Path, max_tokens_per_prot=10):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    col_uni = cols.get('uniprot_id') or cols.get('entry') or cols.get('uniprot')
    if not col_uni:
        col_uni = df.columns[0]
    cand_cols = []
    for k in ['gene_names', 'gene name', 'genes', 'entry_name', 'entry name', 'protein_name', 'protein names', 'id']:
        if k in cols:
            cand_cols.append(cols[k])
    m = {}
    for _, r in df.iterrows():
        uni = norm_str(r.get(col_uni))
        if not uni:
            continue
        raw = []
        for c in cand_cols:
            raw.append(r.get(c))
        tokens = []
        for x in raw:
            s = norm_str(x)
            if not s:
                continue
            s = s.replace('_HUMAN', ' ')
            parts = re.split('[;,|/ ]+', s)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if len(p) < 3:
                    continue
                tokens.append(p)
        phrases = build_phrase_list(tokens, max_keep=max_tokens_per_prot, min_len=3)
        if phrases:
            m[uni] = phrases
    return m

def load_ligand_mentions(path: Path, max_tokens_per_lig=25):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    col_ik = cols.get('inchikey') or cols.get('inchi_key') or cols.get('ik')
    if not col_ik:
        return {}
    cand_cols = []
    for k in ['name', 'canonical_name', 'ligand_name', 'synonyms', 'pubchem_synonyms', 'all_synonyms']:
        if k in cols:
            cand_cols.append(cols[k])
    m = {}
    for _, r in df.iterrows():
        ik = norm_str(r.get(col_ik))
        if not ik:
            continue
        tokens = []
        for c in cand_cols:
            s = norm_str(r.get(c))
            if not s:
                continue
            parts = re.split('[|;]\\s*|,\\s*', s)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if len(p) > 60:
                    continue
                tokens.append(p)
        phrases = build_phrase_list(tokens, max_keep=max_tokens_per_lig, min_len=3)
        if phrases:
            m[ik] = phrases
    return m

class LRUCache:

    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.d = OrderedDict()

    def get(self, k):
        if k not in self.d:
            return None
        v = self.d.pop(k)
        self.d[k] = v
        return v

    def put(self, k, v):
        if k in self.d:
            self.d.pop(k)
        self.d[k] = v
        if len(self.d) > self.capacity:
            self.d.popitem(last=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paironly_candidates', default='../data_intermediate/pair_paper_candidates_paironly.csv')
    ap.add_argument('--paper_pool', default='../data_intermediate/paper_pool.csv')
    ap.add_argument('--fulltext_index', default='../data_intermediate/paper_fulltext_index.csv')
    ap.add_argument('--main', default='../data_intermediate/triples_with_ligand_text_full_with_docking.csv')
    ap.add_argument('--main_docking_col', default='docking_text')
    ap.add_argument('--protein_index', default='../data_intermediate/protein_search_index.csv')
    ap.add_argument('--ligand_index', default='../data_intermediate/ligand_search_index.csv')
    ap.add_argument('--max_rank', type=int, default=30)
    ap.add_argument('--max_sent_per_paper', type=int, default=8)
    ap.add_argument('--max_sent_total_per_pair', type=int, default=40)
    ap.add_argument('--window', type=int, default=0)
    ap.add_argument('--min_score', type=float, default=2.0)
    ap.add_argument('--require_cooccur', action='store_true', default=True)
    ap.add_argument('--no_require_cooccur', action='store_true', default=False)
    ap.add_argument('--flush_every', type=int, default=50)
    ap.add_argument('--sleep', type=float, default=0.0)
    ap.add_argument('--out_csv', default='../data_intermediate/pair_docking_sentence_candidates_paironly_backfill.csv')
    ap.add_argument('--out_stats', default='../data_intermediate/pair_docking_sentence_candidates_paironly_backfill_stats.json')
    args = ap.parse_args()
    require_cooccur = args.require_cooccur and (not args.no_require_cooccur)
    main_df = pd.read_csv(args.main)
    if args.main_docking_col not in main_df.columns:
        raise ValueError(f'main missing docking col: {args.main_docking_col}')
    docking = main_df[args.main_docking_col].fillna('').astype(str).str.strip()
    missing_pairs = set(main_df.loc[docking == '', 'pair_key'].astype(str).tolist())
    pairs_total = main_df['pair_key'].nunique()
    print(f"main pairs_total: {pairs_total} pairs_with_docking: {(docking != '').sum()} pairs_missing_docking: {len(missing_pairs)}")
    cand = pd.read_csv(args.paironly_candidates)
    cand = cand[cand['pair_key'].astype(str).isin(missing_pairs)].copy()
    if 'rank' in cand.columns:
        cand = cand[cand['rank'] <= args.max_rank].copy()
    if 'source_tag' in cand.columns:
        cand = cand[cand['source_tag'].astype(str) == 'pair'].copy()
    print('paironly candidate rows (missing pairs only):', len(cand), 'pairs:', cand['pair_key'].nunique())
    out_csv = Path(args.out_csv)
    already_done_pairs = set()
    if out_csv.exists() and out_csv.stat().st_size > 100:
        try:
            old = pd.read_csv(out_csv, usecols=['pair_key'])
            already_done_pairs = set(old['pair_key'].astype(str).unique().tolist())
        except Exception:
            already_done_pairs = set()
    print('already_done_pairs:', len(already_done_pairs))
    pool = pd.read_csv(args.paper_pool)
    if 'paper_key' not in pool.columns:
        raise ValueError('paper_pool missing paper_key')
    pool['paper_key'] = pool['paper_key'].astype(str)
    pool_abs = {}
    if 'abstract' in pool.columns:
        pool_abs = dict(zip(pool['paper_key'], pool['abstract']))
    else:
        pool_abs = {k: '' for k in pool['paper_key'].tolist()}
    fti = pd.DataFrame(columns=['paper_key', 'fulltext_path'])
    fti_path = Path(args.fulltext_index)
    paper2ft = {}
    if fti_path.exists():
        fti = pd.read_csv(fti_path)
        if 'fulltext_path' not in fti.columns and 'fulltext' in fti.columns:
            fti['fulltext_path'] = fti['fulltext']
        if 'paper_key' in fti.columns and 'fulltext_path' in fti.columns:
            paper2ft = dict(zip(fti['paper_key'].astype(str), fti['fulltext_path'].fillna('').astype(str)))
    prot_map = load_protein_mentions(Path(args.protein_index))
    lig_map = load_ligand_mentions(Path(args.ligand_index))
    pair2rows = defaultdict(list)
    for _, r in cand.iterrows():
        pair2rows[str(r['pair_key'])].append(r)
    pairs_to_process = [p for p in pair2rows.keys() if p not in already_done_pairs]
    pbar = tqdm(pairs_to_process, desc='24b paironly backfill (tight)')
    cache = LRUCache(capacity=6000)
    pairs_done = 0
    pairs_hit = 0
    total_sent = 0
    miss_reason = defaultdict(int)
    buffer_rows = []

    def flush():
        nonlocal buffer_rows
        if not buffer_rows:
            return
        df = pd.DataFrame(buffer_rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if out_csv.exists() and out_csv.stat().st_size > 100:
            df.to_csv(out_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(out_csv, index=False)
        buffer_rows = []
    for pair_key in pbar:
        rows = pair2rows[pair_key]
        uni = norm_str(rows[0].get('uniprot_id'))
        ik = norm_str(rows[0].get('inchikey'))
        prot_phr = prot_map.get(uni, [])
        lig_phr = lig_map.get(ik, [])
        if require_cooccur and (not prot_phr or not lig_phr):
            miss_reason['missing_mentions'] += 1
            pairs_done += 1
            pbar.set_postfix(pairs_done=pairs_done, pairs_hit=pairs_hit, hit_ratio=round(pairs_hit / max(pairs_done, 1), 3), sent=total_sent)
            continue
        prot_sub, prot_rgx = compile_matchers(prot_phr)
        lig_sub, lig_rgx = compile_matchers(lig_phr)
        chosen = []
        chosen_set = set()
        any_text_used = False
        rows_sorted = sorted(rows, key=lambda x: int(x.get('rank', 10 ** 9)))
        for rr in rows_sorted:
            paper_key = norm_str(rr.get('paper_key'))
            rank = int(rr.get('rank', 0)) if str(rr.get('rank', '')).strip() != '' else 0
            source_tag = norm_str(rr.get('source_tag', 'pair'))
            text = ''
            text_source = ''
            ft_path = norm_str(paper2ft.get(paper_key, ''))
            if ft_path:
                cached = cache.get(('ft', ft_path))
                if cached is None:
                    text = safe_read_text(Path(ft_path))
                    cache.put(('ft', ft_path), text)
                else:
                    text = cached
                if text:
                    text_source = 'fulltext'
            if not text:
                abs_txt = norm_str(pool_abs.get(paper_key, ''))
                if abs_txt:
                    text = abs_txt
                    text_source = 'abstract'
            if not text:
                continue
            any_text_used = True
            sents = split_sentences(text)
            if not sents:
                continue
            if args.window > 0:
                ext = []
                for i, s in enumerate(sents):
                    l = max(0, i - args.window)
                    r = min(len(sents), i + args.window + 1)
                    ext.append(' '.join(sents[l:r]))
                sents_use = ext
            else:
                sents_use = sents
            scored = []
            for si, s in enumerate(sents_use):
                if not TRIG_BIND.search(s) and (not TRIG_MEASURE.search(s)) and (not TRIG_SITE.search(s)):
                    continue
                score, reasons = score_sentence(s)
                if score < args.min_score:
                    continue
                s_lc = s.lower()
                prot_hit = ''
                lig_hit = ''
                if require_cooccur:
                    prot_hit = any_phrase_hit(s_lc, prot_sub, prot_rgx)
                    if not prot_hit:
                        continue
                    lig_hit = any_phrase_hit(s_lc, lig_sub, lig_rgx)
                    if not lig_hit:
                        continue
                key = re.sub('\\s+', ' ', s_lc).strip()
                if key in chosen_set:
                    continue
                scored.append((score, si, s, prot_hit, lig_hit, '|'.join(reasons)))
            if not scored:
                continue
            scored.sort(key=lambda x: (-x[0], x[1]))
            take = 0
            for sc, si, s, prot_hit, lig_hit, reasons in scored:
                if take >= args.max_sent_per_paper:
                    break
                key = re.sub('\\s+', ' ', s.lower()).strip()
                if key in chosen_set:
                    continue
                chosen_set.add(key)
                chosen.append({'pair_key': pair_key, 'uniprot_id': uni, 'inchikey': ik, 'paper_key': paper_key, 'rank': rank, 'source_tag': source_tag, 'text_source': text_source, 'score': sc, 'reasons': reasons, 'protein_hit': prot_hit, 'ligand_hit': lig_hit, 'sentence': s})
                take += 1
                if len(chosen) >= args.max_sent_total_per_pair:
                    break
            if len(chosen) >= args.max_sent_total_per_pair:
                break
            if args.sleep > 0:
                time.sleep(args.sleep)
        pairs_done += 1
        if chosen:
            pairs_hit += 1
            total_sent += len(chosen)
            buffer_rows.extend(chosen)
        elif not any_text_used:
            miss_reason['no_text'] += 1
        else:
            miss_reason['pair_no_hit'] += 1
        if args.flush_every > 0 and pairs_done % args.flush_every == 0:
            flush()
        pbar.set_postfix(pairs_done=pairs_done, pairs_hit=pairs_hit, hit_ratio=round(pairs_hit / max(pairs_done, 1), 3), sent=total_sent)
    flush()
    stats = {'pairs_total_in_main': int(pairs_total), 'pairs_with_docking_in_main': int((docking != '').sum()), 'pairs_missing_docking_in_main': int(len(missing_pairs)), 'paironly_pairs_scanned': int(len(pair2rows)), 'pairs_processed_now': int(pairs_done), 'pairs_with_hits_now': int(pairs_hit), 'pair_hit_ratio_now': float(pairs_hit / max(pairs_done, 1)), 'total_sentences_now': int(total_sent), 'args': vars(args), 'require_cooccur': bool(require_cooccur), 'miss_reason': dict(sorted(miss_reason.items(), key=lambda x: -x[1]))}
    Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print('Saved:', args.out_csv)
    print('Saved:', args.out_stats)
    print('Summary:', stats)
if __name__ == '__main__':
    main()
