from __future__ import annotations
import argparse
import json
import ast
from pathlib import Path
import pandas as pd
TRIPLES = Path('../data_intermediate/triples_with_ligand_text_full.csv')
PROT_IDX = Path('../data_intermediate/protein_search_index.csv')
IN_LIG_WITH = Path('../data_intermediate/ligand_search_index.with_synonyms.csv')
OUT_LIG = Path('../data_intermediate/ligand_search_index.csv')
OUT_PAIR = Path('../data_intermediate/pair_search_index.csv')

def _norm_str(x) -> str:
    if pd.isna(x):
        return ''
    return str(x).strip()

def _ensure_json_array_str(x) -> str:
    if not isinstance(x, str):
        return '[]'
    s = x.strip()
    if not s:
        return '[]'
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return json.dumps(v, ensure_ascii=False)
        return '[]'
    except Exception:
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return json.dumps(v, ensure_ascii=False)
        except Exception:
            pass
        return '[]'

def _parse_synonyms_cell(cell) -> list[str]:
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    s = str(cell).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        pass
    if '|' in s:
        return [t.strip() for t in s.split('|') if t.strip()]
    return [s]

def build_ligand_search_index_from_with_synonyms(in_path: Path, max_terms: int=60, force_rebuild_terms: bool=True) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    if 'syn_status' not in df.columns:
        if 'syn_status_y' in df.columns:
            df['syn_status'] = df['syn_status_y']
        elif 'syn_status_x' in df.columns:
            df['syn_status'] = df['syn_status_x']
        else:
            df['syn_status'] = ''
    if 'pubchem_cid' not in df.columns:
        if 'cid' in df.columns:
            df['pubchem_cid'] = df['cid']
        else:
            df['pubchem_cid'] = ''
    if 'synonyms' in df.columns and force_rebuild_terms:

        def make_terms(row):
            terms = []
            name = _norm_str(row.get('ligand_name', ''))
            if name and name.upper() != 'NAN':
                terms.append(name)
            syn_list = _parse_synonyms_cell(row.get('synonyms', ''))
            for t in syn_list:
                t = _norm_str(t)
                if t:
                    terms.append(t)
            seen = set()
            uniq = []
            for t in terms:
                t = _norm_str(t)
                if not t:
                    continue
                k = t.lower()
                if k in seen:
                    continue
                uniq.append(t)
                seen.add(k)
                if len(uniq) >= max_terms:
                    break
            return json.dumps(uniq, ensure_ascii=False)
        df['ligand_terms_json'] = df.apply(make_terms, axis=1)
    if 'ligand_terms_json' not in df.columns:
        df['ligand_terms_json'] = '[]'
    required = ['ligand_smiles_canon', 'inchikey', 'ligand_name', 'ligand_chebi', 'ligand_annotation', 'pubchem_cid', 'syn_status', 'ligand_terms_json']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'[ERROR] {in_path} missing columns: {missing}\nAvailable: {df.columns.tolist()}')
    out = df[required].copy()
    out['ligand_smiles_canon'] = out['ligand_smiles_canon'].apply(_norm_str)
    out['inchikey'] = out['inchikey'].apply(_norm_str)
    out['ligand_name'] = out['ligand_name'].apply(_norm_str)
    out['ligand_chebi'] = out['ligand_chebi'].apply(_norm_str)
    out['ligand_annotation'] = out['ligand_annotation'].apply(_norm_str)
    out['pubchem_cid'] = out['pubchem_cid'].apply(_norm_str)
    out['syn_status'] = out['syn_status'].apply(_norm_str)
    out['ligand_terms_json'] = out['ligand_terms_json'].apply(_ensure_json_array_str)
    out = out[out['ligand_smiles_canon'] != ''].drop_duplicates(subset=['ligand_smiles_canon'], keep='first')
    return out

def patch_pair_search_index(pair_df: pd.DataFrame, lig_out: pd.DataFrame) -> pd.DataFrame:
    lig_map = lig_out[['ligand_smiles_canon', 'inchikey', 'ligand_terms_json']].copy()
    lig_map['ligand_smiles_canon'] = lig_map['ligand_smiles_canon'].apply(_norm_str)
    for c in ['inchikey', 'ligand_terms_json']:
        if c in pair_df.columns:
            pair_df = pair_df.drop(columns=[c])
    pair_df['ligand_smiles_canon'] = pair_df['ligand_smiles_canon'].apply(_norm_str)
    pair_df['uniprot_id'] = pair_df['uniprot_id'].apply(_norm_str)
    merged = pair_df.merge(lig_map, on='ligand_smiles_canon', how='left')
    merged['inchikey'] = merged['inchikey'].fillna('').apply(_norm_str)
    merged['ligand_terms_json'] = merged['ligand_terms_json'].fillna('[]').apply(_ensure_json_array_str)
    merged['pair_id'] = merged.apply(lambda r: f"{r['uniprot_id']}__{r['inchikey']}", axis=1)
    return merged

def rebuild_pair_from_triples(lig_out: pd.DataFrame) -> pd.DataFrame:
    if not TRIPLES.exists():
        raise FileNotFoundError(f'[ERROR] missing TRIPLES: {TRIPLES}')
    if not PROT_IDX.exists():
        raise FileNotFoundError(f'[ERROR] missing PROT_IDX: {PROT_IDX}')
    df = pd.read_csv(TRIPLES)
    pair_df = df[['uniprot_id', 'ligand_smiles_canon', 'ligand_name']].dropna(subset=['uniprot_id', 'ligand_smiles_canon']).astype(str)
    pair_df['uniprot_id'] = pair_df['uniprot_id'].str.strip()
    pair_df['ligand_smiles_canon'] = pair_df['ligand_smiles_canon'].str.strip()
    pair_df = pair_df.groupby(['uniprot_id', 'ligand_smiles_canon'])['ligand_name'].agg(lambda x: x.value_counts().index[0]).reset_index()
    print('Unique (protein, ligand) pairs:', len(pair_df))
    prot = pd.read_csv(PROT_IDX)
    prot['uniprot_id'] = prot['uniprot_id'].apply(_norm_str)
    lig_map = lig_out[['ligand_smiles_canon', 'inchikey', 'ligand_terms_json']].copy()
    lig_map['ligand_smiles_canon'] = lig_map['ligand_smiles_canon'].apply(_norm_str)
    pair = pair_df.merge(prot[['uniprot_id', 'has_pdb', 'af_available', 'pdb_id', 'protein_terms_json']], on='uniprot_id', how='left')
    pair = pair.merge(lig_map, on='ligand_smiles_canon', how='left')
    pair['uniprot_id'] = pair['uniprot_id'].apply(_norm_str)
    pair['inchikey'] = pair['inchikey'].fillna('').apply(_norm_str)
    pair['ligand_terms_json'] = pair['ligand_terms_json'].fillna('[]').apply(_ensure_json_array_str)
    pair['pair_id'] = pair.apply(lambda r: f"{r['uniprot_id']}__{r['inchikey']}", axis=1)
    return pair

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_lig_with', default=str(IN_LIG_WITH), help='--in_lig_with input path or value.')
    ap.add_argument('--out_lig', default=str(OUT_LIG), help='--out_lig output path.')
    ap.add_argument('--out_pair', default=str(OUT_PAIR), help='--out_pair output path.')
    ap.add_argument('--pair_in', default='', help='--pair_in option.')
    ap.add_argument('--max_terms', type=int, default=60, help='--max_terms numeric setting.')
    ap.add_argument('--no_force_rebuild_terms', action='store_true', help='--no_force_rebuild_terms option.')
    args = ap.parse_args()
    in_lig_with = Path(args.in_lig_with)
    out_lig = Path(args.out_lig)
    out_pair = Path(args.out_pair)
    pair_in = Path(args.pair_in) if args.pair_in else out_pair
    print('[1/3] Build ligand_search_index.csv from with_synonyms ...')
    lig_out = build_ligand_search_index_from_with_synonyms(in_lig_with, max_terms=args.max_terms, force_rebuild_terms=not args.no_force_rebuild_terms)
    out_lig.parent.mkdir(parents=True, exist_ok=True)
    lig_out.to_csv(out_lig, index=False)
    print('[OK] Saved:', out_lig, 'n=', len(lig_out))
    print('syn_status counts (top):\n', lig_out['syn_status'].value_counts().head(10))
    lens = lig_out['ligand_terms_json'].apply(lambda s: len(json.loads(s)) if isinstance(s, str) and s.strip() else 0)
    print('ligands terms_len>1:', int((lens > 1).sum()), '/', len(lig_out))
    if int((lens > 1).sum()) == 0:
        print('[WARN] All ligands have terms_len<=1. This usually means synonyms column was empty or not parsed.')
    print('[2/3] Patch or rebuild pair_search_index.csv ...')
    if pair_in.exists() and pair_in.stat().st_size > 0:
        pair_df = pd.read_csv(pair_in)
        need_cols = {'uniprot_id', 'ligand_smiles_canon'}
        if not need_cols.issubset(set(pair_df.columns)):
            raise ValueError(f'[ERROR] pair_in missing columns: {sorted(list(need_cols - set(pair_df.columns)))}')
        pair_df2 = patch_pair_search_index(pair_df, lig_out)
        out_pair.parent.mkdir(parents=True, exist_ok=True)
        pair_df2.to_csv(out_pair, index=False)
        print('[OK] Patched pair:', out_pair, 'n=', len(pair_df2))
    else:
        pair_df2 = rebuild_pair_from_triples(lig_out)
        out_pair.parent.mkdir(parents=True, exist_ok=True)
        pair_df2.to_csv(out_pair, index=False)
        print('[OK] Rebuilt pair:', out_pair, 'n=', len(pair_df2))
    if 'has_pdb' in pair_df2.columns:
        print('Pairs has_pdb ratio:', pair_df2['has_pdb'].fillna(0).mean())
    if 'af_available' in pair_df2.columns:
        print('Pairs af_available ratio:', pair_df2['af_available'].fillna(0).mean())
    print('[3/3] Quick sanity check: random 5 ligands terms length ...')
    import random
    for i in random.sample(range(len(lig_out)), min(5, len(lig_out))):
        terms = json.loads(lig_out.iloc[i]['ligand_terms_json'])
        print('  -', lig_out.iloc[i]['ligand_name'], 'terms_len=', len(terms), 'first5=', terms[:5])
if __name__ == '__main__':
    main()
