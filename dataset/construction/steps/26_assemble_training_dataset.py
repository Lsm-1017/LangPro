import json
import argparse
from pathlib import Path
import pandas as pd

def norm_str(x) -> str:
    if x is None:
        return ''
    try:
        if isinstance(x, float) and pd.isna(x):
            return ''
    except Exception:
        pass
    s = str(x).strip()
    return '' if s.lower() == 'nan' else s

def ratio_nonempty(series: pd.Series) -> float:
    s = series.fillna('').astype(str).str.strip()
    return float((s != '').mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', required=True, help='--main input path or value.')
    ap.add_argument('--out_dir', default='../data_dataset/train_pairs_2440', help='--out_dir output path.')
    ap.add_argument('--lig3d_index', default='../data_intermediate/ligands_3d_rdkit_index.csv')
    ap.add_argument('--lig3d_npz', default='../data_intermediate/ligands_3d_rdkit.npz')
    ap.add_argument('--lig3d_sdf', default='../data_intermediate/ligands_3d_rdkit.sdf')
    ap.add_argument('--only_with_docking', action='store_true', help='--only_with_docking option.')
    ap.add_argument('--dedup_by_pair', action='store_true', help='--dedup_by_pair option.')
    ap.add_argument('--validate_files', action='store_true', help='--validate_files option.')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_df = pd.read_csv(args.main)
    required = ['uniprot_id', 'sequence_clean', 'function_text', 'ligand_smiles_canon', 'ligand_text', 'inchikey', 'pair_key', 'docking_text']
    missing = [c for c in required if c not in main_df.columns]
    if missing:
        raise ValueError(f':{missing}:{main_df.columns.tolist()}')
    for c in required + ['docking_sentence_count', 'docking_paper_count']:
        if c in main_df.columns:
            main_df[c] = main_df[c].map(norm_str)
    if args.only_with_docking:
        main_df = main_df[main_df['docking_text'].str.strip() != ''].copy()
    if args.dedup_by_pair:
        main_df = main_df.drop_duplicates(subset=['pair_key'], keep='first').copy()
    lig3d_index_path = Path(args.lig3d_index)
    lig3d_npz_path = Path(args.lig3d_npz)
    lig3d_sdf_path = Path(args.lig3d_sdf)
    lig_map = {}
    if lig3d_index_path.exists():
        idx_df = pd.read_csv(lig3d_index_path)
        if 'ligand_smiles_canon' not in idx_df.columns or 'idx' not in idx_df.columns:
            raise ValueError(f'{lig3d_index_path}ligand_smiles_canon idx cols={idx_df.columns.tolist()}')
        idx_df['ligand_smiles_canon'] = idx_df['ligand_smiles_canon'].map(norm_str)
        lig_map = dict(zip(idx_df['ligand_smiles_canon'], idx_df['idx']))
    else:
        print(f'[WARN] lig3d_index not found: {lig3d_index_path}. ligand_3d_idx')
    if args.validate_files:
        if not lig3d_npz_path.exists():
            raise FileNotFoundError(f'lig3d_npz not found: {lig3d_npz_path}')
        if not lig3d_sdf_path.exists():
            print(f'[WARN] lig3d_sdf not found: {lig3d_sdf_path}Status:')
    rows = []
    for _, r in main_df.iterrows():
        smi = norm_str(r['ligand_smiles_canon'])
        lig3d_idx = lig_map.get(smi, '')
        item = {'pair_key': norm_str(r['pair_key']), 'uniprot_id': norm_str(r['uniprot_id']), 'inchikey': norm_str(r['inchikey']), 'protein_sequence': norm_str(r['sequence_clean']), 'protein_function_text': norm_str(r['function_text']), 'docking_text': norm_str(r['docking_text']), 'ligand_smiles': smi, 'ligand_text': norm_str(r['ligand_text']), 'ligand_3d_npz_path': str(lig3d_npz_path) if lig3d_npz_path.exists() else '', 'ligand_3d_idx': lig3d_idx, 'ligand_3d_sdf_path': str(lig3d_sdf_path) if lig3d_sdf_path.exists() else ''}
        if 'docking_sentence_count' in main_df.columns:
            item['docking_sentence_count'] = r['docking_sentence_count']
        if 'docking_paper_count' in main_df.columns:
            item['docking_paper_count'] = r['docking_paper_count']
        rows.append(item)
    out_df = pd.DataFrame(rows)
    out_csv = out_dir / 'dataset.csv'
    out_jsonl = out_dir / 'dataset.jsonl'
    out_report = out_dir / 'report.json'
    out_df.to_csv(out_csv, index=False)
    with out_jsonl.open('w', encoding='utf-8') as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    lig3d_hit = out_df['ligand_3d_idx'].fillna('').astype(str).str.strip()
    report = {'input_main': args.main, 'rows_out': int(len(out_df)), 'pairs_out_unique': int(out_df['pair_key'].nunique()), 'nonempty_ratio': {'protein_sequence': ratio_nonempty(out_df['protein_sequence']), 'protein_function_text': ratio_nonempty(out_df['protein_function_text']), 'docking_text': ratio_nonempty(out_df['docking_text']), 'ligand_smiles': ratio_nonempty(out_df['ligand_smiles']), 'ligand_text': ratio_nonempty(out_df['ligand_text']), 'ligand_3d_idx': float((lig3d_hit != '').mean())}, 'lig3d_paths': {'lig3d_index': str(lig3d_index_path), 'lig3d_npz': str(lig3d_npz_path), 'lig3d_sdf': str(lig3d_sdf_path)}}
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', out_csv)
    print('Saved:', out_jsonl)
    print('Saved:', out_report)
    print('Report:', json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
