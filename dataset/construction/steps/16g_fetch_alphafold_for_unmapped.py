import time
import random
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
PROTEIN_LIST = Path('../data_intermediate/protein_list.csv')
UNIPROT2PDB0 = Path('../data_intermediate/uniprot2pdb_long.csv')
UNIPROT2PDB1 = Path('../data_intermediate/uniprot2pdb_sifts_long.csv')
UNIPROT2PDB2 = Path('../data_intermediate/uniprot2pdb_rcsb_long.csv')
OUT_INDEX = Path('../data_intermediate/uniprot2alphafold_index.csv')
AF_API = 'https://alphafold.ebi.ac.uk/api/prediction/{}'
SAVE_EVERY = 500
SLEEP_SEC = 0.05
MAX_TRIES = 5

def read_uniprot_list() -> list[str]:
    df = pd.read_csv(PROTEIN_LIST)
    return df['uniprot_id'].astype(str).tolist()

def proteins_with_any_pdb() -> set[str]:
    have = set()
    if UNIPROT2PDB0.exists():
        have |= set(pd.read_csv(UNIPROT2PDB0)['uniprot_id'].astype(str).unique())
    if UNIPROT2PDB1.exists() and UNIPROT2PDB1.stat().st_size > 0:
        have |= set(pd.read_csv(UNIPROT2PDB1)['uniprot_id'].astype(str).unique())
    if UNIPROT2PDB2.exists() and UNIPROT2PDB2.stat().st_size > 0:
        df2 = pd.read_csv(UNIPROT2PDB2)
        if len(df2) > 0:
            have |= set(df2['uniprot_id'].astype(str).unique())
    return have

def get_missing_proteins() -> list[str]:
    proteins = set(read_uniprot_list())
    have = proteins_with_any_pdb()
    missing = sorted(list(proteins - have))
    return missing

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({'User-Agent': 'langpro-alphafold-indexer/1.0'})
    return s

def load_done_ids() -> set[str]:
    if not OUT_INDEX.exists() or OUT_INDEX.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(OUT_INDEX, usecols=['uniprot_id'])
        return set(df['uniprot_id'].astype(str).tolist())
    except Exception:
        return set()

def safe_write_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def fetch_alphafold_meta(session: requests.Session, uniprot_id: str, timeout: int=30) -> dict:
    url = AF_API.format(uniprot_id)
    last_exc = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            r = session.get(url, timeout=timeout)
            http_status = r.status_code
            if r.status_code == 200:
                js = r.json()
                if not js:
                    return {'af_available': 0, 'cif_url': None, 'pdb_url': None, 'pae_url': None, 'status': 'empty', 'http_status': http_status, 'n_items': 0}
                item = js[0]
                cif_url = item.get('cifUrl')
                pdb_url = item.get('pdbUrl')
                pae_url = item.get('paeDocUrl') or item.get('paeJsonUrl')
                if not cif_url and (not pdb_url):
                    return {'af_available': 0, 'cif_url': cif_url, 'pdb_url': pdb_url, 'pae_url': pae_url, 'status': 'no_file_url', 'http_status': http_status, 'n_items': len(js)}
                return {'af_available': 1, 'cif_url': cif_url, 'pdb_url': pdb_url, 'pae_url': pae_url, 'status': 'ok', 'http_status': http_status, 'n_items': len(js)}
            if r.status_code in (400, 404):
                return {'af_available': 0, 'cif_url': None, 'pdb_url': None, 'pae_url': None, 'status': 'api_not_found', 'http_status': http_status, 'n_items': 0}
            last_exc = RuntimeError(f'http {r.status_code}')
        except Exception as e:
            last_exc = e
        wait = min(60, 2 ** min(attempt, 6)) + random.uniform(0, 1)
        time.sleep(wait)
    return {'af_available': 0, 'cif_url': None, 'pdb_url': None, 'pae_url': None, 'status': 'exception', 'http_status': None, 'n_items': 0}

def main():
    missing = get_missing_proteins()
    print('proteins_missing_pdb_total:', len(missing))
    done = load_done_ids()
    if done:
        print(f'[RESUME] already saved: {len(done)} rows in {OUT_INDEX}')
    missing = [u for u in missing if u not in done]
    print('remaining_to_query:', len(missing))
    if not missing:
        print('[DONE] nothing to do.')
        return
    session = make_session()
    if OUT_INDEX.exists() and OUT_INDEX.stat().st_size > 0:
        try:
            out_df = pd.read_csv(OUT_INDEX)
        except Exception:
            out_df = pd.DataFrame()
    else:
        out_df = pd.DataFrame()
    rows_buf = []
    processed = 0
    for uni in tqdm(missing, desc='alphafold api'):
        meta = fetch_alphafold_meta(session, uni, timeout=30)
        rows_buf.append({'uniprot_id': uni, 'af_available': int(meta['af_available']), 'cif_url': meta['cif_url'], 'pdb_url': meta['pdb_url'], 'pae_url': meta['pae_url'], 'local_path': '', 'status': meta['status'], 'http_status': meta['http_status'], 'n_items': meta['n_items']})
        processed += 1
        if processed % SAVE_EVERY == 0:
            buf_df = pd.DataFrame(rows_buf)
            if len(out_df) == 0:
                out_df = buf_df
            else:
                out_df = pd.concat([out_df, buf_df], ignore_index=True)
            safe_write_csv(out_df, OUT_INDEX)
            rows_buf = []
            print(f'[SAVED] total_rows={len(out_df)}')
        time.sleep(SLEEP_SEC)
    if rows_buf:
        buf_df = pd.DataFrame(rows_buf)
        out_df = pd.concat([out_df, buf_df], ignore_index=True) if len(out_df) else buf_df
        safe_write_csv(out_df, OUT_INDEX)
    print('Saved:', OUT_INDEX)
    if 'af_available' in out_df.columns:
        print('AF available:', int(out_df['af_available'].sum()), '/', len(out_df))
        print('Status counts:\n', out_df['status'].value_counts())
if __name__ == '__main__':
    main()
