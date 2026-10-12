import time
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

def read_uniprot_list() -> list[str]:
    df = pd.read_csv(PROTEIN_LIST)
    return df['uniprot_id'].astype(str).tolist()

def proteins_with_any_pdb() -> set[str]:
    have = set()
    if UNIPROT2PDB0.exists():
        have |= set(pd.read_csv(UNIPROT2PDB0)['uniprot_id'].astype(str).unique())
    if UNIPROT2PDB1.exists():
        have |= set(pd.read_csv(UNIPROT2PDB1)['uniprot_id'].astype(str).unique())
    if UNIPROT2PDB2.exists():
        df2 = pd.read_csv(UNIPROT2PDB2) if UNIPROT2PDB2.stat().st_size > 0 else None
        if df2 is not None and len(df2) > 0:
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

def fetch_alphafold_meta(session: requests.Session, uniprot_id: str, timeout: int=30) -> dict:
    url = AF_API.format(uniprot_id)
    try:
        r = session.get(url, timeout=timeout)
        http_status = r.status_code
        if r.status_code != 200:
            return {'af_available': 0, 'cif_url': None, 'pdb_url': None, 'pae_url': None, 'status': 'api_not_found', 'http_status': http_status, 'n_items': 0}
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
    except Exception:
        return {'af_available': 0, 'cif_url': None, 'pdb_url': None, 'pae_url': None, 'status': 'exception', 'http_status': None, 'n_items': 0}

def main():
    missing = get_missing_proteins()
    print('proteins_missing_pdb_total:', len(missing))
    session = make_session()
    rows = []
    for uni in tqdm(missing, desc='alphafold api'):
        meta = fetch_alphafold_meta(session, uni, timeout=30)
        rows.append({'uniprot_id': uni, 'af_available': int(meta['af_available']), 'cif_url': meta['cif_url'], 'pdb_url': meta['pdb_url'], 'pae_url': meta['pae_url'], 'local_path': '', 'status': meta['status'], 'http_status': meta['http_status'], 'n_items': meta['n_items']})
        time.sleep(0.05)
    out = pd.DataFrame(rows)
    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_INDEX, index=False)
    print('Saved:', OUT_INDEX)
    print('AF available:', int(out['af_available'].sum()), '/', len(out))
    print('Status counts:\n', out['status'].value_counts())
if __name__ == '__main__':
    main()
