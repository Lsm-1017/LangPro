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

def get_missing_proteins():
    proteins = set(pd.read_csv(PROTEIN_LIST)['uniprot_id'].astype(str).tolist())
    have = set()
    if UNIPROT2PDB0.exists():
        have |= set(pd.read_csv(UNIPROT2PDB0)['uniprot_id'].astype(str).unique())
    if UNIPROT2PDB1.exists():
        have |= set(pd.read_csv(UNIPROT2PDB1)['uniprot_id'].astype(str).unique())
    if UNIPROT2PDB2.exists():
        have |= set(pd.read_csv(UNIPROT2PDB2)['uniprot_id'].astype(str).unique())
    return sorted(list(proteins - have))

def safe_get_json(url: str, timeout: int=60, retries: int=3, backoff: float=1.5):
    last_err = None
    for k in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                return (None, f'http_{r.status_code}')
            return (r.json(), 'ok')
        except Exception as e:
            last_err = e
            time.sleep(backoff ** k)
    return (None, f'exception_{type(last_err).__name__}')

def main():
    missing = get_missing_proteins()
    print('proteins_missing_pdb_total:', len(missing))
    rows = []
    for uni in tqdm(missing, desc='alphafold api'):
        api = AF_API.format(uni)
        js, status = safe_get_json(api, timeout=60, retries=3)
        if js is None:
            rows.append({'uniprot_id': uni, 'af_available': 0, 'status': status, 'cif_url': None, 'pdb_url': None, 'pae_url': None, 'model_version': None, 'entry_id': None})
            time.sleep(0.05)
            continue
        if not js:
            rows.append({'uniprot_id': uni, 'af_available': 0, 'status': 'empty', 'cif_url': None, 'pdb_url': None, 'pae_url': None, 'model_version': None, 'entry_id': None})
            time.sleep(0.05)
            continue
        item = js[0]
        cif_url = item.get('cifUrl')
        pdb_url = item.get('pdbUrl')
        pae_url = item.get('paeDocUrl') or item.get('paeJsonUrl')
        model_version = item.get('modelVersion') or item.get('model_version')
        entry_id = item.get('entryId') or item.get('entry_id') or item.get('uniprotAccession')
        af_available = 1 if cif_url or pdb_url else 0
        final_status = 'ok' if af_available else 'no_file_url'
        rows.append({'uniprot_id': uni, 'af_available': af_available, 'status': final_status, 'cif_url': cif_url, 'pdb_url': pdb_url, 'pae_url': pae_url, 'model_version': model_version, 'entry_id': entry_id})
        time.sleep(0.05)
    df = pd.DataFrame(rows)
    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_INDEX, index=False)
    print('Saved:', OUT_INDEX)
    print('AF available:', int(df['af_available'].sum()), '/', len(df))
    print('Status counts:\n', df['status'].value_counts().head(10))
if __name__ == '__main__':
    main()
