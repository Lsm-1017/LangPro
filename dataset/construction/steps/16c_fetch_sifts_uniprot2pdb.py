import csv
import gzip
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
PROTEIN_LIST = Path('../data_intermediate/protein_list.csv')
UNIPROT2PDB0 = Path('../data_intermediate/uniprot2pdb_long.csv')
SIFTS_GZ = Path('../data_external/uniprot_pdb.csv.gz')
OUT_LONG = Path('../data_intermediate/uniprot2pdb_sifts_long.csv')
SIFTS_URL = 'https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/uniprot_pdb.csv.gz'

def download_if_needed():
    SIFTS_GZ.parent.mkdir(parents=True, exist_ok=True)
    if SIFTS_GZ.exists() and SIFTS_GZ.stat().st_size > 1024 * 1024:
        return
    print('Downloading SIFTS:', SIFTS_URL)
    with requests.get(SIFTS_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(SIFTS_GZ, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print('Saved:', SIFTS_GZ)

def main():
    proteins = pd.read_csv(PROTEIN_LIST)['uniprot_id'].astype(str).tolist()
    proteins_set = set(proteins)
    have0 = set(pd.read_csv(UNIPROT2PDB0)['uniprot_id'].astype(str).unique())
    missing = proteins_set - have0
    print('proteins:', len(proteins_set))
    print('already_with_pdb (uniprot xref):', len(have0))
    print('missing_for_sifts:', len(missing))
    download_if_needed()
    records = []
    with gzip.open(SIFTS_GZ, 'rt', newline='') as f:
        reader = csv.reader(f)
        for row in tqdm(reader, desc='scan sifts'):
            if not row:
                continue
            uni = row[0].strip()
            if uni not in missing:
                continue
            if len(row) < 2:
                continue
            pdbs = row[1].strip()
            if not pdbs:
                continue
            for pdb in pdbs.split(';'):
                pdb = pdb.strip()
                if pdb:
                    records.append((uni, pdb, 'sifts_uniprot_pdb'))
    out = pd.DataFrame(records, columns=['uniprot_id', 'pdb_id', 'source'])
    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_LONG, index=False)
    print('Saved:', OUT_LONG, 'pairs=', len(out))
    print('proteins_hit_by_sifts:', out['uniprot_id'].nunique())
if __name__ == '__main__':
    main()
