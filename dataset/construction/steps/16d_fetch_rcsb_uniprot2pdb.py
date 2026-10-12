import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter
PROTEIN_LIST = Path('../data_intermediate/protein_list.csv')
UNIPROT2PDB0 = Path('../data_intermediate/uniprot2pdb_long.csv')
UNIPROT2PDB1 = Path('../data_intermediate/uniprot2pdb_sifts_long.csv')
OUT_LONG = Path('../data_intermediate/uniprot2pdb_rcsb_long.csv')
RCSB_ENDPOINT = 'https://search.rcsb.org/rcsbsearch/v2/query'
ATTR = 'rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession'

def rcsb_query(uniprot_id: str, rows: int=10000, retries: int=5, base_sleep: float=0.5):
    q = {'query': {'type': 'terminal', 'service': 'text', 'parameters': {'attribute': ATTR, 'operator': 'exact_match', 'value': uniprot_id}}, 'return_type': 'entry', 'request_options': {'paginate': {'start': 0, 'rows': min(int(rows), 10000)}}}
    last_status = None
    for k in range(retries):
        try:
            r = requests.post(RCSB_ENDPOINT, json=q, timeout=60)
            last_status = r.status_code
            if r.status_code == 200:
                js = r.json()
                hits = js.get('result_set', [])
                return [h.get('identifier') for h in hits if h.get('identifier')]
            if r.status_code == 429 or 500 <= r.status_code < 600:
                time.sleep(base_sleep * 2 ** k)
                continue
            msg = r.text[:200].replace('\n', ' ')
            print(f'[RCSB 4xx] {uniprot_id} status={r.status_code} msg={msg}')
            return []
        except Exception as e:
            if k == retries - 1:
                print(f'[RCSB EXC] {uniprot_id} final_fail: {type(e).__name__}: {e}')
                return []
            time.sleep(base_sleep * 2 ** k)
    if last_status is not None:
        print(f'[RCSB FAIL] {uniprot_id} last_status={last_status}')
    return []

def main():
    proteins = pd.read_csv(PROTEIN_LIST)['uniprot_id'].astype(str).tolist()
    proteins_set = set(proteins)
    have0 = set(pd.read_csv(UNIPROT2PDB0)['uniprot_id'].astype(str).unique()) if UNIPROT2PDB0.exists() else set()
    have1 = set(pd.read_csv(UNIPROT2PDB1)['uniprot_id'].astype(str).unique()) if UNIPROT2PDB1.exists() else set()
    missing = sorted(list(proteins_set - have0 - have1))
    print('missing_for_rcsb:', len(missing))
    records = []
    stat = Counter()
    for uni in tqdm(missing, desc='rcsb query'):
        uni_q = uni.split('-')[0]
        pdbs = rcsb_query(uni_q)
        stat['queried'] += 1
        if pdbs:
            stat['hit'] += 1
            for pdb in pdbs:
                records.append((uni, pdb, 'rcsb_search'))
        else:
            stat['miss'] += 1
        time.sleep(0.12)
    out = pd.DataFrame(records, columns=['uniprot_id', 'pdb_id', 'source'])
    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_LONG, index=False)
    print('Saved:', OUT_LONG, 'pairs=', len(out))
    print('proteins_hit_by_rcsb:', out['uniprot_id'].nunique() if len(out) else 0)
    print('stats:', dict(stat))
if __name__ == '__main__':
    main()
