import time
import random
from pathlib import Path
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
BASE = 'https://rest.uniprot.org/uniprotkb/search'
IN_PATH = Path('../data_intermediate/protein_list.csv')
OUT_PATH = Path('../data_intermediate/uniprot_index.tsv')
FIELDS = 'accession,id,protein_name,gene_names,ec,organism_id,xref_pdb,reviewed'
CHUNK_SIZE = 100
TIMEOUT = 90
MAX_TRIES_PER_CHUNK = 30
BACKOFF_CAP = 120

def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=10, connect=10, read=10, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=['GET'], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    return s

def load_done_accessions() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    done = set()
    with OUT_PATH.open('r', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip().split('\t')
        try:
            idx = header.index('Entry')
        except ValueError:
            try:
                idx = header.index('accession')
            except ValueError:
                idx = 0
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) > idx and parts[idx]:
                done.add(parts[idx])
    return done

def fetch_chunk(session: requests.Session, chunk_ids: list[str]) -> str:
    q = '(' + ' OR '.join([f'accession_id:{x}' for x in chunk_ids]) + ')'
    params = {'query': q, 'format': 'tsv', 'fields': FIELDS, 'size': CHUNK_SIZE}
    headers = {'User-Agent': 'langpro-uniprot-index/1.0'}
    last_err = None
    for attempt in range(1, MAX_TRIES_PER_CHUNK + 1):
        try:
            r = session.get(BASE, params=params, headers=headers, timeout=TIMEOUT)
            if r.status_code != 200:
                last_err = RuntimeError(f'HTTP {r.status_code}: {r.text[:200]}')
                wait = min(BACKOFF_CAP, 2 ** min(attempt, 7)) + random.uniform(0, 2)
                print(f'[WARN] HTTP {r.status_code} for chunk, sleep {wait:.1f}s then retry')
                time.sleep(wait)
                continue
            return r.text
        except requests.exceptions.RequestException as e:
            last_err = e
            wait = min(BACKOFF_CAP, 2 ** min(attempt, 7)) + random.uniform(0, 2)
            print(f'[WARN] network/DNS error attempt {attempt}/{MAX_TRIES_PER_CHUNK}: {e}')
            print(f'       -> sleep {wait:.1f}s then retry')
            time.sleep(wait)
    raise RuntimeError(f'Failed to fetch chunk after retries. last_err={last_err}')

def main():
    df = pd.read_csv(IN_PATH)
    if 'uniprot_id' not in df.columns:
        raise ValueError(f'IN_PATH missing column uniprot_id, got columns={list(df.columns)[:20]}')
    proteins = [str(x).strip() for x in df['uniprot_id'].tolist() if str(x).strip()]
    print('proteins:', len(proteins))
    done = load_done_accessions()
    if done:
        print(f'[RESUME] already have {len(done)} accessions in {OUT_PATH}, will skip them.')
    proteins_left = [p for p in proteins if p not in done]
    print('remaining:', len(proteins_left))
    if not proteins_left:
        print('[DONE] nothing to do.')
        return
    session = build_session()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = OUT_PATH.exists()
    fetched_chunks = 0
    for i in range(0, len(proteins_left), CHUNK_SIZE):
        chunk = proteins_left[i:i + CHUNK_SIZE]
        txt = fetch_chunk(session, chunk)
        lines = txt.splitlines()
        if not lines:
            continue
        if not wrote_header:
            with OUT_PATH.open('w', encoding='utf-8') as f:
                f.write(lines[0].rstrip('\n') + '\n')
                for ln in lines[1:]:
                    if ln.strip():
                        f.write(ln.rstrip('\n') + '\n')
            wrote_header = True
        else:
            with OUT_PATH.open('a', encoding='utf-8') as f:
                for ln in lines[1:]:
                    if ln.strip():
                        f.write(ln.rstrip('\n') + '\n')
        fetched_chunks += 1
        if fetched_chunks % 5 == 0:
            print('fetched chunks:', fetched_chunks)
        time.sleep(0.05 + random.uniform(0, 0.1))
    print('[OK] saved:', OUT_PATH)
if __name__ == '__main__':
    main()
