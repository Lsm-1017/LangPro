import random
import time
from pathlib import Path
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
IN_PATH = Path('../data_intermediate/ligand_names.csv')
OUT_PATH = Path('../data_intermediate/ligand_smiles.csv')
BASE_SLEEP = 0.6
JITTER_SLEEP = (0.2, 0.8)
MAX_TRIES = 10
BACKOFF_CAP = 120
PCP_TIMEOUT = 30

def _sleep_with_jitter(base: float, jitter_range=(0.0, 0.0)):
    time.sleep(base + random.uniform(*jitter_range))

def _is_retryable_error(e: Exception) -> bool:
    msg = str(e)
    retry_signals = ['503', 'ServerBusy', 'Too many requests', '429', 'timed out', 'Timeout', 'temporarily', 'Temporary', 'Connection', 'RemoteDisconnected', 'Name or service not known', 'Could not resolve host', 'Bad Gateway', '502', '504']
    return any((s in msg for s in retry_signals))

def fetch_smiles_for_name(name: str) -> str | None:
    name = str(name).strip()
    if not name:
        return None
    last_err = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            compounds = pcp.get_compounds(name, 'name', timeout=PCP_TIMEOUT)
        except Exception as e:
            last_err = e
            if _is_retryable_error(e):
                wait = min(BACKOFF_CAP, 2 ** min(attempt, 7)) + random.uniform(0, 2)
                print(f'[WARN] PubChem ({name}) attempt {attempt}/{MAX_TRIES}: {e}')
                print(f'       -> sleep {wait:.1f}s then retry')
                time.sleep(wait)
                continue
            else:
                print(f'[WARN] PubChem ({name}): {e}')
                return None
        if not compounds:
            print(f'[WARN] PubChem :{name}')
            return None
        for c in compounds[:5]:
            smiles = getattr(c, 'canonical_smiles', None) or getattr(c, 'isomeric_smiles', None) or getattr(c, 'smiles', None)
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            return Chem.MolToSmiles(mol, canonical=True)
        if attempt < MAX_TRIES:
            wait = min(10, 1.5 * attempt) + random.uniform(0, 1)
            print(f'[WARN] SMILES RDKit :{name}')
            print(f'       -> sleep {wait:.1f}s then retry (attempt {attempt + 1}/{MAX_TRIES})')
            time.sleep(wait)
            continue
        print(f'[WARN] SMILES:{name}')
        return None
    if last_err is not None:
        print(f'[WARN]{name} last_err={last_err}')
    return None

def main():
    df_names = pd.read_csv(IN_PATH)
    df_names['ligand_name'] = df_names['ligand_name'].astype(str)
    done = {}
    if OUT_PATH.exists():
        try:
            old = pd.read_csv(OUT_PATH)
            if 'ligand_name' in old.columns and 'smiles' in old.columns:
                for _, r in old.iterrows():
                    n = str(r['ligand_name']).strip()
                    s = str(r['smiles']).strip() if pd.notna(r['smiles']) else ''
                    if n and s:
                        done[n] = s
                print(f'[RESUME]{len(done)}Status:')
        except Exception as e:
            print(f'[WARN] OUT{e}')
    total = len(df_names)
    print(f'Status:{total}ligand_name SMILES PubChem')
    rows = []
    for n, s in done.items():
        rows.append({'ligand_name': n, 'smiles': s})
    done_set = set(done.keys())
    for i, name in enumerate(df_names['ligand_name'], start=1):
        name = str(name).strip()
        if not name:
            continue
        if name in done_set:
            continue
        print(f'[{i}/{total}] {name} ...')
        smiles = fetch_smiles_for_name(name)
        if not smiles:
            print('-> SMILES')
        else:
            print(f'  -> SMILES: {smiles[:60]}...')
            rows.append({'ligand_name': name, 'smiles': smiles})
            done_set.add(name)
            lig_df = pd.DataFrame(rows).drop_duplicates(subset=['ligand_name'], keep='first')
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            lig_df.to_csv(OUT_PATH, index=False)
        _sleep_with_jitter(BASE_SLEEP, JITTER_SLEEP)
    lig_df = pd.DataFrame(rows).drop_duplicates(subset=['ligand_name'], keep='first')
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lig_df.to_csv(OUT_PATH, index=False)
    print('SMILES', len(lig_df))
    print('Status:', OUT_PATH)
    print('Status:')
    print(lig_df.head())
if __name__ == '__main__':
    main()
