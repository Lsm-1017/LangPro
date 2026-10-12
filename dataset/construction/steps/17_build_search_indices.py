import time
import json
import re
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
TRIPLES = Path('../data_intermediate/triples_with_ligand_text_full.csv')
UNI_TSV = Path('../data_intermediate/uniprot_index.tsv')
U2PDB = Path('../data_intermediate/uniprot2pdb_long.csv')
AFIDX = Path('../data_intermediate/uniprot2alphafold_index.csv')
LIGCORE = Path('../data_intermediate/ligand_core.csv')
OUT_PROT = Path('../data_intermediate/protein_search_index.csv')
OUT_LIG = Path('../data_intermediate/ligand_search_index.csv')
OUT_PAIR = Path('../data_intermediate/pair_search_index.csv')
PUBCHEM_CID_BY_INCHIKEY = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{}/cids/JSON'
PUBCHEM_SYNONYMS_BY_CID = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/synonyms/JSON'
PUBCHEM_IUPAC_BY_CID = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/IUPACName/JSON'

def norm_space(s: str) -> str:
    return re.sub('\\s+', ' ', s).strip()

def split_gene_names(x):
    if not isinstance(x, str) or not x.strip():
        return []
    parts = re.split('[;\\s]+', x.strip())
    out = []
    for p in parts:
        p = p.strip()
        if p and p.upper() != 'NAN':
            out.append(p)
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq

def protein_primary_name(x):
    if not isinstance(x, str) or not x.strip():
        return ''
    s = x.strip()
    s = s.split(';')[0]
    s = re.sub('\\([^)]*\\)', '', s)
    return norm_space(s)

def parse_ec_numbers(x):
    if not isinstance(x, str) or not x.strip():
        return []
    parts = [p.strip() for p in x.split(';')]
    parts = [p for p in parts if p and p.upper() != 'NAN']
    seen, uniq = (set(), [])
    for p in parts:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

def make_session():
    s = requests.Session()
    s.headers.update({'User-Agent': 'langpro-search-index-builder/1.0'})
    return s

def pubchem_fetch_synonyms(session: requests.Session, inchikey: str, sleep_s: float=0.08, max_syn: int=40):
    inchikey = str(inchikey).strip()
    if not inchikey or inchikey.upper() == 'NAN':
        return (None, [], None, 'no_inchikey')
    try:
        r = session.get(PUBCHEM_CID_BY_INCHIKEY.format(inchikey), timeout=30)
        if r.status_code != 200:
            return (None, [], None, f'cid_http_{r.status_code}')
        js = r.json()
        cids = js.get('IdentifierList', {}).get('CID', [])
        if not cids:
            return (None, [], None, 'cid_empty')
        cid = int(cids[0])
    except Exception:
        return (None, [], None, 'cid_exception')
    time.sleep(sleep_s)
    synonyms = []
    try:
        r2 = session.get(PUBCHEM_SYNONYMS_BY_CID.format(cid), timeout=30)
        if r2.status_code == 200:
            js2 = r2.json()
            synonyms = js2.get('InformationList', {}).get('Information', [{}])[0].get('Synonym', []) or []
        else:
            pass
    except Exception:
        pass
    time.sleep(sleep_s)
    iupac = None
    try:
        r3 = session.get(PUBCHEM_IUPAC_BY_CID.format(cid), timeout=30)
        if r3.status_code == 200:
            js3 = r3.json()
            props = js3.get('PropertyTable', {}).get('Properties', [])
            if props and 'IUPACName' in props[0]:
                iupac = props[0]['IUPACName']
    except Exception:
        pass
    cleaned = []
    seen = set()
    for t in synonyms:
        if not isinstance(t, str):
            continue
        t = norm_space(t)
        if not t:
            continue
        if len(t) > 120:
            continue
        if t.lower() in seen:
            continue
        cleaned.append(t)
        seen.add(t.lower())
        if len(cleaned) >= max_syn:
            break
    return (cid, cleaned, iupac, 'ok')

def main():
    df = pd.read_csv(TRIPLES)
    pair_df = df[['uniprot_id', 'ligand_smiles_canon', 'ligand_name']].dropna(subset=['uniprot_id', 'ligand_smiles_canon']).astype(str)
    pair_df['uniprot_id'] = pair_df['uniprot_id'].str.strip()
    pair_df['ligand_smiles_canon'] = pair_df['ligand_smiles_canon'].str.strip()
    pair_df = pair_df.groupby(['uniprot_id', 'ligand_smiles_canon'])['ligand_name'].agg(lambda x: x.value_counts().index[0]).reset_index()
    print('Unique (protein, ligand) pairs:', len(pair_df))
    uni = pd.read_csv(UNI_TSV, sep='\t')
    uni = uni.rename(columns={'Entry': 'uniprot_id', 'Protein names': 'protein_names', 'Gene Names': 'gene_names', 'EC number': 'ec_number', 'Organism (ID)': 'organism_id', 'Entry Name': 'entry_name'})
    uni['uniprot_id'] = uni['uniprot_id'].astype(str).str.strip()
    uni['protein_primary_name'] = uni['protein_names'].apply(protein_primary_name)
    uni['gene_list'] = uni['gene_names'].apply(split_gene_names)
    uni['ec_list'] = uni['ec_number'].apply(parse_ec_numbers)
    if U2PDB.exists() and U2PDB.stat().st_size > 0:
        u2p = pd.read_csv(U2PDB)
        u2p['uniprot_id'] = u2p['uniprot_id'].astype(str).str.strip()
        u2p['pdb_id'] = u2p['pdb_id'].astype(str).str.strip()
        pdb_agg = u2p.groupby('uniprot_id')['pdb_id'].agg(lambda x: sorted(set(x))).reset_index()
        pdb_agg['has_pdb'] = 1
    else:
        pdb_agg = pd.DataFrame(columns=['uniprot_id', 'pdb_id', 'has_pdb'])
    af = pd.read_csv(AFIDX)
    af['uniprot_id'] = af['uniprot_id'].astype(str).str.strip()
    if 'af_available' not in af.columns:
        af['af_available'] = 0
    prot = uni[['uniprot_id', 'entry_name', 'protein_primary_name', 'protein_names', 'gene_names', 'organism_id', 'protein_primary_name', 'gene_list', 'ec_list']].copy()
    prot = prot.merge(pdb_agg[['uniprot_id', 'pdb_id', 'has_pdb']], on='uniprot_id', how='left')
    prot = prot.merge(af[['uniprot_id', 'af_available', 'cif_url', 'pdb_url', 'pae_url', 'status']], on='uniprot_id', how='left')
    prot['has_pdb'] = prot['has_pdb'].fillna(0).astype(int)
    prot['af_available'] = prot['af_available'].fillna(0).astype(int)
    prot['pdb_id'] = prot['pdb_id'].apply(lambda x: x if isinstance(x, list) else [])

    def build_protein_terms(row):
        terms = []
        uid = row.get('uniprot_id', '')
        if uid:
            terms.append(uid)
        en = row.get('entry_name', '')
        if isinstance(en, str) and en.strip():
            terms.append(en.strip())
        pn = row.get('protein_primary_name', '')
        if isinstance(pn, str) and pn.strip():
            terms.append(pn.strip())
        for g in row.get('gene_list', []) or []:
            terms.append(g)
        for ec in row.get('ec_list', []) or []:
            terms.append(ec)
        seen = set()
        uniq = []
        for t in terms:
            t = norm_space(str(t))
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            uniq.append(t)
            seen.add(key)
        return uniq[:25]
    prot['protein_terms'] = prot.apply(build_protein_terms, axis=1)
    prot['protein_terms_json'] = prot['protein_terms'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    OUT_PROT.parent.mkdir(parents=True, exist_ok=True)
    prot_out = prot[['uniprot_id', 'entry_name', 'protein_primary_name', 'gene_names', 'organism_id', 'has_pdb', 'af_available', 'pdb_id', 'status', 'cif_url', 'pdb_url', 'pae_url', 'protein_terms_json']].copy()
    prot_out.to_csv(OUT_PROT, index=False)
    print('Saved:', OUT_PROT, 'n=', len(prot_out))
    lig = pd.read_csv(LIGCORE)
    lig['ligand_smiles_canon'] = lig['ligand_smiles_canon'].astype(str).str.strip()
    lig['ligand_name'] = lig['ligand_name'].astype(str).str.strip()
    lig['inchikey'] = lig['inchikey'].astype(str).str.strip()
    session = make_session()
    records = []
    for _, r in tqdm(lig.iterrows(), total=len(lig), desc='pubchem synonyms'):
        smi = r['ligand_smiles_canon']
        name = r['ligand_name']
        ik = r['inchikey']
        cid, syns, iupac, st = pubchem_fetch_synonyms(session, ik)
        terms = []
        if name and name.upper() != 'NAN':
            terms.append(name)
        if iupac and isinstance(iupac, str) and iupac.strip():
            terms.append(norm_space(iupac))
        for t in syns:
            terms.append(t)
        seen = set()
        uniq = []
        for t in terms:
            t = norm_space(str(t))
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            uniq.append(t)
            seen.add(key)
        records.append({'ligand_smiles_canon': smi, 'inchikey': ik, 'ligand_name': name, 'ligand_chebi': r.get('ligand_chebi', ''), 'ligand_annotation': r.get('ligand_annotation', ''), 'pubchem_cid': cid if cid is not None else '', 'syn_status': st, 'ligand_terms_json': json.dumps(uniq[:60], ensure_ascii=False)})
    lig_out = pd.DataFrame(records)
    lig_out.to_csv(OUT_LIG, index=False)
    print('Saved:', OUT_LIG, 'n=', len(lig_out))
    print('PubChem syn_status counts:\n', lig_out['syn_status'].value_counts())
    pair = pair_df.merge(prot_out[['uniprot_id', 'has_pdb', 'af_available', 'pdb_id', 'protein_terms_json']], on='uniprot_id', how='left')
    pair = pair.merge(lig_out[['ligand_smiles_canon', 'inchikey', 'ligand_terms_json']], on='ligand_smiles_canon', how='left')
    pair['pair_id'] = pair.apply(lambda x: f"{x['uniprot_id']}__{x['inchikey']}", axis=1)
    pair.to_csv(OUT_PAIR, index=False)
    print('Saved:', OUT_PAIR, 'n=', len(pair))
    print('Pairs has_pdb ratio:', pair['has_pdb'].fillna(0).mean())
    print('Pairs af_available ratio:', pair['af_available'].fillna(0).mean())
if __name__ == '__main__':
    main()
