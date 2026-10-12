import pandas as pd
from pathlib import Path
from rdkit import Chem
CCD_SDF = Path('../CCD/components-pub.sdf')
MY_LIGAND = Path('../data_intermediate/my_ligands_inchikey.csv')
OUT_SDF = Path('../data_intermediate/ligands_ccd_3d_relaxed.sdf')
OUT_INDEX = Path('../data_intermediate/ligands_ccd_index_relaxed.csv')

def inchikey_prefix(ikey: str | None) -> str | None:
    if not isinstance(ikey, str):
        return None
    ikey = ikey.strip()
    if not ikey:
        return None
    return ikey.split('-')[0]

def main():
    my_df = pd.read_csv(MY_LIGAND)
    my_df = my_df.dropna(subset=['inchikey'])
    my_df['ikey_prefix'] = my_df['inchikey'].apply(inchikey_prefix)
    my_df = my_df[my_df['ikey_prefix'].notna()]
    print('ligand ( InChIKey ):', len(my_df))
    prefix_to_smiles = {}
    for _, row in my_df.iterrows():
        pfx = row['ikey_prefix']
        prefix_to_smiles.setdefault(pfx, set()).add(row['ligand_smiles_canon'])
    suppl = Chem.SDMolSupplier(str(CCD_SDF), sanitize=False)
    ccd_hits = {}
    n_valid = 0
    for mol in suppl:
        if mol is None:
            continue
        n_valid += 1
        ikey = None
        for pname in mol.GetPropNames():
            if 'INCHIKEY' in pname.upper():
                ikey = mol.GetProp(pname)
                break
        pfx = inchikey_prefix(ikey)
        if not pfx:
            continue
        if pfx in prefix_to_smiles and pfx not in ccd_hits:
            ccd_hits[pfx] = mol
    print('CCD :', n_valid)
    print('14 CCD prefix :', len(ccd_hits))
    writer = Chem.SDWriter(str(OUT_SDF))
    index_records = []
    for pfx, mol in ccd_hits.items():
        my_smiles_list = sorted(list(prefix_to_smiles.get(pfx, [])))
        my_smiles_str = ' || '.join(my_smiles_list)
        mol.SetProp('_Name', f'MYLIG_PREF_{pfx}')
        mol.SetProp('my_ikey_prefix', pfx)
        mol.SetProp('my_smiles_list', my_smiles_str)
        writer.write(mol)
        index_records.append({'ikey_prefix': pfx, 'my_smiles_list': my_smiles_str})
    writer.close()
    pd.DataFrame(index_records).to_csv(OUT_INDEX, index=False)
    print('CCD( ) 3D :', OUT_SDF)
    print(':', OUT_INDEX)
if __name__ == '__main__':
    main()
