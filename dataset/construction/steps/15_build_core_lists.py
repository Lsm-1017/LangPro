import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import inchi
IN_MAIN = Path('../data_intermediate/triples_with_ligand_text_full.csv')
OUT_PROT = Path('../data_intermediate/protein_list.csv')
OUT_LIG = Path('../data_intermediate/ligand_core.csv')

def smiles_to_inchikey(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        ik = inchi.MolToInchiKey(m)
        return ik
    except Exception:
        return None

def main():
    df = pd.read_csv(IN_MAIN)
    prots = df['uniprot_id'].dropna().astype(str).unique()
    pd.DataFrame({'uniprot_id': sorted(prots)}).to_csv(OUT_PROT, index=False)
    print('Saved:', OUT_PROT, 'n=', len(prots))
    lig = df[['ligand_smiles_canon', 'ligand_name', 'ligand_chebi', 'ligand_annotation']].copy()
    lig['ligand_smiles_canon'] = lig['ligand_smiles_canon'].astype(str)
    lig_u = lig.drop_duplicates(subset=['ligand_smiles_canon']).reset_index(drop=True)
    lig_u['inchikey'] = lig_u['ligand_smiles_canon'].apply(smiles_to_inchikey)
    print('Unique ligands:', len(lig_u))
    print('InChIKey ok:', lig_u['inchikey'].notna().sum(), 'fail:', lig_u['inchikey'].isna().sum())
    lig_u.to_csv(OUT_LIG, index=False)
    print('Saved:', OUT_LIG)
    print(lig_u.head(5))
if __name__ == '__main__':
    main()
