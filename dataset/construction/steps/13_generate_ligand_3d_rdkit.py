import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
TRIPLE_PATH = Path('../data_intermediate/triples_with_ligand_text_full.csv')
SDF_OUT = Path('../data_intermediate/ligands_3d_rdkit.sdf')
NPZ_OUT = Path('../data_intermediate/ligands_3d_rdkit.npz')
INDEX_OUT = Path('../data_intermediate/ligands_3d_rdkit_index.csv')
EMBED_TRIES = 20
NUM_CONFS = 10
RNG_SEED = 42

def build_rdkit_3d(smiles: str):
    if not isinstance(smiles, str):
        return (None, None)
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        return (None, None)
    if mol0.GetNumAtoms() <= 2:
        conf = Chem.Conformer(mol0.GetNumAtoms())
        for i in range(mol0.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(0.0, 0.0, float(i)))
        mol0.AddConformer(conf, assignId=True)
        return (mol0, 0)
    mol = Chem.AddHs(mol0)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = RNG_SEED
    ps.numThreads = 0
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFS, params=ps)
    if len(cids) == 0:
        return (None, None)
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    energies = np.array([e for ok, e in res])
    min_idx = int(energies.argmin())
    min_conf_id = cids[min_idx]
    return (mol, min_conf_id)

def main():
    df = pd.read_csv(TRIPLE_PATH)
    smiles_list = df['ligand_smiles_canon'].dropna().astype(str).unique()
    print('SMILES :', len(smiles_list))
    writer = Chem.SDWriter(str(SDF_OUT))
    z_list = []
    coord_list = []
    smiles_idx = []
    src_list = []
    index_records = []
    for i, smi in enumerate(smiles_list):
        mol, conf_id = build_rdkit_3d(smi)
        if mol is None:
            print(f'[WARN] 3D :{smi}')
            continue
        mol.SetProp('_Name', f'LIG_{i}')
        mol.SetProp('smiles', smi)
        writer.write(mol, confId=conf_id)
        conf = mol.GetConformer(conf_id)
        zs = []
        coords = []
        for atom in mol.GetAtoms():
            zs.append(atom.GetAtomicNum())
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
        z_arr = np.array(zs, dtype=np.int64)
        coord_arr = np.array(coords, dtype=np.float32)
        z_list.append(z_arr)
        coord_list.append(coord_arr)
        smiles_idx.append(smi)
        src_list.append('rdkit_etkdg')
        index_records.append({'idx': len(z_list) - 1, 'ligand_smiles_canon': smi, 'source': 'rdkit_etkdg', 'num_atoms': len(z_arr)})
        if (i + 1) % 50 == 0:
            print(f'Status:{i + 1}/{len(smiles_list)}Status:')
    writer.close()
    Z_arr = np.array(z_list, dtype=object)
    coords_arr = np.array(coord_list, dtype=object)
    smiles_arr = np.array(smiles_idx, dtype=object)
    source_arr = np.array(src_list, dtype=object)
    print('3D', len(Z_arr))
    np.savez_compressed(NPZ_OUT, Z=Z_arr, coords=coords_arr, smiles=smiles_arr, source=source_arr)
    pd.DataFrame(index_records).to_csv(INDEX_OUT, index=False)
    print('SDF :', SDF_OUT)
    print('NPZ :', NPZ_OUT)
    print(':', INDEX_OUT)
if __name__ == '__main__':
    main()
