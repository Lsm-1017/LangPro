"""Retrieve AlphaFold structures and compute structural annotations.

The stage derives burial depth, optional DSSP assignments, pocket labels,
and solvent-accessible surface area. It can use the external mkdssp
program when available and otherwise retains the geometry-based features.
Prefer the public pipeline entry point: python -m dataset.expansion.run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request

import numpy as np
import pandas as pd


# ============================================================
# 常量
# ============================================================
AF_API_TEMPLATE = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"

# AA 3-letter → 1-letter mapping
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Van der Waals radii for heavy atoms (Å)
VDW_RADII = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
}

DSSP_CLASSES = {
    "H": 0, "G": 1, "I": 2, "E": 3, "B": 4, "T": 5, "S": 6, "-": 7,
}


# ============================================================
# 3a. AlphaFold 结构下载
# ============================================================
def download_alphafold_structures(
    uniprot_ids: List[str],
    af_dir: Path,
    num_workers: int = 4,
    delay_per_1000: float = 5.0,
) -> Tuple[int, int]:
    """
    下载 AlphaFold 结构文件（PDB 或 CIF）。
    已存在则跳过。
    返回 (downloaded, skipped)。
    """
    af_dir.mkdir(parents=True, exist_ok=True)
    uniq_ids = list(dict.fromkeys(uniprot_ids))  # 去重

    # 先检查已存在的 (AF-{uid}-F1-model_v*.pdb 或 .cif)
    existing = set()
    for u in uniq_ids:
        prefix = f"AF-{u}-F1-model_v"
        for cand in af_dir.glob(prefix + "*"):
            if cand.suffix in (".pdb", ".cif"):
                existing.add(u)
                break
    missing = [u for u in uniq_ids if u not in existing]

    if not missing:
        print(f"[phase3a] All {len(uniq_ids)} structures already cached.")
        return 0, len(uniq_ids)

    print(f"[phase3a] Missing {len(missing)}/{len(uniq_ids)} structures. Downloading ...")

    downloaded = 0
    failed = 0

    _first_errors: List[str] = []
    _max_error_log = 5

    def _get_download_url(uid: str) -> Optional[str]:
        """通过 AlphaFold API 获取 PDB 下载链接，失败返回 None。"""
        api_url = AF_API_TEMPLATE.format(uniprot=uid)
        try:
            import requests
            r = requests.get(api_url, timeout=20,
                           headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            if r.status_code != 200:
                return None
            data = r.json()
            # API 返回结构: [{...}] 或 {...}，pdbUrl / cifUrl / bcifUrl
            items = data if isinstance(data, list) else [data]
            for item in items:
                for key in ("pdbUrl", "cifUrl", "bcifUrl"):
                    url = item.get(key, "")
                    if url:
                        return url
            return None
        except Exception:
            return None

    def _download_one(uid: str) -> Tuple[str, bool]:
        # 1) 通过 API 获取下载链接
        pdb_url = _get_download_url(uid)

        if pdb_url is None:
            if len(_first_errors) < _max_error_log:
                _first_errors.append(f"  {uid}: API returned no pdbUrl")
            return uid, False

        # 从 URL 中提取文件名 (例如 AF-P04637-F1-model_v6.pdb)
        fname = pdb_url.split("/")[-1]
        fpath = af_dir / fname

        # 已存在则跳过
        if fpath.exists():
            return uid, True

        try:
            import requests
            r = requests.get(pdb_url, timeout=60,
                           headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                raise Exception(f"HTTP {r.status_code}")

            fpath.write_bytes(r.content)
            return uid, True
        except Exception as e:
            if len(_first_errors) < _max_error_log:
                _first_errors.append(f"  {uid}: {e}")
            return uid, False

    for i in range(0, len(missing), 1000):
        chunk = missing[i:i + 1000]
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_download_one, uid): uid for uid in chunk}
            for fut in as_completed(futures):
                uid, ok = fut.result()
                if ok:
                    downloaded += 1
                else:
                    failed += 1

        if i + 1000 < len(missing):
            print(f"[phase3a]  progress: {min(i+1000, len(missing))}/{len(missing)}  "
                  f"ok={downloaded} fail={failed}  sleeping {delay_per_1000}s ...")
            time.sleep(delay_per_1000)

    print(f"[phase3a] Done. downloaded={downloaded} failed={failed} skipped={len(existing)}")
    if failed > 0:
        print(f"[phase3a] First {min(_max_error_log, len(_first_errors))} errors:")
        for e in _first_errors:
            print(e)
    return downloaded, failed + len(existing)


# ============================================================
# 结构文件解析 (Bio.PDB)
# ============================================================
def _parse_structure(struct_path: Path) -> Optional[Dict]:
    """解析蛋白结构文件 (PDB 或 CIF)。先尝试 Bio.PDB，失败则用纯 Python。"""
    # 优先使用 Bio.PDB（支持 PDB 和 CIF）
    try:
        from Bio.PDB import PDBParser, MMCIFParser

        if struct_path.suffix.lower() in (".cif", ".mmcif"):
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("prot", str(struct_path))
        else:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("prot", str(struct_path))

        model = structure[0]
        chain = next(model.get_chains())  # 取第一条链

        ca_coords = []
        cb_coords = []
        all_coords = []
        res_ids = []
        res_names = []
        atom_res_idx = []
        atom_elements = []

        for ri, residue in enumerate(chain):
            if residue.id[0] != " ":  # 跳过 HETATM
                continue
            res_names.append(residue.resname.strip())
            res_ids.append(ri)
            has_ca = False
            for atom in residue:
                el = atom.element.strip() if atom.element else atom.name[0]
                if el == "H":
                    continue
                coord = list(atom.coord)
                all_coords.append(coord)
                atom_res_idx.append(ri)
                atom_elements.append(el if el in VDW_RADII else "C")
                if atom.name.strip() == "CA":
                    ca_coords.append(coord)
                    has_ca = True
                if atom.name.strip() == "CB":
                    cb_coords.append(coord)

            if not has_ca:
                ca_coords.append([0.0, 0.0, 0.0])
            if len(cb_coords) <= ri:
                cb_coords.append(ca_coords[-1] if ca_coords else [0.0, 0.0, 0.0])

        N = len(res_ids)
        if N < 1:
            return None

        seq = "".join(AA_3TO1.get(rn, "X") for rn in res_names)

        return {
            "coords_ca": np.array(ca_coords[:N], dtype=np.float64),
            "coords_cb": np.array(cb_coords[:N], dtype=np.float64),
            "coords_all": np.array(all_coords, dtype=np.float64),
            "res_ids": res_ids[:N],
            "res_names": res_names[:N],
            "seq": seq,
            "atom_res_idx": atom_res_idx,
            "atom_elements": atom_elements,
            "N": N,
        }
    except ImportError:
        pass
    except Exception:
        pass

    # 回退：纯 Python PDB 解析
    return _parse_pdb_basic(struct_path)


def _parse_pdb_basic(pdb_path: Path) -> Optional[Dict]:
    """
    纯 Python PDB 解析 (不依赖 Bio.PDB，可独立运行)。
    返回:
      {
        "coords_ca": np.ndarray [N, 3]  # CA 原子坐标
        "coords_cb": np.ndarray [N, 3]  # CB 原子坐标 (GLY 用 CA 代替)
        "coords_all": np.ndarray [M, 3] # 所有重原子坐标
        "res_ids":   List[int]           # 残基序号
        "res_names": List[str]           # 三字母残基名
        "seq":       str                 # 单字母序列
        "atom_res_idx": List[int]        # 每个重原子对应的残基索引
        "atom_elements": List[str]       # 每个重原子的元素
        "N": int                         # 残基数
      }
    失败返回 None。
    """
    lines = pdb_path.read_text(errors="ignore").splitlines()

    ca_coords: List[List[float]] = []
    cb_coords: List[List[float]] = []
    all_coords: List[List[float]] = []
    res_ids: List[int] = []
    res_names: List[str] = []
    atom_res_idx: List[int] = []
    atom_elements: List[str] = []

    current_res = None
    res_idx = -1

    # Gly CB 用 CA 坐标
    _gly_ca: Optional[List[float]] = None

    for line in lines:
        if not line.startswith("ATOM") and not line.startswith("HETATM"):
            continue
        if len(line) < 54:
            continue

        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain = line[21:22]
        res_seq = line[22:26].strip()
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue

        element = atom_name[0] if atom_name else "C"
        if element not in VDW_RADII and element != "H":
            element = "C"

        # 只处理 ATOM 记录，跳过 HETATM 和氢原子
        if line.startswith("HETATM"):
            continue
        if element == "H":
            continue

        this_res = (chain, res_seq, res_name)
        if this_res != current_res:
            # 处理上一个残基的 CB
            if res_idx >= 0 and _gly_ca is not None:
                cb_coords.append(_gly_ca[:])
            current_res = this_res
            res_idx += 1
            res_ids.append(res_idx)
            res_names.append(res_name)
            _gly_ca = None

        coord = [x, y, z]
        all_coords.append(coord)
        atom_res_idx.append(res_idx)
        atom_elements.append(element)

        if atom_name == "CA":
            ca_coords.append(coord)
            _gly_ca = coord  # 缓存 CA 用于 GLY
            if len(cb_coords) < res_idx:
                cb_coords.append(coord)  # 暂时填入 CA
        elif atom_name == "CB":
            # 更新 CB
            if len(cb_coords) <= res_idx:
                cb_coords.append(coord)
            else:
                cb_coords[res_idx] = coord
            _gly_ca = None  # 有 CB 说明不是 GLY

    # 最后一个残基
    if res_idx >= 0 and _gly_ca is not None:
        cb_coords.append(_gly_ca[:])

    N = res_idx + 1
    if N < 1:
        return None

    # 确保 ca_coords/cb_coords 长度一致
    while len(ca_coords) < N:
        ca_coords.append([0.0, 0.0, 0.0])
    while len(cb_coords) < N:
        cb_coords.append(ca_coords[min(len(cb_coords), len(ca_coords) - 1)])

    ca_np = np.array(ca_coords[:N], dtype=np.float64)
    cb_np = np.array(cb_coords[:N], dtype=np.float64)
    all_np = np.array(all_coords, dtype=np.float64)

    seq = "".join(AA_3TO1.get(rn, "X") for rn in res_names[:N])

    return {
        "coords_ca": ca_np,
        "coords_cb": cb_np,
        "coords_all": all_np,
        "res_ids": res_ids[:N],
        "res_names": res_names[:N],
        "seq": seq,
        "atom_res_idx": atom_res_idx,
        "atom_elements": atom_elements,
        "N": N,
    }


# ============================================================
# 3b. 结构特征计算 (纯 Python，无外部依赖)
# ============================================================

def calc_burial_depth_ca(parsed: Dict, radius: float = 12.0) -> List[float]:
    """
    基于 CA-CA 距离的埋藏深度:
      depth[i] = 1 - (n_neighbors[i] / n_total)
    其中 n_neighbors[i] 是 CA_i 在 radius 内的 CA 原子数。
    埋藏越深 → depth 越大。
    """
    ca = parsed["coords_ca"]  # [N, 3]
    N = ca.shape[0]
    if N < 2:
        return [0.5] * N

    # 用 numpy 广播计算距离矩阵的上三角 → 避免 O(N^3) 内存
    # 但 1000+ 残基的蛋白矩阵可能很大，改用逐点计算
    depths = []
    for i in range(N):
        dists = np.sqrt(((ca - ca[i]) ** 2).sum(axis=1))
        n_neighbors = int((dists < radius).sum()) - 1
        # 归一化：最多 N-1 个邻居
        depth = 1.0 - (n_neighbors / max(N - 1, 1))
        depths.append(round(depth, 4))
    return depths


def calc_sasa_ca(parsed: Dict, probe_radius: float = 1.4) -> List[float]:
    """
    简化的溶剂可及表面积 (SASA)：基于 CA 坐标 + 近似球体。
    使用 Shrake-Rupley 的简化版：每个残基的暴露比例。
    """
    ca = parsed["coords_ca"]
    N = ca.shape[0]
    if N < 1:
        return [0.0] * N

    # 简化：计算每个 CA 的局部密度
    densities = []
    for i in range(N):
        dists = np.sqrt(((ca - ca[i]) ** 2).sum(axis=1))
        # 在 10Å 内的邻居越多 → ASA 越低
        n_local = int((dists < 10.0).sum())
        density = n_local / N
        # 模拟 ASA: 50 * (1 - density) 大致范围 0-50
        asa = max(0.0, 50.0 * (1.0 - density))
        densities.append(round(asa, 2))
    return densities


def calc_pocket_labels(parsed: Dict, sasa: List[float],
                       depth: List[float]) -> List[int]:
    """
    口袋残基检测:
      1. 残基有一定的埋藏性 (depth > 0.3)
      2. 但不完全埋藏 (depth < 0.9)
      3. SASA 不太高 (sasa < 20)
      4. 周围残基中有 SASA 更高的（处于口袋边缘）

    返回 [0/1]。
    """
    N = parsed["N"]
    ca = parsed["coords_ca"]
    labels = [0] * N

    for i in range(N):
        if depth[i] < 0.25 or depth[i] > 0.88:
            continue
        if sasa[i] > 25.0:
            continue
        # 检查附近是否有暴露残基
        dists = np.sqrt(((ca - ca[i]) ** 2).sum(axis=1))
        nearby = dists < 8.0
        nearby_sasa = [sasa[j] for j in range(N) if nearby[j] and j != i]
        if nearby_sasa and max(nearby_sasa) > 15.0:
            labels[i] = 1

    return labels


def calc_dssp_via_subprocess(pdb_path: Path) -> Tuple[Optional[List[int]], Optional[List[float]]]:
    """
    通过调用外部 mkdssp/dssp 二进制计算 DSSP。
    返回 (dssp_classes: List[int] 可选, burial: List[float] 可选)。
    找不到工具或失败则返回 (None, None)。
    """
    import subprocess

    for cmd_name in ["mkdssp", "dssp", "dsspcmbi"]:
        try:
            proc = subprocess.run(
                [cmd_name, str(pdb_path)],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0 and proc.stdout:
                dssp_classes, burial = _parse_dssp_output(proc.stdout)
                return dssp_classes, burial
        except Exception:
            continue
    return None, None


def _parse_dssp_output(dssp_text: str) -> Tuple[List[int], List[float]]:
    """Parse DSSP output text, return (classes, burial)."""
    dssp_classes: List[int] = []
    burial: List[float] = []

    in_data = False
    for line in dssp_text.splitlines():
        if line.startswith("  #  RESIDUE"):
            in_data = True
            continue
        if not in_data:
            continue
        if len(line) < 110:
            continue

        try:
            ss = line[16]
            if ss == " ":
                ss = "-"
            dssp_classes.append(DSSP_CLASSES.get(ss, 7))

            # solvent accessibility in DSSP column
            acc_str = line[34:38].strip()
            acc = float(acc_str) if acc_str else 0.0
            burial.append(round(acc, 2))
        except (ValueError, IndexError):
            continue

    return dssp_classes, burial


# ============================================================
# 单条蛋白计算
# ============================================================
def process_one_protein(
    uniprot_id: str,
    af_dir: Path,
    use_dssp: bool = True,
) -> Tuple[str, Optional[Dict]]:
    """
    处理一个蛋白的结构。
    返回 (uniprot_id, features_dict) 或 (uniprot_id, None)。
    features_dict:
      {
        "burial_depth":     "[0.5,0.6,...]"  JSON string
        "dssp":             "[3,0,...]"      JSON string (若可用)
        "pocket_labels":    "[0,1,...]"      JSON string
        "pocket_asa":       "[12.3,8.5,...]" JSON string
        "af_pdb_path":      "AF-xxx.pdb"              pdb文件名
        "af_residue_count": 310                       残基数
      }
    """
    # 查找结构文件 (v4/v6, .pdb/.cif 均可)
    pdb_path = None
    prefix = f"AF-{uniprot_id}-F1-model_v"
    for cand in af_dir.glob(prefix + "*"):
        if cand.suffix in (".pdb", ".cif"):
            pdb_path = cand
            break
    if pdb_path is None:
        return uniprot_id, None

    parsed = _parse_structure(pdb_path)
    if parsed is None or parsed["N"] < 1:
        return uniprot_id, None

    # ---- burial_depth (CA 距离法，始终可用) ----
    ca_depth = calc_burial_depth_ca(parsed)

    # ---- SASA (简化) ----
    sasa = calc_sasa_ca(parsed)

    # ---- pocket_labels ----
    pocket_lbl = calc_pocket_labels(parsed, sasa, ca_depth)

    # ---- DSSP (尝试) ----
    dssp_classes = None
    dssp_burial = None
    if use_dssp:
        dssp_classes, dssp_burial = calc_dssp_via_subprocess(pdb_path)

    # ----- 合并为 JSON -----
    result = {
        "burial_depth":      json.dumps(ca_depth),
        "pocket_labels":     json.dumps(pocket_lbl),
        "pocket_asa":        json.dumps(sasa),
        "af_residue_count":  parsed["N"],
        "af_pdb_path":       pdb_path.name,
    }

    if dssp_classes is not None:
        result["dssp"] = json.dumps(dssp_classes)
    if dssp_burial is not None:
        result["dssp_burial"] = json.dumps(dssp_burial)

    return uniprot_id, result


# ============================================================
# 批量处理
# ============================================================
def compute_features_batch(
    df: pd.DataFrame,
    af_dir: Path,
    output_csv: Path,
    num_workers: int = 4,
    use_dssp: bool = True,
    log_interval: int = 500,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """批量计算结构特征，结果写回 df，保存到 CSV。"""
    uniprot_ids = df["uniprot_id"].tolist()
    if limit:
        uniprot_ids = uniprot_ids[:limit]

    n_total = len(uniprot_ids)
    print(f"[phase3b] Computing features for {n_total} proteins ...")
    print(f"[phase3b] AF dir: {af_dir}")
    print(f"[phase3b] workers: {num_workers}, use_dssp: {use_dssp}")

    results: Dict[str, Dict] = {}
    failed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(process_one_protein, uid, af_dir, use_dssp): uid
            for uid in uniprot_ids
        }

        for i, fut in enumerate(as_completed(futures)):
            uid, feat = fut.result()
            if feat is not None:
                results[uid] = feat
            else:
                failed += 1

            if (i + 1) % log_interval == 0:
                elapsed = time.time() - t_start
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_total - i - 1) / speed if speed > 0 else 0
                print(f"[phase3b] {i+1}/{n_total}  ok={len(results)} fail={failed}  "
                      f"speed={speed:.1f}/s  eta={eta/60:.1f}min")

    # 合并结果
    burial_list = []
    dssp_list = []
    dssp_burial_list = []
    pkt_list = []
    asa_list = []
    af_count_list = []
    af_path_list = []

    for uid in df["uniprot_id"]:
        if limit and len(burial_list) >= limit:
            break

        feat = results.get(uid, {})
        burial_list.append(feat.get("burial_depth", ""))
        dssp_list.append(feat.get("dssp", ""))
        dssp_burial_list.append(feat.get("dssp_burial", ""))
        pkt_list.append(feat.get("pocket_labels", ""))
        asa_list.append(feat.get("pocket_asa", ""))
        af_count_list.append(feat.get("af_residue_count", 0))
        af_path_list.append(feat.get("af_pdb_path", ""))

    df_out = df.copy()
    if limit:
        df_out = df_out.iloc[:limit].copy()

    df_out["burial_depth"] = burial_list
    df_out["pocket_labels"] = pkt_list
    df_out["pocket_asa"] = asa_list
    df_out["af_residue_count"] = af_count_list
    df_out["af_pdb_path"] = af_path_list

    if any(d for d in dssp_list if d):
        df_out["dssp"] = dssp_list
    if any(d for d in dssp_burial_list if d):
        df_out["dssp_burial"] = dssp_burial_list

    # 统计
    n_success = len(results)
    n_af_missing = failed
    print(f"\n[phase3b] Completed.")
    print(f"[phase3b]   success: {n_success}/{n_total}")
    print(f"[phase3b]   AF missing: {n_af_missing}")
    print(f"[phase3b]   DSSP available: {bool(any(d for d in dssp_list if d))}")

    if n_success > 0:
        sample = results[next(iter(results))]
        print(f"[phase3b]   sample burial_depth: {json.loads(sample['burial_depth'])[:5]}...")
        print(f"[phase3b]   sample pocket_labels: {json.loads(sample['pocket_labels'])[:5]}...")
        print(f"[phase3b]   total pocket residues in sample: "
              f"{sum(json.loads(sample['pocket_labels']))}")

    df_out.to_csv(output_csv, index=False)
    print(f"[phase3b]   saved to {output_csv}")

    elapsed = time.time() - t_start
    print(f"[phase3b]   total time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
    return df_out


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Retrieve AlphaFold structures and compute structural features."
    )
    ap.add_argument("--download_af", action="store_true",
                    help="下载 AlphaFold 结构 (需要联网)")
    ap.add_argument("--compute_features", action="store_true",
                    help="计算结构特征")
    ap.add_argument("--input_csv", type=str, required=True,
                    help="Input dataset CSV.")
    ap.add_argument("--output_csv", type=str, required=True,
                    help="Output CSV for --compute_features.")
    ap.add_argument("--af_dir", type=str, required=True,
                    help="AlphaFold 结构 PDB 目录")
    ap.add_argument("--num_workers", type=int, default=4,
                    help="并行 worker 数")
    ap.add_argument("--use_dssp", action="store_true", default=True,
                    help="尝试使用外部 DSSP 工具 (需要安装 mkdssp)")
    ap.add_argument("--no_dssp", action="store_false", dest="use_dssp",
                    help="不使用外部 DSSP，仅用纯 Python 方法")
    ap.add_argument("--limit", type=int, default=None,
                    help="只处理前 N 条 (用于测试)")
    ap.add_argument("--log_interval", type=int, default=500,
                    help="日志间隔")
    args = ap.parse_args()

    if not (args.download_af or args.compute_features):
        ap.print_help()
        print("\n[ERROR] 必须指定 --download_af 或 --compute_features")
        sys.exit(1)

    af_dir = Path(args.af_dir)
    input_csv = Path(args.input_csv)

    if not input_csv.exists():
        print(f"[ERROR] 输入 CSV 不存在: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    print(f"[phase3] Read {len(df)} rows from {input_csv}")

    if "uniprot_id" not in df.columns:
        print("[ERROR] CSV 缺少 uniprot_id 列")
        sys.exit(1)

    # ---- 3a: 下载 ----
    if args.download_af:
        uniq = df["uniprot_id"].dropna().unique().tolist()
        print(f"[phase3a] {len(uniq)} unique UniProt IDs to download")
        download_alphafold_structures(
            uniq, af_dir, num_workers=args.num_workers,
        )

    # ---- 3b: 计算 ----
    if args.compute_features:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        compute_features_batch(
            df=df,
            af_dir=af_dir,
            output_csv=output_csv,
            num_workers=args.num_workers,
            use_dssp=args.use_dssp,
            log_interval=args.log_interval,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
