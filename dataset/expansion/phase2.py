"""Compute residue-level conservation scores with ESM-2.

For each wild-type residue, the stage standardizes its model logit against
the amino-acid logits at the same position and maps the score to [0, 1].
The resulting JSON vector is stored in conservation_scores. Prefer the
public pipeline entry point: python -m dataset.expansion.run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Dataset
# -----------------------------
class ProteinSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_col: str = "protein_sequence"):
        self.seqs = df[seq_col].tolist()

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> Tuple[int, str]:
        return i, str(self.seqs[i] or "")


# -----------------------------
# ESM batch converter wrapper
# -----------------------------
class ESMBatchConverter:
    """Wraps ESM alphabet's batch_converter for DataLoader collation."""

    def __init__(self, alphabet):
        self.converter = alphabet.get_batch_converter()

    def __call__(self, batch):
        indices = [item[0] for item in batch]
        labels = [str(idx) for idx in indices]
        seqs = [item[1] for item in batch]
        data = list(zip(labels, seqs))
        _, _, tokens = self.converter(data)
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        return tokens, lengths, torch.tensor(indices, dtype=torch.long)


# -----------------------------
# Core computation
# -----------------------------
@torch.no_grad()
def compute_conservation_zscore(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> List[str]:
    """
    tokens: [B, L+2] — includes BOS at pos 0, EOS at pos L+1
    lengths: [B] — real AA count (L)
    Returns: List of JSON strings, each a list of floats in [0, 1], length = L.
    """
    B = tokens.size(0)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[], return_contacts=False)
    logits = out["logits"]  # [B, L+2, V]

    results: List[str] = []

    for b in range(B):
        L = int(lengths[b].item())
        if L < 1:
            results.append(json.dumps([]))
            continue

        scores = []
        for pos in range(1, L + 1):
            # logits at this position
            l = logits[b, pos]  # [V]
            mean_l = l.mean()
            std_l = l.std() + 1e-8

            # z-score for wildtype amino acid
            wt_token = int(tokens[b, pos].item())
            z = (l[wt_token] - mean_l) / std_l
            z_clamped = float(z.clamp(-3.0, 3.0).item())

            # map z-score from [-3, 3] to [0, 1] via linear scaling
            normalized = (z_clamped + 3.0) / 6.0  # in [0, 1]
            scores.append(round(normalized, 6))

        results.append(json.dumps(scores))

    return results


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seq_col", type=str, default="protein_sequence")
    ap.add_argument("--esm_model", type=str, default="esm2_t33_650M_UR50D",
                    help="ESM-2 model name. Smaller options: esm2_t30_150M_UR50D, esm2_t12_35M_UR50D")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_interval", type=int, default=200)
    ap.add_argument("--esm_cache_dir", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    # ---- Validate ----
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[phase2] ERROR: input not found: {input_path}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[phase2] device = {device}")

    # ---- 设置 ESM 模型下载目录 (必须在 import esm 之前) ----
    if args.esm_cache_dir:
        cache_dir = Path(args.esm_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(cache_dir)
        print(f"[phase2] TORCH_HOME = {cache_dir}")
        print(f"[phase2] 模型将缓存至: {cache_dir / 'hub' / 'checkpoints'}")

    # ---- Import ESM (在 TORCH_HOME 设置之后) ----
    try:
        import esm
    except ImportError:
        print("[phase2] ERROR: 'fair-esm' package not installed.")
        print("[phase2] Install:  pip install fair-esm")
        sys.exit(1)

    # ---- Load model + alphabet ----
    print(f"[phase2] loading ESM model: {args.esm_model}")

    try:
        # fair-esm API: load_model_and_alphabet(model_name)
        model, alphabet = esm.pretrained.load_model_and_alphabet(args.esm_model)
    except AttributeError:
        # Alternative API: direct function call
        try:
            model, alphabet = esm.pretrained.load_model_and_alphabet_hub(args.esm_model)
        except Exception:
            print(f"[phase2] ERROR: Cannot load model '{args.esm_model}'.")
            print("[phase2] Make sure 'fair-esm' is installed: pip install fair-esm")
            print("[phase2] Or try a different model name from ESM Atlas.")
            sys.exit(1)
    except Exception as e:
        print(f"[phase2] ERROR loading model: {e}")
        print("[phase2] Available: esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, "
              "esm2_t30_150M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D")
        sys.exit(1)

    model = model.to(device)
    model.eval()
    print(f"[phase2] model loaded. Embed dim = {model.embed_dim}, layers = {model.num_layers}")

    # ---- Read data ----
    print(f"[phase2] reading {input_path}")
    df = pd.read_csv(input_path)
    n_total = len(df)
    print(f"[phase2] total rows = {n_total}")

    if args.seq_col not in df.columns:
        print(f"[phase2] ERROR: column '{args.seq_col}' not found. "
              f"Available: {list(df.columns)}")
        sys.exit(1)

    # Pre-truncate
    max_len = args.max_seq_len
    original_lengths = df[args.seq_col].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    n_truncated = (original_lengths > max_len).sum()
    if n_truncated > 0:
        print(f"[phase2] truncating {n_truncated} sequences (> {max_len} aa)")

    df[args.seq_col] = df[args.seq_col].apply(
        lambda x: str(x)[:max_len] if pd.notna(x) else ""
    )

    # Sort by length for efficient batching
    df["_sort_len"] = df[args.seq_col].apply(lambda x: len(x) if pd.notna(x) else 0)
    df = df.sort_values("_sort_len").reset_index(drop=True)
    seq_col = args.seq_col

    # ---- DataLoader ----
    ds = ProteinSequenceDataset(df, seq_col=seq_col)
    collator = ESMBatchConverter(alphabet)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collator, pin_memory=True,
    )

    # ---- Compute ----
    all_conservation: List[str] = [""] * n_total
    total_batches = len(loader)

    print(f"[phase2] computing conservation scores over {total_batches} batches "
          f"(~{n_total} seqs) ...")
    t_start = time.time()

    for bi, (batch_tokens, lengths, indices) in enumerate(loader):
        json_list = compute_conservation_zscore(
            batch_tokens, lengths, model, device,
        )

        for idx, jstr in zip(indices.tolist(), json_list):
            all_conservation[idx] = jstr

        if (bi + 1) % args.log_interval == 0:
            elapsed = time.time() - t_start
            processed = (bi + 1) * args.batch_size
            speed = processed / elapsed if elapsed > 0 else 0
            remaining = n_total - processed
            eta = remaining / speed if speed > 0 else 0
            print(f"[phase2] batch {bi+1}/{total_batches}  "
                  f"processed~{min(processed, n_total)}/{n_total}  "
                  f"speed={speed:.1f} seq/s  eta={eta/60:.1f}min")

    # Drop sort column, assign results
    df = df.drop(columns=["_sort_len"])
    df["conservation_scores"] = all_conservation

    # ---- Stats ----
    non_empty = sum(1 for x in all_conservation if x)
    print(f"[phase2] non-empty conservation scores: {non_empty}/{n_total}")

    if non_empty > 0:
        example = next(x for x in all_conservation if x)
        example_parsed = json.loads(example)
        print(f"[phase2] example (len={len(example_parsed)}): "
              f"min={min(example_parsed):.4f}  max={max(example_parsed):.4f}  "
              f"mean={np.mean(example_parsed):.4f}")
        print(f"[phase2] first 10: {example_parsed[:10]}")

    # ---- Save ----
    df.to_csv(output_path, index=False)
    elapsed_total = time.time() - t_start
    print(f"[phase2] saved to {output_path}  ({n_total} rows, {len(df.columns)} cols)")
    print(f"[phase2] total time: {elapsed_total/60:.1f} min ({elapsed_total/3600:.2f} hr)")
    print("[phase2] DONE.")


if __name__ == "__main__":
    main()
