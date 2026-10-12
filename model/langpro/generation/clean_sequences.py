#!/usr/bin/env python3
"""Normalize generated sequences and trim degenerate low-complexity tails."""

import argparse
import os
import re
from collections import Counter

import pandas as pd


def normalize_seq(seq: str) -> str:
    """Remove whitespace and digits without rewriting residue letters."""
    if seq is None:
        return ""
    s = str(seq)
    s = s.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    s = re.sub(r"\d+", "", s)
    return s.strip()


def truncate_from_tail(seq: str, max_len: int = 512) -> str:
    """Keep at most the first ``max_len`` residues."""
    s = normalize_seq(seq)
    if len(s) <= max_len:
        return s
    return s[:max_len]


def max_run_length(seq: str) -> int:
    if not seq:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def trim_exact_repeats_from_tail(
    seq: str,
    min_single_run: int = 6,
    max_motif_len: int = 6,
    min_motif_repeats: int = 4,
) -> str:
    """Trim exact single-residue or short-motif repetitions at the tail."""
    s = normalize_seq(seq)
    if not s:
        return s

    changed = True
    while changed and len(s) > 0:
        changed = False
        n = len(s)

        # Single-residue run.
        last_char = s[-1]
        i = n - 1
        while i >= 0 and s[i] == last_char:
            i -= 1
        run_len = n - 1 - i
        if run_len >= min_single_run:
            s = s[: i + 1]
            changed = True
            continue

        # Repeated short motif.
        best_cut = None
        upper_motif_len = min(max_motif_len, len(s) // max(1, min_motif_repeats))
        for motif_len in range(1, upper_motif_len + 1):
            motif = s[-motif_len:]
            repeats = 1
            pos = len(s) - motif_len

            while pos - motif_len >= 0 and s[pos - motif_len:pos] == motif:
                repeats += 1
                pos -= motif_len

            if repeats >= min_motif_repeats:
                if best_cut is None or pos < best_cut:
                    best_cut = pos

        if best_cut is not None:
            s = s[:best_cut]
            changed = True

    return s


def is_low_complexity_suffix(
    suffix: str,
    min_tail_len: int = 20,
    dominant_ratio_thresh: float = 0.65,
    max_unique_chars: int = 6,
    min_max_run: int = 5,
) -> bool:
    """Return whether a suffix satisfies the low-complexity criteria."""
    if len(suffix) < min_tail_len:
        return False

    cnt = Counter(suffix)
    total = len(suffix)
    dominant_char, dominant_count = cnt.most_common(1)[0]
    dominant_ratio = dominant_count / total
    unique_chars = len(cnt)
    longest_run = max_run_length(suffix)

    return (
        dominant_ratio >= dominant_ratio_thresh
        and unique_chars <= max_unique_chars
        and longest_run >= min_max_run
    )


def trim_low_complexity_tail(
    seq: str,
    min_tail_len: int = 20,
    dominant_ratio_thresh: float = 0.65,
    max_unique_chars: int = 6,
    min_max_run: int = 5,
) -> str:
    """Locate and remove the earliest qualifying low-complexity suffix."""
    s = normalize_seq(seq)
    n = len(s)
    if n < min_tail_len:
        return s

    best_cut = None

    for start in range(n - min_tail_len + 1):
        suffix = s[start:]
        if is_low_complexity_suffix(
            suffix,
            min_tail_len=min_tail_len,
            dominant_ratio_thresh=dominant_ratio_thresh,
            max_unique_chars=max_unique_chars,
            min_max_run=min_max_run,
        ):
            best_cut = start
            break

    if best_cut is not None:
        return s[:best_cut]

    return s


def clean_tail(
    seq: str,
    truncate_len: int = 512,
    min_len: int = 30,
    min_single_run: int = 6,
    max_motif_len: int = 6,
    min_motif_repeats: int = 4,
    low_complexity_min_tail_len: int = 20,
    low_complexity_dominant_ratio: float = 0.65,
    low_complexity_max_unique: int = 6,
    low_complexity_min_run: int = 5,
) -> str:
    raw = normalize_seq(seq)
    if not raw:
        return raw

    # Apply the hard maximum first.
    truncated = truncate_from_tail(raw, max_len=truncate_len)

    # Remove exact repeat tails.
    s = trim_exact_repeats_from_tail(
        truncated,
        min_single_run=min_single_run,
        max_motif_len=max_motif_len,
        min_motif_repeats=min_motif_repeats,
    )

    # Remove broader low-complexity tails.
    s = trim_low_complexity_tail(
        s,
        min_tail_len=low_complexity_min_tail_len,
        dominant_ratio_thresh=low_complexity_dominant_ratio,
        max_unique_chars=low_complexity_max_unique,
        min_max_run=low_complexity_min_run,
    )

    # Fall back to the truncated sequence if cleanup is too aggressive.
    if len(s) < min_len:
        return truncated

    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--out-csv", default=None, help="Output CSV; overwrites input when omitted")
    ap.add_argument("--seq-col", default="pred_sequence", help="Sequence column")

    ap.add_argument("--truncate-len", type=int, default=512, help="Maximum retained length")
    ap.add_argument("--min-len", type=int, default=30, help="Minimum accepted cleaned length")

    # Exact-repeat tail parameters.
    ap.add_argument("--min-single-run", type=int, default=6)
    ap.add_argument("--max-motif-len", type=int, default=6)
    ap.add_argument("--min-motif-repeats", type=int, default=4)

    # Low-complexity tail parameters.
    ap.add_argument("--low-complexity-min-tail-len", type=int, default=20)
    ap.add_argument("--low-complexity-dominant-ratio", type=float, default=0.65)
    ap.add_argument("--low-complexity-max-unique", type=int, default=6)
    ap.add_argument("--low-complexity-min-run", type=int, default=5)

    args = ap.parse_args()

    in_csv = args.csv
    out_csv = args.out_csv if args.out_csv else args.csv

    df = pd.read_csv(in_csv)

    if args.seq_col not in df.columns:
        raise ValueError(f"Column {args.seq_col!r} is missing; available={list(df.columns)}")

    df[args.seq_col] = df[args.seq_col].apply(
        lambda x: clean_tail(
            x,
            truncate_len=args.truncate_len,
            min_len=args.min_len,
            min_single_run=args.min_single_run,
            max_motif_len=args.max_motif_len,
            min_motif_repeats=args.min_motif_repeats,
            low_complexity_min_tail_len=args.low_complexity_min_tail_len,
            low_complexity_dominant_ratio=args.low_complexity_dominant_ratio,
            low_complexity_max_unique=args.low_complexity_max_unique,
            low_complexity_min_run=args.low_complexity_min_run,
        )
    )

    if os.path.dirname(out_csv):
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df.to_csv(out_csv, index=False)
    print(f"Done. total={len(df)}, saved -> {out_csv}")


if __name__ == "__main__":
    main()
