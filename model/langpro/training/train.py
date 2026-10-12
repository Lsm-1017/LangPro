"""Train LangPro from paired condition/protein data and ligand coordinates."""

from __future__ import annotations

import os
import re
import json
import math
import time
import random
import argparse
import gc
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from typing import Optional

from transformers import AutoTokenizer
from transformers.optimization import Adafactor


def autocast_context(args, device: torch.device):
    """Return a fresh mixed-precision context without changing legacy defaults."""
    amp_dtype = str(getattr(args, "amp_dtype", "none")).lower()
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


PHASE3_CHEM_COLUMNS = [
    "chem_MW",
    "chem_logP",
    "chem_HBD",
    "chem_HBA",
    "chem_TPSA",
    "chem_charge",
    "chem_aromatic_ratio",
    "chem_rotatable_bonds",
    "chem_num_rings",
    "chem_num_heavy_atoms",
]


CHEM_FEATURE_SCALE = {
    "chem_MW": ("log", 5.8, 0.45),
    "chem_logP": ("linear", 2.5, 2.5),
    "chem_HBD": ("log", 1.0, 0.7),
    "chem_HBA": ("log", 1.8, 0.7),
    "chem_TPSA": ("log", 4.2, 0.65),
    "chem_charge": ("linear", 0.0, 2.0),
    "chem_aromatic_ratio": ("linear", 0.35, 0.35),
    "chem_rotatable_bonds": ("log", 1.6, 0.8),
    "chem_num_rings": ("log", 1.0, 0.7),
    "chem_num_heavy_atoms": ("log", 3.0, 0.55),
}


TEXT_CHUNK_LEN_DEFAULTS = {
    "docking": 256,
    "function": 256,
    "ligand_text": 192,
    "smiles": 192,
}

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {residue: index for index, residue in enumerate(AA_ORDER)}

RECONSTRUCTIBLE_PRETRAINED_PREFIXES = (
    "encoder.bert_dock.model.",
    "encoder.bert_func.model.",
    "encoder.bert_ltxt.model.",
    "encoder.smiles_lm.model.",
    "decoder.backbone.",
    "decoder.blocks.",
    "decoder.input_embeddings.",
)
OPTIONAL_NEW_HEAD_PREFIXES = (
    "protein_length_head.",
    "structure_contact_left.",
    "structure_contact_right.",
    "fold_planner.",
    "structure_token_planner.",
    "decoder.fold_plan_proj.",
)


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_full_injection_stop_choices(value: str) -> List[int]:
    """Parse a comma-separated injection curriculum, preserving ``-1`` as full length."""
    text = str(value or "").strip()
    if not text:
        return []
    choices = [int(item.strip()) for item in text.split(",") if item.strip()]
    if any(item < -1 for item in choices):
        raise ValueError("--full-injection-train-stop-choices accepts -1 or non-negative token counts")
    return choices


def parse_full_injection_scale_choices(value: str) -> List[float]:
    """Parse non-negative full-memory injection scales sampled per minibatch."""
    text = str(value or "").strip()
    if not text:
        return []
    choices = [float(item.strip()) for item in text.split(",") if item.strip()]
    if any(not math.isfinite(item) or item < 0.0 for item in choices):
        raise ValueError("--full-injection-train-scale-choices accepts finite non-negative values")
    return choices


def parse_snapshot_steps(value: str) -> List[int]:
    text = str(value or "").strip()
    if not text:
        return []
    steps = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if any(step <= 0 for step in steps):
        raise ValueError("--snapshot-steps accepts comma-separated positive steps")
    return steps


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_slot_residue_composition(
    sequences: List[str], num_slots: int
) -> torch.Tensor:
    """Pool target residue composition into ordered normalized-length slots."""
    target = torch.zeros(len(sequences), int(num_slots), len(AA_ORDER), dtype=torch.float32)
    for batch_index, raw_sequence in enumerate(sequences):
        sequence = str(raw_sequence).strip().upper()
        length = len(sequence)
        if length == 0:
            continue
        for slot in range(int(num_slots)):
            start = (slot * length) // int(num_slots)
            end = max(start + 1, ((slot + 1) * length) // int(num_slots))
            for residue in sequence[start:min(end, length)]:
                residue_index = AA_TO_INDEX.get(residue)
                if residue_index is not None:
                    target[batch_index, slot, residue_index] += 1.0
            target[batch_index, slot] /= target[batch_index, slot].sum().clamp_min(1.0)
    return target


def build_slot_scalar_targets(
    values: List[Optional[List[float]]], sequences: List[str], num_slots: int
) -> torch.Tensor:
    """Max-pool per-residue geometric signals into normalized-length slots."""
    target = torch.zeros(len(sequences), int(num_slots), dtype=torch.float32)
    for batch_index, (row_values, raw_sequence) in enumerate(zip(values, sequences)):
        if row_values is None:
            continue
        length = min(len(str(raw_sequence).strip()), len(row_values))
        if length <= 0:
            continue
        row = torch.as_tensor(row_values[:length], dtype=torch.float32)
        row = torch.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        for slot in range(int(num_slots)):
            start = (slot * length) // int(num_slots)
            end = max(start + 1, ((slot + 1) * length) // int(num_slots))
            target[batch_index, slot] = row[start:min(end, length)].max()
    return target


def checkpoint_model_state(model: nn.Module, full_checkpoint: bool = False) -> Dict[str, torch.Tensor]:
    """Return a compact state dict by omitting locally reconstructible backbones."""
    state = model.state_dict()
    if full_checkpoint:
        return state
    trainable_keys = {name for name, param in model.named_parameters() if param.requires_grad}
    return {
        key: value
        for key, value in state.items()
        if (
            key in trainable_keys
            or not any(key.startswith(prefix) for prefix in RECONSTRUCTIBLE_PRETRAINED_PREFIXES)
        )
    }


def load_checkpoint_model_state(model: nn.Module, state: Dict[str, torch.Tensor], compact: bool) -> None:
    incompatible = model.load_state_dict(state, strict=not compact)
    if not compact:
        return
    bad_missing = [
        key for key in incompatible.missing_keys
        if not any(
            key.startswith(prefix)
            for prefix in RECONSTRUCTIBLE_PRETRAINED_PREFIXES + OPTIONAL_NEW_HEAD_PREFIXES
        )
    ]
    # Compact checkpoints may contain an optional planner/head that is
    # intentionally disabled in the new run.  The inverse case (a newly
    # enabled optional planner missing from an older checkpoint) is already
    # accepted above.  Keep both directions symmetric while remaining strict
    # for the shared encoder/perceiver/decoder surface.
    bad_unexpected = [
        key for key in incompatible.unexpected_keys
        if not any(key.startswith(prefix) for prefix in OPTIONAL_NEW_HEAD_PREFIXES)
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Compact checkpoint mismatch: "
            f"missing={bad_missing[:20]} unexpected={bad_unexpected[:20]}"
        )


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def split_df(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    group_by: str = "row",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministically split rows, optionally keeping groups in one split.

    ``group_by`` accepts ``row`` (the historical behavior) or a comma-separated
    list of dataframe columns, such as ``uniprot_id,inchikey``. Missing group
    values become row-unique so unrelated incomplete records are not collapsed.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    group_columns = [x.strip() for x in str(group_by).split(",") if x.strip()]
    if not group_columns or group_columns == ["row"]:
        shuffled = df.sample(frac=1.0, random_state=seed)
        n_train = int(len(shuffled) * train_ratio)
        n_val = int(len(shuffled) * val_ratio)
        return (
            shuffled.iloc[:n_train].copy(),
            shuffled.iloc[n_train:n_train + n_val].copy(),
            shuffled.iloc[n_train + n_val:].copy(),
        )

    missing = [col for col in group_columns if col not in df.columns]
    if missing:
        raise KeyError(f"--split_group columns missing from dataset: {missing}")

    work = df.copy()
    row_ids = work.index.to_series().astype(str)
    parts = []
    for col in group_columns:
        values = work[col].where(work[col].notna(), "").astype(str).str.strip()
        # A missing key must not merge all missing records into one giant group.
        values = values.mask(values.eq(""), "__missing_row__" + row_ids)
        parts.append(values)
    work["_split_group"] = parts[0]
    for values in parts[1:]:
        work["_split_group"] = work["_split_group"] + "||" + values

    groups = list(work.groupby("_split_group", sort=False).groups.values())
    active_splits = sum(x > 0 for x in (train_ratio, val_ratio, test_ratio))
    if len(groups) < active_splits:
        raise ValueError(
            f"Need at least {active_splits} groups for split_group={group_by!r}, found {len(groups)}."
        )
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    targets = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64) * len(work)
    assigned: List[List[Any]] = [[], [], []]
    counts = np.zeros(3, dtype=np.int64)
    for group_idx in groups:
        size = len(group_idx)
        # Assign the next whole group to the currently most under-filled split.
        split_idx = int(np.argmax(targets - counts))
        assigned[split_idx].extend(group_idx)
        counts[split_idx] += size

    return tuple(work.loc[idxs].drop(columns=["_split_group"]).copy() for idxs in assigned)  # type: ignore[return-value]


def _split_key_columns(df: pd.DataFrame) -> List[str]:
    preferred = ["pair_key", "uniprot_id", "inchikey", "scaffold_smiles", "ligand_smiles"]
    return [col for col in preferred if col in df.columns]


def write_split_artifacts(
    out_dir: Path,
    source_df: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    group_by: str,
    strict_sequence_disjoint: bool = False,
) -> Dict[str, Any]:
    """Persist split membership and a compact leakage audit before training."""
    split_dir = out_dir / "splits"
    ensure_dir(split_dir)
    key_columns = _split_key_columns(source_df)
    group_columns = [x.strip() for x in str(group_by).split(",") if x.strip() and x.strip() != "row"]
    audit: Dict[str, Any] = {"split_group": group_by, "splits": {}, "overlap": {}}
    split_values: Dict[str, set] = {}
    audit_columns = [
        col for col in (
            "protein_sequence",
            "uniprot_id",
            "inchikey",
            "ligand_smiles",
            "scaffold_smiles",
            "docking_text",
            "protein_function_text",
        )
        if col in source_df.columns
    ]

    for name, part in splits.items():
        manifest = pd.DataFrame({"source_row": part.index.astype(int)})
        for col in key_columns:
            manifest[col] = part[col].astype(str).to_numpy()
        manifest.to_csv(split_dir / f"{name}_keys.csv", index=False)
        stats = {"rows": int(len(part))}
        for col in key_columns:
            stats[f"unique_{col}"] = int(part[col].dropna().astype(str).nunique())
        if "protein_sequence" in part.columns:
            lengths = part["protein_sequence"].fillna("").astype(str).str.len()
            stats["protein_length_mean"] = float(lengths.mean()) if len(lengths) else 0.0
            stats["protein_length_p90"] = float(lengths.quantile(0.9)) if len(lengths) else 0.0
        audit["splits"][name] = stats
        if group_columns:
            split_values[name] = set(
                part[group_columns].fillna("").astype(str).agg("||".join, axis=1).tolist()
            )

    if group_columns:
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = split_values[left] & split_values[right]
            audit["overlap"][f"{left}__{right}"] = {"group_overlap_count": len(overlap)}
            if overlap:
                raise RuntimeError(f"Grouped split leakage detected for {left}/{right}: {len(overlap)} groups")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        column_overlap: Dict[str, Any] = {}
        for col in audit_columns:
            left_values = splits[left][col].fillna("").astype(str).str.strip()
            right_values = splits[right][col].fillna("").astype(str).str.strip()
            overlap = (set(left_values) & set(right_values)) - {""}
            column_overlap[col] = {
                "unique_overlap_count": int(len(overlap)),
                f"{left}_rows_covered": int(left_values.isin(overlap).sum()),
                f"{right}_rows_covered": int(right_values.isin(overlap).sum()),
            }
        audit["overlap"].setdefault(f"{left}__{right}", {}).update(
            {"column_overlap": column_overlap}
        )
        sequence_overlap = column_overlap.get("protein_sequence", {}).get(
            "unique_overlap_count", 0
        )
        if strict_sequence_disjoint and sequence_overlap:
            raise RuntimeError(
                f"Exact protein-sequence leakage detected for {left}/{right}: "
                f"{sequence_overlap} shared sequences. Use --split_group protein_sequence "
                "or explicitly disable --strict_sequence_disjoint."
            )
    with open(split_dir / "split_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    return audit


def iter_minibatches(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    seed: int,
    sample_weight_column: str = "",
    balance_group_columns: str = "",
    balance_power: float = 0.5,
) -> List[pd.DataFrame]:
    if shuffle:
        balance_columns = [
            x.strip() for x in str(balance_group_columns).split(",") if x.strip()
        ]
        combined_weights = pd.Series(1.0, index=df.index, dtype=float)
        if balance_columns:
            missing = [col for col in balance_columns if col not in df.columns]
            if missing:
                raise KeyError(
                    f"--balance_group_columns missing from dataset: {missing}"
                )
            keys = df[balance_columns].fillna("").astype(str).agg("||".join, axis=1)
            counts = keys.map(keys.value_counts()).astype(float)
            combined_weights *= counts.pow(-float(balance_power))
        if sample_weight_column:
            if sample_weight_column not in df.columns:
                raise KeyError(f"--sample_weight_column not found in dataset: {sample_weight_column}")
            sample_weights = pd.to_numeric(df[sample_weight_column], errors="coerce").fillna(0.0).clip(lower=0.0)
            combined_weights *= sample_weights
            if float(combined_weights.sum()) <= 0.0:
                raise ValueError(f"--sample_weight_column={sample_weight_column!r} has no positive finite weights")
        if balance_columns or sample_weight_column:
            # Multiplication lets a biological curriculum and source balancing
            # coexist: e.g. true complexes can be upweighted while the larger
            # natural-fold pool is prevented from overwhelming them.
            df = df.sample(
                n=len(df), replace=True, weights=combined_weights,
                random_state=seed,
            ).reset_index(drop=True)
        else:
            df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    batches = []
    for i in range(0, len(df), batch_size):
        batches.append(df.iloc[i:i + batch_size])
    return batches


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def normalize_chem_features_array(arr: np.ndarray, columns: List[str]) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    for j, col in enumerate(columns[:out.shape[1]]):
        mode, center, scale = CHEM_FEATURE_SCALE.get(col, ("linear", 0.0, 1.0))
        x = out[:, j]
        valid = np.isfinite(x)
        if not valid.any():
            continue
        if mode == "log":
            x_valid = np.log1p(np.clip(x[valid], a_min=0.0, a_max=None))
        else:
            x_valid = x[valid]
            if col == "chem_logP":
                x_valid = np.clip(x_valid, -5.0, 10.0)
            elif col == "chem_charge":
                x_valid = np.clip(x_valid, -4.0, 4.0)
            elif col == "chem_aromatic_ratio":
                x_valid = np.clip(x_valid, 0.0, 1.0)
        out[valid, j] = np.clip((x_valid - float(center)) / max(float(scale), 1e-6), -5.0, 5.0)
    return out


def text_max_len(args, field: str) -> int:
    if field == "docking":
        base = int(getattr(args, "max_docking_len", 512))
        chunks = getattr(args, "dock_max_chunks", None)
    elif field == "function":
        base = int(getattr(args, "max_func_len", 256))
        chunks = getattr(args, "func_max_chunks", None)
    elif field == "ligand_text":
        base = int(getattr(args, "max_ligtext_len", 256))
        chunks = getattr(args, "ligtext_max_chunks", None)
    else:
        base = int(getattr(args, "max_smiles_len", 256))
        chunks = getattr(args, "smiles_max_chunks", None)
    if chunks is None:
        return base
    chunk_len = TEXT_CHUNK_LEN_DEFAULTS[field]
    return max(base, int(chunk_len) * max(1, int(chunks)))


TEXT_IMPORTANCE_KEYWORDS = {
    "docking": [
        "docking", "dock", "vina", "score", "binding affinity", "binding energy", "binding pose",
        "binding pocket", "active site", "pocket", "residue", "contact", "interaction",
        "hydrogen bond", "hbond", "hydrophobic", "salt bridge", "ionic", "electrostatic",
        "pi-stacking", "pi stacking", "π", "aromatic", "van der waals", "vdw", "donor", "acceptor",
    ],
    "function": [
        "function", "activity", "catalytic", "enzyme", "binding", "domain", "motif", "family",
        "localization", "membrane", "secreted", "receptor", "transporter", "kinase", "hydrolase",
        "oxidoreductase", "transferase", "ligase", "isomerase", "lyase", "protein", "design",
    ],
    "ligand_text": [
        "ligand", "smiles", "molecular", "property", "mw", "logp", "tpsa", "hbd", "hba",
        "charge", "ring", "aromatic", "rotatable", "scaffold", "functional group",
    ],
    "smiles": [],
}


DEFAULT_NEURAL_INSTRUCTION = (
    "Design a protein that binds the specified ligand in a stable, "
    "geometrically complementary pocket while preserving the requested "
    "biological or catalytic function. Favor physically plausible "
    "noncovalent contacts and avoid steric clashes."
)


def _split_text_units(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？.!?;；])\s+|[\r\n]+", text)
    units: List[str] = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            units.append(p)
    return units or [text]


def _keyword_score(text: str, field: str) -> int:
    low = text.lower()
    return sum(1 for k in TEXT_IMPORTANCE_KEYWORDS.get(field, []) if k in low)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = re.sub(r"\s+", " ", item.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def smart_preserve_text(text: str, field: str, max_chars: int) -> str:
    text = str(text or "").strip()
    max_chars = int(max_chars)
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text
    if field == "smiles":
        return text[:max_chars]

    units = _split_text_units(text)
    head_budget = max(256, int(max_chars * 0.30))
    tail_budget = max(128, int(max_chars * 0.15))
    middle_budget = max(0, max_chars - head_budget - tail_budget - 64)

    head = text[:head_budget].strip()
    tail = text[-tail_budget:].strip()
    ranked = sorted(
        [(u, _keyword_score(u, field), i) for i, u in enumerate(units)],
        key=lambda x: (-x[1], x[2]),
    )
    selected: List[str] = []
    used = 0
    for unit, score, _idx in ranked:
        if score <= 0 and selected:
            continue
        extra = len(unit) + 1
        if used + extra > middle_budget:
            continue
        selected.append(unit)
        used += extra
        if used >= middle_budget:
            break

    parts = _dedupe_keep_order([head] + selected + [tail])
    out = "\n".join(parts)
    if len(out) > max_chars:
        keep_head = max_chars // 2
        keep_tail = max_chars - keep_head - 16
        out = out[:keep_head].rstrip() + "\n...\n" + out[-keep_tail:].lstrip()
    return out


# -----------------------------
# Token helpers
# -----------------------------
def encode_hf_text(tok, texts: List[str], max_len: int, field: str = "function") -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = int(max_len)
    # Compress only over-budget text, retaining keyword-bearing segments.
    processed: List[str] = []
    for text in texts:
        raw = str(text or "")
        ids = tok(raw, add_special_tokens=True, truncation=False)["input_ids"]
        if len(ids) <= max_len:
            processed.append(raw)
            continue
        # Recheck token length after each compression pass.
        compressed = raw
        for factor in (5, 4, 3, 2, 1, 0.5):
            char_budget = max(128, int(max_len * factor))
            compressed = smart_preserve_text(raw, field, char_budget)
            if len(tok(compressed, add_special_tokens=True, truncation=False)["input_ids"]) <= max_len:
                break
        processed.append(compressed)

    enc = tok(
        processed,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )
    return enc["input_ids"], enc["attention_mask"]


def _token_id(tok, attr: str, token: Optional[str], fallback: Optional[int] = None) -> Optional[int]:
    # ProGen's GPT2 tokenizer reports <|endoftext|> for both bos/eos, while
    # its protein vocabulary has explicit <|bos|>/<|eos|>/<|pad|> tokens.
    # Prefer the requested explicit token when it is in vocabulary.
    if token is not None:
        try:
            tid = tok.convert_tokens_to_ids(token)
            unk_id = getattr(tok, "unk_token_id", None)
            if tid is not None and int(tid) >= 0 and (unk_id is None or int(tid) != int(unk_id)):
                return int(tid)
        except Exception:
            pass
    tid = getattr(tok, attr, None)
    if tid is not None:
        return int(tid)
    return fallback


def clean_protein_sequence(seq: Any) -> str:
    s = str(seq or "").upper()
    return "".join(ch for ch in s if "A" <= ch <= "Z")


def randomize_smiles(smiles: str) -> str:
    """Return an equivalent randomized SMILES when RDKit is available."""
    text = str(smiles or "").strip()
    if not text:
        return text
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(text)
        return Chem.MolToSmiles(mol, doRandom=True) if mol is not None else text
    except Exception:
        return text


def encode_protein(
    tok_protein,
    seqs: List[str],
    max_len: int,
    start_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Build decoder inputs/labels explicitly as:
      input_ids = [BOS] + protein_tokens[:max_len-2] + [EOS] + PAD...
      labels    = [-100] + protein_tokens[:max_len-2] + [EOS] + -100...

    This avoids relying on tokenizer(add_special_tokens=True) to append EOS. Without
    explicit EOS labels, the decoder learns to continue until max_new_tokens.
    """
    max_len = int(max_len)
    if max_len < 3:
        raise ValueError("max_protein_len must be >= 3 to fit BOS, at least one residue, and EOS")

    pad_id = _token_id(tok_protein, "pad_token_id", "<|pad|>", fallback=None)
    bos_id = (
        int(start_token_id)
        if start_token_id is not None
        else _token_id(tok_protein, "bos_token_id", "<|bos|>", fallback=1)
    )
    eos_id = _token_id(tok_protein, "eos_token_id", "<|eos|>", fallback=2)
    if pad_id is None:
        # ProGen2 local tokenizer has <|pad|>=0. This fallback keeps pad distinct from EOS.
        pad_id = 0
    if bos_id is None or eos_id is None:
        raise ValueError(f"Protein tokenizer missing BOS/EOS ids: bos={bos_id}, eos={eos_id}")
    if int(pad_id) == int(eos_id):
        raise ValueError("Protein pad_token_id equals eos_token_id; EOS labels would be masked. Use a tokenizer with distinct PAD/EOS.")
    if int(bos_id) == int(eos_id):
        raise ValueError("Protein bos_token_id equals eos_token_id; use distinct explicit BOS/EOS tokens for autoregressive training.")

    B = len(seqs)
    input_ids = torch.full((B, max_len), int(pad_id), dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)

    raw_token_lens: List[int] = []
    used_token_lens: List[int] = []
    truncated = 0
    empty = 0
    capacity = max_len - 2

    for i, seq in enumerate(seqs):
        clean = clean_protein_sequence(seq)
        if not clean:
            empty += 1
        ids = tok_protein(clean, add_special_tokens=False)["input_ids"]
        ids = [int(x) for x in ids]
        raw_len = len(ids)
        raw_token_lens.append(raw_len)
        if raw_len > capacity:
            truncated += 1
        ids = ids[:capacity]
        used_len = len(ids)
        used_token_lens.append(used_len)

        full = [int(bos_id)] + ids + [int(eos_id)]
        n = min(len(full), max_len)
        input_ids[i, :n] = torch.tensor(full[:n], dtype=torch.long)
        attn[i, :n] = 1

        lab = [-100] + ids + [int(eos_id)]
        labels[i, :len(lab)] = torch.tensor(lab, dtype=torch.long)

    eos_count = int((labels == int(eos_id)).sum().item())
    stats = {
        "batch_size": float(B),
        "raw_len_mean": float(np.mean(raw_token_lens)) if raw_token_lens else 0.0,
        "raw_len_max": float(np.max(raw_token_lens)) if raw_token_lens else 0.0,
        "used_len_mean": float(np.mean(used_token_lens)) if used_token_lens else 0.0,
        "truncated_count": float(truncated),
        "truncated_frac": float(truncated) / max(1, B),
        "empty_count": float(empty),
        "eos_count": float(eos_count),
        "eos_per_sample": float(eos_count) / max(1, B),
        "bos_id": float(bos_id),
        "eos_id": float(eos_id),
        "pad_id": float(pad_id),
    }
    return input_ids, attn, labels, stats


def _looks_like_array_literal(s: str) -> bool:
    st = s.lstrip()
    return st.startswith("[") or st.startswith("(") or st.startswith("{")


def _looks_like_numeric_path(s: str) -> bool:
    if len(s) > 512 or _looks_like_array_literal(s):
        return False
    suffix = Path(s).suffix.lower()
    return suffix in {".npy", ".npz"}


def parse_numeric_list(value: Any) -> Optional[List[float]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in np.asarray(value).reshape(-1).tolist()]
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, (list, tuple)):
            return [float(x) for x in np.asarray(obj).reshape(-1).tolist()]
    except Exception:
        pass
    if _looks_like_numeric_path(s):
        try:
            p = Path(s)
            if p.exists():
                arr = np.load(p, allow_pickle=True)
                if isinstance(arr, np.lib.npyio.NpzFile):
                    arr = arr[arr.files[0]]
                return [float(x) for x in np.asarray(arr).reshape(-1).tolist()]
        except OSError:
            pass
        except Exception:
            return None
    s = s.strip("[]")
    sep = "," if "," in s else None
    try:
        return [float(x) for x in s.split(sep) if str(x).strip()]
    except Exception:
        return None


def parse_numeric_matrix(value: Any) -> Optional[List[List[float]]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, np.ndarray):
        arr = value
    else:
        s = str(value).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        try:
            arr = np.asarray(json.loads(s), dtype=np.float32)
        except Exception:
            if not _looks_like_numeric_path(s):
                return None
            try:
                p = Path(s)
                if not p.exists():
                    return None
                arr = np.load(p, allow_pickle=True)
                if isinstance(arr, np.lib.npyio.NpzFile):
                    arr = arr[arr.files[0]]
            except OSError:
                return None
            except Exception:
                return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        return None
    return arr.tolist()


def parse_contact_pairs(value: Any) -> Optional[List[Tuple[int, int]]]:
    """Parse sparse ``[[res_i, res_j], ...]`` contact annotations."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        obj = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        try:
            obj = json.loads(text)
        except Exception:
            return None
    pairs: List[Tuple[int, int]] = []
    try:
        for item in obj:
            if len(item) != 2:
                continue
            i, j = int(item[0]), int(item[1])
            if i >= 0 and j >= 0 and i != j:
                pairs.append((min(i, j), max(i, j)))
    except (TypeError, ValueError):
        return None
    return pairs


def pad_numeric_rows(
    rows: List[Optional[List[float]]],
    length: int,
    pad_value: float,
) -> torch.Tensor:
    arr = np.full((len(rows), int(length)), float(pad_value), dtype=np.float32)
    for i, vals in enumerate(rows):
        if vals is None:
            continue
        # Shorter rows are the normal case: annotations are residue-length
        # while ``length`` is the batch's fixed padded length.  Only report a
        # real truncation, otherwise this emitted thousands of misleading
        # warnings per epoch and obscured useful diagnostics.
        if len(vals) > int(length):
            print(f"[warn] per-residue field length {len(vals)} > max_protein_len {length}; safely truncate.")
        n = min(len(vals), int(length))
        if n > 0:
            arr[i, :n] = np.asarray(vals[:n], dtype=np.float32)
    return torch.from_numpy(arr).float()


def pad_class_rows(
    rows: List[Optional[List[float]]],
    length: int,
    pad_value: int = -100,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    arr = np.full((len(rows), int(length)), int(pad_value), dtype=np.int64)
    for i, vals in enumerate(rows):
        if vals is None:
            continue
        if len(vals) > int(length):
            print(f"[warn] class field length {len(vals)} > max_protein_len {length}; safely truncate.")
        n = min(len(vals), int(length))
        if n > 0:
            v = np.asarray(vals[:n], dtype=np.int64)
            if num_classes is not None:
                # Preserve the ignore index for structurally uncertain or
                # unavailable residues instead of silently turning it into a
                # valid class target.
                v = np.where(
                    (v >= 0) & (v < int(num_classes)),
                    v,
                    int(pad_value),
                )
            arr[i, :n] = v
    return torch.from_numpy(arr).long()


def build_structure_contact_batch(
    pair_rows: List[Optional[List[Tuple[int, int]]]],
    confidence_rows: List[Optional[List[float]]],
    length: int,
    min_sequence_separation: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand sparse confidence-filtered contacts into dense batch maps.

    The dense supervision mask must use the same minimum sequence separation
    as the offline contact-label builder.  Otherwise, labels made for
    long-range topology can silently gain 4--11-residue negative pairs during
    batching and reintroduce a dominant local-geometry objective.
    """
    if int(min_sequence_separation) < 1:
        raise ValueError("min_sequence_separation must be positive")
    target = np.zeros((len(pair_rows), int(length), int(length)), dtype=np.float32)
    mask = np.zeros_like(target)
    residue_index = np.arange(int(length))
    sequence_far = np.abs(residue_index[:, None] - residue_index[None, :]) >= int(min_sequence_separation)
    for row_idx, (pairs, confidence) in enumerate(zip(pair_rows, confidence_rows)):
        if pairs is None or confidence is None:
            continue
        n = min(len(confidence), int(length))
        if n <= 0:
            continue
        valid = np.zeros(int(length), dtype=bool)
        valid[:n] = np.asarray(confidence[:n], dtype=np.float32) > 0.5
        pair_mask = valid[:, None] & valid[None, :] & sequence_far
        mask[row_idx, pair_mask] = 1.0
        for i, j in pairs:
            if i < int(length) and j < int(length) and pair_mask[i, j]:
                target[row_idx, i, j] = 1.0
                target[row_idx, j, i] = 1.0
    return torch.from_numpy(target).float(), torch.from_numpy(mask).float()


def pad_matrix_rows(
    rows: List[Optional[List[List[float]]]],
    length: int,
    width: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    arr = np.full((len(rows), int(length), int(width)), float(pad_value), dtype=np.float32)
    for i, vals in enumerate(rows):
        if vals is None:
            continue
        m = np.asarray(vals, dtype=np.float32)
        if m.ndim == 1:
            m = m[:, None]
        if m.shape[0] != int(length):
            print(f"[warn] interaction_labels length {m.shape[0]} != max_protein_len {length}; safely truncate/pad.")
        n = min(m.shape[0], int(length))
        k = min(m.shape[1], int(width))
        if n > 0 and k > 0:
            arr[i, :n, :k] = m[:n, :k]
    return torch.from_numpy(arr).float()


def build_weak_interaction_labels(
    texts: List[str],
    pocket_rows: List[Optional[List[float]]],
    length: int,
    width: int,
) -> torch.Tensor:
    keywords = [
        (0, ["hydrophobic", "nonpolar", "lipophilic", "alkyl"]),
        (1, ["salt bridge", "ionic", "charged", "electrostatic", "cation", "anion"]),
        (2, ["pi stacking", "π", "aromatic", "phenyl", "ring stacking"]),
        (3, ["donor", "hbond donor", "hydrogen bond donor"]),
        (4, ["acceptor", "hbond acceptor", "hydrogen bond acceptor"]),
        (5, ["vdw", "van der waals", "contact", "close contact"]),
    ]
    arr = np.zeros((len(texts), int(length), int(width)), dtype=np.float32)
    for i, text in enumerate(texts):
        t = str(text).lower()
        active = [idx for idx, keys in keywords if idx < width and any(k in t for k in keys)]
        if not active and width > 5:
            active = [5]
        elif not active and width > 0:
            active = [0]
        pockets = pocket_rows[i] if i < len(pocket_rows) else None
        if pockets is None:
            continue
        n = min(len(pockets), int(length))
        if n <= 0:
            continue
        mask = np.asarray(pockets[:n], dtype=np.float32) > 0.5
        for idx in active:
            arr[i, :n, idx] = mask.astype(np.float32)
    return torch.from_numpy(arr).float()


def pad_chem_feature_rows(
    rows: List[Optional[List[float]]],
    n_features: int,
) -> Optional[torch.Tensor]:
    if not any(vals is not None for vals in rows):
        return None
    arr = np.full((len(rows), int(n_features)), np.nan, dtype=np.float32)
    for i, vals in enumerate(rows):
        if vals is None:
            continue
        n = min(len(vals), int(n_features))
        if n > 0:
            arr[i, :n] = np.asarray(vals[:n], dtype=np.float32)
    return torch.from_numpy(arr).float()


# -----------------------------
# Lig3D (ragged) handling
# -----------------------------
def _as_int64_1d(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype == np.object_:
        a = np.asarray(a.tolist(), dtype=np.int64)
    a = np.asarray(a, dtype=np.int64).reshape(-1)
    return a


def _as_f32_2d3(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype == np.object_:
        a = np.asarray(a.tolist(), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32).reshape(-1, 3)
    return a


def build_lig3d_batch(
    lig_npz: np.lib.npyio.NpzFile,
    lig_indices: List[int],
    max_atoms: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      lig3d_Z:        [B, A]
      lig3d_coords:   [B, A, 3]
      lig3d_atom_mask:[B, A]  (1=valid atom, 0=pad)
    """
    if "Z" not in lig_npz or "coords" not in lig_npz:
        raise KeyError(f"lig3d_npz must contain keys 'Z' and 'coords'. has={list(lig_npz.keys())}")

    arr_Z = lig_npz["Z"]
    arr_coords = lig_npz["coords"]

    zs: List[np.ndarray] = []
    cs: List[np.ndarray] = []

    for idx in lig_indices:
        z = _as_int64_1d(arr_Z[idx])
        c = _as_f32_2d3(arr_coords[idx])

        A = min(len(z), len(c))
        z = z[:A]
        c = c[:A]

        if max_atoms is not None:
            z = z[:max_atoms]
            c = c[:max_atoms]

        zs.append(z)
        cs.append(c)

    B = len(zs)
    Amax = max(len(z) for z in zs) if B > 0 else 0
    if max_atoms is not None:
        Amax = min(Amax, int(max_atoms))

    Z_pad = np.zeros((B, Amax), dtype=np.int64)
    C_pad = np.zeros((B, Amax, 3), dtype=np.float32)
    M_pad = np.zeros((B, Amax), dtype=np.int64)

    for i, (z, c) in enumerate(zip(zs, cs)):
        a = min(len(z), Amax)
        if a <= 0:
            continue
        Z_pad[i, :a] = z[:a]
        C_pad[i, :a, :] = c[:a, :]
        mask_z = Z_pad[i, :a] > 0
        mask_c = np.isfinite(C_pad[i, :a, :]).all(axis=-1)
        M_pad[i, :a] = (mask_z & mask_c).astype(np.int64)

    return torch.from_numpy(Z_pad).long(), torch.from_numpy(C_pad).float(), torch.from_numpy(M_pad).long()


# -----------------------------
# Batch builder (matches assert_batch_sanity)
# -----------------------------
def build_model_batch(
    rows: pd.DataFrame,
    lig_npz: np.lib.npyio.NpzFile,
    tok_pubmed,
    tok_smiles,
    tok_protein,
    args,
    randomize_smiles_input: bool = False,
) -> Dict[str, torch.Tensor]:
    def col_pick(r: pd.Series, keys: List[str], default: str = "") -> str:
        for k in keys:
            if k in r and pd.notna(r[k]):
                return str(r[k])
        return default

    prot_seqs = [col_pick(r, ["protein_sequence", "sequence", "prot_seq"]) for _, r in rows.iterrows()]
    instruction_txts = [
        col_pick(r, ["instruction", "design_instruction"], default=DEFAULT_NEURAL_INSTRUCTION)
        or DEFAULT_NEURAL_INSTRUCTION
        for _, r in rows.iterrows()
    ]
    docking_txts = [col_pick(r, ["docking_text", "dock_text", "docking"]) for _, r in rows.iterrows()]
    func_txts = [col_pick(r, ["protein_function_text", "function_text", "prot_func_text"]) for _, r in rows.iterrows()]
    lig_txts = [col_pick(r, ["ligand_text", "lig_text", "ligand_description"]) for _, r in rows.iterrows()]
    smiles = [col_pick(r, ["ligand_smiles", "smiles", "lig_smiles"]) for _, r in rows.iterrows()]
    canonical_smiles = list(smiles)
    if bool(getattr(args, "neural_condition_only", False)):
        # Match generation preprocessing: no literature docking passage and no
        # ligand-description channel is available at inference time.
        docking_txts = instruction_txts
        lig_txts = [""] * len(rows)
    if randomize_smiles_input:
        smiles = [
            randomize_smiles(s) if random.random() < float(args.random_smiles_aug_prob) else s
            for s in smiles
        ]

    lig_indices: List[int] = []
    for _, r in rows.iterrows():
        if "ligand_3d_idx" not in r:
            raise KeyError("dataset.csv missing column 'ligand_3d_idx'")
        lig_indices.append(int(r["ligand_3d_idx"]))

    docking_input_ids, docking_attention_mask = encode_hf_text(tok_pubmed, docking_txts, text_max_len(args, "docking"), field="docking")
    function_input_ids, function_attention_mask = encode_hf_text(tok_pubmed, func_txts, text_max_len(args, "function"), field="function")
    ligand_text_input_ids, ligand_text_attention_mask = encode_hf_text(tok_pubmed, lig_txts, text_max_len(args, "ligand_text"), field="ligand_text")

    smiles_input_ids, smiles_attention_mask = encode_hf_text(tok_smiles, smiles, text_max_len(args, "smiles"), field="smiles")

    protein_input_ids, protein_attention_mask, labels, protein_eos_stats = encode_protein(
        tok_protein,
        prot_seqs,
        args.max_protein_len,
        start_token_id=getattr(args, "protein_start_token_id", None),
    )

    lig3d_Z, lig3d_coords, lig3d_atom_mask = build_lig3d_batch(
        lig_npz,
        lig_indices,
        max_atoms=args.lig_max_atoms,
    )

    extra: Dict[str, torch.Tensor] = {}
    if "position_weights" in rows.columns:
        weight_rows = [parse_numeric_list(r.get("position_weights", None)) for _, r in rows.iterrows()]
        extra["position_weights"] = pad_numeric_rows(
            weight_rows,
            length=args.max_protein_len,
            pad_value=1.0,
        )
    if "conservation_scores" in rows.columns:
        conservation_rows = [parse_numeric_list(r.get("conservation_scores", None)) for _, r in rows.iterrows()]
        extra["conservation_scores"] = pad_numeric_rows(
            conservation_rows,
            length=args.max_protein_len,
            pad_value=0.0,
        )

    chem_col = None
    if "chem_features" in rows.columns:
        chem_col = "chem_features"
    elif "ligand_chem_features" in rows.columns:
        chem_col = "ligand_chem_features"
    if chem_col is not None:
        chem_rows = [parse_numeric_list(r.get(chem_col, None)) for _, r in rows.iterrows()]
        chem_tensor = pad_chem_feature_rows(chem_rows, n_features=args.n_chem_features)
        if chem_tensor is not None:
            cols = PHASE3_CHEM_COLUMNS[:chem_tensor.size(1)]
            arr = normalize_chem_features_array(chem_tensor.numpy(), cols)
            extra["chem_features"] = torch.from_numpy(arr).float()
    elif any(c in rows.columns for c in PHASE3_CHEM_COLUMNS):
        chem_cols = [c for c in PHASE3_CHEM_COLUMNS if c in rows.columns]
        chem_arr = rows[chem_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        chem_arr = normalize_chem_features_array(chem_arr, chem_cols)
        out = np.full((len(rows), int(args.n_chem_features)), np.nan, dtype=np.float32)
        k = min(out.shape[1], chem_arr.shape[1])
        out[:, :k] = chem_arr[:, :k]
        extra["chem_features"] = torch.from_numpy(out).float()

    if "biochemical_residue_prior" in rows.columns:
        prior_array = np.full((len(rows), 20), np.nan, dtype=np.float32)
        for row_index, value in enumerate(rows["biochemical_residue_prior"]):
            parsed = parse_numeric_list(value)
            if parsed is None or len(parsed) != 20:
                continue
            prior_array[row_index] = np.asarray(parsed, dtype=np.float32)
        extra["biochemical_residue_prior"] = torch.from_numpy(prior_array).float()
        if "biochemical_prior_strength" in rows.columns:
            strengths = pd.to_numeric(
                rows["biochemical_prior_strength"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=np.float32)
            extra["biochemical_prior_strength"] = torch.from_numpy(strengths).float()

    pocket_rows_for_weak_labels: Optional[List[Optional[List[float]]]] = None
    for cols, key in [
        (["pocket_labels"], "pocket_labels"),
        (["ligand_contact_labels"], "ligand_contact_labels"),
        (["solvent_accessibility", "rsa", "pocket_asa"], "solvent_accessibility"),
        (["burial_depth"], "burial_depth"),
        (["residue_depth"], "residue_depth"),
    ]:
        col = next((c for c in cols if c in rows.columns), None)
        if col is not None:
            vals = [parse_numeric_list(r.get(col, None)) for _, r in rows.iterrows()]
            if key == "pocket_labels":
                pocket_rows_for_weak_labels = vals
            pad = 0.0 if key in {"pocket_labels", "ligand_contact_labels"} else float("nan")
            tensor = pad_numeric_rows(vals, length=args.max_protein_len, pad_value=pad)
            if key == "solvent_accessibility":
                # RSA is 0-100+ (percent), normalize to ~[0, 1.5]
                tensor = tensor / 100.0
            elif key == "residue_depth":
                # depth is 0-50+ angstroms, normalize to ~[0, 1]
                tensor = tensor / 20.0
            extra[key] = tensor

    ss_col = "secondary_structure_labels" if "secondary_structure_labels" in rows.columns else ("dssp" if "dssp" in rows.columns else None)
    if ss_col is not None:
        vals = [parse_numeric_list(r.get(ss_col, None)) for _, r in rows.iterrows()]
        extra["secondary_structure_labels"] = pad_class_rows(
            vals,
            length=args.max_protein_len,
            pad_value=-100,
            num_classes=3,
        )

    if {
        "structure_contact_pairs",
        "structure_confidence_mask",
    }.issubset(rows.columns):
        pair_rows = [parse_contact_pairs(r.get("structure_contact_pairs", None)) for _, r in rows.iterrows()]
        confidence_rows = [parse_numeric_list(r.get("structure_confidence_mask", None)) for _, r in rows.iterrows()]
        contact_map, contact_mask = build_structure_contact_batch(
            pair_rows,
            confidence_rows,
            length=args.max_protein_len,
            min_sequence_separation=int(args.structure_contact_min_sequence_separation),
        )
        extra["structure_contact_map"] = contact_map
        extra["structure_contact_mask"] = contact_mask

    if "interaction_labels" in rows.columns:
        vals = [parse_numeric_matrix(r.get("interaction_labels", None)) for _, r in rows.iterrows()]
        extra["interaction_labels"] = pad_matrix_rows(
            vals,
            length=args.max_protein_len,
            width=args.n_interaction_types,
            pad_value=0.0,
        )
    elif pocket_rows_for_weak_labels is not None:
        weak_interactions = build_weak_interaction_labels(
            docking_txts,
            pocket_rows_for_weak_labels,
            length=args.max_protein_len,
            width=args.n_interaction_types,
        )
        # Placeholder all-zero pocket labels cannot identify interaction
        # residues. Feeding their derived all-zero matrix into BCE would train
        # away every ligand-contact signal.
        if bool((weak_interactions > 0).any()):
            extra["interaction_labels"] = weak_interactions
            extra["interaction_labels_is_weak"] = torch.ones(len(rows), dtype=torch.bool)

    if "scaffold_smiles" in rows.columns:
        extra["scaffold_smiles"] = rows["scaffold_smiles"].astype(str).tolist()
    if "af_residue_count" in rows.columns:
        extra["af_residue_count"] = torch.tensor(
            pd.to_numeric(rows["af_residue_count"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64),
            dtype=torch.long,
        )
    if "af_pdb_path" in rows.columns:
        extra["af_pdb_path"] = rows["af_pdb_path"].astype(str).tolist()
    if "hard_negative_indices" in rows.columns:
        hn_rows = [parse_numeric_list(r.get("hard_negative_indices", None)) for _, r in rows.iterrows()]
        max_h = max([len(v) for v in hn_rows if v is not None] + [1])
        hn = np.full((len(hn_rows), max_h), -1, dtype=np.int64)
        for i, vals in enumerate(hn_rows):
            if vals is None:
                continue
            n = min(len(vals), max_h)
            hn[i, :n] = np.asarray(vals[:n], dtype=np.int64)
        extra["hard_negative_indices"] = torch.from_numpy(hn).long()
    if "hard_negative_ids" in rows.columns:
        extra["hard_negative_ids"] = rows["hard_negative_ids"].astype(str).tolist()

    if bool(getattr(args, "enable_condition_fold_planner", False)):
        extra["slot_residue_composition"] = build_slot_residue_composition(
            prot_seqs, int(args.fold_plan_slots)
        )
        pocket_source = None
        if "geometric_pocket_residue_scores" in rows.columns:
            pocket_source = "geometric_pocket_residue_scores"
        elif "pocket_labels" in rows.columns:
            pocket_source = "pocket_labels"
        if pocket_source is not None:
            pocket_rows = [
                parse_numeric_list(value)
                for value in rows[pocket_source]
            ]
            extra["slot_pocket_targets"] = build_slot_scalar_targets(
                pocket_rows, prot_seqs, int(args.fold_plan_slots)
            )
        if "fold_archetype_id" in rows.columns:
            archetype_ids = pd.to_numeric(
                rows["fold_archetype_id"], errors="coerce"
            ).fillna(-100).to_numpy(dtype=np.int64)
            extra["fold_archetype_id"] = torch.from_numpy(archetype_ids).long()

    if bool(getattr(args, "enable_structure_token_planner", False)):
        slots = int(args.structure_token_slots)
        codes = int(args.structure_token_codes)
        required_structure_columns = {
            "structure_token_ids",
            "structure_token_mask",
        }
        missing_structure = sorted(required_structure_columns - set(rows.columns))
        if missing_structure:
            raise ValueError(
                "structure-token planner requires dataset columns: "
                + ", ".join(missing_structure)
            )
        token_rows = [parse_numeric_list(value) for value in rows["structure_token_ids"]]
        mask_rows = [parse_numeric_list(value) for value in rows["structure_token_mask"]]
        extra["structure_token_targets"] = pad_class_rows(
            token_rows, length=slots, pad_value=-100, num_classes=codes
        )
        extra["structure_token_mask"] = pad_numeric_rows(
            mask_rows, length=slots, pad_value=0.0
        ).bool()
        if "structure_pocket_slot_targets" in rows.columns:
            values = [
                parse_numeric_list(value)
                for value in rows["structure_pocket_slot_targets"]
            ]
            extra["structure_pocket_slot_targets"] = pad_numeric_rows(
                values, length=slots, pad_value=float("nan")
            )
        if "structure_interaction_slot_targets" in rows.columns:
            values = [
                parse_numeric_matrix(value)
                for value in rows["structure_interaction_slot_targets"]
            ]
            extra["structure_interaction_slot_targets"] = pad_matrix_rows(
                values,
                length=slots,
                width=int(args.n_interaction_types),
                pad_value=float("nan"),
            )
        for column, key in (
            ("structure_slot_contact_map", "structure_slot_contact_map"),
            ("structure_slot_contact_mask", "structure_slot_contact_mask"),
        ):
            if column in rows.columns:
                values = [parse_numeric_matrix(value) for value in rows[column]]
                extra[key] = pad_matrix_rows(
                    values, length=slots, width=slots, pad_value=0.0
                )

    if extra:
        extra["_detected_extra_fields"] = sorted([k for k in extra.keys() if not k.startswith("_")])

    def _factorize_groups(column: str, fallback_values: List[str]) -> torch.Tensor:
        if column in rows.columns:
            values = rows[column].fillna("").astype(str).str.strip().tolist()
            values = [
                value if value else (
                    str(fallback_values[i]).strip()
                    if i < len(fallback_values) and str(fallback_values[i]).strip()
                    else f"__missing_row_{i}__"
                )
                for i, value in enumerate(values)
            ]
        else:
            values = fallback_values
        codes, _ = pd.factorize(np.asarray(values, dtype=object), sort=False)
        return torch.tensor(codes, dtype=torch.long)

    if bool(getattr(args, "neural_condition_only", False)):
        ligand_condition_values = (
            rows["inchikey"].fillna("").astype(str).str.strip().tolist()
            if "inchikey" in rows.columns else [""] * len(rows)
        )
        ligand_condition_values = [
            value if value else str(canonical_smiles[index]).strip()
            for index, value in enumerate(ligand_condition_values)
        ]
        condition_values = [
            "||".join((ligand_key, str(function), str(instruction)))
            for ligand_key, function, instruction in zip(
                ligand_condition_values, func_txts, instruction_txts
            )
        ]
        condition_columns = []
    else:
        condition_columns = [
            col
            for col in (
                "inchikey",
                "protein_function_text",
                "docking_text",
                "ligand_text",
            )
            if col in rows.columns
        ]
    if bool(getattr(args, "neural_condition_only", False)):
        pass
    elif condition_columns:
        condition_values = (
            rows[condition_columns]
            .fillna("")
            .astype(str)
            .agg("||".join, axis=1)
            .tolist()
        )
    else:
        condition_values = [
            "||".join(parts)
            for parts in zip(smiles, func_txts, docking_txts, lig_txts)
        ]
    condition_codes, _ = pd.factorize(
        np.asarray(condition_values, dtype=object),
        sort=False,
    )

    base = {
        "docking_input_ids": docking_input_ids,
        "docking_attention_mask": docking_attention_mask,
        "function_input_ids": function_input_ids,
        "function_attention_mask": function_attention_mask,
        "ligand_text_input_ids": ligand_text_input_ids,
        "ligand_text_attention_mask": ligand_text_attention_mask,
        "smiles_input_ids": smiles_input_ids,
        "smiles_attention_mask": smiles_attention_mask,
        "lig3d_Z": lig3d_Z,
        "lig3d_coords": lig3d_coords,
        "lig3d_atom_mask": lig3d_atom_mask,
        "protein_input_ids": protein_input_ids,
        "protein_attention_mask": protein_attention_mask,
        "protein_labels": labels,
        "ligand_group_ids": _factorize_groups(
            "inchikey",
            [str(x) for x in canonical_smiles],
        ),
        "protein_group_ids": _factorize_groups(
            "protein_sequence",
            [str(x) for x in prot_seqs],
        ),
        "condition_group_ids": torch.tensor(condition_codes, dtype=torch.long),
        "_protein_eos_stats": protein_eos_stats,
    }
    base.update(extra)
    return base


# -----------------------------
# Optimizer helpers
# -----------------------------
def build_param_groups(
    model: nn.Module,
    weight_decay: float,
    base_lr: float,
    lr_encoder: float = 0.0,
    lr_perceiver: float = 0.0,
    lr_decoder_new: float = 0.0,
    lr_decoder_backbone: float = 0.0,
    lr_fold_planner: float = 0.0,
    lr_length_head: float = 0.0,
) -> List[Dict[str, Any]]:
    """Build real module-wise LR groups and separate decay/no-decay params."""
    lr_by_kind = {
        "encoder": float(lr_encoder or base_lr),
        "perceiver": float(lr_perceiver or base_lr),
        "decoder_new": float(lr_decoder_new or base_lr),
        "decoder_backbone": float(lr_decoder_backbone or (base_lr * 0.1)),
        "fold_planner": float(lr_fold_planner or lr_decoder_new or base_lr),
        "length_head": float(lr_length_head or lr_decoder_new or base_lr),
    }
    buckets: Dict[Tuple[str, bool], List[nn.Parameter]] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("protein_length_head."):
            kind = "length_head"
        elif n.startswith(("fold_planner.", "structure_token_planner.")):
            kind = "fold_planner"
        elif n.startswith("encoder."):
            kind = "encoder"
        elif n.startswith("perceiver."):
            kind = "perceiver"
        elif n.startswith("decoder.backbone."):
            kind = "decoder_backbone"
        else:
            kind = "decoder_new"
        lname = n.lower()
        no_decay = (
            p.ndim < 2
            or n.endswith(".bias")
            or "layer_norm" in lname
            or "layernorm" in lname
            or ".ln" in lname
        )
        buckets.setdefault((kind, no_decay), []).append(p)
    groups = []
    for (kind, no_decay), params in buckets.items():
        groups.append(
            {
                "params": params,
                "lr": lr_by_kind[kind],
                "weight_decay": 0.0 if no_decay else float(weight_decay),
                "group_name": f"{kind}_{'no_decay' if no_decay else 'decay'}",
            }
        )
    return groups


def _length_bucket(length: int) -> str:
    for upper in (128, 256, 512, 768, 1024):
        if length <= upper:
            return f"<= {upper}"
    return "> 1024"


def _teacher_forced_sequence_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prefix_len: int,
    eos_token_id: int,
    eos_stats: Dict[str, float],
) -> Dict[str, Any]:
    """Measure EOS and predicted stop positions from the existing LM logits.

    These are teacher-forced diagnostics, not autoregressive generation metrics.
    They are cheap enough to run on validation and final test without changing
    the model's training objective or decoding behavior.
    """
    L = labels.size(1)
    start = max(0, int(prefix_len) - 1)
    step_logits = logits[:, start:start + L, :]
    if step_logits.size(1) != L:
        raise RuntimeError(
            f"Cannot align decoder logits to labels: logits={tuple(logits.shape)}, "
            f"prefix_len={prefix_len}, labels={tuple(labels.shape)}"
        )
    pred = step_logits.argmax(dim=-1)
    valid = labels.ne(-100)
    eos_target = valid & labels.eq(int(eos_token_id))
    correct = pred.eq(labels) & valid
    eos_correct = pred.eq(int(eos_token_id)) & eos_target
    pred_eos = pred.eq(int(eos_token_id)) & valid

    result: Dict[str, Any] = {
        "valid_tokens": int(valid.sum().item()),
        "correct_tokens": int(correct.sum().item()),
        "eos_targets": int(eos_target.sum().item()),
        "eos_correct": int(eos_correct.sum().item()),
        "predicted_eos": int(pred_eos.sum().item()),
        "target_truncated_samples": int(round(float(eos_stats.get("truncated_count", 0.0)))),
        "samples": int(labels.size(0)),
        "length_buckets": {},
    }
    for i in range(labels.size(0)):
        valid_positions = torch.nonzero(valid[i], as_tuple=False).flatten()
        target_len = max(0, int(valid_positions.numel()) - 1)  # exclude EOS
        eos_positions = torch.nonzero(pred_eos[i], as_tuple=False).flatten()
        predicted_len = int(eos_positions[0].item()) if eos_positions.numel() else None
        bucket = _length_bucket(target_len)
        bucket_stats = result["length_buckets"].setdefault(
            bucket,
            {"samples": 0, "target_length_sum": 0, "predicted_eos_count": 0, "predicted_length_sum": 0},
        )
        bucket_stats["samples"] += 1
        bucket_stats["target_length_sum"] += target_len
        if predicted_len is not None:
            bucket_stats["predicted_eos_count"] += 1
            bucket_stats["predicted_length_sum"] += predicted_len
    return result


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    df_val: pd.DataFrame,
    lig_npz,
    tok_pubmed,
    tok_smiles,
    tok_protein,
    args,
    device: torch.device,
    max_batches: Optional[int] = 20,
) -> Dict[str, float]:
    model.eval()
    batches = iter_minibatches(df_val, args.val_batch_size, shuffle=False, seed=args.seed)

    losses = []
    lm_losses = []
    aux_losses: Dict[str, List[float]] = {}
    sequence_totals: Dict[str, Any] = {
        "valid_tokens": 0, "correct_tokens": 0, "eos_targets": 0, "eos_correct": 0,
        "predicted_eos": 0, "target_truncated_samples": 0, "samples": 0, "length_buckets": {},
    }

    for bi, bdf in enumerate(batches):
        if max_batches is not None and bi >= max_batches:
            break
        batch = build_model_batch(bdf, lig_npz, tok_pubmed, tok_smiles, tok_protein, args)
        batch = to_device(batch, device)
        with autocast_context(args, device):
            out = model(batch, return_debug=False)

        if out.loss_total is not None:
            losses.append(float(out.loss_total.detach().cpu()))
        if out.loss_lm is not None:
            lm_losses.append(float(out.loss_lm.detach().cpu()))
        for k, v in (out.loss_aux or {}).items():
            aux_losses.setdefault(k, []).append(float(v.detach().cpu()))
        eos_id = int(getattr(model, "eos_token_id"))
        seq_metrics = _teacher_forced_sequence_metrics(
            out.logits,
            batch["protein_labels"],
            int(out.prefix_len or 0),
            eos_id,
            batch.get("_protein_eos_stats", {}),
        )
        for key in ("valid_tokens", "correct_tokens", "eos_targets", "eos_correct", "predicted_eos", "target_truncated_samples", "samples"):
            sequence_totals[key] += seq_metrics[key]
        for bucket, stats in seq_metrics["length_buckets"].items():
            dst = sequence_totals["length_buckets"].setdefault(
                bucket,
                {"samples": 0, "target_length_sum": 0, "predicted_eos_count": 0, "predicted_length_sum": 0},
            )
            for key, value in stats.items():
                dst[key] += value

    model.train()

    def mean(x: List[float]) -> float:
        return float(np.mean(x)) if len(x) else float("nan")

    outm = {"loss_total": mean(losses), "loss_lm": mean(lm_losses)}
    for k, arr in aux_losses.items():
        outm[f"loss_aux/{k}"] = mean(arr)
    outm["teacher_forced/token_accuracy"] = sequence_totals["correct_tokens"] / max(1, sequence_totals["valid_tokens"])
    outm["teacher_forced/eos_recall"] = sequence_totals["eos_correct"] / max(1, sequence_totals["eos_targets"])
    outm["teacher_forced/eos_precision"] = sequence_totals["eos_correct"] / max(1, sequence_totals["predicted_eos"])
    outm["teacher_forced/target_truncation_rate"] = sequence_totals["target_truncated_samples"] / max(1, sequence_totals["samples"])
    outm["teacher_forced/length_buckets"] = sequence_totals["length_buckets"]
    return outm


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--code_dir",
        default=str(Path(__file__).resolve().parent),
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--lig3d_npz", required=True)
    ap.add_argument("--hf_cache_root", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--run_label",
        default="standard",
        help="Metadata only, e.g. smoke, quick_not_for_quality, or production.",
    )
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.05)
    ap.add_argument(
        "--split_group",
        default="row",
        help="row (default) or comma-separated grouping columns, e.g. uniprot_id,inchikey",
    )
    ap.add_argument(
        "--strict_sequence_disjoint",
        action="store_true",
        help="Abort if an exact protein sequence occurs in more than one split.",
    )
    ap.add_argument(
        "--sample_weight_column",
        default="",
        help="Optional positive numeric dataset column for weighted training resampling; disabled by default.",
    )
    ap.add_argument(
        "--balance_group_columns",
        default="",
        help="Optional columns for inverse-frequency training resampling, e.g. inchikey.",
    )
    ap.add_argument(
        "--balance_power",
        type=float,
        default=0.5,
        help="Inverse-frequency exponent (0.5 moderate, 1.0 fully balanced).",
    )
    ap.add_argument("--random_smiles_augmentation", action="store_true", help="Randomize equivalent SMILES only for training batches.")
    ap.add_argument("--random_smiles_aug_prob", type=float, default=1.0, help="Per-sample probability for --random_smiles_augmentation.")
    ap.add_argument(
        "--neural_condition_only",
        action="store_true",
        help="Use only SMILES + instruction + protein function; replace training docking/ligand text channels to match generation.",
    )

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--val_batch_size", type=int, default=2)
    ap.add_argument(
        "--amp_dtype",
        choices=("none", "bf16", "fp16"),
        default="none",
        help="CUDA autocast precision. The legacy FP32 path remains the default.",
    )
    ap.add_argument(
        "--model_dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Stored model parameter dtype; BF16 halves parameter/gradient memory on supported CUDA GPUs.",
    )
    ap.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable decoder-backbone activation checkpointing for long proteins.",
    )

    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr_encoder", type=float, default=0.0)
    ap.add_argument("--lr_perceiver", type=float, default=0.0)
    ap.add_argument("--lr_decoder_new", type=float, default=0.0)
    ap.add_argument("--lr_decoder_backbone", type=float, default=0.0)
    ap.add_argument("--lr_fold_planner", type=float, default=0.0)
    ap.add_argument(
        "--lr_length_head",
        type=float,
        default=0.0,
        help="Optional dedicated LR for the condition-specific protein length head.",
    )
    ap.add_argument(
        "--optimizer",
        choices=("adamw", "adafactor"),
        default="adamw",
        help="Optimizer; Adafactor substantially reduces state memory for long-chain continuation.",
    )
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--max_protein_len", type=int, default=1024)
    ap.add_argument(
        "--protein_control_tag",
        default="",
        help=(
            "Optional ProGen control tag used instead of <|bos|>; the local "
            "ProGen2-base documentation uses '1'."
        ),
    )
    ap.add_argument("--dock_max_chunks", type=int, default=24)
    ap.add_argument("--func_max_chunks", type=int, default=8)
    ap.add_argument("--ligtext_max_chunks", type=int, default=8)
    ap.add_argument("--smiles_max_chunks", type=int, default=2)
    ap.add_argument("--use_param_groups", action="store_true")

    ap.add_argument("--max_docking_len", type=int, default=2048)
    ap.add_argument("--max_func_len", type=int, default=1024)
    ap.add_argument("--max_ligtext_len", type=int, default=1024)
    ap.add_argument("--max_smiles_len", type=int, default=384)
    ap.add_argument("--lig_max_atoms", type=int, default=128)

    ap.add_argument("--decoder_backbone", type=str, default="hugohrban/progen2-base")
    ap.add_argument("--decoder_unfreeze_last_n_blocks", type=int, default=0)
    ap.add_argument(
        "--prefix_boundary_scale",
        type=float,
        default=1.0,
        help="Scale learned prefix boundary embeddings; use a small value when matching the frozen protein prior.",
    )
    ap.add_argument("--local_files_only", action="store_true")

    # Optional sequence and route regularizers.
    ap.add_argument("--eos_loss_weight", type=float, default=1.0, help="EOS token loss weight (>1.0 for stronger supervision)")
    ap.add_argument("--lambda_diversity", type=float, default=0.0, help="Diversity regularization weight")
    ap.add_argument("--lambda-local-composition-unlikelihood", type=float, default=0.0)
    ap.add_argument("--local-composition-window", type=int, default=32)
    ap.add_argument("--local-composition-min-context", type=int, default=12)
    ap.add_argument("--local-composition-dominance-threshold", type=float, default=0.35)
    ap.add_argument("--lambda-local-group-unlikelihood", type=float, default=0.0)
    ap.add_argument("--local-positive-group-threshold", type=float, default=0.35)
    ap.add_argument("--local-negative-group-threshold", type=float, default=0.35)
    ap.add_argument("--local-hydrophobic-group-threshold", type=float, default=0.65)
    ap.add_argument("--lambda_route_comp", type=float, default=0.0, help="Docking-ligand route contrastive loss weight")
    ap.add_argument("--route_comp_temperature", type=float, default=0.07, help="Temperature for route contrastive InfoNCE")
    ap.add_argument("--lambda_chem_align", type=float, default=0.0, help="Ligand chemical feature prediction loss weight")
    ap.add_argument("--lambda_query_diversity", type=float, default=0.0, help="Interaction-query diversity regularization weight")
    ap.add_argument("--n_chem_features", type=int, default=10, help="Number of ligand chemical feature targets")
    ap.add_argument("--chem_loss_type", type=str, default="auto", choices=["auto", "bce", "mse", "smooth_l1"])
    ap.add_argument("--use_hard_negative_contrastive", action="store_true")
    ap.add_argument("--hard_negative_weight", type=float, default=0.0)
    ap.add_argument("--ligand_memory_bank_size", type=int, default=0)
    ap.add_argument("--ligand_contrastive_temperature", type=float, default=0.07)
    ap.add_argument("--n_interaction_types", type=int, default=6)
    ap.add_argument("--lambda_pocket", type=float, default=0.0)
    ap.add_argument("--lambda_contact", type=float, default=0.0)
    ap.add_argument("--lambda_structure_contacts", type=float, default=0.0)
    ap.add_argument("--structure_contact_proj_dim", type=int, default=128)
    ap.add_argument("--structure_contact_negative_ratio", type=float, default=4.0)
    ap.add_argument(
        "--structure_contact_min_sequence_separation",
        type=int,
        default=4,
        help=(
            "Minimum |i-j| retained in the dense structure-contact loss mask. "
            "Set this to the same value used by build_af_structure_supervision.py."
        ),
    )
    ap.add_argument("--lambda_interaction", type=float, default=0.0)
    ap.add_argument("--lambda_secondary_structure", type=float, default=0.0)
    ap.add_argument("--enable_condition_fold_planner", action="store_true")
    ap.add_argument("--fold_plan_slots", type=int, default=16)
    ap.add_argument("--fold_plan_prototypes", type=int, default=64)
    ap.add_argument("--fold_plan_archetypes", type=int, default=32)
    ap.add_argument("--fold_plan_layers", type=int, default=2)
    ap.add_argument("--fold_plan_heads", type=int, default=8)
    ap.add_argument("--fold_plan_dropout", type=float, default=0.10)
    ap.add_argument("--fold_plan_prototype_temperature", type=float, default=0.50)
    ap.add_argument("--fold_plan_condition_dropout", type=float, default=0.10)
    ap.add_argument("--fold_plan_local_chemistry_scale", type=float, default=0.15)
    ap.add_argument("--fold_plan_archetype_scale", type=float, default=0.20)
    ap.add_argument("--fold_plan_prefix_scale", type=float, default=0.35)
    ap.add_argument("--fold_plan_full_memory_scale", type=float, default=1.0)
    ap.add_argument("--fold_plan_injection_warmup_steps", type=int, default=800)
    ap.add_argument("--lambda_fold_plan_ss", type=float, default=0.0)
    ap.add_argument("--lambda_fold_plan_contacts", type=float, default=0.0)
    ap.add_argument("--lambda_fold_plan_global", type=float, default=0.0)
    ap.add_argument("--lambda_fold_plan_codebook", type=float, default=0.0)
    ap.add_argument("--lambda_fold_plan_chemistry", type=float, default=0.0)
    ap.add_argument("--lambda_fold_plan_sequence", type=float, default=0.0)
    ap.add_argument("--lambda_fold_plan_pocket", type=float, default=0.0)
    ap.add_argument("--lambda_fold_plan_archetype", type=float, default=0.0)
    ap.add_argument("--enable_structure_token_planner", action="store_true")
    ap.add_argument("--structure_token_slots", type=int, default=64)
    ap.add_argument("--structure_token_codes", type=int, default=128)
    ap.add_argument("--structure_token_prefix_slots", type=int, default=16)
    ap.add_argument("--structure_token_layers", type=int, default=3)
    ap.add_argument("--structure_token_heads", type=int, default=8)
    ap.add_argument("--structure_token_dropout", type=float, default=0.10)
    ap.add_argument("--structure_token_temperature", type=float, default=0.50)
    ap.add_argument("--structure_token_posterior_teacher_forcing", type=float, default=0.75)
    ap.add_argument("--structure_token_posterior_final_mix", type=float, default=0.15)
    ap.add_argument("--structure_token_posterior_decay_steps", type=int, default=900)
    ap.add_argument("--structure_token_geometry_feature_dim", type=int, default=10)
    ap.add_argument(
        "--structure_token_codebook_npz",
        default="",
        help="Optional builder codebook whose normalized centers initialize persistent geometry features.",
    )
    ap.add_argument("--structure_token_prefix_scale", type=float, default=0.30)
    ap.add_argument("--structure_token_full_memory_scale", type=float, default=1.0)
    ap.add_argument(
        "--structure-token-train-scale-choices",
        default="",
        help="Comma-separated factors sampled for both structure-token injection routes.",
    )
    ap.add_argument("--lambda_structure_token_prior", type=float, default=0.0)
    ap.add_argument("--lambda_structure_token_posterior", type=float, default=0.0)
    ap.add_argument("--lambda_structure_token_kl", type=float, default=0.0)
    ap.add_argument("--lambda_structure_token_pocket", type=float, default=0.0)
    ap.add_argument("--lambda_structure_token_interaction", type=float, default=0.0)
    ap.add_argument("--lambda_structure_token_topology", type=float, default=0.0)
    ap.add_argument("--lambda_solvent_accessibility", type=float, default=0.0)
    ap.add_argument("--lambda_residue_depth", type=float, default=0.0)
    ap.add_argument("--lambda_complementarity", type=float, default=0.0)
    ap.add_argument("--complementarity_loss_type", type=str, default="mse", choices=["bce", "mse"])
    ap.add_argument("--lambda_condition_awareness", type=float, default=0.0, help="Penalize docking-influenced memory collapsing to function memory")
    ap.add_argument("--cond_awareness_prob", type=float, default=0.15, help="Probability of computing condition-awareness loss each step")
    ap.add_argument("--condition_awareness_max_similarity", type=float, default=0.80)
    ap.add_argument("--lambda_condition_contrastive", type=float, default=0.0)
    ap.add_argument("--condition_contrastive_temperature", type=float, default=0.07)
    ap.add_argument("--lambda_length_prediction", type=float, default=0.0)
    ap.add_argument("--perceiver_layerscale_init", type=float, default=1e-3)
    ap.add_argument("--perceiver_cross_every_block", action="store_true")
    ap.add_argument("--use_bio_position_weights", action="store_true")
    ap.add_argument("--alpha_conservation", type=float, default=0.0)
    ap.add_argument("--alpha_pocket", type=float, default=0.0)
    ap.add_argument("--alpha_contact", type=float, default=0.0)
    ap.add_argument("--enable_pocket_aware_injection", action="store_true")
    ap.add_argument("--pocket_gate_strength", type=float, default=0.5)
    ap.add_argument("--aux_loss_warmup_steps", type=int, default=0)
    ap.add_argument("--aux_loss_warmup_epochs", type=int, default=0)
    ap.add_argument("--aux_loss_warmup_mode", type=str, default="none", choices=["linear", "cosine", "none"])
    ap.add_argument("--disable_full_injection_train", action="store_true", help="Disable full-memory injection during training ablation")
    ap.add_argument("--disable_full_injection_generate", action="store_true", help="Save config with generation full injection disabled")
    ap.add_argument("--full_injection_layer_policy", type=str, default="all", choices=["all", "last_half", "last_k", "every_other", "none"])
    ap.add_argument("--full_injection_last_k", type=int, default=0)
    ap.add_argument("--share_full_injection_weights", action="store_true")
    ap.add_argument(
        "--full-injection-train-stop-choices",
        default="",
        help="Comma-separated protein-token cutoffs sampled per training batch; -1 keeps full injection.",
    )
    ap.add_argument(
        "--full-injection-train-scale-choices",
        default="",
        help="Comma-separated full-memory injection scales sampled per training batch.",
    )
    ap.add_argument(
        "--full_injection_eval_scale",
        type=float,
        default=1.0,
        help="Full-memory injection scale used during validation when training injection is enabled.",
    )
    ap.add_argument(
        "--prefix-condition-train-scale-choices",
        default="",
        help="Comma-separated prefix-condition scales sampled per training batch.",
    )
    ap.add_argument(
        "--pool-condition-memory",
        action="store_true",
        help="Pool each prefix memory to one token to preserve ProGen residue positions.",
    )
    ap.add_argument(
        "--init_checkpoint",
        type=str,
        default="",
        help="Initialize model weights from a prior best.pt; optimizer/scheduler start fresh.",
    )
    ap.add_argument(
        "--init_skip_decoder_backbone",
        action="store_true",
        help="Reuse adapters/encoders from --init_checkpoint but restore the local pretrained ProGen backbone.",
    )
    ap.add_argument(
        "--selection_metric",
        choices=["loss_lm", "loss_total"],
        default="loss_lm",
        help="Validation metric used to select best.pt; loss_lm is less confounded by optional auxiliary losses.",
    )
    ap.add_argument("--save_last_checkpoint", action="store_true", help="Also save last.pt with optimizer state; disabled to conserve disk.")
    ap.add_argument("--full_checkpoint", action="store_true", help="Include reconstructible frozen pretrained backbones in checkpoints.")
    ap.add_argument("--use_cosine_schedule", action="store_true", help="Use cosine annealing LR schedule")
    ap.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps for LR schedule")

    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=0, help="Deprecated; use --snapshot-steps for explicit external-selection checkpoints.")
    ap.add_argument(
        "--snapshot-steps",
        default="",
        help="Comma-separated steps saved as compact model-only checkpoints for external sentinel selection.",
    )

    args = ap.parse_args()
    if int(args.structure_contact_min_sequence_separation) < 1:
        raise ValueError("--structure_contact_min_sequence_separation must be positive")
    if int(args.local_composition_window) < 2:
        raise ValueError("--local-composition-window must be at least 2")
    if not 2 <= int(args.local_composition_min_context) <= int(args.local_composition_window):
        raise ValueError("--local-composition-min-context must be in [2, window]")
    if not 0.0 < float(args.local_composition_dominance_threshold) < 1.0:
        raise ValueError("--local-composition-dominance-threshold must be in (0, 1)")
    for name in (
        "local_positive_group_threshold",
        "local_negative_group_threshold",
        "local_hydrophobic_group_threshold",
    ):
        if not 0.0 < float(getattr(args, name)) < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1)")
    if args.enable_condition_fold_planner and int(args.fold_plan_slots) < 4:
        raise ValueError("--fold_plan_slots must be at least 4")
    if int(args.fold_plan_prototypes) < int(args.fold_plan_slots):
        raise ValueError("--fold_plan_prototypes must be >= --fold_plan_slots")
    if int(args.fold_plan_archetypes) < 4:
        raise ValueError("--fold_plan_archetypes must be at least 4")
    if args.enable_structure_token_planner:
        if int(args.structure_token_slots) < 16:
            raise ValueError("--structure_token_slots must be at least 16")
        if int(args.structure_token_codes) < 16:
            raise ValueError("--structure_token_codes must be at least 16")
        if not 1 <= int(args.structure_token_prefix_slots) <= int(args.structure_token_slots):
            raise ValueError("--structure_token_prefix_slots must be in [1, structure_token_slots]")
        if not 0.0 <= float(args.structure_token_posterior_teacher_forcing) <= 1.0:
            raise ValueError("--structure_token_posterior_teacher_forcing must be in [0, 1]")
        if not 0.0 <= float(args.structure_token_posterior_final_mix) <= 1.0:
            raise ValueError("--structure_token_posterior_final_mix must be in [0, 1]")
        if int(args.structure_token_posterior_decay_steps) < 0:
            raise ValueError("--structure_token_posterior_decay_steps must be non-negative")
        if int(args.structure_token_geometry_feature_dim) <= 0:
            raise ValueError("--structure_token_geometry_feature_dim must be positive")
    args.full_injection_train_stop_choices_parsed = parse_full_injection_stop_choices(
        args.full_injection_train_stop_choices
    )
    args.full_injection_train_scale_choices_parsed = parse_full_injection_scale_choices(
        args.full_injection_train_scale_choices
    )
    args.prefix_condition_train_scale_choices_parsed = parse_full_injection_scale_choices(
        args.prefix_condition_train_scale_choices
    )
    args.structure_token_train_scale_choices_parsed = parse_full_injection_scale_choices(
        args.structure_token_train_scale_choices
    )
    args.snapshot_steps_parsed = parse_snapshot_steps(args.snapshot_steps)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ckpt_dir = out_dir / "ckpts"
    ensure_dir(ckpt_dir)

    os.environ["HF_HOME"] = str(args.hf_cache_root)
    os.environ["TRANSFORMERS_CACHE"] = str(args.hf_cache_root)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[data] reading dataset.csv from {args.pairs_dir}")
    df = pd.read_csv(Path(args.pairs_dir) / "dataset.csv")
    if bool(args.enable_structure_token_planner) and "protein_sequence" in df.columns:
        longest_residues = int(
            df["protein_sequence"].fillna("").astype(str).str.replace(
                r"\s+", "", regex=True
            ).str.len().max()
        )
        residue_capacity = int(args.max_protein_len) - 2
        if longest_residues > residue_capacity:
            raise ValueError(
                "structure-token training must not truncate residue-aligned targets: "
                f"longest sequence={longest_residues}, residue capacity={residue_capacity}. "
                "Increase --max_protein_len by at least two for BOS/EOS."
            )
    if "pocket_labels" in df.columns:
        has_pocket_positive = any(
            values is not None and any(value > 0.5 for value in values)
            for values in (parse_numeric_list(value) for value in df["pocket_labels"])
        )
        if not has_pocket_positive:
            disabled = []
            for name in ("lambda_pocket", "lambda_interaction", "lambda_complementarity"):
                if float(getattr(args, name)) != 0.0:
                    setattr(args, name, 0.0)
                    disabled.append(name)
            if bool(args.enable_pocket_aware_injection):
                args.enable_pocket_aware_injection = False
                disabled.append("enable_pocket_aware_injection")
            if float(args.alpha_pocket) != 0.0:
                args.alpha_pocket = 0.0
                disabled.append("alpha_pocket")
            if disabled:
                print(
                    "[data] pocket_labels contain no positive residues; disabled "
                    + ", ".join(disabled)
                    + " to avoid all-negative pocket/interface supervision."
                )
    df_train, df_val, df_test = split_df(
        df,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        group_by=args.split_group,
    )
    required_splits = {"train": df_train, "val": df_val, "test": df_test}
    empty_splits = [name for name, part in required_splits.items() if part.empty]
    if empty_splits:
        raise ValueError(
            f"Empty required split(s): {empty_splits}. Increase dataset size, adjust ratios, or choose a finer --split_group."
        )
    split_audit = write_split_artifacts(
        out_dir,
        df,
        {"train": df_train, "val": df_val, "test": df_test},
        args.split_group,
        strict_sequence_disjoint=bool(args.strict_sequence_disjoint),
    )
    print(f"[data] train={len(df_train)} val={len(df_val)} test={len(df_test)} split_group={args.split_group}")
    print(f"[data] split audit -> {out_dir / 'splits' / 'split_audit.json'}")
    print(f"[init] device={device}")

    try:
        from ..architecture.model import DockingWizardModel, DockingWizardModelConfig
    except ImportError:  # Support direct execution from a source checkout.
        import sys
        sys.path.insert(0, args.code_dir)
        from langpro.architecture.model import (  # type: ignore
            DockingWizardModel,
            DockingWizardModelConfig,
        )

    # ``--hf_cache_root`` is the public training interface and must not be
    # shadowed by workstation-specific paths baked into the source.  Export
    # the resolved paths as well because the model submodules consume these
    # environment variables during construction.
    hf_cache_root = Path(args.hf_cache_root).expanduser().resolve()
    PUB_MED_BERT_DIR = str(hf_cache_root / "PubMedBERT")
    SMILES_MODEL_DIR = str(hf_cache_root / "SMILES_tokenized_PubChem_shard00_160k")
    PROGEN_MODEL_DIR = str(hf_cache_root / "progen2-base-hf")
    os.environ["PUB_MED_BERT_DIR"] = PUB_MED_BERT_DIR
    os.environ["SMILES_MODEL_DIR"] = SMILES_MODEL_DIR
    os.environ["PROGEN_MODEL_DIR"] = PROGEN_MODEL_DIR

    tok_pubmed = AutoTokenizer.from_pretrained(
        PUB_MED_BERT_DIR,
        local_files_only=True,
        trust_remote_code=True
    )
    print("[tok] pubmedbert: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    tok_smiles = AutoTokenizer.from_pretrained(
        SMILES_MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True
    )
    print("[tok] smiles_lm: seyonec/SMILES_tokenized_PubChem_shard00_160k")

    tok_protein = AutoTokenizer.from_pretrained(
        PROGEN_MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True
    )
    print(f"[tok] protein(decoder): {args.decoder_backbone}")
    if args.protein_control_tag:
        start_ids = tok_protein(
            str(args.protein_control_tag), add_special_tokens=False
        )["input_ids"]
        if len(start_ids) != 1:
            raise ValueError(
                f"--protein_control_tag={args.protein_control_tag!r} maps to "
                f"{start_ids}; expected exactly one token"
            )
        args.protein_start_token_id = int(start_ids[0])
        print(
            f"[tok] protein control tag {args.protein_control_tag!r} -> "
            f"start_token_id={args.protein_start_token_id}"
        )
    else:
        args.protein_start_token_id = None

    local_residue_token_groups = []
    for residue_group in ("KR", "DE", "AILMFWVY"):
        token_ids = []
        for residue in residue_group:
            encoded = tok_protein(residue, add_special_tokens=False)["input_ids"]
            if len(encoded) != 1:
                raise ValueError(
                    f"protein residue {residue!r} maps to {encoded}; expected one token"
                )
            token_ids.append(int(encoded[0]))
        local_residue_token_groups.append(token_ids)

    cfg = DockingWizardModelConfig(
        decoder_backbone_name_or_path=args.decoder_backbone,
        gen_bos_token_id=args.protein_start_token_id,
        decoder_unfreeze_last_n_blocks=int(args.decoder_unfreeze_last_n_blocks),
        decoder_prefix_boundary_scale=float(args.prefix_boundary_scale),
        eos_loss_weight=float(args.eos_loss_weight),
        lambda_diversity=float(args.lambda_diversity),
        lambda_local_composition_unlikelihood=float(
            args.lambda_local_composition_unlikelihood
        ),
        local_composition_window=int(args.local_composition_window),
        local_composition_min_context=int(args.local_composition_min_context),
        local_composition_dominance_threshold=float(
            args.local_composition_dominance_threshold
        ),
        lambda_local_group_unlikelihood=float(
            args.lambda_local_group_unlikelihood
        ),
        local_residue_token_groups=local_residue_token_groups,
        local_residue_group_thresholds=[
            float(args.local_positive_group_threshold),
            float(args.local_negative_group_threshold),
            float(args.local_hydrophobic_group_threshold),
        ],
        lambda_route_comp=float(args.lambda_route_comp),
        route_comp_temperature=float(args.route_comp_temperature),
        lambda_chem_align=float(args.lambda_chem_align),
        lambda_query_diversity=float(args.lambda_query_diversity),
        n_chem_features=int(args.n_chem_features),
        chem_loss_type=str(args.chem_loss_type),
        use_hard_negative_contrastive=bool(args.use_hard_negative_contrastive),
        hard_negative_weight=float(args.hard_negative_weight),
        ligand_memory_bank_size=int(args.ligand_memory_bank_size),
        ligand_contrastive_temperature=float(args.ligand_contrastive_temperature),
        n_interaction_types=int(args.n_interaction_types),
        lambda_pocket=float(args.lambda_pocket),
        lambda_contact=float(args.lambda_contact),
        lambda_structure_contacts=float(args.lambda_structure_contacts),
        structure_contact_proj_dim=int(args.structure_contact_proj_dim),
        structure_contact_negative_ratio=float(args.structure_contact_negative_ratio),
        lambda_interaction=float(args.lambda_interaction),
        lambda_secondary_structure=float(args.lambda_secondary_structure),
        enable_condition_fold_planner=bool(args.enable_condition_fold_planner),
        fold_plan_slots=int(args.fold_plan_slots),
        fold_plan_prototypes=int(args.fold_plan_prototypes),
        fold_plan_archetypes=int(args.fold_plan_archetypes),
        fold_plan_layers=int(args.fold_plan_layers),
        fold_plan_heads=int(args.fold_plan_heads),
        fold_plan_dropout=float(args.fold_plan_dropout),
        fold_plan_prototype_temperature=float(args.fold_plan_prototype_temperature),
        fold_plan_condition_dropout=float(args.fold_plan_condition_dropout),
        fold_plan_local_chemistry_scale=float(args.fold_plan_local_chemistry_scale),
        fold_plan_archetype_scale=float(args.fold_plan_archetype_scale),
        fold_plan_prefix_scale=float(args.fold_plan_prefix_scale),
        fold_plan_full_memory_scale=float(args.fold_plan_full_memory_scale),
        fold_plan_injection_warmup_steps=int(args.fold_plan_injection_warmup_steps),
        lambda_fold_plan_ss=float(args.lambda_fold_plan_ss),
        lambda_fold_plan_contacts=float(args.lambda_fold_plan_contacts),
        lambda_fold_plan_global=float(args.lambda_fold_plan_global),
        lambda_fold_plan_codebook=float(args.lambda_fold_plan_codebook),
        lambda_fold_plan_chemistry=float(args.lambda_fold_plan_chemistry),
        lambda_fold_plan_sequence=float(args.lambda_fold_plan_sequence),
        lambda_fold_plan_pocket=float(args.lambda_fold_plan_pocket),
        lambda_fold_plan_archetype=float(args.lambda_fold_plan_archetype),
        enable_structure_token_planner=bool(args.enable_structure_token_planner),
        structure_token_slots=int(args.structure_token_slots),
        structure_token_codes=int(args.structure_token_codes),
        structure_token_prefix_slots=int(args.structure_token_prefix_slots),
        structure_token_layers=int(args.structure_token_layers),
        structure_token_heads=int(args.structure_token_heads),
        structure_token_dropout=float(args.structure_token_dropout),
        structure_token_temperature=float(args.structure_token_temperature),
        structure_token_posterior_teacher_forcing=float(
            args.structure_token_posterior_teacher_forcing
        ),
        structure_token_posterior_final_mix=float(
            args.structure_token_posterior_final_mix
        ),
        structure_token_posterior_decay_steps=int(
            args.structure_token_posterior_decay_steps
        ),
        structure_token_geometry_feature_dim=int(
            args.structure_token_geometry_feature_dim
        ),
        structure_token_prefix_scale=float(args.structure_token_prefix_scale),
        structure_token_full_memory_scale=float(args.structure_token_full_memory_scale),
        structure_token_train_scale_choices=list(
            args.structure_token_train_scale_choices_parsed
        ),
        lambda_structure_token_prior=float(args.lambda_structure_token_prior),
        lambda_structure_token_posterior=float(args.lambda_structure_token_posterior),
        lambda_structure_token_kl=float(args.lambda_structure_token_kl),
        lambda_structure_token_pocket=float(args.lambda_structure_token_pocket),
        lambda_structure_token_interaction=float(args.lambda_structure_token_interaction),
        lambda_structure_token_topology=float(args.lambda_structure_token_topology),
        lambda_solvent_accessibility=float(args.lambda_solvent_accessibility),
        lambda_residue_depth=float(args.lambda_residue_depth),
        lambda_complementarity=float(args.lambda_complementarity),
        complementarity_loss_type=str(args.complementarity_loss_type),
        lambda_condition_awareness=float(args.lambda_condition_awareness),
        cond_awareness_prob=float(args.cond_awareness_prob),
        condition_awareness_max_similarity=float(args.condition_awareness_max_similarity),
        lambda_condition_contrastive=float(args.lambda_condition_contrastive),
        condition_contrastive_temperature=float(args.condition_contrastive_temperature),
        lambda_length_prediction=float(args.lambda_length_prediction),
        protein_length_scale=float(args.max_protein_len),
        perceiver_layerscale_init=float(args.perceiver_layerscale_init),
        perceiver_cross_every_block=bool(args.perceiver_cross_every_block),
        use_bio_position_weights=bool(args.use_bio_position_weights),
        alpha_conservation=float(args.alpha_conservation),
        alpha_pocket=float(args.alpha_pocket),
        alpha_contact=float(args.alpha_contact),
        enable_pocket_aware_injection=bool(args.enable_pocket_aware_injection),
        pocket_gate_strength=float(args.pocket_gate_strength),
        aux_loss_warmup_steps=int(args.aux_loss_warmup_steps),
        aux_loss_warmup_epochs=int(args.aux_loss_warmup_epochs),
        aux_loss_warmup_mode=str(args.aux_loss_warmup_mode),
        enable_full_injection_train=not bool(args.disable_full_injection_train),
        enable_full_injection_generate=not bool(args.disable_full_injection_generate),
        full_injection_layer_policy=str(args.full_injection_layer_policy),
        full_injection_last_k=int(args.full_injection_last_k),
        share_full_injection_weights=bool(args.share_full_injection_weights),
        full_injection_train_stop_choices=list(args.full_injection_train_stop_choices_parsed),
        full_injection_train_scale_choices=list(args.full_injection_train_scale_choices_parsed),
        full_injection_eval_scale=float(args.full_injection_eval_scale),
        prefix_condition_train_scale_choices=list(args.prefix_condition_train_scale_choices_parsed),
        pool_condition_memory=bool(args.pool_condition_memory),
    )
    model = DockingWizardModel(cfg).to(device)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint).expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        init_payload = torch.load(init_path, map_location="cpu")
        init_state = init_payload.get("model", init_payload)
        compact_init = str(init_payload.get("checkpoint_type", "")).startswith("compact")
        skipped_backbone_keys = 0
        if args.init_skip_decoder_backbone:
            backbone_keys = [
                key for key in init_state
                if key.startswith("decoder.backbone.")
            ]
            for key in backbone_keys:
                del init_state[key]
            skipped_backbone_keys = len(backbone_keys)
            compact_init = True
        load_checkpoint_model_state(model, init_state, compact=compact_init)
        del init_state, init_payload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[init] loaded model weights from {init_path} "
            f"(compact={compact_init}, skipped_decoder_backbone_keys={skipped_backbone_keys})"
        )
    if args.model_dtype == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("--model_dtype bf16 requires a BF16-capable CUDA GPU.")
        model.to(dtype=torch.bfloat16)
        print("[memory] model parameters cast to BF16")
    if args.gradient_checkpointing:
        decoder_backbone = getattr(getattr(model, "decoder", None), "backbone", None)
        if decoder_backbone is None:
            raise RuntimeError("Decoder backbone does not support gradient checkpointing.")
        if bool(getattr(decoder_backbone, "supports_gradient_checkpointing", False)):
            decoder_backbone.gradient_checkpointing_enable()
            if hasattr(decoder_backbone, "enable_input_require_grads"):
                decoder_backbone.enable_input_require_grads()
            if hasattr(decoder_backbone, "config") and hasattr(decoder_backbone.config, "use_cache"):
                decoder_backbone.config.use_cache = False
            print("[memory] decoder gradient checkpointing enabled")
        else:
            print(
                "[memory] decoder does not expose gradient checkpointing; "
                "continuing with last-hidden-only capture and BF16"
            )
    if args.enable_structure_token_planner and args.structure_token_codebook_npz:
        codebook_path = Path(args.structure_token_codebook_npz).expanduser().resolve()
        if not codebook_path.is_file():
            raise FileNotFoundError(f"structure token codebook not found: {codebook_path}")
        with np.load(codebook_path, allow_pickle=False) as codebook_payload:
            centers = np.asarray(codebook_payload["centers"], dtype=np.float32)
        model.structure_token_planner.set_geometry_code_features(
            torch.from_numpy(centers)
        )
        print(
            f"[init] loaded geometry code features {tuple(centers.shape)} "
            f"from {codebook_path}"
        )
    model.train()

    if args.use_param_groups:
        param_groups = build_param_groups(
            model,
            args.weight_decay,
            base_lr=args.lr,
            lr_encoder=args.lr_encoder,
            lr_perceiver=args.lr_perceiver,
            lr_decoder_new=args.lr_decoder_new,
            lr_decoder_backbone=args.lr_decoder_backbone,
            lr_fold_planner=args.lr_fold_planner,
            lr_length_head=args.lr_length_head,
        )
        if args.optimizer == "adafactor":
            opt = Adafactor(
                param_groups,
                lr=args.lr,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
        else:
            opt = AdamW(param_groups, lr=args.lr)
        print(
            f"[opt] using module-wise {args.optimizer} groups: "
            + ", ".join(
                f"{g['group_name']}={g['lr']:.3g} (n={len(g['params'])})"
                for g in param_groups
            )
        )
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if args.optimizer == "adafactor":
            opt = Adafactor(
                trainable_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
        else:
            opt = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        print(f"[opt] using flat {args.optimizer}")

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and str(args.amp_dtype).lower() == "fp16")
    )
    print(f"[memory] amp_dtype={args.amp_dtype}; fp16_grad_scaler={scaler.is_enabled()}")

    # Learning-rate scheduler.
    total_steps = len(df_train) // args.batch_size * args.epochs
    warmup_scheduler = None
    cosine_scheduler = None
    if args.use_cosine_schedule:
        warmup_scheduler = LinearLR(opt, start_factor=0.1, total_iters=args.warmup_steps)
        cosine_scheduler = CosineAnnealingLR(opt, T_max=total_steps - args.warmup_steps)
        print(f"[opt] using cosine annealing LR (warmup={args.warmup_steps}, total={total_steps})")

    lig_npz = np.load(args.lig3d_npz, allow_pickle=True)

    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    with open(out_dir / "model_cfg.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    log_path = out_dir / "train.log.txt"

    def log(msg: str) -> None:
        s = f"[{now_str()}] {msg}"
        print(s)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    best_val = float("inf")
    global_step = 0

    def maybe_save_best(metrics: Dict[str, float], epoch: int, step: int) -> None:
        nonlocal best_val
        val = metrics.get(args.selection_metric, float("nan"))
        if not math.isnan(val) and val < best_val:
            best_val = val
            ckpt_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "step": step,
                    "epoch": epoch,
                    "best_val": best_val,
                    "model": checkpoint_model_state(model, full_checkpoint=bool(args.full_checkpoint)),
                    "cfg": asdict(cfg),
                    "args": vars(args),
                    "split_group": args.split_group,
                    "checkpoint_type": (
                        "full_best_model_only"
                        if args.full_checkpoint
                        else "compact_best_model_only"
                    ),
                },
                ckpt_path,
            )
            log(
                f"[ckpt] saved best to {ckpt_path} "
                f"({args.selection_metric}={best_val:.6f})"
            )
        
    def save_last(epoch: int, step: int) -> None:
        ckpt_path = ckpt_dir / "last.pt"
        torch.save(
            {
                "step": step,
                "epoch": epoch,
                "best_val": best_val,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "cfg": asdict(cfg),
                "args": vars(args),
                "split_group": args.split_group,
            },
            ckpt_path,
        )
        log(f"[ckpt] saved last to {ckpt_path}")

    def save_snapshot(epoch: int, step: int) -> None:
        ckpt_path = ckpt_dir / f"snapshot_step_{int(step):08d}.pt"
        torch.save(
            {
                "step": int(step),
                "epoch": int(epoch),
                "best_val": best_val,
                "model": checkpoint_model_state(model, full_checkpoint=False),
                "cfg": asdict(cfg),
                "args": vars(args),
                "split_group": args.split_group,
                "checkpoint_type": "compact_external_sentinel_snapshot",
            },
            ckpt_path,
        )
        log(f"[ckpt] saved external sentinel snapshot to {ckpt_path}")

    log("start training")
    seen_extra_fields = set()

    for epoch in range(args.epochs):
        epoch_seed = args.seed + epoch
        batches = iter_minibatches(
            df_train,
            args.batch_size,
            shuffle=True,
            seed=epoch_seed,
            sample_weight_column=args.sample_weight_column,
            balance_group_columns=args.balance_group_columns,
            balance_power=args.balance_power,
        )

        for bdf in batches:
            batch = build_model_batch(
                bdf,
                lig_npz,
                tok_pubmed,
                tok_smiles,
                tok_protein,
                args,
                randomize_smiles_input=bool(args.random_smiles_augmentation),
            )
            protein_eos_stats = batch.get("_protein_eos_stats", {})
            detected = batch.get("_detected_extra_fields", [])
            new_fields = sorted(set(detected) - seen_extra_fields)
            if new_fields:
                seen_extra_fields.update(new_fields)
                log(f"[data] detected enhanced fields: {new_fields}")
            batch = to_device(batch, device)

            with autocast_context(args, device):
                out = model(batch, return_debug=False, global_step=global_step, epoch=epoch)
                loss = out.loss_total
            if loss is None:
                raise RuntimeError("Model returned loss_total=None; check forward() and batch content.")

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            # Advance the learning-rate scheduler.
            if warmup_scheduler is not None and global_step < args.warmup_steps:
                warmup_scheduler.step()
            elif cosine_scheduler is not None and global_step >= args.warmup_steps:
                cosine_scheduler.step()

            global_step += 1

            if global_step == 1 or global_step % args.print_every == 0:
                eos_diag = ""
                if isinstance(protein_eos_stats, dict) and protein_eos_stats:
                    eos_diag = (
                        " protein_eos="
                        f"eos_per_sample={protein_eos_stats.get('eos_per_sample', float('nan')):.3f},"
                        f"trunc_frac={protein_eos_stats.get('truncated_frac', float('nan')):.3f},"
                        f"raw_len_mean={protein_eos_stats.get('raw_len_mean', float('nan')):.1f},"
                        f"raw_len_max={protein_eos_stats.get('raw_len_max', float('nan')):.0f},"
                        f"bos/eos/pad={int(protein_eos_stats.get('bos_id', -1))}/"
                        f"{int(protein_eos_stats.get('eos_id', -1))}/"
                        f"{int(protein_eos_stats.get('pad_id', -1))}"
                    )
                if global_step % args.print_every == 0:
                    loss_total = float(out.loss_total.detach().cpu())
                    loss_lm = float(out.loss_lm.detach().cpu()) if out.loss_lm is not None else float("nan")
                    aux = {k: float(v.detach().cpu()) for k, v in (out.loss_aux or {}).items()}
                    eff = model._effective_aux_lambdas(global_step=global_step, epoch=epoch)
                    log(f"epoch={epoch+1}/{args.epochs} step={global_step} loss_total={loss_total:.6f} loss_lm={loss_lm:.6f} aux={aux} effective_lambdas={eff}{eos_diag}")
                else:
                    log(f"[data] protein EOS supervision enabled.{eos_diag}")

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                metrics = eval_loss(model, df_val, lig_npz, tok_pubmed, tok_smiles, tok_protein, args, device, max_batches=20)
                log(f"[val] step={global_step} {metrics}")
                maybe_save_best(metrics, epoch=epoch, step=global_step)

            if global_step in args.snapshot_steps_parsed:
                save_snapshot(epoch=epoch, step=global_step)

        # epoch end: run one more validation to avoid missing the best
        metrics = eval_loss(model, df_val, lig_npz, tok_pubmed, tok_smiles, tok_protein, args, device, max_batches=None)
        log(f"[val] epoch_end epoch={epoch+1}/{args.epochs} step={global_step} {metrics}")
        maybe_save_best(metrics, epoch=epoch, step=global_step)
        if args.save_last_checkpoint:
            save_last(epoch=epoch, step=global_step)

    best_path = ckpt_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("Training finished without best.pt; validation did not produce a finite loss.")
    # Loading a full checkpoint directly on CUDA duplicates the model and may
    # also materialize the optimizer state on GPU. Keep checkpoint tensors on
    # CPU, discard optimizer state, then copy parameters into the existing
    # model one at a time through load_state_dict.
    best_checkpoint = torch.load(best_path, map_location="cpu")
    best_model_state = best_checkpoint.pop("model")
    best_checkpoint.pop("opt", None)
    del best_checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    load_checkpoint_model_state(
        model,
        best_model_state,
        compact=not bool(args.full_checkpoint),
    )
    del best_model_state
    gc.collect()
    test_metrics = eval_loss(
        model, df_test, lig_npz, tok_pubmed, tok_smiles, tok_protein, args, device, max_batches=None
    )
    test_payload = {
        "checkpoint": str(best_path),
        "selected_by": f"validation {args.selection_metric}",
        "run_label": args.run_label,
        "best_val": best_val,
        "metrics": test_metrics,
        "split_audit": split_audit,
    }
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_payload, f, indent=2, ensure_ascii=False)
    log(f"[test] best_checkpoint={best_path} metrics={test_metrics}")
    log(f"[done] training finished. best_val={best_val:.6f}; final test metrics -> {out_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
