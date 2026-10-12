"""Multimodal condition encoder used before the LangPro Perceiver.

Long text and SMILES inputs are chunked and hierarchically pooled. Ligand text,
SMILES, and 3D geometry are aligned into a shared context space and fused with
learned gates. The module emits context tokens, masks, and optional auxiliary
alignment losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import math
import torch
import os
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel, AutoModelForMaskedLM


# -----------------------------
# Config
# -----------------------------
@dataclass
class PrePerceiverEncoderConfig:
    # ---------- pretrained backbone names ----------
    pubmedbert_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    smiles_lm_name: str = "seyonec/SMILES_tokenized_PubChem_shard00_160k"

    # ---------- d_ctx (Perceiver input dim) ----------
    # Use >=768 to avoid dimensional compression. 1024 is a good default.
    d_ctx: int = 1024

    # ---------- chunking (keep long info, avoid hard truncation) ----------
    # Text chunk settings
    dock_chunk_len: int = 256
    dock_chunk_overlap: int = 64
    func_chunk_len: int = 256
    func_chunk_overlap: int = 0
    ligtext_chunk_len: int = 192
    ligtext_chunk_overlap: int = 0

    # SMILES chunk settings
    smiles_chunk_len: int = 192
    smiles_chunk_overlap: int = 32

    # Optional maximum chunks (to bound worst-case compute; set None to keep all)
    dock_max_chunks: Optional[int] = None
    func_max_chunks: Optional[int] = None
    ligtext_max_chunks: Optional[int] = None
    smiles_max_chunks: Optional[int] = None

    # ---------- hierarchical pooling tokens ----------
    # per-chunk summary tokens and final summary tokens
    text_chunk_summary_tokens: int = 12
    dock_final_tokens: int = 96
    func_final_tokens: int = 48
    # ligand_text is a ligand view -> should match view tokens
    ligand_view_tokens: int = 32
    smiles_view_tokens: int = 32
    lig3d_view_tokens: int = 32
    pocket_final_tokens: int = 32
    pocket_interaction_types: int = 6
    chem_requirement_dim: int = 10

    # ---------- ligand 3D atom handling ----------
    lig3d_keep_all_threshold: int = 192    # keep all atoms if N<=threshold
    lig3d_fps_cap: int = 256              # if N>cap, FPS down to cap before pooling

    # ---------- attention pooling / adapter details ----------
    attn_heads: int = 8
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0
    adapter_mlp_ratio: float = 2.0
    num_pool_layers: int = 3

    # ---------- contrastive learning ----------
    contrastive_proj_dim: int = 256
    contrastive_temperature: float = 0.07
    aux_contrastive_weight: float = 0.05

    # ---------- ligand fusion constraint ----------
    # weight = (1-alpha)*softmax(gate) + alpha*(1/3) ensures no view is ever dropped
    ligand_weight_floor_alpha: float = 0.20

    # ---------- type embeddings ----------
    use_type_embeddings: bool = True

    # ---------- freezing ----------
    freeze_backbones: bool = True
    no_grad_backbone_forward: bool = True


# -----------------------------
# Small utilities
# -----------------------------
def freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [B,L,D], mask: [B,L] with 1 valid else 0
    """
    mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
    s = (x * mask_f).sum(dim=1)
    d = mask_f.sum(dim=1).clamp_min(eps)
    return s / d


def split_into_chunks(
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    chunk_len: int,
    overlap: int,
    max_chunks: Optional[int] = None
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Split tokenized sequence into overlapping chunks.

    Returns list of tuples:
      (chunk_ids [B, Lc], chunk_mask [B, Lc], chunk_valid [B]) where chunk_valid indicates any token exists.
    """
    assert chunk_len > 0
    step = chunk_len - overlap
    assert step > 0, "chunk_len must be > overlap"
    B, L = input_ids.shape

    chunks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    starts = list(range(0, L, step))
    if max_chunks is not None:
        starts = starts[:max_chunks]

    for s in starts:
        e = min(s + chunk_len, L)
        ids = input_ids[:, s:e]
        m = attn_mask[:, s:e]
        valid = (m.sum(dim=1) > 0)  # [B]
        chunks.append((ids, m, valid))
        if e >= L:
            break
    return chunks


def clip_infonce(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.07,
    group_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Symmetric CLIP-style InfoNCE with optional multi-positive groups.

    Repeated ligands are common in DockingWizard datasets. Treating another row
    containing the same molecule as a negative creates a contradictory target.
    ``group_ids`` marks all rows describing the same ligand as positives.
    z_a, z_b: [B,D] already normalized
    """
    logits = (z_a @ z_b.t()) / temperature
    if group_ids is None:
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a + loss_b)

    ids = group_ids.to(device=logits.device).view(-1)
    if ids.numel() != logits.size(0):
        raise ValueError(
            f"group_ids length={ids.numel()} does not match batch={logits.size(0)}"
        )
    positive = ids[:, None].eq(ids[None, :])
    positive.fill_diagonal_(True)

    def _directional_multi_positive_loss(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        log_prob = x - torch.logsumexp(x, dim=1, keepdim=True)
        positive_log_prob = torch.logsumexp(
            log_prob.masked_fill(~pos, torch.finfo(log_prob.dtype).min),
            dim=1,
        )
        return -positive_log_prob.mean()

    return 0.5 * (
        _directional_multi_positive_loss(logits, positive)
        + _directional_multi_positive_loss(logits.t(), positive.t())
    )


# -----------------------------
# Information-preserving projection & calibrators
# -----------------------------
class IdentityInitUpProject(nn.Module):
    """
    Linear up-projection from in_dim -> out_dim with identity-like init to preserve information.
    - Works best when out_dim >= in_dim.
    - bias=False.
    - Initializes weight so first in_dim rows form identity, remaining rows zeros.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if out_dim < in_dim:
            raise ValueError(f"IdentityInitUpProject requires out_dim>=in_dim, got {out_dim}<{in_dim}.")
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_parameters_identity()

    def reset_parameters_identity(self) -> None:
        with torch.no_grad():
            self.lin.weight.zero_()
            # weight: [out_dim, in_dim]
            for i in range(self.lin.in_features):
                self.lin.weight[i, i] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class ResidualAdapterBlock(nn.Module):
    """
    Near-identity trainable adapter block:
      y = x + MLP(LN(x))
    with MLP last layer zero-init for stability.
    """
    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        # zero init to start as identity
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class Stage1AlignProjector(nn.Module):
    """
    Stage-1 alignment: backbone hidden -> d_ctx with minimal information loss:
      x_up = IdentityInitUpProject(in_dim -> d_ctx)(x)
      x_out = Adapter(x_up)  (near-identity)
    """
    def __init__(self, in_dim: int, d_ctx: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.up = IdentityInitUpProject(in_dim, d_ctx) if d_ctx != in_dim else nn.Identity()
        self.adapt = ResidualAdapterBlock(d_ctx, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.adapt(x)
        return x


class FinalCalibrator(nn.Module):
    """
    Stage-2 alignment: very light per-group calibrator (near identity).
    """
    def __init__(self, d_ctx: int, mlp_ratio: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.block = ResidualAdapterBlock(d_ctx, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# -----------------------------
# Learnable Query Pooling
# -----------------------------
class QueryPool(nn.Module):
    """
    Learnable pooling with K trainable queries to softly select information from a variable-length token set.
    Implements stacked cross-attention + residual MLP.

    Inputs:
      kv: [B, L, D]
      kv_mask: [B, L] (1 valid, 0 pad)
    Output:
      q_out: [B, K, D]  (K learnable summary tokens)
    """
    def __init__(
        self,
        d_model: int,
        num_queries: int,
        num_heads: int = 8,
        num_layers: int = 2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        mlp_ratio: float = 2.0
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(_QueryPoolLayer(d_model, num_heads, attn_dropout, ffn_dropout, mlp_ratio))

    def forward(self, kv: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        B = kv.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B,K,D]

        # key_padding_mask expects True for padding positions
        key_padding = ~kv_mask.bool()  # [B,L]
        for layer in self.layers:
            q = layer(q, kv, key_padding)
        return q


class ExplicitPocketInteractionExtractor(nn.Module):
    """Extract fixed pocket-interaction tokens with typed learned queries."""
    def __init__(
        self,
        d_model: int,
        num_interaction_types: int,
        num_pocket_tokens: int,
        num_heads: int = 8,
        num_layers: int = 2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.type_queries = nn.Parameter(torch.randn(num_interaction_types, d_model) * 0.02)
        self.pocket_queries = nn.Parameter(torch.randn(num_pocket_tokens, d_model) * 0.02)
        self.type_layers = nn.ModuleList([
            _QueryPoolLayer(d_model, num_heads, attn_dropout, ffn_dropout, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.expand_layers = nn.ModuleList([
            _QueryPoolLayer(d_model, num_heads, attn_dropout, ffn_dropout, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.dock_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, dock_tokens: torch.Tensor, dock_mask: torch.Tensor, ligand_tokens: torch.Tensor, ligand_mask: torch.Tensor) -> torch.Tensor:
        B = dock_tokens.size(0)
        kv = torch.cat([dock_tokens, ligand_tokens], dim=1)
        kv_mask = torch.cat([dock_mask, ligand_mask], dim=1).bool()
        key_padding = ~kv_mask

        q_type = self.type_queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.type_layers:
            q_type = layer(q_type, kv, key_padding)

        # Docking text controls the sample-specific interaction-type weights.
        dock_summary = masked_mean_pool(dock_tokens, dock_mask)
        type_gate = self.dock_gate(q_type + dock_summary.unsqueeze(1))
        q_type = q_type * type_gate

        q_pocket = self.pocket_queries.unsqueeze(0).expand(B, -1, -1)
        type_padding = torch.zeros(B, q_type.size(1), device=q_type.device, dtype=torch.bool)
        for layer in self.expand_layers:
            q_pocket = layer(q_pocket, q_type, type_padding)
        return self.out_norm(q_pocket)


class _QueryPoolLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_dropout: float, ffn_dropout: float, mlp_ratio: float):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.drop = nn.Dropout(ffn_dropout)
        self.ffn = ResidualAdapterBlock(d_model, mlp_ratio=mlp_ratio, dropout=ffn_dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        qn = self.ln_q(q)
        kvn = self.ln_kv(kv)
        # attn: query attends to kv
        attn_out, _ = self.attn(qn, kvn, kvn, key_padding_mask=key_padding_mask, need_weights=False)
        q = q + self.drop(attn_out)
        q = self.ffn(q)
        return q


# -----------------------------
# Frozen Backbones
# -----------------------------
class FrozenPubMedBERT(nn.Module):
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        model_source = os.environ.get("PUB_MED_BERT_DIR", model_name)
        self.model = AutoModel.from_pretrained(
            model_source,
            local_files_only=os.path.isdir(model_source),
            trust_remote_code=True
        )
        if freeze:
            freeze_module(self.model)
            self.model.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, no_grad: bool = True) -> torch.Tensor:
        if no_grad:
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        else:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return out.last_hidden_state


class FrozenSmilesMaskedLM(nn.Module):
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        model_source = os.environ.get("SMILES_MODEL_DIR", model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_source,
            local_files_only=os.path.isdir(model_source),
            trust_remote_code=True
        )
        if freeze:
            freeze_module(self.model)
            self.model.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, no_grad: bool = True) -> torch.Tensor:
        if no_grad:
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
        else:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        return out.hidden_states[-1]


# -----------------------------
# 3D Encoder (fallback EGNN + loadable weights)
# -----------------------------
class EGNN3DEncoder(nn.Module):
    """
    Dependency-free EGNN-like 3D encoder.
    You can load a pretrained checkpoint via load_state_dict externally.
    """
    def __init__(self, out_dim: int = 768, n_layers: int = 4, max_z: int = 128, edge_hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.atom_embed = nn.Embedding(max_z, out_dim)
        self.layers = nn.ModuleList([_EGNNLayer(out_dim, edge_hidden=edge_hidden, dropout=dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(out_dim)

    @property
    def hidden_size(self) -> int:
        return int(self.atom_embed.embedding_dim)

    def forward(self, Z: torch.Tensor, coords: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        Zc = Z.clamp(min=0, max=self.atom_embed.num_embeddings - 1)
        h = self.atom_embed(Zc)  # [B,N,D]
        x = coords
        m = atom_mask.bool()
        for layer in self.layers:
            h, x = layer(h, x, m)
        return self.ln(h)


class _EGNNLayer(nn.Module):
    def __init__(self, dim: int, edge_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1, edge_hidden),
            nn.SiLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(dim + edge_hidden, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(edge_hidden, edge_hidden),
            nn.SiLU(),
            nn.Linear(edge_hidden, 1),
            nn.Tanh(),
        )
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = h.shape
        dx = x.unsqueeze(2) - x.unsqueeze(1)            # [B,N,N,3]
        dist2 = (dx * dx).sum(dim=-1, keepdim=True)     # [B,N,N,1]
        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)
        e_in = torch.cat([hi, hj, dist2], dim=-1)       # [B,N,N,2D+1]
        e = self.edge_mlp(e_in)                         # [B,N,N,Eh]

        edge_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1)  # [B,N,N,1]
        e = e * edge_mask.to(dtype=e.dtype)

        m_agg = e.sum(dim=2)                            # [B,N,Eh]
        h_up = self.node_mlp(torch.cat([h, m_agg], dim=-1))
        h = self.ln(h + self.drop(h_up))

        dx_norm = dx / (dx.norm(dim=-1, keepdim=True) + 1e-6)  # [B,N,N,3]
        w = self.coord_mlp(e)                                  # [B,N,N,1]
        x_up = (dx_norm * w).sum(dim=2)                        # [B,N,3]
        x = x + x_up * mask.to(dtype=x.dtype).unsqueeze(-1)
        return h, x


# -----------------------------
# Geometric sampling (FPS)
# -----------------------------
def farthest_point_sampling_single(coords: torch.Tensor, k: int) -> torch.Tensor:
    """
    coords: [N,3] float
    returns indices: [k]
    Greedy FPS. O(N*k). Good for moderate N (<=~1000).
    """
    N = coords.size(0)
    if k >= N:
        return torch.arange(N, device=coords.device)

    # pick first point as 0
    selected = torch.zeros(k, dtype=torch.long, device=coords.device)
    selected[0] = 0
    dist = torch.full((N,), float("inf"), device=coords.device)

    last = coords[0:1]  # [1,3]
    for i in range(1, k):
        d = torch.cdist(coords, last).squeeze(-1)  # [N]
        dist = torch.minimum(dist, d)
        selected[i] = torch.argmax(dist)
        last = coords[selected[i]:selected[i]+1]
    return selected


def fps_batch(coords: torch.Tensor, mask: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    coords: [B,N,3]
    mask:   [B,N] (1 valid)
    returns:
      coords_k: [B,k,3]
      idx_k:    [B,k] indices in original N (padded with 0 if not enough)
      mask_k:   [B,k] (1 valid)
    """
    B, N, _ = coords.shape
    idx_out = torch.zeros(B, k, dtype=torch.long, device=coords.device)
    mask_out = torch.zeros(B, k, dtype=mask.dtype, device=mask.device)
    coords_out = torch.zeros(B, k, 3, dtype=coords.dtype, device=coords.device)

    for b in range(B):
        valid_idx = torch.nonzero(mask[b].bool(), as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        c = coords[b, valid_idx]  # [Nv,3]
        kk = min(k, c.size(0))
        sel_local = farthest_point_sampling_single(c, kk)  # [kk]
        sel = valid_idx[sel_local]  # map back to original indices

        idx_out[b, :kk] = sel
        mask_out[b, :kk] = 1
        coords_out[b, :kk] = coords[b, sel]

        # If kk<k, remaining are zeros masked out.
    return coords_out, idx_out, mask_out


# -----------------------------
# Chunked encoding + hierarchical pooling
# -----------------------------
class ChunkedHierPool(nn.Module):
    """
    Generic: chunk -> backbone -> stage1_align -> per-chunk QueryPool -> concat -> global QueryPool.
    Returns fixed summary tokens [B, M_final, d_ctx].
    """
    def __init__(
        self,
        d_ctx: int,
        chunk_summary_tokens: int,
        final_tokens: int,
        num_heads: int,
        num_layers: int,
        attn_dropout: float,
        ffn_dropout: float,
        mlp_ratio: float,
    ):
        super().__init__()
        self.chunk_pool = QueryPool(
            d_model=d_ctx,
            num_queries=chunk_summary_tokens,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            mlp_ratio=mlp_ratio
        )
        self.global_pool = QueryPool(
            d_model=d_ctx,
            num_queries=final_tokens,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            mlp_ratio=mlp_ratio
        )

    def forward(
        self,
        chunks_tokens: List[torch.Tensor],   # each [B,Lc,d_ctx]
        chunks_masks: List[torch.Tensor],    # each [B,Lc]
        chunks_valid: List[torch.Tensor],    # each [B] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          summary_tokens: [B, M_final, d_ctx]
          summary_mask:   [B, M_final] (1 valid)
        """
        B = chunks_tokens[0].size(0)
        device = chunks_tokens[0].device

        # pool each chunk -> [B, M_chunk, d]
        per_chunk_summaries: List[torch.Tensor] = []
        per_chunk_masks: List[torch.Tensor] = []

        for x, m, v in zip(chunks_tokens, chunks_masks, chunks_valid):
            # If a sample has v=False (empty chunk), we mask out its chunk summaries.
            chunk_sum = self.chunk_pool(x, m)  # [B,M_chunk,d]
            chunk_sum_mask = v.to(dtype=torch.long, device=device).unsqueeze(1).repeat(1, chunk_sum.size(1))
            per_chunk_summaries.append(chunk_sum)
            per_chunk_masks.append(chunk_sum_mask)

        # concat chunk summaries
        cat = torch.cat(per_chunk_summaries, dim=1)  # [B, n_chunks*M_chunk, d]
        cat_mask = torch.cat(per_chunk_masks, dim=1) # [B, n_chunks*M_chunk]

        # global pool to fixed M_final
        out = self.global_pool(cat, cat_mask)  # [B,M_final,d]
        out_mask = torch.ones(B, out.size(1), device=device, dtype=torch.long)
        return out, out_mask


# -----------------------------
# Ligand tri-view contrastive + non-drop fusion (token-level)
# -----------------------------
class ContrastiveProjection(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proj_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, dim=-1)


class LigandTriViewFuseContrast(nn.Module):
    """
    Inputs:
      S_tokens: [B,K,d], G_tokens: [B,K,d], T_tokens: [B,K,d]  (K same across views)
    Outputs:
      L_tokens: [B,K,d] fused tokens
      loss: scalar
      weights: [B,3] view weights (with floor)
    """
    def __init__(self, d_ctx: int, proj_dim: int, temperature: float, alpha_floor: float):
        super().__init__()
        self.temp = temperature
        self.alpha = alpha_floor

        self.head_s = ContrastiveProjection(d_ctx, proj_dim)
        self.head_g = ContrastiveProjection(d_ctx, proj_dim)
        self.head_t = ContrastiveProjection(d_ctx, proj_dim)

        # Gate to learn preferences; init to equal importance.
        self.gate = nn.Sequential(
            nn.LayerNorm(d_ctx * 3),
            nn.Linear(d_ctx * 3, d_ctx),
            nn.GELU(),
            nn.Linear(d_ctx, 3),
        )
        # initialize last layer to output ~0 => uniform softmax initially
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        # small post-fuse adapter so fused tokens can become more expressive
        self.post = ResidualAdapterBlock(d_ctx, mlp_ratio=2.0, dropout=0.0)

    def forward(
        self,
        S: torch.Tensor,
        G: torch.Tensor,
        T: torch.Tensor,
        compute_loss: bool = True,
        ligand_group_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # vector summaries for contrastive (stable)
        e_s = S.mean(dim=1)  # [B,d]
        e_g = G.mean(dim=1)
        e_t = T.mean(dim=1)

        loss: Optional[torch.Tensor]
        if compute_loss:
            z_s = self.head_s(e_s)
            z_g = self.head_g(e_g)
            z_t = self.head_t(e_t)

            loss = (
                clip_infonce(z_s, z_g, temperature=self.temp, group_ids=ligand_group_ids) +
                clip_infonce(z_s, z_t, temperature=self.temp, group_ids=ligand_group_ids) +
                clip_infonce(z_g, z_t, temperature=self.temp, group_ids=ligand_group_ids)
            ) / 3.0
        else:
            loss = None

        # gate weights with non-drop floor
        g = self.gate(torch.cat([e_s, e_g, e_t], dim=-1))  # [B,3]
        w_soft = F.softmax(g, dim=-1)                      # [B,3]
        w = (1.0 - self.alpha) * w_soft + self.alpha * (1.0 / 3.0)

        # token-level fusion (ensures all three views contribute)
        wS = w[:, 0].view(-1, 1, 1)
        wG = w[:, 1].view(-1, 1, 1)
        wT = w[:, 2].view(-1, 1, 1)
        L = wS * S + wG * G + wT * T  # [B,K,d]
        L = self.post(L)
        return L, loss, w


# -----------------------------
# Type embeddings
# -----------------------------
class ChemicalFieldRequirement(nn.Module):
    def __init__(self, d_ctx: int, n_properties: int = 10):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_ctx),
            nn.Linear(d_ctx, max(32, d_ctx // 2)),
            nn.GELU(),
            nn.Linear(max(32, d_ctx // 2), n_properties),
        )

    def forward(self, ligand_fused_tokens: torch.Tensor) -> torch.Tensor:
        ligand_repr = ligand_fused_tokens.mean(dim=1)
        return self.proj(ligand_repr)


class ModalityTypeEmbedding(nn.Module):
    def __init__(self, d_ctx: int):
        super().__init__()
        self.emb = nn.Embedding(4, d_ctx)  # 0=dock, 1=ligand_chem, 2=pocket, 3=func

    def forward(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:
        return tokens + self.emb.weight[type_id].view(1, 1, -1)


# -----------------------------
# Main Encoder
# -----------------------------
class PrePerceiverConditionEncoder(nn.Module):
    """
    Improved Pre-Perceiver encoder implementing your 4 requirements.
    """

    def __init__(
        self,
        cfg: PrePerceiverEncoderConfig,
        lig3d_encoder: Optional[nn.Module] = None,
        lig3d_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = cfg

        # ---- backbones ----
        self.bert_dock = FrozenPubMedBERT(cfg.pubmedbert_name, freeze=cfg.freeze_backbones)
        self.bert_func = FrozenPubMedBERT(cfg.pubmedbert_name, freeze=cfg.freeze_backbones)
        self.bert_ltxt = FrozenPubMedBERT(cfg.pubmedbert_name, freeze=cfg.freeze_backbones)

        self.smiles_lm = FrozenSmilesMaskedLM(cfg.smiles_lm_name, freeze=cfg.freeze_backbones)

        # 3D encoder: allow user to inject a pretrained one; fallback to EGNN.
        if lig3d_encoder is not None:
            self.lig3d = lig3d_encoder
        else:
            self.lig3d = EGNN3DEncoder(out_dim=768, n_layers=4, edge_hidden=256, dropout=cfg.ffn_dropout)

        if lig3d_ckpt is not None:
            sd = torch.load(lig3d_ckpt, map_location="cpu")
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            self.lig3d.load_state_dict(sd, strict=False)

        if cfg.freeze_backbones:
            freeze_module(self.lig3d)
            self.lig3d.eval()

        # ---- stage1 align projectors (backbone hidden -> d_ctx), near-identity init ----
        self.align_dock = Stage1AlignProjector(self.bert_dock.hidden_size, cfg.d_ctx, mlp_ratio=cfg.adapter_mlp_ratio, dropout=cfg.ffn_dropout)
        self.align_func = Stage1AlignProjector(self.bert_func.hidden_size, cfg.d_ctx, mlp_ratio=cfg.adapter_mlp_ratio, dropout=cfg.ffn_dropout)
        self.align_ltxt = Stage1AlignProjector(self.bert_ltxt.hidden_size, cfg.d_ctx, mlp_ratio=cfg.adapter_mlp_ratio, dropout=cfg.ffn_dropout)
        self.align_smi  = Stage1AlignProjector(self.smiles_lm.hidden_size, cfg.d_ctx, mlp_ratio=cfg.adapter_mlp_ratio, dropout=cfg.ffn_dropout)
        self.align_3d   = Stage1AlignProjector(self.lig3d.hidden_size, cfg.d_ctx, mlp_ratio=cfg.adapter_mlp_ratio, dropout=cfg.ffn_dropout)

        # ---- hierarchical pooling modules ----
        self.pool_dock = ChunkedHierPool(
            d_ctx=cfg.d_ctx,
            chunk_summary_tokens=cfg.text_chunk_summary_tokens,
            final_tokens=cfg.dock_final_tokens,
            num_heads=cfg.attn_heads,
            num_layers=cfg.num_pool_layers,
            attn_dropout=cfg.attn_dropout,
            ffn_dropout=cfg.ffn_dropout,
            mlp_ratio=cfg.adapter_mlp_ratio,
        )
        self.pool_pocket = ExplicitPocketInteractionExtractor(
            d_model=cfg.d_ctx,
            num_interaction_types=cfg.pocket_interaction_types,
            num_pocket_tokens=cfg.pocket_final_tokens,
            num_heads=cfg.attn_heads,
            num_layers=cfg.num_pool_layers,
            attn_dropout=cfg.attn_dropout,
            ffn_dropout=cfg.ffn_dropout,
            mlp_ratio=cfg.adapter_mlp_ratio,
        )
        self.pool_func = ChunkedHierPool(
            d_ctx=cfg.d_ctx,
            chunk_summary_tokens=cfg.text_chunk_summary_tokens,
            final_tokens=cfg.func_final_tokens,
            num_heads=cfg.attn_heads,
            num_layers=cfg.num_pool_layers,
            attn_dropout=cfg.attn_dropout,
            ffn_dropout=cfg.ffn_dropout,
            mlp_ratio=cfg.adapter_mlp_ratio,
        )
        # ligand_text is a ligand view -> final tokens = ligand_view_tokens
        self.pool_ltxt = ChunkedHierPool(
            d_ctx=cfg.d_ctx,
            chunk_summary_tokens=cfg.text_chunk_summary_tokens,
            final_tokens=cfg.ligand_view_tokens,
            num_heads=cfg.attn_heads,
            num_layers=cfg.num_pool_layers,
            attn_dropout=cfg.attn_dropout,
            ffn_dropout=cfg.ffn_dropout,
            mlp_ratio=cfg.adapter_mlp_ratio,
        )
        self.pool_smiles = ChunkedHierPool(
            d_ctx=cfg.d_ctx,
            chunk_summary_tokens=cfg.text_chunk_summary_tokens,
            final_tokens=cfg.smiles_view_tokens,
            num_heads=cfg.attn_heads,
            num_layers=cfg.num_pool_layers,
            attn_dropout=cfg.attn_dropout,
            ffn_dropout=cfg.ffn_dropout,
            mlp_ratio=cfg.adapter_mlp_ratio,
        )
        # 3D pool (direct QueryPool; no chunking needed)
        self.pool_3d = QueryPool(
            d_model=cfg.d_ctx,
            num_queries=cfg.lig3d_view_tokens,
            num_heads=cfg.attn_heads,
            num_layers=cfg.num_pool_layers,
            attn_dropout=cfg.attn_dropout,
            ffn_dropout=cfg.ffn_dropout,
            mlp_ratio=cfg.adapter_mlp_ratio,
        )

        # ---- ligand fuse + contrastive ----
        assert cfg.smiles_view_tokens == cfg.ligand_view_tokens == cfg.lig3d_view_tokens, \
            "To do token-level weighted fusion, set all ligand view token counts equal."
        self.lig_fuse = LigandTriViewFuseContrast(
            d_ctx=cfg.d_ctx,
            proj_dim=cfg.contrastive_proj_dim,
            temperature=cfg.contrastive_temperature,
            alpha_floor=cfg.ligand_weight_floor_alpha,
        )
        self.chem_field_req = ChemicalFieldRequirement(cfg.d_ctx, cfg.chem_requirement_dim)

        # ---- stage2 calibrators (pre-perceiver) ----
        self.calib_dock = FinalCalibrator(cfg.d_ctx, mlp_ratio=1.0, dropout=cfg.ffn_dropout)
        self.calib_lig  = FinalCalibrator(cfg.d_ctx, mlp_ratio=1.0, dropout=cfg.ffn_dropout)
        self.calib_pocket = FinalCalibrator(cfg.d_ctx, mlp_ratio=1.0, dropout=cfg.ffn_dropout)
        self.calib_func = FinalCalibrator(cfg.d_ctx, mlp_ratio=1.0, dropout=cfg.ffn_dropout)

        # ---- type embeddings ----
        self.type_emb = ModalityTypeEmbedding(cfg.d_ctx) if cfg.use_type_embeddings else None

    # ---------- backbone forward helpers ----------
    def _bert_forward(self, bert: FrozenPubMedBERT, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        no_grad = (self.cfg.freeze_backbones and self.cfg.no_grad_backbone_forward)
        return bert(ids, mask, no_grad=no_grad)

    def _smiles_forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        no_grad = (self.cfg.freeze_backbones and self.cfg.no_grad_backbone_forward)
        return self.smiles_lm(ids, mask, no_grad=no_grad)

    def _lig3d_forward(self, Z: torch.Tensor, coords: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        no_grad = (self.cfg.freeze_backbones and self.cfg.no_grad_backbone_forward)
        if no_grad:
            with torch.no_grad():
                return self.lig3d(Z, coords, atom_mask)
        return self.lig3d(Z, coords, atom_mask)

    # ---------- main forward ----------
    def forward(
        self,
        docking_input_ids: torch.Tensor,
        docking_attention_mask: torch.Tensor,
        function_input_ids: torch.Tensor,
        function_attention_mask: torch.Tensor,
        ligand_text_input_ids: torch.Tensor,
        ligand_text_attention_mask: torch.Tensor,
        smiles_input_ids: torch.Tensor,
        smiles_attention_mask: torch.Tensor,
        lig3d_Z: torch.Tensor,
        lig3d_coords: torch.Tensor,
        lig3d_atom_mask: torch.Tensor,
        compute_aux_losses: Optional[bool] = None,
        ligand_group_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        cfg = self.cfg
        if compute_aux_losses is None:
            compute_aux_losses = self.training

        aux: Dict[str, torch.Tensor] = {}

        # -------------------------
        # 1) docking_text: chunk -> BERT -> stage1 align -> hier pool -> D_tokens
        # -------------------------
        dock_chunks = split_into_chunks(
            docking_input_ids, docking_attention_mask,
            chunk_len=cfg.dock_chunk_len,
            overlap=cfg.dock_chunk_overlap,
            max_chunks=cfg.dock_max_chunks
        )
        dock_chunk_tokens: List[torch.Tensor] = []
        dock_chunk_masks: List[torch.Tensor] = []
        dock_chunk_valid: List[torch.Tensor] = []

        for ids, m, v in dock_chunks:
            H = self._bert_forward(self.bert_dock, ids, m)   # [B,Lc,768]
            X = self.align_dock(H)                            # [B,Lc,d_ctx]
            dock_chunk_tokens.append(X)
            dock_chunk_masks.append(m)
            dock_chunk_valid.append(v)

        D_tokens, D_mask = self.pool_dock(dock_chunk_tokens, dock_chunk_masks, dock_chunk_valid)  # [B,64,d], [B,64]

        # -------------------------
        # 2) function_text: chunk -> BERT -> align -> hier pool -> F_tokens
        # -------------------------
        func_chunks = split_into_chunks(
            function_input_ids, function_attention_mask,
            chunk_len=cfg.func_chunk_len,
            overlap=cfg.func_chunk_overlap,
            max_chunks=cfg.func_max_chunks
        )
        func_chunk_tokens: List[torch.Tensor] = []
        func_chunk_masks: List[torch.Tensor] = []
        func_chunk_valid: List[torch.Tensor] = []

        for ids, m, v in func_chunks:
            H = self._bert_forward(self.bert_func, ids, m)   # [B,Lc,768]
            X = self.align_func(H)                            # [B,Lc,d_ctx]
            func_chunk_tokens.append(X)
            func_chunk_masks.append(m)
            func_chunk_valid.append(v)

        F_tokens, F_mask = self.pool_func(func_chunk_tokens, func_chunk_masks, func_chunk_valid)  # [B,32,d]

        # -------------------------
        # 3) ligand_text view: chunk -> BERT -> align -> hier pool -> T_tokens (K tokens)
        # -------------------------
        ltxt_chunks = split_into_chunks(
            ligand_text_input_ids, ligand_text_attention_mask,
            chunk_len=cfg.ligtext_chunk_len,
            overlap=cfg.ligtext_chunk_overlap,
            max_chunks=cfg.ligtext_max_chunks
        )
        ltxt_chunk_tokens: List[torch.Tensor] = []
        ltxt_chunk_masks: List[torch.Tensor] = []
        ltxt_chunk_valid: List[torch.Tensor] = []

        for ids, m, v in ltxt_chunks:
            H = self._bert_forward(self.bert_ltxt, ids, m)   # [B,Lc,768]
            X = self.align_ltxt(H)                            # [B,Lc,d_ctx]
            ltxt_chunk_tokens.append(X)
            ltxt_chunk_masks.append(m)
            ltxt_chunk_valid.append(v)

        T_tokens, T_mask = self.pool_ltxt(ltxt_chunk_tokens, ltxt_chunk_masks, ltxt_chunk_valid)  # [B,K,d], [B,K]

        # -------------------------
        # 4) SMILES view: chunk -> SMILES LM -> align -> hier pool -> S_tokens (K tokens)
        # -------------------------
        smi_chunks = split_into_chunks(
            smiles_input_ids, smiles_attention_mask,
            chunk_len=cfg.smiles_chunk_len,
            overlap=cfg.smiles_chunk_overlap,
            max_chunks=cfg.smiles_max_chunks
        )
        smi_chunk_tokens: List[torch.Tensor] = []
        smi_chunk_masks: List[torch.Tensor] = []
        smi_chunk_valid: List[torch.Tensor] = []

        for ids, m, v in smi_chunks:
            H = self._smiles_forward(ids, m)                 # [B,Lc,hidden]
            X = self.align_smi(H)                            # [B,Lc,d_ctx]
            smi_chunk_tokens.append(X)
            smi_chunk_masks.append(m)
            smi_chunk_valid.append(v)

        S_tokens, S_mask = self.pool_smiles(smi_chunk_tokens, smi_chunk_masks, smi_chunk_valid)  # [B,K,d], [B,K]

        # -------------------------
        # 5) 3D view: keep all if small; if large, FPS cap; then align + QueryPool -> G_tokens (K tokens)
        # -------------------------
        # Count atoms per sample
        atom_counts = lig3d_atom_mask.sum(dim=1)  # [B]
        B, N = lig3d_Z.shape

        # If any sample has N > fps_cap, do FPS on that sample (batched helper handles all)
        # Strategy:
        #   - If max atoms <= keep_all_threshold: keep original
        #   - Else if max atoms > fps_cap: FPS down to fps_cap
        #   - Else keep original (QueryPool can handle)
        max_atoms = int(atom_counts.max().item()) if atom_counts.numel() > 0 else 0

        Z_use, coords_use, mask_use = lig3d_Z, lig3d_coords, lig3d_atom_mask
        if max_atoms > cfg.lig3d_fps_cap:
            coords_fps, idx_fps, mask_fps = fps_batch(lig3d_coords, lig3d_atom_mask, cfg.lig3d_fps_cap)
            # gather Z using idx_fps
            Z_fps = torch.zeros(B, cfg.lig3d_fps_cap, dtype=lig3d_Z.dtype, device=lig3d_Z.device)
            for b in range(B):
                if mask_fps[b].sum() > 0:
                    Z_fps[b] = lig3d_Z[b, idx_fps[b]]
            Z_use, coords_use, mask_use = Z_fps, coords_fps, mask_fps
        else:
            # keep all (even if >keep_all_threshold but <=fps_cap)
            # no hard truncation here
            pass

        H3 = self._lig3d_forward(Z_use, coords_use, mask_use)  # [B,N',768]
        X3 = self.align_3d(H3)                                  # [B,N',d_ctx]
        G_tokens = self.pool_3d(X3, mask_use)                   # [B,K,d_ctx]
        G_mask = torch.ones(B, G_tokens.size(1), device=G_tokens.device, dtype=torch.long)

        # -------------------------
        # 6) Ligand tri-view contrast + non-drop fusion -> L_tokens (K tokens)
        # -------------------------
        L_tokens, ctr_loss, w = self.lig_fuse(
            S_tokens,
            G_tokens,
            T_tokens,
            compute_loss=compute_aux_losses,
            ligand_group_ids=ligand_group_ids,
        )
        if compute_aux_losses and (ctr_loss is not None):
            # Loss weights are owned by DockingWizardModel.forward so logs stay
            # interpretable and auxiliary losses are not scaled twice.
            aux["aux_ligand_contrastive"] = ctr_loss
        aux["ligand_view_weights"] = w  # for logging/analysis (optional use)

        L_mask = torch.ones(B, L_tokens.size(1), device=L_tokens.device, dtype=torch.long)
        chem_requirements = self.chem_field_req(L_tokens)

        # Typed queries extract docking/ligand interaction semantics before
        # expansion into fixed-length pocket tokens.
        P_tokens = self.pool_pocket(D_tokens, D_mask, L_tokens, L_mask)
        P_mask = torch.ones(B, P_tokens.size(1), device=P_tokens.device, dtype=torch.long)

        # -------------------------
        # 7) Stage-2 alignment (FinalCalibrator) + type embedding + pack ctx
        # -------------------------
        D_tokens = self.calib_dock(D_tokens)
        L_tokens = self.calib_lig(L_tokens)
        P_tokens = self.calib_pocket(P_tokens)
        F_tokens = self.calib_func(F_tokens)

        if self.type_emb is not None:
            D_tokens = self.type_emb(D_tokens, type_id=0)
            L_tokens = self.type_emb(L_tokens, type_id=1)
            P_tokens = self.type_emb(P_tokens, type_id=2)
            F_tokens = self.type_emb(F_tokens, type_id=3)

        ctx_tokens = torch.cat([D_tokens, L_tokens, P_tokens, F_tokens], dim=1)
        ctx_mask = torch.cat([D_mask, L_mask, P_mask, F_mask], dim=1)

        out: Dict[str, torch.Tensor] = {
            "ctx_tokens": ctx_tokens,
            "ctx_mask": ctx_mask,
            "chem_requirements": chem_requirements,
        }
        out.update(aux)
        return out
