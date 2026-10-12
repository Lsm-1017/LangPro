"""Perceiver resampler and condition-memory routing for LangPro."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class PerceiverModuleConfig:
    d_ctx: int = 1024

    k_dock: int = 96
    k_lig: int = 32
    k_pocket: int = 32
    k_func: int = 48

    n_func: int = 64
    n_dock: int = 64
    n_ligand_chem: int = 32
    n_pocket: int = 32
    n_full: int = 128
    n_light: int = 16
    n_interaction_types: int = 6
    n_chem_features: int = 10

    func_num_blocks: int = 4
    func_num_self_attends_per_block: int = 2
    dl_num_blocks: int = 4
    dl_num_self_attends_per_block: int = 2

    heads: int = 8
    dim_head: int = 64
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0

    cross_every_block: bool = False
    share_self_attn_weights: bool = True

    use_bridge: bool = True
    bridge_gate_tokenwise: bool = True
    bridge_gate_init_bias: float = -2.0
    enable_ligand_chemical_memory: bool = True
    enable_pocket_interaction_memory: bool = True
    enable_interaction_type_queries: bool = True
    enable_biological_bridge: bool = True
    bridge_use_ffn: bool = True

    use_inter_branch_attn: bool = False
    inter_branch_num_heads: int = 8

    resampler_depth: int = 3
    resampler_use_self_attn: bool = True

    use_layerscale: bool = True
    layerscale_init: float = 1e-3


def _as_bool_mask(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bool:
        return x
    return x != 0


def _safe_masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    if mask is None:
        return F.softmax(logits, dim=dim)
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return F.softmax(logits, dim=dim)


def _ensure_non_empty_kv(kv: torch.Tensor, kv_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if kv_mask is None:
        return kv, kv_mask
    kv_mask = _as_bool_mask(kv_mask)
    any_valid = kv_mask.any(dim=1)
    if bool(any_valid.all()):
        return kv, kv_mask
    B, _, D = kv.shape
    dummy = torch.zeros(B, 1, D, device=kv.device, dtype=kv.dtype)
    kv = torch.cat([kv, dummy], dim=1)
    kv_mask = torch.cat([kv_mask, (~any_valid).unsqueeze(1)], dim=1)
    return kv, kv_mask


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiheadAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, attn_dropout: float, proj_dropout: float):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.attn_dropout = attn_dropout
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(proj_dropout))

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Nq, _ = q.shape
        _, Nk, _ = kv.shape
        h = self.heads
        q = self.to_q(q).view(B, Nq, h, self.dim_head).transpose(1, 2)
        k = self.to_k(kv).view(B, Nk, h, self.dim_head).transpose(1, 2)
        v = self.to_v(kv).view(B, Nk, h, self.dim_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        keep = _as_bool_mask(kv_mask)[:, None, None, :] if kv_mask is not None else None
        attn = _safe_masked_softmax(attn, keep, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, Nq, h * self.dim_head)
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float, dropout: float, attn_dropout: float, use_layerscale: bool, layerscale_init: float, use_ffn: bool = True):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = MultiheadAttention(dim, heads, dim_head, attn_dropout, dropout)
        self.use_ffn = use_ffn
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, int(dim * mlp_ratio), dropout) if use_ffn else None
        self.ls_attn = LayerScale(dim, layerscale_init) if use_layerscale else nn.Identity()
        self.ls_ff = LayerScale(dim, layerscale_init) if (use_layerscale and use_ffn) else nn.Identity()

    def forward(self, z: torch.Tensor, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        kv, kv_mask = _ensure_non_empty_kv(kv, kv_mask)
        z = z + self.ls_attn(self.attn(self.norm_q(z), self.norm_kv(kv), kv_mask=kv_mask))
        if self.use_ffn:
            z = z + self.ls_ff(self.ff(self.norm_ff(z)))
        return z


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float, dropout: float, attn_dropout: float, use_layerscale: bool, layerscale_init: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MultiheadAttention(dim, heads, dim_head, attn_dropout, dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, int(dim * mlp_ratio), dropout)
        self.ls_attn = LayerScale(dim, layerscale_init) if use_layerscale else nn.Identity()
        self.ls_ff = LayerScale(dim, layerscale_init) if use_layerscale else nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z + self.ls_attn(self.attn(self.norm(z), self.norm(z), kv_mask=None))
        z = z + self.ls_ff(self.ff(self.norm_ff(z)))
        return z


class TrainableLatents(nn.Module):
    def __init__(self, n_latents: int, dim: int, init_std: float = 0.02):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, dim) * init_std)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.latents.unsqueeze(0).expand(batch_size, -1, -1)


class PerceiverBranch(nn.Module):
    def __init__(self, n_latents: int, dim: int, num_blocks: int, num_self_attends_per_block: int, heads: int, dim_head: int, mlp_ratio: float, dropout: float, attn_dropout: float, cross_every_block: bool, share_self_attn_weights: bool, use_layerscale: bool, layerscale_init: float):
        super().__init__()
        self.latents = TrainableLatents(n_latents, dim)
        self.cross = CrossAttentionBlock(dim, heads, dim_head, mlp_ratio, dropout, attn_dropout, use_layerscale, layerscale_init, use_ffn=True)
        self.num_blocks = num_blocks
        self.num_self = num_self_attends_per_block
        self.cross_every_block = cross_every_block
        self.share_self = share_self_attn_weights
        n_self_layers = num_self_attends_per_block if share_self_attn_weights else num_blocks * num_self_attends_per_block
        self.self_layers = nn.ModuleList([
            SelfAttentionBlock(dim, heads, dim_head, mlp_ratio, dropout, attn_dropout, use_layerscale, layerscale_init)
            for _ in range(n_self_layers)
        ])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, kv: torch.Tensor, kv_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B = kv.size(0)
        z = self.latents(B)
        kv, kv_mask = _ensure_non_empty_kv(kv, kv_mask)
        z = self.cross(z, kv, kv_mask=kv_mask)
        if self.share_self:
            for _ in range(self.num_blocks):
                if self.cross_every_block:
                    z = self.cross(z, kv, kv_mask=kv_mask)
                for layer in self.self_layers:
                    z = layer(z)
        else:
            idx = 0
            for _ in range(self.num_blocks):
                if self.cross_every_block:
                    z = self.cross(z, kv, kv_mask=kv_mask)
                for _ in range(self.num_self):
                    z = self.self_layers[idx](z)
                    idx += 1
        return self.out_norm(z)


class PerceiverResampler(nn.Module):
    def __init__(self, n_latents: int, dim: int, depth: int, heads: int, dim_head: int, mlp_ratio: float, dropout: float, attn_dropout: float, use_layerscale: bool, layerscale_init: float, use_self_attn: bool):
        super().__init__()
        self.latents = TrainableLatents(n_latents, dim)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(CrossAttentionBlock(dim, heads, dim_head, mlp_ratio, dropout, attn_dropout, use_layerscale, layerscale_init, use_ffn=True))
            if use_self_attn:
                self.blocks.append(SelfAttentionBlock(dim, heads, dim_head, mlp_ratio, dropout, attn_dropout, use_layerscale, layerscale_init))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, full_mem: torch.Tensor) -> torch.Tensor:
        B = full_mem.size(0)
        z = self.latents(B)
        for blk in self.blocks:
            if isinstance(blk, CrossAttentionBlock):
                z = blk(z, full_mem, kv_mask=None)
            else:
                z = blk(z)
        return self.out_norm(z)


class BiologicalBridge(nn.Module):
    def __init__(self, cfg: PerceiverModuleConfig):
        super().__init__()
        args = (cfg.d_ctx, cfg.heads, cfg.dim_head, cfg.mlp_ratio, cfg.dropout, cfg.attn_dropout, cfg.use_layerscale, cfg.layerscale_init)
        self.func_to_chem = CrossAttentionBlock(*args, use_ffn=True)
        self.chem_to_func = CrossAttentionBlock(*args, use_ffn=True)
        self.chem_to_pocket = CrossAttentionBlock(*args, use_ffn=True)
        self.pocket_to_chem = CrossAttentionBlock(*args, use_ffn=True)
        self.func_to_pocket = CrossAttentionBlock(*args, use_ffn=True)
        self.pocket_to_func = CrossAttentionBlock(*args, use_ffn=True)
        self.gate_func = nn.Parameter(torch.tensor(float(cfg.bridge_gate_init_bias)))
        self.gate_chem = nn.Parameter(torch.tensor(float(cfg.bridge_gate_init_bias)))
        self.gate_pocket = nn.Parameter(torch.tensor(float(cfg.bridge_gate_init_bias)))

    def forward(self, mem_func: torch.Tensor, mem_ligand_chem: torch.Tensor, mem_pocket: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        f2c = self.func_to_chem(mem_ligand_chem, mem_func)
        c2f = self.chem_to_func(mem_func, mem_ligand_chem)
        c2p = self.chem_to_pocket(mem_pocket, mem_ligand_chem)
        p2c = self.pocket_to_chem(mem_ligand_chem, mem_pocket)
        f2p = self.func_to_pocket(mem_pocket, mem_func)
        p2f = self.pocket_to_func(mem_func, mem_pocket)
        gf = torch.sigmoid(self.gate_func)
        gc = torch.sigmoid(self.gate_chem)
        gp = torch.sigmoid(self.gate_pocket)
        mem_func = mem_func + gf * (c2f + p2f)
        mem_ligand_chem = mem_ligand_chem + gc * (f2c + p2c)
        mem_pocket = mem_pocket + gp * (c2p + f2p)
        return mem_func, mem_ligand_chem, mem_pocket, {"gate_func": gf.detach(), "gate_chem": gc.detach(), "gate_pocket": gp.detach()}


class InteractionTypeQueries(nn.Module):
    def __init__(self, n_types: int, dim: int, heads: int, dim_head: int, mlp_ratio: float, dropout: float, attn_dropout: float, use_layerscale: bool, layerscale_init: float):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_types, dim) * 0.02)
        self.attend_ligand = CrossAttentionBlock(dim, heads, dim_head, mlp_ratio, dropout, attn_dropout, use_layerscale, layerscale_init, use_ffn=True)
        self.attend_pocket = CrossAttentionBlock(dim, heads, dim_head, mlp_ratio, dropout, attn_dropout, use_layerscale, layerscale_init, use_ffn=True)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, ligand_mem: torch.Tensor, pocket_mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = ligand_mem.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        q = self.attend_ligand(q, ligand_mem, kv_mask=None)
        q = self.attend_pocket(q, pocket_mem, kv_mask=None)
        q = self.out_norm(q)
        scores = torch.matmul(F.normalize(q, dim=-1), F.normalize(pocket_mem, dim=-1).transpose(1, 2))
        return q, scores


class DockingWizardPerceiver(nn.Module):
    def __init__(self, cfg: PerceiverModuleConfig):
        super().__init__()
        self.cfg = cfg
        branch_args = dict(
            dim=cfg.d_ctx,
            num_blocks=cfg.func_num_blocks,
            num_self_attends_per_block=cfg.func_num_self_attends_per_block,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
            cross_every_block=cfg.cross_every_block,
            share_self_attn_weights=cfg.share_self_attn_weights,
            use_layerscale=cfg.use_layerscale,
            layerscale_init=cfg.layerscale_init,
        )
        dl_branch_args = dict(branch_args)
        dl_branch_args["num_blocks"] = cfg.dl_num_blocks
        dl_branch_args["num_self_attends_per_block"] = cfg.dl_num_self_attends_per_block

        self.func_branch = PerceiverBranch(n_latents=cfg.n_func, **branch_args)
        self.ligand_chem_branch = PerceiverBranch(n_latents=cfg.n_ligand_chem, **branch_args)
        self.pocket_branch = PerceiverBranch(n_latents=cfg.n_pocket, **branch_args)
        self.docking_branch = PerceiverBranch(n_latents=cfg.n_dock, **branch_args)
        self.full_branch = PerceiverBranch(n_latents=cfg.n_full, **dl_branch_args)

        self.resampler = PerceiverResampler(cfg.n_light, cfg.d_ctx, cfg.resampler_depth, cfg.heads, cfg.dim_head, cfg.mlp_ratio, cfg.dropout, cfg.attn_dropout, cfg.use_layerscale, cfg.layerscale_init, cfg.resampler_use_self_attn)
        self.bio_bridge = BiologicalBridge(cfg)
        self.interaction_queries = InteractionTypeQueries(cfg.n_interaction_types, cfg.d_ctx, cfg.heads, cfg.dim_head, cfg.mlp_ratio, cfg.dropout, cfg.attn_dropout, cfg.use_layerscale, cfg.layerscale_init)
        self.chem_feature_proj = nn.Sequential(nn.LayerNorm(cfg.n_chem_features), nn.Linear(cfg.n_chem_features, cfg.d_ctx), nn.GELU(), nn.Linear(cfg.d_ctx, cfg.d_ctx))
        self.chem_feature_gate = nn.Sequential(nn.LayerNorm(cfg.d_ctx), nn.Linear(cfg.d_ctx, 1), nn.Sigmoid())

    def _split_ctx(self, ctx_tokens: torch.Tensor, ctx_mask: torch.Tensor):
        cfg = self.cfg
        B, L, D = ctx_tokens.shape
        assert D == cfg.d_ctx, f"d_ctx mismatch: got {D}, expect {cfg.d_ctx}"
        expect_L = cfg.k_dock + cfg.k_lig + cfg.k_pocket + cfg.k_func
        assert L == expect_L, f"L_ctx mismatch: got {L}, expect {expect_L}"
        kD, kL, kP = cfg.k_dock, cfg.k_lig, cfg.k_pocket
        D_tok = ctx_tokens[:, :kD, :]
        L_tok = ctx_tokens[:, kD:kD + kL, :]
        P_tok = ctx_tokens[:, kD + kL:kD + kL + kP, :]
        F_tok = ctx_tokens[:, kD + kL + kP:, :]
        D_m = ctx_mask[:, :kD]
        L_m = ctx_mask[:, kD:kD + kL]
        P_m = ctx_mask[:, kD + kL:kD + kL + kP]
        F_m = ctx_mask[:, kD + kL + kP:]
        return (D_tok, L_tok, P_tok, F_tok), (D_m, L_m, P_m, F_m)

    def forward(self, ctx_tokens: torch.Tensor, ctx_mask: torch.Tensor, return_debug: bool = False, chem_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        ctx_mask = _as_bool_mask(ctx_mask)
        (D_tok, L_tok, P_tok, F_tok), (D_m, L_m, P_m, F_m) = self._split_ctx(ctx_tokens, ctx_mask)

        mem_func = self.func_branch(F_tok, F_m)
        mem_ligand_chem = self.ligand_chem_branch(L_tok, L_m) if cfg.enable_ligand_chemical_memory else L_tok
        mem_pocket = self.pocket_branch(P_tok, P_m) if cfg.enable_pocket_interaction_memory else P_tok
        mem_docking = self.docking_branch(D_tok, D_m)

        debug: Dict[str, torch.Tensor] = {}
        chem_features_used = False
        chem_feature_embedding = None
        if chem_features is not None and cfg.enable_ligand_chemical_memory:
            chem = chem_features.to(device=ctx_tokens.device, dtype=ctx_tokens.dtype)
            chem = torch.nan_to_num(chem, nan=0.0, posinf=0.0, neginf=0.0)
            if chem.dim() != 2 or chem.size(0) != ctx_tokens.size(0) or chem.size(1) != int(cfg.n_chem_features):
                raise ValueError(f"chem_features must be [B,{cfg.n_chem_features}], got {tuple(chem.shape)}")
            chem_feature_embedding = self.chem_feature_proj(chem)
            chem_gate = self.chem_feature_gate(mem_ligand_chem)
            mem_ligand_chem = mem_ligand_chem + chem_gate * chem_feature_embedding.unsqueeze(1)
            chem_features_used = True

        if cfg.enable_biological_bridge:
            mem_func, mem_ligand_chem, mem_pocket, bridge_debug = self.bio_bridge(mem_func, mem_ligand_chem, mem_pocket)
            if return_debug:
                debug["biological_bridge_gate"] = bridge_debug

        interaction_query_memory = None
        interaction_query_scores = None
        if cfg.enable_interaction_type_queries:
            interaction_query_memory, interaction_query_scores = self.interaction_queries(mem_ligand_chem, mem_pocket)
            if return_debug:
                debug["interaction_query_scores"] = interaction_query_scores

        full_kv = torch.cat([mem_docking, mem_ligand_chem, mem_pocket], dim=1)
        mem_dl_full = self.full_branch(full_kv, None)
        mem_dl_light = self.resampler(mem_dl_full)

        out = {
            "mem_func": mem_func,
            "mem_dl_full": mem_dl_full,
            "mem_dl_light": mem_dl_light,
            "mem_docking": mem_docking,
            "mem_ligand_chem": mem_ligand_chem,
            "mem_pocket": mem_pocket,
        }
        if interaction_query_memory is not None:
            out["interaction_query_memory"] = interaction_query_memory
            out["interaction_embeds"] = interaction_query_memory
        if return_debug:
            debug["chem_features_used"] = bool(chem_features_used)
            debug["chem_feature_embedding_shape"] = tuple(chem_feature_embedding.shape) if chem_feature_embedding is not None else None
            debug["memory_shapes"] = {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)}
            out["debug"] = debug
        return out
