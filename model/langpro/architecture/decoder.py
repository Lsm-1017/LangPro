"""ProGen2 decoder conditioned by LangPro prefix and full-memory routes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModelForCausalLM


# -----------------------------
# Small utilities
# -----------------------------
def _get_attr_path(obj: nn.Module, path: str):
    """Safely get nested attribute by dotted path; return None if missing."""
    cur = obj
    for p in path.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur


def _resolve_input_embedding(model: nn.Module) -> nn.Embedding:
    """
    Robustly locate the token embedding layer for HF / trust_remote_code models.
    Some remote-code models (e.g., ProGenModel) do not implement get_input_embeddings().
    """
    # 1) Try official API first (catch NotImplementedError)
    if hasattr(model, "get_input_embeddings"):
        try:
            emb = model.get_input_embeddings()
            if isinstance(emb, nn.Embedding):
                return emb
        except NotImplementedError:
            pass

    # 2) Try base_model if present
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "get_input_embeddings"):
        try:
            emb = base.get_input_embeddings()
            if isinstance(emb, nn.Embedding):
                return emb
        except NotImplementedError:
            pass

    # 3) Common attribute paths (GPT-style / LLaMA-style / ProGen remote code)
    candidates = [
        "transformer.wte",
        "model.transformer.wte",
        "progen.transformer.wte",
        "model.progen.transformer.wte",
        "model.embed_tokens",
        "model.decoder.embed_tokens",
        "transformer.embed_tokens",
        "embed_tokens",
        "wte",
        "gpt_neox.embed_in",
    ]
    for path in candidates:
        m = _get_attr_path(model, path)
        if isinstance(m, nn.Embedding):
            return m

    # 4) Fallback: scan all Embedding modules, prefer vocab match and tied lm_head
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    lm_head = getattr(model, "lm_head", None)
    lm_head_ptr = None
    if lm_head is not None and hasattr(lm_head, "weight"):
        lm_head_ptr = lm_head.weight.data_ptr()

    embs = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Embedding)]
    if len(embs) == 0:
        raise RuntimeError("Cannot find any nn.Embedding inside decoder backbone.")

    if vocab_size is not None:
        exact = [(n, m) for (n, m) in embs if m.num_embeddings == int(vocab_size)]
        if len(exact) == 1:
            return exact[0][1]
        if len(exact) > 1:
            if lm_head_ptr is not None:
                for _, m in exact:
                    if m.weight.data_ptr() == lm_head_ptr:
                        return m
            exact.sort(key=lambda x: x[1].embedding_dim, reverse=True)
            return exact[0][1]

    # No vocab_size info: choose the embedding with largest num_embeddings
    embs.sort(key=lambda x: x[1].num_embeddings, reverse=True)
    return embs[0][1]


def _find_progen_blocks(model: nn.Module) -> nn.ModuleList:
    """
    Try to locate the Transformer block list (ModuleList) inside a ProGen/ProGen2 HF model.
    Different repos may nest it differently (remote code).
    """
    candidates = [
        "transformer.h",
        "model.transformer.h",
        "progen.transformer.h",
        "model.progen.transformer.h",
        "gpt_neox.layers",          # just in case (unlikely)
        "transformer.layers",       # just in case
        "model.layers",             # some LLMs
    ]

    def _get_attr_path_local(obj: nn.Module, path: str):
        cur = obj
        for p in path.split("."):
            if not hasattr(cur, p):
                return None
            cur = getattr(cur, p)
        return cur

    for path in candidates:
        m = _get_attr_path_local(model, path)
        if isinstance(m, nn.ModuleList) and len(m) > 0:
            return m

    # Fallback: scan all submodules for the "largest" ModuleList of blocks
    best = None
    best_len = 0
    for _, sub in model.named_modules():
        if isinstance(sub, nn.ModuleList) and len(sub) > best_len:
            # heuristics: block module usually has parameters and is not just embeddings
            if len(list(sub.parameters())) > 0:
                best = sub
                best_len = len(sub)

    if best is None:
        raise RuntimeError(
            "Cannot locate transformer blocks (ModuleList) in the loaded backbone. "
            "Please inspect model.named_modules() and update _find_progen_blocks()."
        )
    return best


def _make_full_labels(prefix_len: int, protein_labels: torch.Tensor) -> torch.Tensor:
    """
    protein_labels: [B, Lp], tokens for protein region only.
    Return full_labels: [B, Lprefix+Lp], prefix part is -100.
    """
    B, Lp = protein_labels.shape
    full = torch.full((B, prefix_len + Lp), -100, device=protein_labels.device, dtype=torch.long)
    full[:, prefix_len:] = protein_labels
    return full


def _make_full_position_weights(
    prefix_len: int,
    protein_weights: Optional[torch.Tensor],
    protein_labels: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Build full-sequence target weights aligned to full_labels positions.
    Prefix positions are zero because they are always ignored by LM loss.
    """
    if protein_weights is None:
        return None
    if protein_weights.shape != protein_labels.shape:
        raise ValueError(
            f"position_weights shape must match protein labels: "
            f"got {tuple(protein_weights.shape)} vs {tuple(protein_labels.shape)}"
        )
    full = torch.zeros(
        (protein_weights.size(0), prefix_len + protein_weights.size(1)),
        device=protein_weights.device,
        dtype=torch.float32,
    )
    full[:, prefix_len:] = protein_weights.to(dtype=torch.float32)
    return full


# -----------------------------
# Gated full-memory injector (post-block)
# -----------------------------
class MultiHeadCrossAttention(nn.Module):
    """
    Simple cross-attention (Q from x, K/V from mem).
    Uses explicit projections + SDPA-like computation.
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, D] -> [B, H, L, Dh]
        B, L, D = x.shape
        x = x.view(B, L, self.n_head, self.d_head).permute(0, 2, 1, 3).contiguous()
        return x

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, L, Dh] -> [B, L, D]
        B, H, L, Dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)

    def forward(self, x_q: torch.Tensor, mem_kv: torch.Tensor) -> torch.Tensor:
        # x_q: [B,L,D], mem_kv: [B,N,D]
        q = self._split(self.q(x_q))         # [B,H,L,Dh]
        k = self._split(self.k(mem_kv))      # [B,H,N,Dh]
        v = self._split(self.v(mem_kv))      # [B,H,N,Dh]

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_head ** 0.5)  # [B,H,L,N]
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)          # [B,H,L,Dh]
        out = self._merge(out)               # [B,L,D]
        return self.o(out)


class GatedFullMemoryInjector(nn.Module):
    """Gated cross-attention/FFN injection after the condition prefix."""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.ln_mem = nn.LayerNorm(d_model)
        
        # Multi-head cross-attention.
        self.xattn = MultiHeadCrossAttention(d_model, n_head, dropout=dropout)
        self.interaction_xattn = MultiHeadCrossAttention(d_model, n_head, dropout=dropout)
        self.ln_interaction = nn.LayerNorm(d_model)
        self.interaction_update = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        
        # Feed-forward residual path.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        
        # Token-wise memory gate.
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        
        # Layer scaling stabilizes early training.
        self.layer_scale = nn.Parameter(torch.ones(1) * 0.30)
        self.interaction_scale = nn.Parameter(torch.ones(1) * 0.10)

        # A positive bias prevents the gate from closing at initialization.
        with torch.no_grad():
            last_gate_module = list(self.gate.modules())[-1]
            if isinstance(last_gate_module, nn.Linear):
                last_gate_module.bias.data += 1.0

    def forward(
        self,
        h: torch.Tensor,
        mem_full: Optional[torch.Tensor],
        prefix_len: int,
        start_pos: int = 0,
        pocket_gate_signal: Optional[torch.Tensor] = None,
        pocket_gate_strength: float = 0.0,
    ) -> torch.Tensor:
        """Inject full memory into hidden states.

        Args:
            h: [B, L, D] hidden states returned by a decoder block.
            mem_full: [B, N, D] projected full memory (already in backbone hidden size). If None -> no-op.
            prefix_len: length of prefix in the *global* sequence (0-index positions < prefix_len are prefix).
            start_pos: global position index of h[:, 0, :]. For full-seq forward this is 0.
                       For KV-cache step forward (L==1) this should be the global position of that token.
        """
        if mem_full is None:
            return h

        B, L, D = h.shape
        # local split index: how many positions in `h` still belong to the prefix?
        # global positions covered by `h` are [start_pos, start_pos + L - 1]
        split = max(0, int(prefix_len) - int(start_pos))
        if split >= L:
            return h

        h_prefix = h[:, :split, :]
        h_prot = h[:, split:, :]

        q = self.ln(h_prot)
        mem_norm = self.ln_mem(mem_full)
        
        # Multi-head cross-attention followed by an FFN.
        upd = self.xattn(q, mem_norm)  # [B, L_prot, D]
        upd = upd + self.ffn(upd)

        # Route each residue position across all interaction-type embeddings.
        mem_sum = mem_full.mean(dim=1, keepdim=True).expand(-1, q.size(1), -1)  # [B, L_prot, D]
        gate_context = mem_sum
        interaction_direct = None
        if pocket_gate_signal is not None and pocket_gate_signal.dim() == 3:
            ctx = pocket_gate_signal.to(device=h.device, dtype=h.dtype)
            if ctx.size(-1) == q.size(-1):
                if ctx.size(1) == h_prot.size(1):
                    gate_context = ctx
                else:
                    inter = self.ln_interaction(ctx)
                    gate_context = self.interaction_xattn(q, inter)
                interaction_direct = self.interaction_update(gate_context)
        g = self.gate(torch.cat([q, gate_context], dim=-1))  # [B, L_prot, 1]
        if pocket_gate_signal is not None and pocket_gate_signal.dim() != 3 and float(pocket_gate_strength) != 0.0:
            pg = pocket_gate_signal.to(device=h.device, dtype=h.dtype)
            local_start = max(0, int(start_pos) - int(prefix_len))
            pg = pg[:, local_start:local_start + h_prot.size(1)]
            if pg.size(1) < h_prot.size(1):
                pad = torch.zeros(pg.size(0), h_prot.size(1) - pg.size(1), device=pg.device, dtype=pg.dtype)
                pg = torch.cat([pg, pad], dim=1)
            pg = pg.unsqueeze(-1).clamp(0.0, 1.0)
            g = g * (1.0 + float(pocket_gate_strength) * pg)

        # Gate the memory update and add the routed interaction context.
        h_prot = h_prot + self.layer_scale * g * upd
        if interaction_direct is not None:
            h_prot = h_prot + self.interaction_scale * g * interaction_direct
        
        if split == 0:
            return h_prot
        return torch.cat([h_prefix, h_prot], dim=1)


# -----------------------------
# Output
# -----------------------------
@dataclass
class DecoderOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]
    prefix_len: int
    hidden_states: Optional[torch.Tensor] = None
    protein_hidden_states: Optional[torch.Tensor] = None
    full_injection_layer_indices: Optional[List[int]] = None


# -----------------------------
# Route-1 ProGen conditioned decoder (wrapper)
# -----------------------------
class ProGenConditionedDecoder(nn.Module):
    """
    Route-1:
      - Load ProGen2 backbone by from_pretrained()
      - Build prefix embeds from mem_func + mem_dl_light (plus optional boundary tokens)
      - Concat prefix embeds + protein token embeds -> inputs_embeds
      - attention_mask concat
      - labels: prefix positions set to -100
      - Optional: per-layer gated injection of mem_dl_full via forward hooks

    NOTE:
      - We do NOT re-implement ProGen's rotary/attention/cache.
      - We only add a lightweight "post-block" injector.
    """
    def __init__(
        self,
        backbone_name_or_path: str = "hugohrban/progen2-base",
        d_ctx: int = 768,
        add_boundary_tokens: bool = True,
        enable_full_injection: bool = True,
        inject_n_head: int = 8,
        inject_dropout: float = 0.0,
        freeze_backbone: bool = True,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        n_interaction_types: int = 6,
        full_injection_layer_policy: str = "all",
        full_injection_last_k: int = 0,
        share_injection_weights: bool = False,
        unfreeze_last_n_blocks: int = 0,
        prefix_boundary_scale: float = 1.0,
    ):
        super().__init__()

        model_source = os.environ.get("PROGEN_MODEL_DIR", backbone_name_or_path)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_source,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if int(unfreeze_last_n_blocks) > 0:
                backbone_blocks = _find_progen_blocks(self.backbone)
                for block in backbone_blocks[-int(unfreeze_last_n_blocks):]:
                    for p in block.parameters():
                        p.requires_grad = True

        # hidden size
        cfg = self.backbone.config
        # Resolve token embedding layer (remote-code ProGenModel may not implement get_input_embeddings())
        # Keep aliases as plain attributes. Registering the same embedding and
        # transformer ModuleList again under this wrapper duplicated the frozen
        # ProGen weights in state_dict/checkpoints.
        object.__setattr__(self, "input_embeddings", _resolve_input_embedding(self.backbone))

        d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None)
        if d_model is None:
            # fallback: read embedding dim
            d_model = int(self.input_embeddings.weight.shape[1])
        self.d_model = int(d_model)
        self.d_ctx = int(d_ctx)
        self.n_interaction_types = int(n_interaction_types)
        self.full_injection_layer_policy = str(full_injection_layer_policy or "all").lower()
        self.full_injection_last_k = int(full_injection_last_k or 0)
        self.share_injection_weights = bool(share_injection_weights)
        self.prefix_boundary_scale = float(prefix_boundary_scale)

        # projections from perceiver memory dim -> progen hidden dim
        self.func_proj = nn.Linear(self.d_ctx, self.d_model, bias=False) if self.d_ctx != self.d_model else nn.Identity()
        self.light_proj = nn.Linear(self.d_ctx, self.d_model, bias=False) if self.d_ctx != self.d_model else nn.Identity()
        self.fold_plan_proj = nn.Sequential(
            nn.LayerNorm(self.d_ctx),
            nn.Linear(self.d_ctx, self.d_model, bias=False),
        )
        self.full_proj = nn.Linear(self.d_ctx, self.d_model, bias=False) if self.d_ctx != self.d_model else nn.Identity()
        self.pocket_proj = nn.Linear(self.d_ctx, self.d_model, bias=False) if self.d_ctx != self.d_model else nn.Identity()
        self.ligand_chem_proj = nn.Linear(self.d_ctx, self.d_model, bias=False) if self.d_ctx != self.d_model else nn.Identity()
        self.interaction_proj = nn.Linear(self.d_ctx, self.d_model, bias=False) if self.d_ctx != self.d_model else nn.Identity()

        self.add_boundary_tokens = add_boundary_tokens
        if add_boundary_tokens:
            # 0: <FUNC>, 1: </FUNC>, 2: <DL_LIGHT>, 3: </DL_LIGHT>
            self.boundary = nn.Embedding(4, self.d_model)
        else:
            self.boundary = None

        head_hidden = max(16, self.d_model // 4)
        self.pocket_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        self.contact_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        self.interaction_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, self.n_interaction_types),
        )
        self.secondary_structure_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 3),
        )
        self.solvent_accessibility_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        self.residue_depth_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

        # locate blocks and register hooks for full-memory injection (optional)
        self.enable_full_injection = enable_full_injection
        object.__setattr__(
            self,
            "blocks",
            _find_progen_blocks(self.backbone) if enable_full_injection else None,
        )
        self.active_injection_layer_indices: List[int] = []

        self.injectors: Optional[nn.ModuleList] = None
        self.shared_injector: Optional[GatedFullMemoryInjector] = None
        self.injection_layer_scales: Optional[nn.Parameter] = None
        self._hooks = []
        if enable_full_injection:
            n_blocks = len(self.blocks)
            self.active_injection_layer_indices = self._select_injection_layers(n_blocks)
            if self.share_injection_weights:
                # A separate 37.8M-parameter injector at every ProGen layer made
                # the adapter larger than the frozen language model and is poorly
                # identified by small/medium datasets. Share the transformation
                # while retaining one learned scalar per layer.
                self.shared_injector = GatedFullMemoryInjector(
                    self.d_model, inject_n_head, dropout=inject_dropout
                )
                self.injection_layer_scales = nn.Parameter(torch.ones(n_blocks))
            else:
                self.injectors = nn.ModuleList([
                    GatedFullMemoryInjector(self.d_model, inject_n_head, dropout=inject_dropout)
                    for _ in range(n_blocks)
                ])
            self._register_injection_hooks()

        # runtime buffers for hooks
        self._rt_prefix_len: Optional[int] = None
        self._rt_mem_full: Optional[torch.Tensor] = None
        self._rt_mem_pocket: Optional[torch.Tensor] = None
        self._rt_mem_ligand_chem: Optional[torch.Tensor] = None
        self._rt_interaction_embeds: Optional[torch.Tensor] = None
        self._rt_start_pos: Optional[int] = None  # global start position for hook outputs
        self._rt_pocket_gate_signal: Optional[torch.Tensor] = None
        self._rt_pocket_gate_strength: float = 0.0
        self._rt_global_injection_scale: float = 1.0
        # Number of protein tokens that receive full-memory injection. ``None``
        # means inject for the whole sequence. This is used during training to
        # match hybrid decoding, where the injector is deliberately disabled
        # after an early prefix of generated residues.
        self._rt_injection_stop_after: Optional[int] = None

        # model max length (best effort)
        self.max_len = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None)

    def _select_injection_layers(self, n_layers: int) -> List[int]:
        policy = str(self.full_injection_layer_policy or "all").lower()
        if n_layers <= 0 or policy == "none":
            return []
        if policy == "all":
            return list(range(n_layers))
        if policy == "last_half":
            start = n_layers // 2
            return list(range(start, n_layers))
        if policy == "last_k":
            k = self.full_injection_last_k if self.full_injection_last_k > 0 else 1
            k = min(int(k), n_layers)
            return list(range(n_layers - k, n_layers))
        if policy == "every_other":
            return list(range(0, n_layers, 2))
        raise ValueError(
            f"Unknown full_injection_layer_policy={self.full_injection_layer_policy!r}; "
            "expected all/last_half/last_k/every_other/none"
        )

    def get_active_injection_layer_indices(self) -> List[int]:
        return list(self.active_injection_layer_indices)

    def _register_injection_hooks(self):
        # Hook after each block: output hidden -> inject full memory -> return same structure.
        assert self.blocks is not None
        assert self.injectors is not None or self.shared_injector is not None

        for i, blk in enumerate(self.blocks):
            injector = self.shared_injector if self.shared_injector is not None else self.injectors[i]

            def _hook_fn(module, inputs, output, idx=i, inj=injector):
                # output can be tuple or tensor depending on remote code; try to handle both.
                # We only adjust hidden states (first item).
                if self._rt_prefix_len is None or self._rt_mem_full is None:
                    return output
                if idx not in self.active_injection_layer_indices:
                    return output

                prefix_len = int(self._rt_prefix_len)
                start_pos = int(self._rt_start_pos or 0)
                n_layers = max(1, len(self.blocks))
                early_end = max(1, n_layers // 3)
                mid_end = max(early_end + 1, 2 * n_layers // 3)

                mem = self._rt_mem_full
                if idx < early_end and self._rt_mem_pocket is not None:
                    mem = self._rt_mem_pocket
                elif idx < mid_end and self._rt_mem_ligand_chem is not None:
                    mem = self._rt_mem_ligand_chem

                gate_signal = self._rt_interaction_embeds if self._rt_interaction_embeds is not None else self._rt_pocket_gate_signal
                layer_scale = (
                    self.injection_layer_scales[idx]
                    if self.injection_layer_scales is not None
                    else None
                )
                global_scale = float(self._rt_global_injection_scale)

                def _apply_runtime_window(h: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
                    stop_after = self._rt_injection_stop_after
                    if stop_after is None or int(stop_after) < 0:
                        return h2
                    positions = torch.arange(
                        h.size(1), device=h.device, dtype=torch.long
                    ) + int(start_pos)
                    active = (
                        (positions >= int(prefix_len))
                        & (positions < int(prefix_len) + int(stop_after))
                    ).view(1, -1, 1)
                    return torch.where(active, h2, h)

                if isinstance(output, tuple):
                    h = output[0]
                    h2 = inj(
                        h,
                        mem,
                        prefix_len=prefix_len,
                        start_pos=start_pos,
                        pocket_gate_signal=gate_signal,
                        pocket_gate_strength=self._rt_pocket_gate_strength,
                    )
                    if layer_scale is not None:
                        h2 = h + layer_scale.to(dtype=h.dtype) * (h2 - h)
                    if global_scale != 1.0:
                        h2 = h + global_scale * (h2 - h)
                    h2 = _apply_runtime_window(h, h2)
                    return (h2,) + output[1:]
                else:
                    h = output
                    h2 = inj(
                        h,
                        mem,
                        prefix_len=prefix_len,
                        start_pos=start_pos,
                        pocket_gate_signal=gate_signal,
                        pocket_gate_strength=self._rt_pocket_gate_strength,
                    )
                    if layer_scale is not None:
                        h2 = h + layer_scale.to(dtype=h.dtype) * (h2 - h)
                    if global_scale != 1.0:
                        h2 = h + global_scale * (h2 - h)
                    h2 = _apply_runtime_window(h, h2)
                    return h2

            self._hooks.append(blk.register_forward_hook(_hook_fn))

    def _maybe_project_mem_full(self, mem_full: torch.Tensor) -> torch.Tensor:
        """
        Ensure mem_full last dim == decoder hidden size (self.d_model).
        Compatibility wrapper; the projection layer is registered in __init__.
        """
        return self._project_full_mem(mem_full)


    def remove_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def _project_memory(self, mem: Optional[torch.Tensor], proj: nn.Module, name: str) -> Optional[torch.Tensor]:
        if mem is None:
            return None
        if mem.dim() != 3:
            raise ValueError(f"{name} must be [B,N,D], got {tuple(mem.shape)}")
        in_dim = int(mem.size(-1))
        if in_dim == int(self.d_model):
            return mem
        if in_dim == int(self.d_ctx):
            return proj(mem)
        raise ValueError(f"Unsupported {name} dim={in_dim}; expected {self.d_model} or {self.d_ctx}.")

    def _project_full_mem(self, mem_full: torch.Tensor) -> torch.Tensor:
        return self._project_memory(mem_full, self.full_proj, "mem_full")

    def _build_interaction_context(self, h_like: torch.Tensor, interaction_embeds: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if interaction_embeds is None:
            return None
        inter = self._project_memory(interaction_embeds, self.interaction_proj, "interaction_embeds")
        # Runtime hooks do not know hidden states before the block, so use a compact global
        # interaction context expanded along protein positions. The injector still computes
        # token-wise gates from q + this interaction context.
        ctx = inter.mean(dim=1, keepdim=True)
        return ctx.expand(h_like.size(0), h_like.size(1), h_like.size(2))


    def build_prefix(
        self,
        mem_func: torch.Tensor,
        mem_dl_light: torch.Tensor,
        mem_fold_plan: Optional[torch.Tensor] = None,
        condition_scale: float = 1.0,
        fold_plan_scale: float = 1.0,
        pool_condition_memory: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """
        Build a hierarchical prefix. Fold-plan slots remain ordered and are
        never pooled because they encode normalized sequence topology.
        """
        B = mem_func.size(0)
        device = mem_func.device

        func = self.func_proj(mem_func) * float(condition_scale)          # [B,Nf,D]
        light = self.light_proj(mem_dl_light) * float(condition_scale)    # [B,Nl,D]
        fold_plan = (
            self.fold_plan_proj(mem_fold_plan) * float(fold_plan_scale)
            if mem_fold_plan is not None
            else func.new_zeros((B, 0, self.d_model))
        )
        if pool_condition_memory:
            # Keep a compact, separately tagged summary of function and
            # ligand/docking memories. This avoids shifting ProGen residues by
            # dozens of unobserved prefix positions.
            func = func.mean(dim=1, keepdim=True)
            light = light.mean(dim=1, keepdim=True)

        if self.boundary is None:
            prefix = torch.cat([func, fold_plan, light], dim=1)
            return prefix, prefix.size(1)

        # boundary tokens
        ids = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)  # [4]
        bnd = self.boundary(ids) * float(self.prefix_boundary_scale)  # [4,D]
        b0 = bnd[0].view(1, 1, -1).expand(B, -1, -1)
        b1 = bnd[1].view(1, 1, -1).expand(B, -1, -1)
        b2 = bnd[2].view(1, 1, -1).expand(B, -1, -1)
        b3 = bnd[3].view(1, 1, -1).expand(B, -1, -1)

        prefix = torch.cat([b0, func, b1, fold_plan, b2, light, b3], dim=1)
        return prefix, prefix.size(1)

    def enable_runtime_full_injection(
        self,
        prefix_len: int,
        mem_full: torch.Tensor,
        start_pos: int = 0,
        pocket_gate_signal: Optional[torch.Tensor] = None,
        pocket_gate_strength: float = 0.0,
        mem_pocket: Optional[torch.Tensor] = None,
        mem_ligand_chem: Optional[torch.Tensor] = None,
        interaction_embeds: Optional[torch.Tensor] = None,
        global_injection_scale: float = 1.0,
        injection_stop_after: Optional[int] = None,
    ) -> None:
        """Enable full injection and set runtime buffers."""
        if not self.enable_full_injection or not self.active_injection_layer_indices:
            return
        self._rt_prefix_len = int(prefix_len)
        self._rt_mem_full = self._project_full_mem(mem_full)
        self._rt_mem_pocket = self._project_memory(mem_pocket, self.pocket_proj, "mem_pocket")
        self._rt_mem_ligand_chem = self._project_memory(mem_ligand_chem, self.ligand_chem_proj, "mem_ligand_chem")
        self._rt_interaction_embeds = self._project_memory(interaction_embeds, self.interaction_proj, "interaction_embeds")
        self._rt_start_pos = int(start_pos)
        self._rt_pocket_gate_signal = pocket_gate_signal
        self._rt_pocket_gate_strength = float(pocket_gate_strength)
        self._rt_global_injection_scale = float(global_injection_scale)
        self._rt_injection_stop_after = (
            None if injection_stop_after is None or int(injection_stop_after) < 0
            else int(injection_stop_after)
        )

    def update_runtime_start_pos(self, start_pos: int) -> None:
        """Update global start position for cached one-token generation hooks."""
        self._rt_start_pos = int(start_pos)

    def disable_runtime_full_injection(self) -> None:
        """Disable full injection and clear runtime buffers."""
        self._rt_prefix_len = None
        self._rt_mem_full = None
        self._rt_mem_pocket = None
        self._rt_mem_ligand_chem = None
        self._rt_interaction_embeds = None
        self._rt_start_pos = None
        self._rt_pocket_gate_signal = None
        self._rt_pocket_gate_strength = 0.0
        self._rt_global_injection_scale = 1.0
        self._rt_injection_stop_after = None

    def forward(
        self,
        protein_input_ids: torch.Tensor,          # [B,Lp]
        protein_attention_mask: torch.Tensor,     # [B,Lp] 1=valid 0=pad
        mem_func: torch.Tensor,                   # [B,Nf,d_ctx]
        mem_dl_light: torch.Tensor,               # [B,Nl,d_ctx]
        mem_fold_plan: Optional[torch.Tensor] = None,  # [B,Nplan,d_ctx]
        mem_dl_full: Optional[torch.Tensor] = None,  # [B,Nfull,d_ctx]
        mem_pocket: Optional[torch.Tensor] = None,
        mem_ligand_chem: Optional[torch.Tensor] = None,
        interaction_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,    # [B,Lp] protein-only labels
        position_weights: Optional[torch.Tensor] = None,  # [B,Lp] optional token CE weights
        pocket_gate_signal: Optional[torch.Tensor] = None,
        pocket_gate_strength: float = 0.0,
        use_cache: bool = False,                  # training usually False
        eos_weight: Optional[float] = None,       # if >1.0, upweight EOS token loss
        eos_token_id: Optional[int] = None,       # required when eos_weight is used
        full_injection_stop_after: Optional[int] = None,
        full_injection_scale: float = 1.0,
        prefix_condition_scale: float = 1.0,
        fold_plan_scale: float = 1.0,
        pool_condition_memory: bool = False,
    ) -> DecoderOutput:
        B, Lp = protein_input_ids.shape
        device = protein_input_ids.device

        # prefix embeds
        prefix, prefix_len = self.build_prefix(
            mem_func,
            mem_dl_light,
            mem_fold_plan=mem_fold_plan,
            condition_scale=prefix_condition_scale,
            fold_plan_scale=fold_plan_scale,
            pool_condition_memory=pool_condition_memory,
        )  # [B,Lprefix,D]

        # protein token embeds from backbone (robust for remote-code models)
        prot_emb = self.input_embeddings(protein_input_ids)  # [B,Lp,D]

        # concat into inputs_embeds
        inputs_embeds = torch.cat([prefix, prot_emb], dim=1)  # [B,Ltotal,D]
        Ltotal = inputs_embeds.size(1)

        if self.max_len is not None and Ltotal > int(self.max_len):
            raise ValueError(f"Ltotal={Ltotal} exceeds backbone max_len={self.max_len}. Reduce prefix/protein length.")

        # attention_mask: prefix all ones + protein mask
        prefix_mask = torch.ones((B, prefix_len), device=device, dtype=protein_attention_mask.dtype)
        attention_mask = torch.cat([prefix_mask, protein_attention_mask], dim=1)  # [B,Ltotal]

        # labels: prefix -> -100
        full_labels = None
        full_position_weights = None
        if labels is not None:
            full_labels = _make_full_labels(prefix_len, labels)
            full_position_weights = _make_full_position_weights(prefix_len, position_weights, labels)

        # setup runtime buffers for hooks
        if self.enable_full_injection and self.active_injection_layer_indices and mem_dl_full is not None:
            self._rt_prefix_len = prefix_len
            self._rt_mem_full = self._project_full_mem(mem_dl_full)  # [B,Nfull,D]
            self._rt_mem_pocket = self._project_memory(mem_pocket, self.pocket_proj, "mem_pocket")
            self._rt_mem_ligand_chem = self._project_memory(mem_ligand_chem, self.ligand_chem_proj, "mem_ligand_chem")
            self._rt_interaction_embeds = self._project_memory(interaction_embeds, self.interaction_proj, "interaction_embeds")
            self._rt_start_pos = 0
            self._rt_pocket_gate_signal = pocket_gate_signal
            self._rt_pocket_gate_strength = float(pocket_gate_strength)
            self._rt_global_injection_scale = float(full_injection_scale)
            self._rt_injection_stop_after = (
                None
                if full_injection_stop_after is None or int(full_injection_stop_after) < 0
                else int(full_injection_stop_after)
            )
        else:
            self._rt_prefix_len = None
            self._rt_mem_full = None
            self._rt_mem_pocket = None
            self._rt_mem_ligand_chem = None
            self._rt_interaction_embeds = None
            self._rt_start_pos = None
            self._rt_pocket_gate_signal = None
            self._rt_pocket_gate_strength = 0.0
            self._rt_global_injection_scale = 1.0
            self._rt_injection_stop_after = None

        # We only need the final normalized hidden state for auxiliary heads.
        # Asking Hugging Face for ``output_hidden_states=True`` retains all 27
        # ProGen block outputs and dominates memory for long proteins. Capture
        # the exact tensor entering the LM head instead; this is the same final
        # normalized state without materializing the full hidden-state tuple.
        captured_final_hidden = {}
        capture_handle = None
        lm_head = getattr(self.backbone, "lm_head", None)
        if lm_head is not None:
            def _capture_lm_head_input(_module, hook_inputs):
                if hook_inputs:
                    captured_final_hidden["value"] = hook_inputs[0]

            capture_handle = lm_head.register_forward_pre_hook(_capture_lm_head_input)

        try:
            try:
                out = self.backbone(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=full_labels,
                    use_cache=use_cache,
                    output_hidden_states=(capture_handle is None),
                    return_dict=True,
                )
            except TypeError:
                out = self.backbone(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=full_labels,
                    use_cache=use_cache,
                    return_dict=True,
                )
            logits = out.logits
            loss = out.loss if hasattr(out, "loss") else None
            hidden_states = getattr(out, "hidden_states", None)
            last_hidden = captured_final_hidden.get("value")
            if last_hidden is None and hidden_states is not None and len(hidden_states) > 0:
                last_hidden = hidden_states[-1]
            protein_hidden_states = last_hidden[:, prefix_len:prefix_len + Lp, :] if last_hidden is not None else None

            if (
                full_labels is not None
                and (
                    full_position_weights is not None
                    or (eos_weight is not None and float(eos_weight) != 1.0 and eos_token_id is not None)
                    or loss is None
                )
            ):
                shift_logits = logits[:, :-1, :].contiguous()      # [B, L-1, V]
                shift_labels = full_labels[:, 1:].contiguous()     # [B, L-1]

                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                per_tok_loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                ).view_as(shift_labels)

                if full_position_weights is None:
                    weights = torch.ones_like(
                        shift_labels,
                        dtype=shift_logits.dtype,
                        device=shift_logits.device
                    )
                else:
                    weights = full_position_weights[:, 1:].to(
                        dtype=shift_logits.dtype,
                        device=shift_logits.device
                    ).contiguous()

                if eos_weight is not None and float(eos_weight) != 1.0 and eos_token_id is not None:
                    weights = weights.clone()
                    weights[shift_labels == eos_token_id] *= float(eos_weight)
                weights[shift_labels == -100] = 0.0

                denom = weights.sum().clamp(min=1e-8)
                loss = (per_tok_loss * weights).sum() / denom

        finally:
            if capture_handle is not None:
                capture_handle.remove()
            # always clear runtime buffers to avoid accidental leakage between batches
            self._rt_prefix_len = None
            self._rt_mem_full = None
            self._rt_mem_pocket = None
            self._rt_mem_ligand_chem = None
            self._rt_interaction_embeds = None
            self._rt_start_pos = None
            self._rt_pocket_gate_signal = None
            self._rt_pocket_gate_strength = 0.0
            self._rt_global_injection_scale = 1.0
            self._rt_injection_stop_after = None

        return DecoderOutput(
            logits=logits,
            loss=loss,
            prefix_len=prefix_len,
            hidden_states=last_hidden,
            protein_hidden_states=protein_hidden_states,
            full_injection_layer_indices=self.get_active_injection_layer_indices(),
        )
