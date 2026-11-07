"""Qwen3 / Llama-style decoder built on the custom kernels.

All tokens of a step are flattened into one [num_tokens, hidden] batch; per-sequence structure is carried by the
AttentionMetadata (paged KV addressing), so GEMMs always see the whole batch.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from tinyserve import ops
from tinyserve.config import ModelConfig
from tinyserve.reference import rope_cos_sin_cache


@dataclass
class AttentionMetadata:
    """Paged-attention addressing for one step. Tokens are laid out decodes first: rows [0, num_decodes) are one
    query token per decoding sequence (served by the flash-decoding kernel), the remaining rows are the prompt
    chunks of the prefilling sequences (served by the varlen prefill kernel). Either part may be empty."""

    slot_mapping: torch.Tensor  # [T] int64, -1 = padding
    num_decodes: int = 0
    # decode part
    decode_block_tables: torch.Tensor | None = None  # [num_decodes, max_blocks] int32
    decode_context_lens: torch.Tensor | None = None  # [num_decodes] int32, kv length including this token
    num_partitions: int = 1
    decode_tmp_out: torch.Tensor | None = None
    decode_tmp_ml: torch.Tensor | None = None
    # prefill part
    block_tables: torch.Tensor | None = None  # [P, max_blocks] int32
    context_lens: torch.Tensor | None = None  # [P] int32, kv length including this step's tokens
    cu_seqlens_q: torch.Tensor | None = None  # [P+1] int32, offsets relative to the first prefill row
    max_q_len: int = 0

    @property
    def has_prefill(self) -> bool:
        return self.cu_seqlens_q is not None


class Qwen3Model(torch.nn.Module):
    def __init__(self, cfg: ModelConfig, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        h, d = cfg.hidden_size, cfg.head_dim
        self.hq, self.hk, self.d = cfg.num_attention_heads, cfg.num_key_value_heads, d
        self.scale = d ** -0.5
        e = lambda *s: torch.empty(*s, device=device, dtype=dtype)  # noqa: E731
        self.embed = e(cfg.vocab_size, h)
        self.layers = []
        for _ in range(cfg.num_hidden_layers):
            self.layers.append({
                "in_norm": e(h),
                "qkv": e((self.hq + 2 * self.hk) * d, h),
                "q_norm": e(d) if cfg.use_qk_norm else None,
                "k_norm": e(d) if cfg.use_qk_norm else None,
                "o": e(h, self.hq * d),
                "post_norm": e(h),
                "gate_up": e(2 * cfg.intermediate_size, h),
                "down": e(h, cfg.intermediate_size),
            })
        self.norm = e(h)
        self.lm_head = self.embed if cfg.tie_word_embeddings else e(cfg.vocab_size, h)
        self.cos_sin = rope_cos_sin_cache(d, cfg.max_position_embeddings, cfg.rope_theta, device)
        self.kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []

    @torch.no_grad()
    def load_weights(self, path: str | Path):
        files = sorted(Path(path).glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {path}")
        hq, hk, inter = self.hq * self.d, self.hk * self.d, self.cfg.intermediate_size
        # HF name -> (destination tensor, row offset into a fused weight)
        dest = {"model.embed_tokens.weight": (self.embed, 0), "model.norm.weight": (self.norm, 0)}
        if not self.cfg.tie_word_embeddings:
            dest["lm_head.weight"] = (self.lm_head, 0)
        for i, L in enumerate(self.layers):
            p = f"model.layers.{i}."
            dest |= {
                p + "input_layernorm.weight": (L["in_norm"], 0),
                p + "post_attention_layernorm.weight": (L["post_norm"], 0),
                p + "self_attn.q_proj.weight": (L["qkv"], 0),
                p + "self_attn.k_proj.weight": (L["qkv"], hq),
                p + "self_attn.v_proj.weight": (L["qkv"], hq + hk),
                p + "self_attn.o_proj.weight": (L["o"], 0),
                p + "mlp.gate_proj.weight": (L["gate_up"], 0),
                p + "mlp.up_proj.weight": (L["gate_up"], inter),
                p + "mlp.down_proj.weight": (L["down"], 0),
            }
            if self.cfg.use_qk_norm:
                dest[p + "self_attn.q_norm.weight"] = (L["q_norm"], 0)
                dest[p + "self_attn.k_norm.weight"] = (L["k_norm"], 0)
        loaded = set()
        for f in files:
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                for name in sf.keys():
                    if name not in dest:
                        continue
                    t, off = dest[name]
                    w = sf.get_tensor(name)
                    t[off: off + w.shape[0]].copy_(w)
                    loaded.add(name)
        missing = set(dest) - loaded
        if missing:
            raise ValueError(f"missing weights: {sorted(missing)[:5]} ...")

    def bind_kv_cache(self, kv_cache: torch.Tensor):
        """kv_cache: [num_layers, 2, num_blocks, block_size, Hk, D]."""
        self.kv_caches = [(kv_cache[i, 0], kv_cache[i, 1]) for i in range(len(self.layers))]

    def _attention(self, L, layer_idx, h, positions, meta: AttentionMetadata):
        t = h.shape[0]
        hq, hk, d = self.hq, self.hk, self.d
        qkv = F.linear(h, L["qkv"])
        ops.qk_norm_rope(qkv, positions, self.cos_sin, L["q_norm"], L["k_norm"], hq, hk, d, self.cfg.rms_norm_eps)
        q = qkv[:, : hq * d].view(t, hq, d)
        k = qkv[:, hq * d: (hq + hk) * d].view(t, hk, d)
        v = qkv[:, (hq + hk) * d:].view(t, hk, d)
        k_cache, v_cache = self.kv_caches[layer_idx]
        ops.reshape_and_cache(k, v, k_cache, v_cache, meta.slot_mapping)
        nd = meta.num_decodes
        o = torch.empty(t, hq, d, dtype=h.dtype, device=h.device)
        if nd:
            ops.paged_decode_attention(o[:nd], q[:nd], k_cache, v_cache, meta.decode_block_tables,
                                       meta.decode_context_lens, self.scale, meta.num_partitions,
                                       meta.decode_tmp_out, meta.decode_tmp_ml)
        if meta.has_prefill:
            ops.paged_prefill_attention(q[nd:], k_cache, v_cache, meta.block_tables, meta.cu_seqlens_q,
                                        meta.context_lens, meta.max_q_len, self.scale, out=o[nd:])
        return F.linear(o.view(t, hq * d), L["o"])

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, meta: AttentionMetadata) -> torch.Tensor:
        """Returns final hidden states [T, hidden] (call compute_logits on the rows you want to sample)."""
        eps = self.cfg.rms_norm_eps
        h = F.embedding(input_ids, self.embed)
        residual = None
        for i, L in enumerate(self.layers):
            if residual is None:
                residual = h
                h = ops.rms_norm(h, L["in_norm"], eps)
            else:
                ops.fused_add_rms_norm(h, residual, L["in_norm"], eps)
            h = self._attention(L, i, h, positions, meta)
            ops.fused_add_rms_norm(h, residual, L["post_norm"], eps)
            h = F.linear(ops.silu_and_mul(F.linear(h, L["gate_up"])), L["down"])
        ops.fused_add_rms_norm(h, residual, self.norm, eps)
        return h

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.lm_head).float()
