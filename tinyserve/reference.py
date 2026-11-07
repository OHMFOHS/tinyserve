"""Eager PyTorch reference implementations of every custom kernel, used by tests and benchmarks."""

import torch
import torch.nn.functional as F


def rms_norm(x, w, eps):
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return w * y


def fused_add_rms_norm(x, residual, w, eps):
    residual = x + residual
    return rms_norm(residual, w, eps), residual


def rope_cos_sin_cache(head_dim, max_pos, theta, device="cuda"):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32, device=device), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def apply_rope(x, positions, cos_sin):
    """x: [T, H, D] (any float dtype), NeoX rotate-half convention, computed in fp32."""
    d = x.shape[-1]
    cs = cos_sin[positions]
    cos, sin = cs[:, None, : d // 2], cs[:, None, d // 2:]
    xf = x.float()
    x1, x2 = xf[..., : d // 2], xf[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


def qk_norm_rope(qkv, positions, cos_sin, q_w, k_w, hq, hk, d, eps):
    t = qkv.shape[0]
    q = qkv[:, : hq * d].view(t, hq, d)
    k = qkv[:, hq * d: (hq + hk) * d].view(t, hk, d)
    if q_w is not None:
        q, k = rms_norm(q, q_w, eps), rms_norm(k, k_w, eps)
    out = qkv.clone()
    out[:, : hq * d] = apply_rope(q, positions, cos_sin).reshape(t, -1)
    out[:, hq * d: (hq + hk) * d] = apply_rope(k, positions, cos_sin).reshape(t, -1)
    return out


def silu_and_mul(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def gather_kv(cache, block_table, length):
    """Read `length` tokens of one sequence out of a paged cache [num_blocks, block_size, H, D]."""
    bs = cache.shape[1]
    nblk = (length + bs - 1) // bs
    return cache[block_table[:nblk].long()].flatten(0, 1)[:length]


def paged_attention(q, k_cache, v_cache, block_tables, cu_seqlens_q, context_lens, scale):
    """Causal attention of each sequence's new queries against its full cached context, in fp32."""
    outs = []
    hq, hk = q.shape[1], k_cache.shape[2]
    for b in range(context_lens.numel()):
        s, e = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        ctx = int(context_lens[b])
        qb = q[s:e].float().transpose(0, 1)  # [H, Lq, D]
        k = gather_kv(k_cache, block_tables[b], ctx).float().repeat_interleave(hq // hk, dim=1).transpose(0, 1)
        v = gather_kv(v_cache, block_tables[b], ctx).float().repeat_interleave(hq // hk, dim=1).transpose(0, 1)
        lq = e - s
        qpos = torch.arange(ctx - lq, ctx, device=q.device)[:, None]
        mask = torch.arange(ctx, device=q.device)[None, :] <= qpos
        o = F.scaled_dot_product_attention(qb, k, v, attn_mask=mask, scale=scale)
        outs.append(o.transpose(0, 1))
    return torch.cat(outs).to(q.dtype)
