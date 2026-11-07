"""Python entry points for the custom CUDA kernels (JIT-compiled on first use)."""

import os
from functools import lru_cache
from pathlib import Path

import torch

_CSRC = Path(__file__).resolve().parent.parent / "csrc"
_SOURCES = ["ops.cpp", "norm.cu", "rope.cu", "elementwise.cu", "decode_attention.cu", "prefill_attention.cu"]


@lru_cache(maxsize=None)
def _C():
    from torch.utils.cpp_extension import load

    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    return load(
        name="tinyserve_C",
        sources=[str(_CSRC / s) for s in _SOURCES],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "--use_fast_math", "-lineinfo", "--expt-relaxed-constexpr"],
        verbose=os.environ.get("TINYSERVE_VERBOSE_BUILD") == "1",
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    _C().rms_norm(out, x, weight, eps)
    return out


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    """In place: residual += x; x = rms_norm(residual) * weight."""
    _C().fused_add_rms_norm(x, residual, weight, eps)


def qk_norm_rope(qkv, positions, cos_sin, q_norm_w, k_norm_w, num_q_heads, num_kv_heads, head_dim, eps) -> None:
    """In place on the q and k sections of the packed qkv tensor."""
    _C().qk_norm_rope(qkv, positions, cos_sin, q_norm_w, k_norm_w, num_q_heads, num_kv_heads, head_dim, eps)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[0], x.shape[1] // 2, dtype=x.dtype, device=x.device)
    _C().silu_and_mul(out, x)
    return out


def reshape_and_cache(k, v, k_cache, v_cache, slot_mapping) -> None:
    _C().reshape_and_cache(k, v, k_cache, v_cache, slot_mapping)


def decode_partition_size() -> int:
    return _C().decode_partition_size()


def num_decode_partitions(max_context_len: int) -> int:
    p = decode_partition_size()
    return max(1, (max_context_len + p - 1) // p)


def paged_decode_attention(out, q, k_cache, v_cache, block_tables, context_lens, scale, num_partitions,
                           tmp_out=None, tmp_ml=None) -> torch.Tensor:
    if num_partitions > 1 and tmp_out is None:
        b, h, d = q.shape
        tmp_out = torch.empty(b, h, num_partitions, d, dtype=torch.float32, device=q.device)
        tmp_ml = torch.empty(b, h, num_partitions, 2, dtype=torch.float32, device=q.device)
    if tmp_out is None:
        tmp_out = tmp_ml = torch.empty(0, dtype=torch.float32, device=q.device)
    _C().paged_decode_attention(out, q, k_cache, v_cache, block_tables, context_lens, scale, num_partitions,
                                tmp_out, tmp_ml)
    return out


def paged_prefill_attention(q, k_cache, v_cache, block_tables, cu_seqlens_q, context_lens, max_q_len, scale,
                            out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _C().paged_prefill_attention(out, q, k_cache, v_cache, block_tables, cu_seqlens_q, context_lens, max_q_len,
                                 scale)
    return out
