import pytest
import torch

from tinyserve import ops
from tinyserve import reference as ref

DEV = "cuda"
BF16 = torch.bfloat16


def assert_close(a, b, atol=2e-2, rtol=2e-2):
    torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("tokens,hidden", [(1, 1024), (7, 2048), (128, 2560), (33, 5120)])
def test_rms_norm(tokens, hidden):
    x = torch.randn(tokens, hidden, dtype=BF16, device=DEV)
    w = torch.randn(hidden, dtype=BF16, device=DEV)
    assert_close(ops.rms_norm(x, w, 1e-6), ref.rms_norm(x, w, 1e-6))


@pytest.mark.parametrize("tokens,hidden", [(1, 1024), (64, 2048), (300, 4096)])
def test_fused_add_rms_norm(tokens, hidden):
    x = torch.randn(tokens, hidden, dtype=BF16, device=DEV)
    r = torch.randn(tokens, hidden, dtype=BF16, device=DEV)
    w = torch.randn(hidden, dtype=BF16, device=DEV)
    exp_x, exp_r = ref.fused_add_rms_norm(x, r, w, 1e-6)
    ops.fused_add_rms_norm(x, r, w, 1e-6)
    torch.testing.assert_close(r, exp_r, atol=0, rtol=0)
    assert_close(x, exp_x)


@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("hq,hk,d", [(16, 8, 128), (32, 8, 128), (8, 8, 64)])
def test_qk_norm_rope(norm, hq, hk, d):
    t = 37
    qkv = torch.randn(t, (hq + 2 * hk) * d, dtype=BF16, device=DEV)
    pos = torch.randint(0, 4096, (t,), device=DEV)
    cs = ref.rope_cos_sin_cache(d, 4096, 1e6)
    qw = torch.randn(d, dtype=BF16, device=DEV) if norm else None
    kw = torch.randn(d, dtype=BF16, device=DEV) if norm else None
    expected = ref.qk_norm_rope(qkv, pos, cs, qw, kw, hq, hk, d, 1e-6)
    ops.qk_norm_rope(qkv, pos, cs, qw, kw, hq, hk, d, 1e-6)
    assert_close(qkv, expected)


@pytest.mark.parametrize("tokens,inter", [(1, 3072), (50, 6144), (256, 9728)])
def test_silu_and_mul(tokens, inter):
    x = torch.randn(tokens, 2 * inter, dtype=BF16, device=DEV)
    assert_close(ops.silu_and_mul(x), ref.silu_and_mul(x))


def make_paged_cache(context_lens, hk, d, block_size, extra_blocks=7):
    """Random caches with block tables that use a shuffled (non-contiguous) set of physical blocks."""
    need = [(c + block_size - 1) // block_size for c in context_lens]
    num_blocks = sum(need) + extra_blocks
    k_cache = torch.randn(num_blocks, block_size, hk, d, dtype=BF16, device=DEV)
    v_cache = torch.randn(num_blocks, block_size, hk, d, dtype=BF16, device=DEV)
    perm = torch.randperm(num_blocks, device=DEV).int()
    max_blocks = max(need)
    tables = torch.zeros(len(context_lens), max_blocks, dtype=torch.int32, device=DEV)
    off = 0
    for i, n in enumerate(need):
        tables[i, :n] = perm[off: off + n]
        off += n
    return k_cache, v_cache, tables


def test_reshape_and_cache():
    t, hq, hk, d, bs = 21, 16, 8, 128, 16
    qkv = torch.randn(t, (hq + 2 * hk) * d, dtype=BF16, device=DEV)
    k = qkv[:, hq * d: (hq + hk) * d].view(t, hk, d)
    v = qkv[:, (hq + hk) * d:].view(t, hk, d)
    k_cache = torch.zeros(8, bs, hk, d, dtype=BF16, device=DEV)
    v_cache = torch.zeros_like(k_cache)
    slots = torch.randperm(8 * bs, device=DEV)[:t]
    slots[3] = -1  # padding token must be skipped
    ops.reshape_and_cache(k, v, k_cache, v_cache, slots)
    kc, vc = k_cache.view(-1, hk, d), v_cache.view(-1, hk, d)
    keep = slots >= 0
    torch.testing.assert_close(kc[slots[keep]], k[keep], atol=0, rtol=0)
    torch.testing.assert_close(vc[slots[keep]], v[keep], atol=0, rtol=0)


@pytest.mark.parametrize("hq,hk,d", [(16, 8, 128), (32, 8, 128), (16, 16, 128), (8, 1, 64)])
@pytest.mark.parametrize("block_size", [16, 32])
def test_paged_decode_attention(hq, hk, d, block_size):
    context_lens = [1, 17, 255, 256, 257, 1000, 3001]
    k_cache, v_cache, tables = make_paged_cache(context_lens, hk, d, block_size)
    b = len(context_lens)
    q = torch.randn(b, hq, d, dtype=BF16, device=DEV)
    ctx = torch.tensor(context_lens, dtype=torch.int32, device=DEV)
    cu = torch.arange(b + 1, dtype=torch.int32, device=DEV)
    scale = d ** -0.5
    expected = ref.paged_attention(q, k_cache, v_cache, tables, cu, ctx, scale)
    for parts in [ops.num_decode_partitions(max(context_lens)), ops.num_decode_partitions(8192)]:
        out = torch.empty_like(q)
        ops.paged_decode_attention(out, q, k_cache, v_cache, tables, ctx, scale, parts)
        assert_close(out, expected)


def test_paged_decode_single_partition_strided_q():
    # q as a strided view of a packed qkv buffer, the way the model calls it
    hq, hk, d, bs = 16, 8, 128, 16
    context_lens = [5, 100, 256]
    k_cache, v_cache, tables = make_paged_cache(context_lens, hk, d, bs)
    qkv = torch.randn(3, (hq + 2 * hk) * d, dtype=BF16, device=DEV)
    q = qkv[:, : hq * d].view(3, hq, d)
    ctx = torch.tensor(context_lens, dtype=torch.int32, device=DEV)
    cu = torch.arange(4, dtype=torch.int32, device=DEV)
    out = torch.empty(3, hq, d, dtype=BF16, device=DEV)
    ops.paged_decode_attention(out, q, k_cache, v_cache, tables, ctx, d ** -0.5, 1)
    assert_close(out, ref.paged_attention(q, k_cache, v_cache, tables, cu, ctx, d ** -0.5))


@pytest.mark.parametrize("hq,hk,d", [(16, 8, 128), (32, 8, 128), (8, 2, 64)])
@pytest.mark.parametrize(
    "q_lens,prefix_lens",
    [
        ([1], [0]),
        ([64], [0]),
        ([17, 100, 333], [0, 0, 0]),  # plain varlen prefill
        ([5, 64, 200], [16, 48, 1000]),  # prefix-cache hit / chunked prefill continuation
        ([1, 1, 1], [10, 300, 77]),  # decode-shaped
        ([1024], [512]),
    ],
)
def test_paged_prefill_attention(hq, hk, d, q_lens, prefix_lens):
    bs = 16
    context_lens = [q + p for q, p in zip(q_lens, prefix_lens)]
    k_cache, v_cache, tables = make_paged_cache(context_lens, hk, d, bs)
    total = sum(q_lens)
    qkv = torch.randn(total, (hq + 2 * hk) * d, dtype=BF16, device=DEV)
    q = qkv[:, : hq * d].view(total, hq, d)
    cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=DEV)
    ctx = torch.tensor(context_lens, dtype=torch.int32, device=DEV)
    scale = d ** -0.5
    out = ops.paged_prefill_attention(q, k_cache, v_cache, tables, cu, ctx, max(q_lens), scale)
    assert_close(out, ref.paged_attention(q, k_cache, v_cache, tables, cu, ctx, scale))
