"""Kernel micro-benchmarks against PyTorch baselines. Prints markdown tables (and writes results/kernels.md).

    python benchmarks/bench_kernels.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyserve import ops  # noqa: E402
from tinyserve import reference as ref  # noqa: E402

DEV, BF16 = "cuda", torch.bfloat16
HQ, HK, D = 16, 8, 128  # Qwen3-0.6B / 1.7B attention shape


def bench(fn, warmup=10, iters=50) -> float:
    """Median milliseconds per call, timed with CUDA events; L2 is flushed between calls."""
    flush = torch.empty(64 << 20, dtype=torch.int8, device=DEV)
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def table(title, header, rows):
    out = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    text = "\n".join(out) + "\n"
    print(text)
    return text


def paged_setup(batch, ctx, block_size=16):
    nblk = (ctx + block_size - 1) // block_size
    num_blocks = batch * nblk
    k_cache = torch.randn(num_blocks, block_size, HK, D, dtype=BF16, device=DEV)
    v_cache = torch.randn_like(k_cache)
    tables = torch.randperm(num_blocks, device=DEV).int().view(batch, nblk)
    ctx_lens = torch.full((batch,), ctx, dtype=torch.int32, device=DEV)
    return k_cache, v_cache, tables, ctx_lens


def bench_decode():
    rows = []
    for batch, ctx in [(1, 1024), (1, 4096), (1, 16384), (8, 2048), (32, 1024), (32, 4096), (64, 2048), (128, 1024)]:
        k_cache, v_cache, tables, ctx_lens = paged_setup(batch, ctx)
        q = torch.randn(batch, HQ, D, dtype=BF16, device=DEV)
        out = torch.empty_like(q)
        parts = ops.num_decode_partitions(ctx)
        tmp_o = torch.empty(batch, HQ, parts, D, dtype=torch.float32, device=DEV)
        tmp_m = torch.empty(batch, HQ, parts, 2, dtype=torch.float32, device=DEV)
        ours = bench(lambda: ops.paged_decode_attention(out, q, k_cache, v_cache, tables, ctx_lens, D ** -0.5, parts,
                                                        tmp_o, tmp_m))

        # Baseline 1: what a framework without a paged kernel does - gather the blocks, then SDPA.
        def gather_sdpa():
            k = k_cache[tables.long()].view(batch, -1, HK, D).transpose(1, 2)
            v = v_cache[tables.long()].view(batch, -1, HK, D).transpose(1, 2)
            return F.scaled_dot_product_attention(q[:, :, None], k, v, enable_gqa=True)

        # Baseline 2: SDPA on an already-contiguous cache (no paging at all) - a strong lower bound on cost.
        kc = k_cache[tables.long()].view(batch, -1, HK, D).transpose(1, 2).contiguous()
        vc = v_cache[tables.long()].view(batch, -1, HK, D).transpose(1, 2).contiguous()
        contiguous = bench(lambda: F.scaled_dot_product_attention(q[:, :, None], kc, vc, enable_gqa=True))
        gather = bench(gather_sdpa)
        gbytes = 2 * batch * ctx * HK * D * 2 / 1e9
        rows.append([batch, ctx, f"{ours * 1e3:.1f}", f"{gbytes / (ours / 1e3):.0f}",
                     f"{gather * 1e3:.1f}", f"{contiguous * 1e3:.1f}", f"{gather / ours:.1f}x",
                     f"{contiguous / ours:.2f}x"])
        del k_cache, v_cache, kc, vc
    return table(
        "Paged decode attention (Hq=16, Hk=8, D=128, block=16, bf16)",
        ["batch", "ctx", "ours (us)", "ours GB/s", "gather+SDPA (us)",
         "contiguous SDPA (us)", "vs gather", "vs contiguous"], rows)


def bench_prefill():
    rows = []
    for seqlen, batch in [(512, 8), (1024, 4), (2048, 2), (4096, 1), (8192, 1)]:
        total = seqlen * batch
        k_cache, v_cache, tables, ctx_lens = paged_setup(batch, seqlen)
        q = torch.randn(total, HQ, D, dtype=BF16, device=DEV)
        cu = torch.arange(0, total + 1, seqlen, dtype=torch.int32, device=DEV)
        ours = bench(lambda: ops.paged_prefill_attention(q, k_cache, v_cache, tables, cu, ctx_lens, seqlen,
                                                         D ** -0.5))
        qs = q.view(batch, seqlen, HQ, D).transpose(1, 2).contiguous()
        kc = k_cache[tables.long()].view(batch, -1, HK, D).transpose(1, 2).contiguous()
        vc = v_cache[tables.long()].view(batch, -1, HK, D).transpose(1, 2).contiguous()
        sdpa = bench(lambda: F.scaled_dot_product_attention(qs, kc, vc, is_causal=True, enable_gqa=True))
        flops = 4 * batch * HQ * D * seqlen * seqlen / 2  # causal: half the score matrix
        rows.append([batch, seqlen, f"{ours * 1e3:.0f}", f"{flops / (ours / 1e3) / 1e12:.1f}", f"{sdpa * 1e3:.0f}",
                     f"{flops / (sdpa / 1e3) / 1e12:.1f}", f"{sdpa / ours:.2f}x"])
        del k_cache, v_cache, kc, vc
    return table("Causal prefill attention (paged K/V vs PyTorch SDPA on contiguous K/V)",
                 ["batch", "seqlen", "ours (us)", "ours TFLOPS", "SDPA (us)", "SDPA TFLOPS", "speedup"], rows)


def bench_fused():
    rows = []
    T = 4096
    for name, hidden in [("rms_norm", 2048), ("fused_add_rms_norm", 2048)]:
        x = torch.randn(T, hidden, dtype=BF16, device=DEV)
        r = torch.randn_like(x)
        w = torch.randn(hidden, dtype=BF16, device=DEV)
        if name == "rms_norm":
            ours = bench(lambda: ops.rms_norm(x, w, 1e-6))
            eager = bench(lambda: ref.rms_norm(x, w, 1e-6))
            nbytes = 2 * T * hidden * 2
        else:
            ours = bench(lambda: ops.fused_add_rms_norm(x, r, w, 1e-6))
            eager = bench(lambda: ref.fused_add_rms_norm(x, r, w, 1e-6))
            nbytes = 4 * T * hidden * 2
        rows.append([f"{name} [{T}x{hidden}]", f"{ours * 1e3:.1f}", f"{nbytes / ours / 1e6:.0f}", f"{eager * 1e3:.1f}",
                     f"{eager / ours:.1f}x"])
    x = torch.randn(T, 2 * 6144, dtype=BF16, device=DEV)
    ours = bench(lambda: ops.silu_and_mul(x))
    eager = bench(lambda: ref.silu_and_mul(x))
    rows.append([f"silu_and_mul [{T}x2x6144]", f"{ours * 1e3:.1f}", f"{3 * T * 6144 * 2 / ours / 1e6:.0f}",
                 f"{eager * 1e3:.1f}", f"{eager / ours:.1f}x"])
    qkv = torch.randn(T, (HQ + 2 * HK) * D, dtype=BF16, device=DEV)
    pos = torch.randint(0, 4096, (T,), device=DEV)
    cs = ref.rope_cos_sin_cache(D, 4096, 1e6)
    qw, kw = torch.randn(D, dtype=BF16, device=DEV), torch.randn(D, dtype=BF16, device=DEV)
    ours = bench(lambda: ops.qk_norm_rope(qkv, pos, cs, qw, kw, HQ, HK, D, 1e-6))
    eager = bench(lambda: ref.qk_norm_rope(qkv, pos, cs, qw, kw, HQ, HK, D, 1e-6))
    rows.append([f"qk_norm_rope [{T} tok, 16q+8k heads]", f"{ours * 1e3:.1f}",
                 f"{2 * T * (HQ + HK) * D * 2 / ours / 1e6:.0f}", f"{eager * 1e3:.1f}", f"{eager / ours:.1f}x"])
    return table("Fused memory-bound kernels vs eager PyTorch",
                 ["kernel", "ours (us)", "ours GB/s", "eager (us)", "speedup"], rows)


if __name__ == "__main__":
    torch.manual_seed(0)
    name = torch.cuda.get_device_name()
    header = f"GPU: {name}, torch {torch.__version__}, CUDA {torch.version.cuda}\n\n"
    print(header)
    report = header + bench_decode() + "\n" + bench_prefill() + "\n" + bench_fused()
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "kernels.md").write_text(report)
