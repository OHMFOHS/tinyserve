# tinyserve

A from-scratch LLM inference engine with **hand-written CUDA kernels**: paged KV cache, continuous batching,
chunked prefill, automatic prefix caching, CUDA-graph decode and an OpenAI-compatible streaming server — in about
2.4k lines of CUDA/C++ and Python (plus tests and benchmarks). Runs Qwen3 (0.6B–4B) on a single consumer GPU.

No FlashAttention / FlashInfer / Triton dependency: every attention and fused kernel on the hot path is in
[`csrc/`](csrc/); only the dense GEMMs go to cuBLAS.

## Results (RTX 5070 Ti 16 GB, torch 2.14.1 + CUDA 13.0, vLLM 0.33.0, bf16)

**Offline throughput** — 256 random prompts, input 100–1024 / output 100–1024 tokens, greedy, `ignore_eos`
([`bench_throughput.py`](benchmarks/bench_throughput.py), raw: [`throughput.jsonl`](benchmarks/results/throughput.jsonl)):

| model | tinyserve | vLLM 0.33.0 | HF Transformers (static batch 64) |
|---|---|---|---|
| Qwen3-0.6B | **4829** out tok/s (29.5 s) | 4704 out tok/s (30.3 s) | 34.6 out tok/s (4117 s) |
| Qwen3-1.7B | **3367** out tok/s (42.3 s) | 3433 out tok/s (41.5 s) | – |

**Prefix caching** — same 2000-token system prompt on all 256 requests, + 100–512 unique tokens, 64–128 output
tokens, Qwen3-0.6B ([`prefix.jsonl`](benchmarks/results/prefix.jsonl)):

| | wall time | total tok/s |
|---|---|---|
| tinyserve, prefix cache off | 21.6 s | 28.5k |
| tinyserve, prefix cache on | **6.3 s (3.5x)** | **98.4k** |
| vLLM 0.33.0 (prefix cache on) | 8.1 s | 75.9k |

**Online serving** — Poisson arrivals, 200 requests, streaming `/v1/completions`, Qwen3-0.6B
([`bench_serving.py`](benchmarks/bench_serving.py), raw: [`serving.jsonl`](benchmarks/results/serving.jsonl)):

| req/s | backend | achieved req/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | ITL p99 (ms) |
|---|---|---|---|---|---|
| 4 | tinyserve | 3.74 | 19.1 / 33.6 | 4.0 / 5.3 | 14.1 |
| 4 | vLLM | 3.75 | 20.9 / 32.1 | 3.6 / 4.8 | 10.5 |
| 8 | tinyserve | 7.14 | 21.2 / 47.9 | 5.7 / 6.5 | 20.5 |
| 8 | vLLM | 7.18 | 24.8 / 43.5 | 5.1 / 5.7 | 14.0 |
| 16 | tinyserve | 11.85 | 24.7 / 59.4 | 11.7 / 15.0 | 39.3 |
| 16 | vLLM | 12.74 | 34.6 / 66.0 | 7.9 / 10.0 | 22.4 |
| 32 | tinyserve | 13.42 | 33.2 / 2800 | 23.3 / 27.8 | 64.3 |
| 32 | vLLM | 15.79 | 69.8 / 1761 | 18.6 / 25.6 | 52.1 |

These two tables were measured with the original prefill-first scheduler, where every step was either all-prefill
or all-decode, so each arriving prompt stalled all running decodes for one step and TPOT fell behind vLLM from
~16 req/s. The mixed-step + overlap scheduler below fixes that.

**Mixed prefill/decode steps + overlap scheduling** — same-day A/B on the same GPU, Qwen3-0.6B. "baseline" is
the earlier prefill-first, synchronous engine; "mixed only" is the new scheduler with `--no-overlap-schedule`
(raw: [`serving_v2.jsonl`](benchmarks/results/serving_v2.jsonl),
[`throughput_v2.jsonl`](benchmarks/results/throughput_v2.jsonl)):

| req/s | engine | achieved req/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | ITL p99 (ms) |
|---|---|---|---|---|---|
| 8 | baseline | 7.10 | 23.0 / 46.6 | 6.1 / 7.5 | 21.7 |
| 8 | mixed only | 7.00 | 26.9 / 56.7 | 7.3 / 8.7 | 22.1 |
| 8 | **mixed + overlap** | 7.20 | 27.3 / 49.7 | **4.7 / 5.5** | **15.8** |
| 16 | baseline | 10.98 | 29.1 / 81.0 | 15.7 / 22.2 | 54.2 |
| 16 | mixed only | 9.70 | 56.7 / 151.1 | 23.6 / 28.3 | 64.2 |
| 16 | **mixed + overlap** | **12.78** | 37.3 / 76.1 | **7.8 / 9.6** | **24.1** |
| 32 | baseline | 11.47 | 64.2 / 4800 | 29.1 / 34.1 | 81.0 |
| 32 | mixed only | 11.37 | 73.2 / 4460 | 28.6 / 33.2 | 65.5 |
| 32 | **mixed + overlap** | **16.33** | 66.8 / **1211** | **17.2 / 22.5** | **44.8** |

Offline throughput (same workload as above): baseline 4205 out tok/s → mixed + overlap **4701 out tok/s (+12%)**;
mixed without overlap 4136. (Absolute numbers are ~10% lower than the table at the top because the GPU was also
driving the desktop during this run; only same-day rows are compared.)

The ablation is the interesting part: mixing alone is *slower*, because mixed steps run eagerly (no CUDA graph) and
for a 0.6B model the ~300 kernel launches per step cost more than the GPU work. Overlap scheduling hides that host
time behind the previous step, and only the combination wins — TPOT at 16 req/s halves and sustainable load rises
from ~11 to ~16 req/s.

**Kernels** ([`bench_kernels.py`](benchmarks/bench_kernels.py), full tables: [`kernels.md`](benchmarks/results/kernels.md)):

- Paged decode attention: up to **789 GB/s (88% of the 896 GB/s peak)**; 2.7–3.4x faster than gather + SDPA and
  1.0–1.23x vs SDPA on an already-contiguous cache (0.83x at batch 1 / 1k context, where launch overhead dominates).
- Paged causal prefill attention: 48–69 TFLOPS, **84–89% of PyTorch SDPA** (FlashAttention-2 backend, contiguous K/V).
- Fused kernels vs eager PyTorch: rms_norm 5.8x, fused_add_rms_norm 3.5x, silu_and_mul 1.8x, qk_norm_rope 14.7x
  (720–774 GB/s).


## Architecture

```
            HTTP (OpenAI /v1/completions, /v1/chat/completions, SSE)          /metrics
                                   │                                              ▲
                      ┌────────────▼─────────────┐                                │
 asyncio event loop   │ AsyncEngine              │  thread-safe inbox / per-request asyncio queues
                      └────────────┬─────────────┘
 engine thread        ┌────────────▼─────────────┐
                      │ Scheduler                │  mixed prefill+decode steps in one token budget,
                      │                          │  chunked prefill, recompute preemption,
                      │                          │  overlap scheduling (CPU plans N+1 while GPU runs N)
                      │ BlockManager             │  paged KV blocks, chained-hash prefix cache, LRU eviction
                      └────────────┬─────────────┘
                      ┌────────────▼─────────────┐
                      │ ModelRunner              │  KV memory planning, input packing, decode CUDA graphs,
                      │                          │  batched sampler (greedy / temperature / top-k / top-p)
                      └────────────┬─────────────┘
                      ┌────────────▼─────────────┐
                      │ Qwen3 model              │  cuBLAS GEMMs + custom kernels below
                      └──────────────────────────┘
```

### Kernels ([`csrc/`](csrc/))

| kernel | what it does | technique |
|---|---|---|
| `paged_prefill_attention` | causal varlen attention of new tokens against the paged cache | FlashAttention-2 style: `mma.sync.m16n8k16` bf16 tensor cores via inline PTX, Q fragments in registers, `ldmatrix` / `ldmatrix.trans` fragment loads, `cp.async` prefetch of the next K tile during softmax/P·V and of the next V tile during Q·Kᵀ, online softmax on accumulators, S-accumulator layout reused directly as the A operand of P·V, padded smem to avoid bank conflicts. Reading K/V through the block table makes one kernel serve plain prefill, prefix-cache hits and chunked prefill. |
| `paged_decode_attention` | one query token per sequence against its paged history | flash-decoding: split the context into 256-token partitions across CTAs + log-sum-exp merge kernel; GQA-aware (one CTA serves all query heads of a KV head, so K/V is read once per group); 8 lanes per token so a 128-d dot product is a 3-step shuffle; graph-safe fixed partition count |
| `fused_add_rms_norm` / `rms_norm` | residual add + RMSNorm in one pass | 128-bit vectorized loads, block reduction |
| `qk_norm_rope` | Qwen3 per-head q/k RMSNorm + NeoX RoPE, in place on the packed qkv | one warp per head; the rotate-half partner sits exactly 16 lanes away, so the rotation is one `shfl_xor` |
| `silu_and_mul` | gated-MLP activation | vectorized, fused |
| `reshape_and_cache` | scatter new K/V into cache slots | slot mapping with `-1` padding for graph batches |

All kernels are checked against eager PyTorch references in [`tests/test_kernels.py`](tests/test_kernels.py)
(GQA ratios 1/2/4/8, head dims 64/128, block sizes 16/32, contexts across partition boundaries, prefix offsets).

### Engine

- **Paged KV cache.** `[layers, 2, blocks, block_size, kv_heads, head_dim]`, sized at startup by profiling the
  peak activation of the largest prefill and giving the rest of the memory budget to KV blocks.
- **Prefix caching.** Each full block is keyed by `hash(parent_hash, tokens)`. Freed blocks keep their contents
  in an LRU free list and can be revived by a later request; unhashed tail blocks are evicted first. The block
  holding a prompt's last token is never shared, so a shared block is never written.
- **Mixed prefill/decode steps.** Each step fills one token budget: first one token for every decoding sequence,
  then in-flight prompt chunks, then newly admitted prompts. All rows go through the same GEMMs; inside attention
  the decode rows use the flash-decoding kernel and the prompt rows the varlen prefill kernel, so an arriving
  prompt no longer stalls running decodes for a whole step. Prompts longer than the remaining budget are chunked
  (the prefill kernel attends to the already-cached chunks). When blocks run out, the newest sequences are
  preempted and recomputed later (usually cheap thanks to the prefix cache). Pure-decode steps replay CUDA graphs.
- **Overlap scheduling.** The engine launches step N+1 before it waits for step N. Lengths, slots and block
  allocation only depend on token *counts*, so step N+1 can be planned while step N's sampled tokens are still on
  the GPU; their values are gathered into step N+1's inputs on the device. The host only learns them afterwards
  (one pinned D2H copy + event per step), so stop-token checks lag by one step: a sequence that stopped at step
  N has its step-N+1 token dropped. Sequences that will hit `max_tokens` are not scheduled for that extra step.
  All host-to-device copies on the hot path are pinned and non-blocking, so the CPU never waits for the GPU to
  drain. `--no-overlap-schedule` turns it off for ablation.
- **CUDA graphs.** Decode steps for padded batch sizes {1, 2, 4, 8, 16, 32, …} are captured once; a step is then a
  few H2D copies plus one `graph.replay()`.

## Quick start

```bash
pip install torch transformers safetensors fastapi uvicorn httpx tqdm pytest   # needs nvcc for the JIT build
python -m pytest tests -q                       # kernels, scheduler, end-to-end vs HF
python -m tinyserve.server --model Qwen/Qwen3-0.6B --port 8000
curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is PagedAttention?"}],"stream":true,"max_tokens":200}'
```

```python
from tinyserve import LLM, SamplingParams
llm = LLM("Qwen/Qwen3-1.7B")
print(llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=32))[0]["text"])
```

Benchmarks: `benchmarks/bench_kernels.py`, `benchmarks/bench_throughput.py --backend {tinyserve,vllm,hf}`,
`benchmarks/bench_serving.py --base-url ...`.

## Limitations / roadmap

- Prefill attention reaches ~85% of PyTorch SDPA (FlashAttention-2 backend); the remaining gap is mostly the
  64×64 tile with 4 warps and the per-tile block-table lookup. Larger tiles / more stages would close it.
- Mixed steps run eagerly (only pure-decode steps replay CUDA graphs); a piecewise graph around attention would
  remove that launch overhead for small mixed batches.
- Single GPU, bf16 only. Next: speculative decoding, FP8 KV cache, weight-only int8/int4 GEMV for decode,
  tensor parallelism.
