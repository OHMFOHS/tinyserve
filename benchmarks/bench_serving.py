"""Online serving benchmark against any OpenAI-compatible /v1/completions endpoint (tinyserve or vLLM).

Requests arrive as a Poisson process at --request-rate req/s and stream their output; we record time-to-first-token
(TTFT), time-per-output-token (TPOT, excluding the first token), inter-token latency (ITL) and end-to-end latency.

    python benchmarks/bench_serving.py --base-url http://localhost:8000 --request-rate 8 --num-prompts 200
"""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_throughput import make_workload  # noqa: E402


async def one_request(client, url, model, prompt, max_tokens, sem_start):
    await sem_start
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
               "ignore_eos": True}
    t0 = time.perf_counter()
    stamps = []
    async with client.stream("POST", url, json=payload) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("choices") and chunk["choices"][0].get("text"):
                stamps.append(time.perf_counter())
    end = time.perf_counter()
    if not stamps:
        return None
    ttft = stamps[0] - t0
    itl = [b - a for a, b in zip(stamps, stamps[1:])]
    tpot = (stamps[-1] - stamps[0]) / max(1, max_tokens - 1)
    return {"ttft": ttft, "tpot": tpot, "itl": itl, "e2e": end - t0, "out": max_tokens}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


async def main_async(a):
    work = make_workload(a.num_prompts, *a.input_len, *a.output_len, vocab=150000, seed=a.seed,
                         shared_prefix=a.shared_prefix)
    url = a.base_url.rstrip("/") + "/v1/completions"
    rng = random.Random(a.seed)
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        # warm-up
        await one_request(client, url, a.model, work[0][0], 8, asyncio.sleep(0))
        tasks = []
        t_start = time.perf_counter()
        delay = 0.0
        for prompt, out_len in work:
            tasks.append(asyncio.create_task(one_request(client, url, a.model, prompt, out_len, asyncio.sleep(delay))))
            delay += rng.expovariate(a.request_rate) if a.request_rate != float("inf") else 0.0
        results = [r for r in await asyncio.gather(*tasks) if r]
        wall = time.perf_counter() - t_start
    out_tokens = sum(r["out"] for r in results)
    itl = [x for r in results for x in r["itl"]]
    ms = lambda xs, p: round(1000 * pct(xs, p), 1)  # noqa: E731
    res = {
        "backend": a.label, "request_rate": a.request_rate, "completed": len(results), "duration_s": round(wall, 1),
        "req_s": round(len(results) / wall, 2), "output_tok_s": round(out_tokens / wall, 1),
        "ttft_p50_ms": ms([r["ttft"] for r in results], 50), "ttft_p99_ms": ms([r["ttft"] for r in results], 99),
        "tpot_p50_ms": ms([r["tpot"] for r in results], 50), "tpot_p99_ms": ms([r["tpot"] for r in results], 99),
        "itl_p50_ms": ms(itl, 50), "itl_p99_ms": ms(itl, 99),
        "e2e_p50_s": round(pct([r["e2e"] for r in results], 50), 2),
    }
    print(json.dumps(res))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--label", default="tinyserve")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--request-rate", type=float, default=8.0)
    ap.add_argument("--input-len", type=int, nargs=2, default=[100, 1024])
    ap.add_argument("--output-len", type=int, nargs=2, default=[64, 512])
    ap.add_argument("--shared-prefix", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
