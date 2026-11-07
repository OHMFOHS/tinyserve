"""Offline throughput: N requests with random prompts, generate a fixed number of tokens each (ignore EOS).

    python benchmarks/bench_throughput.py --backend tinyserve --model Qwen/Qwen3-0.6B
    python benchmarks/bench_throughput.py --backend vllm ...     (run inside a venv that has vllm)
    python benchmarks/bench_throughput.py --backend hf --num-prompts 64 ...

Workload is generated from a fixed seed so every backend sees identical token ids.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_workload(n, in_min, in_max, out_min, out_max, vocab, seed=0, shared_prefix=0):
    rng = random.Random(seed)
    prefix = [rng.randrange(100, vocab) for _ in range(shared_prefix)]
    work = []
    for _ in range(n):
        lin = rng.randint(in_min, in_max)
        lout = rng.randint(out_min, out_max)
        work.append((prefix + [rng.randrange(100, vocab) for _ in range(lin)], lout))
    return work


def run_tinyserve(args, work):
    from tinyserve import LLM, SamplingParams

    llm = LLM(args.model, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_num_batched_tokens, gpu_memory_utilization=args.gpu_util,
              enable_prefix_caching=not args.no_prefix_caching, enforce_eager=args.enforce_eager,
              enable_overlap_schedule=not args.no_overlap_schedule)
    params = [SamplingParams(temperature=0, max_tokens=o, ignore_eos=True) for _, o in work]
    llm.generate([p for p, _ in work[:4]], params[:4], use_tqdm=False)  # warm-up
    t0 = time.perf_counter()
    outs = llm.generate([p for p, _ in work], params)
    dt = time.perf_counter() - t0
    extra = {"preemptions": llm.engine.scheduler.num_preemptions, "overlap": not args.no_overlap_schedule,
             "kv_blocks": llm.engine.block_manager.num_blocks}
    return dt, sum(len(o["token_ids"]) for o in outs), extra


def run_vllm(args, work):
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(args.model, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_num_batched_tokens, gpu_memory_utilization=args.gpu_util,
              enable_prefix_caching=not args.no_prefix_caching, enforce_eager=args.enforce_eager)
    prompts = [TokensPrompt(prompt_token_ids=p) for p, _ in work]
    params = [SamplingParams(temperature=0, max_tokens=o, ignore_eos=True) for _, o in work]
    llm.generate(prompts[:4], params[:4], use_tqdm=False)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, params)
    dt = time.perf_counter() - t0
    return dt, sum(len(o.outputs[0].token_ids) for o in outs), {}


def run_hf(args, work):
    """Static batching with HF generate (left padding); the standard 'no serving engine' baseline."""
    import torch
    from transformers import AutoModelForCausalLM

    from tinyserve.engine import resolve_model_path

    path = resolve_model_path(args.model)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
    pad = 0
    total = 0
    t0 = time.perf_counter()
    for i in range(0, len(work), args.hf_batch):
        chunk = work[i: i + args.hf_batch]
        width = max(len(p) for p, _ in chunk)
        ids = torch.tensor([[pad] * (width - len(p)) + p for p, _ in chunk], device="cuda")
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p, _ in chunk], device="cuda")
        new = max(o for _, o in chunk)
        with torch.no_grad():
            model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=new, min_new_tokens=new,
                           do_sample=False, pad_token_id=pad)
        total += sum(o for _, o in chunk)  # tokens the requests asked for (extra padding work is HF's cost)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, total, {"batch": args.hf_batch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tinyserve", "vllm", "hf"], default="tinyserve")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--input-len", type=int, nargs=2, default=[100, 1024], metavar=("MIN", "MAX"))
    ap.add_argument("--output-len", type=int, nargs=2, default=[100, 1024], metavar=("MIN", "MAX"))
    ap.add_argument("--shared-prefix", type=int, default=0, help="tokens of common system prompt per request")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--no-overlap-schedule", action="store_true")
    ap.add_argument("--hf-batch", type=int, default=32)
    ap.add_argument("--out", default=None, help="append a JSON line with the result to this file")
    a = ap.parse_args()

    work = make_workload(a.num_prompts, *a.input_len, *a.output_len, vocab=150000, shared_prefix=a.shared_prefix)
    in_tokens = sum(len(p) for p, _ in work)
    dt, out_tokens, extra = {"tinyserve": run_tinyserve, "vllm": run_vllm, "hf": run_hf}[a.backend](a, work)
    res = {
        "backend": a.backend, "model": a.model, "num_prompts": a.num_prompts, "input_len": a.input_len,
        "output_len": a.output_len, "shared_prefix": a.shared_prefix, "seconds": round(dt, 2),
        "output_tok_s": round(out_tokens / dt, 1), "total_tok_s": round((in_tokens + out_tokens) / dt, 1),
        "req_s": round(a.num_prompts / dt, 2), "prefix_caching": not a.no_prefix_caching,
        "cuda_graphs": not a.enforce_eager, **extra,
    }
    print(json.dumps(res))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
