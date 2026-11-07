"""End-to-end check against HuggingFace transformers.

Exact token equality between two bf16 implementations is not a meaningful bar (tiny rounding differences flip
near-ties and then the sequences diverge), so we teacher-force HF on our greedy output and require that our token
is HF's argmax almost everywhere and inside HF's top-3 everywhere.
"""

import gc
import os

import pytest
import torch

MODEL = os.environ.get("TINYSERVE_TEST_MODEL", "Qwen/Qwen3-0.6B")
PROMPTS = [
    "The capital of France is",
    "Write a Python function that checks whether a number is prime.\n",
    "In one sentence, explain what a KV cache is in LLM inference:",
    "1, 1, 2, 3, 5, 8, 13,",
    "Once upon a time, in a land far away, there lived a dragon who",
    # long enough to span several KV blocks, so prefix caching and chunked prefill are actually exercised
    "Large language model inference has two phases. During prefill, the model processes the entire prompt in "
    "parallel, which is compute-bound and dominated by large matrix multiplications. During decode, the model "
    "generates one token at a time, which is memory-bandwidth-bound because every step must read all weights and "
    "the whole KV cache. Serving systems therefore batch many requests together. Summarize the trade-offs:",
]


@pytest.fixture(scope="module")
def hf():
    from transformers import AutoModelForCausalLM

    from tinyserve.engine import resolve_model_path

    path = resolve_model_path(MODEL)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
    yield model
    del model
    torch.cuda.empty_cache()


def check_against_hf(hf, tokenizer, prompts, outputs, min_argmax_rate=0.95):
    agree = total = 0
    for prompt, out in zip(prompts, outputs):
        p_ids = tokenizer.encode(prompt)
        ids = torch.tensor([p_ids + out["token_ids"]], device="cuda")
        with torch.no_grad():
            logits = hf(ids).logits[0, len(p_ids) - 1: -1].float()
        gen = torch.tensor(out["token_ids"], device="cuda")
        top3 = logits.topk(3, dim=-1).indices
        assert (top3 == gen[:, None]).any(dim=-1).all(), f"token outside HF top-3 for prompt {prompt!r}"
        agree += (logits.argmax(-1) == gen).sum().item()
        total += gen.numel()
    assert agree / total >= min_argmax_rate, f"argmax agreement {agree / total:.3f}"


@pytest.mark.parametrize(
    "engine_kwargs",
    [
        dict(),  # CUDA graphs + prefix caching
        dict(enforce_eager=True, enable_prefix_caching=False),
        dict(max_num_batched_tokens=16),  # forces chunked prefill on every prompt
    ],
    ids=["default", "eager-nocache", "chunked"],
)
def test_matches_hf_greedy(hf, engine_kwargs):
    from tinyserve import LLM, SamplingParams

    llm = LLM(MODEL, max_model_len=1024, max_num_seqs=8, gpu_memory_utilization=0.6, **engine_kwargs)
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=48, ignore_eos=True), use_tqdm=False)
    assert all(len(o["token_ids"]) == 48 for o in outs)
    check_against_hf(hf, llm.tokenizer, PROMPTS, outs)
    # Second round shares every prompt with the first: served from the prefix cache, same greedy result.
    if llm.engine.cfg.enable_prefix_caching:
        again = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=48, ignore_eos=True), use_tqdm=False)
        assert sum(o["num_cached_tokens"] for o in again) > 0
        check_against_hf(hf, llm.tokenizer, PROMPTS, again)
    del llm, outs
    gc.collect()
    torch.cuda.empty_cache()
