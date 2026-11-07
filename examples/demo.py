"""python examples/demo.py [model]  -- offline generation with the chat template."""

import logging
import sys
import time

from tinyserve import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
llm = LLM(model)
questions = ["What is PagedAttention? Answer in two sentences.", "Write a haiku about GPUs."]
prompts = [llm.tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for q in questions]
t0 = time.perf_counter()
outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128), use_tqdm=False)
for q, o in zip(questions, outs):
    print(f"\n>>> {q}\n{o['text'].strip()}")
print(f"\n{sum(len(o['token_ids']) for o in outs)} tokens in {time.perf_counter() - t0:.2f}s")
