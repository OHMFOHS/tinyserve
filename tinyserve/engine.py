import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from tinyserve.block_manager import BlockManager
from tinyserve.config import EngineConfig, ModelConfig
from tinyserve.model_runner import ModelRunner
from tinyserve.sampling_params import SamplingParams
from tinyserve.scheduler import Scheduler
from tinyserve.sequence import Sequence

log = logging.getLogger("tinyserve")


def resolve_model_path(model: str) -> str:
    if Path(model).is_dir():
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])


@dataclass
class EngineStats:
    steps: int = 0
    prefill_steps: int = 0  # steps that computed any prompt tokens (they may carry decodes too)
    mixed_steps: int = 0  # steps that computed prompt tokens and decode tokens together
    prompt_tokens: int = 0  # computed prompt tokens (excludes prefix-cache hits)
    generated_tokens: int = 0


class LLMEngine:
    def __init__(self, cfg: EngineConfig):
        from transformers import AutoTokenizer

        self.cfg = cfg
        path = resolve_model_path(cfg.model)
        self.model_cfg = ModelConfig.from_pretrained(path)
        cfg.max_model_len = min(cfg.max_model_len, self.model_cfg.max_position_embeddings)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        t0 = time.perf_counter()
        self.runner = ModelRunner(cfg, self.model_cfg, path)
        log.info("model ready in %.1fs", time.perf_counter() - t0)
        self.block_manager = BlockManager(self.runner.num_kv_blocks, cfg.block_size, cfg.enable_prefix_caching)
        self.scheduler = Scheduler(cfg, self.block_manager)
        self.eos_ids = set(self.model_cfg.eos_token_id)
        if self.tokenizer.eos_token_id is not None:
            self.eos_ids.add(self.tokenizer.eos_token_id)
        self.stats = EngineStats()
        self._inflight: _InflightStep | None = None

    def add_request(self, prompt: str | list[int], params: SamplingParams, request_id: str | None = None) -> Sequence:
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not ids:
            raise ValueError("empty prompt")
        if len(ids) >= self.cfg.max_model_len:
            raise ValueError(f"prompt has {len(ids)} tokens; max_model_len is {self.cfg.max_model_len}")
        seq = Sequence(ids, params, request_id)
        self.scheduler.add(seq)
        return seq

    def abort(self, request_id: str) -> bool:
        return self.scheduler.abort(request_id)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished() or self._inflight is not None

    def step(self) -> list[Sequence]:
        """Run one scheduler step; returns the sequences that received a new token (finished ones included).

        With overlap scheduling the step launched here is only resolved by the next call: this call schedules and
        launches step N+1, then waits for step N, whose GPU work ran while the CPU was busy with N+1. The sampled
        tokens of step N feed step N+1 on the device, so the GPU never idles waiting for the host."""
        batch = self.scheduler.schedule()
        launched = self._launch(batch) if batch is not None else None
        if not self.cfg.enable_overlap_schedule:
            done, self._inflight = launched, None
        else:
            done, self._inflight = self._inflight, launched
        if done is None:
            if launched is None and self.scheduler.waiting:
                raise RuntimeError("cannot schedule: a waiting request does not fit in the KV cache")
            return []
        return self._resolve(done)

    def _launch(self, batch) -> "_InflightStep":
        prev = self._inflight.tokens if self._inflight is not None else None
        tokens, sampled = self.runner.run(batch, prev)
        self.stats.steps += 1
        if batch.is_prefill:
            self.stats.prefill_steps += 1
            self.stats.mixed_steps += batch.num_decodes > 0
            self.stats.prompt_tokens += batch.num_prefill_tokens
        for seq, n in zip(batch.seqs, batch.num_tokens):
            seq.num_computed_tokens += n
        for i, seq in enumerate(sampled):  # the sampled token now exists, its value is still on the GPU
            seq.num_inflight += 1
            seq.inflight_idx = i
        host = event = None
        if tokens is not None:
            host = torch.empty(tokens.shape, dtype=tokens.dtype, pin_memory=True)
            host.copy_(tokens, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        return _InflightStep(batch.seqs, sampled, tokens, host, event)

    def _resolve(self, step: "_InflightStep") -> list[Sequence]:
        if step.event is not None:
            step.event.synchronize()
        now = time.perf_counter()
        updated = []
        values = step.host.tolist() if step.host is not None else []
        for seq, tok in zip(step.sampled, values):
            seq.num_inflight -= 1
            if seq.is_finished:  # finished (or aborted) while this step was in flight: drop the extra token
                continue
            seq.token_ids.append(tok)
            self.stats.generated_tokens += 1
            if seq.first_token_time is None:
                seq.first_token_time = now
            reason = self._finish_reason(seq, tok)
            if reason:
                seq.finish_time = now
                self.scheduler.finish(seq, reason)
            updated.append(seq)
        for seq in step.seqs:
            if not seq.is_finished:
                self.block_manager.register_computed_blocks(seq)
        return updated

    def _finish_reason(self, seq: Sequence, tok: int) -> str | None:
        p = seq.params
        if not p.ignore_eos and (tok in self.eos_ids or tok in p.stop_token_ids):
            return "stop"
        if seq.num_completion_tokens >= p.max_tokens or len(seq.token_ids) >= self.cfg.max_model_len:
            return "length"
        return None


@dataclass
class _InflightStep:
    seqs: list[Sequence]
    sampled: list[Sequence]  # sequences that get a token from this step, in the order of `tokens`
    tokens: torch.Tensor | None  # sampled ids on the GPU (input of the next step's in-flight tokens)
    host: torch.Tensor | None  # pinned copy, valid once `event` completes
    event: torch.cuda.Event | None


class LLM:
    """Offline batch inference: LLM("Qwen/Qwen3-0.6B").generate(prompts, SamplingParams(...))."""

    def __init__(self, model: str, **kwargs):
        self.engine = LLMEngine(EngineConfig(model=model, **kwargs))
        self.tokenizer = self.engine.tokenizer

    def generate(self, prompts: list[str] | list[list[int]], params: SamplingParams | list[SamplingParams],
                 use_tqdm: bool = True) -> list[dict]:
        if isinstance(params, SamplingParams):
            params = [params] * len(prompts)
        seqs = [self.engine.add_request(p, sp) for p, sp in zip(prompts, params)]
        bar = None
        if use_tqdm:
            from tqdm import tqdm

            bar = tqdm(total=len(seqs), desc="generate", dynamic_ncols=True)
        while self.engine.has_unfinished():
            for s in self.engine.step():
                if s.is_finished and bar:
                    bar.update(1)
        if bar:
            bar.close()
        return [
            {
                "text": self.tokenizer.decode(s.completion_token_ids, skip_special_tokens=True),
                "token_ids": s.completion_token_ids,
                "finish_reason": s.finish_reason,
                "num_cached_tokens": s.num_cached_tokens,
            }
            for s in seqs
        ]
