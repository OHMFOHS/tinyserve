"""Continuous-batching scheduler with mixed prefill/decode steps.

Every step fills one token budget (`max_num_batched_tokens`), in this order:
  1. decode: every running sequence that has finished its prompt advances by one token. If the cache runs out of
     blocks, the newest running sequences are preempted (blocks freed, recomputed later; the prefix cache usually
     makes that cheap).
  2. chunked prefills already in flight continue with whatever budget is left.
  3. waiting requests are admitted while KV blocks, the budget and `max_num_seqs` allow. Prompts longer than the
     remaining budget are split across steps (chunked prefill).
Decodes and prefills share one forward pass, so a newly arriving prompt no longer stalls the running decodes for a
whole step: their inter-token latency only grows by the cost of the prefill chunk that rides along. Decode tokens
come first in the batch; the model runner relies on that layout.
"""

from collections import deque
from dataclasses import dataclass

from tinyserve.block_manager import BlockManager
from tinyserve.config import EngineConfig
from tinyserve.sequence import Sequence, SeqStatus


@dataclass
class ScheduledBatch:
    seqs: list[Sequence]  # decodes first, then prefills
    num_tokens: list[int]  # tokens to compute for each sequence this step
    num_decodes: int  # the first `num_decodes` sequences compute exactly one token each

    @property
    def is_prefill(self) -> bool:
        """True if the step computes any prompt tokens (pure-decode steps can replay CUDA graphs)."""
        return self.num_decodes < len(self.seqs)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(self.num_tokens[self.num_decodes:])


class Scheduler:
    def __init__(self, cfg: EngineConfig, block_manager: BlockManager):
        self.cfg = cfg
        self.bm = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.watermark = max(1, int(0.01 * block_manager.num_blocks))
        self.num_preemptions = 0

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _wants_decode(self, seq: Sequence) -> bool:
        # With overlap scheduling the length limits are known before the last token's value is: don't spend a step
        # on a sequence that will be finished by the token already in flight.
        return (seq.num_pending_tokens == 1 and seq.num_tokens - seq.num_prompt_tokens < seq.params.max_tokens
                and seq.num_tokens < self.cfg.max_model_len)

    def schedule(self) -> ScheduledBatch | None:
        budget = self.cfg.max_num_batched_tokens
        decodes: list[Sequence] = []
        preempted = False

        # 1. decodes. Preemption pops from the tail of `running`, i.e. always at or after index i, so a victim is
        # never a sequence already scheduled in this loop.
        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            if not self._wants_decode(seq):
                i += 1
            elif self.bm.can_append(seq):
                self.bm.append_slot(seq)
                decodes.append(seq)
                budget -= 1
                i += 1
            else:
                self._preempt(self.running.pop())  # newest first; may be `seq` itself, which ends the loop
                preempted = True

        # 2. chunked prefills in flight.
        prefills, counts = [], []
        for seq in self.running:
            if budget == 0:
                break
            if seq.num_pending_tokens > 1:
                n = min(seq.num_pending_tokens, budget)
                prefills.append(seq)
                counts.append(n)
                budget -= n

        # 3. new requests. Not in a step that just preempted (the freed blocks would only be handed straight back),
        # and not for a sequence whose last sampled token is still on the GPU.
        while (not preempted and self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs
               and self.waiting[0].num_inflight == 0):
            seq = self.waiting[0]
            if not self.bm.can_allocate(seq, reserve=self.watermark):
                break
            self.waiting.popleft()
            self.bm.allocate(seq)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            n = min(seq.num_pending_tokens, budget)
            prefills.append(seq)
            counts.append(n)
            budget -= n

        if not decodes and not prefills:
            return None
        return ScheduledBatch(decodes + prefills, [1] * len(decodes) + counts, num_decodes=len(decodes))

    def _preempt(self, seq: Sequence):
        self.bm.free_seq(seq)
        seq.status = SeqStatus.WAITING
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def finish(self, seq: Sequence, reason: str):
        """Works for running sequences and for preempted ones whose in-flight token turned out to be the last."""
        if seq.status == SeqStatus.RUNNING:
            self.running.remove(seq)
        elif seq.status == SeqStatus.WAITING:
            self.waiting.remove(seq)
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason
        self.bm.free_seq(seq)

    def abort(self, request_id: str) -> bool:
        for seq in list(self.waiting) + self.running:
            if seq.request_id == request_id:
                self.finish(seq, "abort")
                return True
        return False
