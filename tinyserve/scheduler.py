"""Continuous-batching scheduler.

Each step is either a prefill step or a decode step:
  * prefill: continue any partially-prefilled sequences, then admit waiting requests while KV blocks and the token
    budget allow. Prompts longer than the budget are split across steps (chunked prefill).
  * decode: every running sequence advances by one token. If the cache runs out of blocks, the newest sequences
    are preempted (their blocks freed, recomputed later; the prefix cache usually makes that cheap).
Prefill is prioritized, which favors time-to-first-token and keeps the decode batch full.
"""

from collections import deque
from dataclasses import dataclass

from tinyserve.block_manager import BlockManager
from tinyserve.config import EngineConfig
from tinyserve.sequence import Sequence, SeqStatus


@dataclass
class ScheduledBatch:
    seqs: list[Sequence]
    num_tokens: list[int]  # tokens to compute for each sequence this step
    is_prefill: bool


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

    def schedule(self) -> ScheduledBatch | None:
        budget = self.cfg.max_num_batched_tokens
        seqs, counts = [], []

        for seq in self.running:  # chunked prefills in flight
            if budget == 0:
                break
            if seq.num_pending_tokens > 1:
                n = min(seq.num_pending_tokens, budget)
                seqs.append(seq)
                counts.append(n)
                budget -= n

        while self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
            seq = self.waiting[0]
            if not self.bm.can_allocate(seq, reserve=self.watermark):
                break
            self.waiting.popleft()
            self.bm.allocate(seq)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            n = min(seq.num_pending_tokens, budget)
            seqs.append(seq)
            counts.append(n)
            budget -= n

        if seqs:
            return ScheduledBatch(seqs, counts, is_prefill=True)

        scheduled = []
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            if self.bm.can_append(seq):
                self.bm.append_slot(seq)
                scheduled.append(seq)
                i += 1
            else:
                self._preempt(self.running.pop())  # newest first; may be `seq` itself, which ends the loop
        if not scheduled:
            return None
        return ScheduledBatch(scheduled, [1] * len(scheduled), is_prefill=False)

    def _preempt(self, seq: Sequence):
        self.bm.free_seq(seq)
        seq.status = SeqStatus.WAITING
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def finish(self, seq: Sequence, reason: str):
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason
        self.bm.free_seq(seq)
        self.running.remove(seq)

    def abort(self, request_id: str) -> bool:
        for seq in list(self.waiting):
            if seq.request_id == request_id:
                self.waiting.remove(seq)
                seq.status = SeqStatus.FINISHED
                seq.finish_reason = "abort"
                return True
        for seq in self.running:
            if seq.request_id == request_id:
                self.finish(seq, "abort")
                return True
        return False
