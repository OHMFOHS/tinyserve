import itertools
import time
from enum import Enum, auto

from tinyserve.sampling_params import SamplingParams

_ids = itertools.count()


class SeqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """One request. `token_ids` holds prompt + generated tokens whose values are known on the host; KV for the first
    `num_computed_tokens` positions is in the paged cache at the blocks listed in `block_table`.

    With overlap scheduling a sampled token can still be on the GPU when the next step is scheduled: it already
    counts towards `num_tokens` (so lengths, slots and block allocation stay exact), and `num_inflight` says how many
    such trailing tokens are not yet in `token_ids`. `inflight_idx` is the row of the most recent one in its step's
    output tensor, which the model runner gathers from on the device."""

    def __init__(self, prompt_token_ids: list[int], params: SamplingParams, request_id: str | None = None):
        self.seq_id = next(_ids)
        self.request_id = request_id or str(self.seq_id)
        self.token_ids = list(prompt_token_ids)
        self.num_prompt_tokens = len(prompt_token_ids)
        self.params = params
        self.status = SeqStatus.WAITING
        self.block_table: list[int] = []
        self.num_computed_tokens = 0
        self.num_cached_tokens = 0  # prompt tokens served from the prefix cache
        self.num_hashed_blocks = 0  # leading full blocks already registered in the prefix cache
        self.block_hashes: list[int] = []
        self.num_inflight = 0
        self.inflight_idx = -1
        self.finish_reason: str | None = None
        self.arrival_time = time.perf_counter()
        self.first_token_time: float | None = None
        self.finish_time: float | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids) + self.num_inflight

    @property
    def num_completion_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_pending_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == SeqStatus.FINISHED

    def __repr__(self):
        return (f"Seq({self.seq_id}, tokens={self.num_tokens}, computed={self.num_computed_tokens}, "
                f"blocks={len(self.block_table)}, {self.status.name})")
