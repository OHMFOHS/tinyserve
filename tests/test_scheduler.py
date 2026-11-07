"""CPU-only tests for the block manager (prefix caching, eviction) and the scheduler (chunking, preemption)."""

from tinyserve.block_manager import BlockManager
from tinyserve.config import EngineConfig
from tinyserve.sampling_params import SamplingParams
from tinyserve.scheduler import Scheduler
from tinyserve.sequence import Sequence

BS = 4


def seq(tokens, max_tokens=8):
    return Sequence(list(tokens), SamplingParams(max_tokens=max_tokens))


def simulate_step(sched, batch, bm, next_token=7):
    """What the engine does after the model runs."""
    for s, n in zip(batch.seqs, batch.num_tokens):
        s.num_computed_tokens += n
        bm.register_computed_blocks(s)
        if s.num_computed_tokens == s.num_tokens:
            s.token_ids.append(next_token)


def test_prefix_cache_hit_and_sharing():
    bm = BlockManager(num_blocks=16, block_size=BS)
    a = seq(range(10))  # blocks: [0-3] [4-7] [8,9]
    bm.allocate(a)
    assert a.num_computed_tokens == 0 and len(a.block_table) == 3
    a.num_computed_tokens = 10
    bm.register_computed_blocks(a)
    assert a.num_hashed_blocks == 2

    b = seq(list(range(8)) + [99, 98, 97])  # same first two blocks
    bm.allocate(b)
    assert b.block_table[:2] == a.block_table[:2]
    assert b.num_computed_tokens == 8
    assert bm.ref_count[a.block_table[0]] == 2


def test_last_token_never_served_from_cache():
    bm = BlockManager(num_blocks=16, block_size=BS)
    a = seq(range(8))
    bm.allocate(a)
    a.num_computed_tokens = 8
    bm.register_computed_blocks(a)
    b = seq(range(8))  # identical prompt, exactly 2 full blocks
    bm.allocate(b)
    assert b.num_computed_tokens == 4  # the block holding the last token is recomputed
    assert b.block_table[1] != a.block_table[1]


def test_freed_blocks_stay_cached_until_evicted():
    bm = BlockManager(num_blocks=4, block_size=BS)
    a = seq(range(9))
    bm.allocate(a)
    a.num_computed_tokens = 9
    bm.register_computed_blocks(a)
    first = a.block_table[0]
    bm.free_seq(a)
    assert bm.num_free_blocks == 4
    b = seq(list(range(8)) + [5])
    assert bm.can_allocate(b)
    bm.allocate(b)
    assert b.block_table[0] == first and b.num_computed_tokens == 8
    # Unhashed tail block of `a` must have been evicted before the cached head blocks.
    c = seq(range(100, 104))
    bm.allocate(c)
    assert first not in c.block_table


def test_chunked_prefill_respects_token_budget():
    cfg = EngineConfig(model="x", max_num_batched_tokens=6, max_num_seqs=4)
    bm = BlockManager(64, BS)
    sched = Scheduler(cfg, bm)
    s = seq(range(15))
    sched.add(s)
    chunks = []
    while s.num_completion_tokens == 0:
        batch = sched.schedule()
        assert batch.is_prefill
        chunks.append(batch.num_tokens[0])
        simulate_step(sched, batch, bm)
    assert chunks == [6, 6, 3]
    batch = sched.schedule()
    assert not batch.is_prefill and batch.num_tokens == [1]


def test_preemption_when_cache_is_full():
    cfg = EngineConfig(model="x", max_num_batched_tokens=64, max_num_seqs=8)
    bm = BlockManager(6, BS)
    sched = Scheduler(cfg, bm)
    sched.watermark = 0
    a, b = seq(range(8)), seq(range(50, 58))
    sched.add(a)
    sched.add(b)
    batch = sched.schedule()
    assert batch.is_prefill and len(batch.seqs) == 2  # 2 + 2 blocks used, 2 free
    simulate_step(sched, batch, bm)  # each now has 9 tokens -> needs a 3rd block
    batch = sched.schedule()
    assert [x.seq_id for x in batch.seqs] == [a.seq_id, b.seq_id]
    simulate_step(sched, batch, bm)
    # grow until a 4th block is needed by both: only one can continue
    for _ in range(3):
        batch = sched.schedule()
        simulate_step(sched, batch, bm)
    batch = sched.schedule()
    assert sched.num_preemptions >= 1
    assert b in sched.waiting and a in sched.running
