"""Paged KV cache allocator with hash-based automatic prefix caching.

Every full block is identified by a chained hash of (parent block hash, its tokens), so equal hashes imply equal
whole prefixes. Blocks with ref_count == 0 stay in an LRU free list and keep their contents, so a later request
with the same prefix can revive them until they are actually reused for something else.
"""

from collections import OrderedDict

from tinyserve.sequence import Sequence


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.ref_count = [0] * num_blocks
        self.block_hash: list[int | None] = [None] * num_blocks
        self.free: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))  # front = evict first
        self.hash_to_block: dict[int, int] = {}
        self.prefix_query_tokens = 0
        self.prefix_hit_tokens = 0

    # ---- internals -------------------------------------------------------------------------------------------
    def _blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def _ensure_hashes(self, seq: Sequence, n: int):
        bs = self.block_size
        while len(seq.block_hashes) < n:
            i = len(seq.block_hashes)
            parent = seq.block_hashes[i - 1] if i else None
            seq.block_hashes.append(hash((parent, tuple(seq.token_ids[i * bs:(i + 1) * bs]))))

    def _lookup_prefix(self, seq: Sequence) -> list[int]:
        if not self.enable_prefix_caching:
            return []
        # Only blocks that end before the last token can be reused: at least one token must be computed to
        # produce logits, and a shared block must never be written to.
        n = (seq.num_tokens - 1) // self.block_size
        self._ensure_hashes(seq, n)
        hits = []
        for i in range(n):
            b = self.hash_to_block.get(seq.block_hashes[i])
            if b is None:
                break
            hits.append(b)
        return hits

    def _acquire(self, b: int):
        if self.ref_count[b] == 0:
            del self.free[b]
        self.ref_count[b] += 1

    def _pop_free(self) -> int:
        b, _ = self.free.popitem(last=False)
        h = self.block_hash[b]
        if h is not None and self.hash_to_block.get(h) == b:
            del self.hash_to_block[h]
        self.block_hash[b] = None
        self.ref_count[b] = 1
        return b

    # ---- public API ------------------------------------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self.free)

    @property
    def usage(self) -> float:
        return 1.0 - len(self.free) / self.num_blocks

    def can_allocate(self, seq: Sequence, reserve: int = 0) -> bool:
        hits = self._lookup_prefix(seq)
        revived = sum(1 for b in hits if self.ref_count[b] == 0)  # cached-but-free blocks also leave the free list
        need = self._blocks_for(seq.num_tokens) - len(hits) + revived
        return need + reserve <= len(self.free)

    def allocate(self, seq: Sequence):
        assert not seq.block_table, "sequence already has blocks"
        hits = self._lookup_prefix(seq)
        for b in hits:
            self._acquire(b)
        table = hits + [self._pop_free() for _ in range(self._blocks_for(seq.num_tokens) - len(hits))]
        seq.block_table = table
        seq.num_computed_tokens = len(hits) * self.block_size
        seq.num_cached_tokens = seq.num_computed_tokens
        seq.num_hashed_blocks = len(hits)
        self.prefix_query_tokens += seq.num_tokens
        self.prefix_hit_tokens += seq.num_computed_tokens

    def free_seq(self, seq: Sequence):
        # Release tail blocks first so they are evicted before the (more shareable) head of the prefix.
        for b in reversed(seq.block_table):
            self.ref_count[b] -= 1
            if self.ref_count[b] == 0:
                self.free[b] = None
                if self.block_hash[b] is None:  # nothing to reuse: evict these first
                    self.free.move_to_end(b, last=False)
        seq.block_table = []
        seq.num_computed_tokens = 0
        seq.num_hashed_blocks = 0

    def can_append(self, seq: Sequence) -> bool:
        return self._blocks_for(seq.num_tokens) <= len(seq.block_table) or len(self.free) > 0

    def append_slot(self, seq: Sequence):
        """Make sure the block table covers every token of the sequence (called before a decode step)."""
        while len(seq.block_table) < self._blocks_for(seq.num_tokens):
            seq.block_table.append(self._pop_free())

    def register_computed_blocks(self, seq: Sequence):
        """Publish newly completed full blocks to the prefix cache (prompt and generated tokens alike)."""
        if not self.enable_prefix_caching:
            return
        n_full = seq.num_computed_tokens // self.block_size
        if n_full <= seq.num_hashed_blocks:
            return
        self._ensure_hashes(seq, n_full)
        for i in range(seq.num_hashed_blocks, n_full):
            b, h = seq.block_table[i], seq.block_hashes[i]
            if self.block_hash[b] is None:
                self.block_hash[b] = h
                self.hash_to_block.setdefault(h, b)
        seq.num_hashed_blocks = n_full

    @property
    def prefix_hit_rate(self) -> float:
        return self.prefix_hit_tokens / max(1, self.prefix_query_tokens)
