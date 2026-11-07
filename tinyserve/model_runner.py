"""Owns the model, the KV cache memory and CUDA graphs; turns a ScheduledBatch into sampled token ids."""

import logging

import torch

from tinyserve import ops
from tinyserve.config import EngineConfig, ModelConfig
from tinyserve.model import AttentionMetadata, Qwen3Model
from tinyserve.sampler import Sampler
from tinyserve.scheduler import ScheduledBatch

log = logging.getLogger("tinyserve")


def _to_gpu(data, dtype):
    return torch.tensor(data, dtype=dtype, pin_memory=True).cuda(non_blocking=True)


class ModelRunner:
    def __init__(self, cfg: EngineConfig, model_cfg: ModelConfig, model_path: str):
        self.cfg = cfg
        self.mcfg = model_cfg
        self.bs = cfg.block_size
        self.max_blocks_per_seq = (cfg.max_model_len + self.bs - 1) // self.bs
        torch.cuda.set_device(0)
        self.model = Qwen3Model(model_cfg)
        self.model.load_weights(model_path)
        self.sampler = Sampler()
        self.num_kv_blocks = self._allocate_kv_cache()
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        if not cfg.enforce_eager:
            self._capture_cuda_graphs()

    # ---- memory planning ---------------------------------------------------------------------------------------
    def _bytes_per_block(self) -> int:
        m = self.mcfg
        return 2 * m.num_hidden_layers * self.bs * m.num_key_value_heads * m.head_dim * 2

    @torch.inference_mode()
    def _profile_peak_activation(self) -> int:
        """Run the largest prefill the scheduler can emit against a 1-block dummy cache; return activation bytes."""
        m = self.mcfg
        t = self.cfg.max_num_batched_tokens
        dummy = torch.zeros(m.num_hidden_layers, 2, 1, self.bs, m.num_key_value_heads, m.head_dim,
                            dtype=torch.bfloat16, device="cuda")
        self.model.bind_kv_cache(dummy)
        meta = AttentionMetadata(
            is_prefill=True,
            slot_mapping=torch.full((t,), -1, dtype=torch.long, device="cuda"),
            block_tables=torch.zeros(1, (t + self.bs - 1) // self.bs, dtype=torch.int32, device="cuda"),
            context_lens=torch.tensor([t], dtype=torch.int32, device="cuda"),
            cu_seqlens_q=torch.tensor([0, t], dtype=torch.int32, device="cuda"),
            max_q_len=t,
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        h = self.model(torch.zeros(t, dtype=torch.long, device="cuda"), torch.zeros(t, dtype=torch.long, device="cuda"),
                       meta)
        self.model.compute_logits(h[: self.cfg.max_num_seqs])
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        del dummy, h
        torch.cuda.empty_cache()
        return peak

    def _allocate_kv_cache(self) -> int:
        m = self.mcfg
        if self.cfg.num_kv_blocks:
            num_blocks = self.cfg.num_kv_blocks
        else:
            activation = self._profile_peak_activation()
            free, total = torch.cuda.mem_get_info()
            used_elsewhere_and_weights = total - free
            graph_headroom = 0 if self.cfg.enforce_eager else 256 << 20
            budget = total * self.cfg.gpu_memory_utilization - used_elsewhere_and_weights - activation - graph_headroom
            num_blocks = int(budget // self._bytes_per_block())
            log.info("memory: total %.2f GiB, in use %.2f GiB, activation peak %.2f GiB -> %d KV blocks (%.2f GiB)",
                     total / 2**30, used_elsewhere_and_weights / 2**30, activation / 2**30, num_blocks,
                     num_blocks * self._bytes_per_block() / 2**30)
        if num_blocks < self.max_blocks_per_seq:
            raise RuntimeError(f"only {num_blocks} KV blocks fit; raise gpu_memory_utilization or lower max_model_len")
        self.kv_cache = torch.empty(m.num_hidden_layers, 2, num_blocks, self.bs, m.num_key_value_heads, m.head_dim,
                                    dtype=torch.bfloat16, device="cuda")
        self.model.bind_kv_cache(self.kv_cache)
        return num_blocks

    # ---- input preparation -------------------------------------------------------------------------------------
    def _prepare_prefill(self, batch: ScheduledBatch):
        bs = self.bs
        input_ids, positions, slots, cu, ctx_lens, tables, sample_rows = [], [], [], [0], [], [], []
        for seq, n in zip(batch.seqs, batch.num_tokens):
            start, end = seq.num_computed_tokens, seq.num_computed_tokens + n
            input_ids.extend(seq.token_ids[start:end])
            positions.extend(range(start, end))
            table = seq.block_table
            p = start
            while p < end:  # one contiguous run of slots per block touched
                blk, off = divmod(p, bs)
                run = min(bs - off, end - p)
                base = table[blk] * bs + off
                slots.extend(range(base, base + run))
                p += run
            cu.append(cu[-1] + n)
            ctx_lens.append(end)
            tables.append(table)
            if end == seq.num_tokens:
                sample_rows.append(cu[-1] - 1)
        width = max(len(t) for t in tables)
        tables = [t + [0] * (width - len(t)) for t in tables]
        meta = AttentionMetadata(
            is_prefill=True,
            slot_mapping=_to_gpu(slots, torch.long),
            block_tables=_to_gpu(tables, torch.int32),
            context_lens=_to_gpu(ctx_lens, torch.int32),
            cu_seqlens_q=_to_gpu(cu, torch.int32),
            max_q_len=max(batch.num_tokens),
        )
        return _to_gpu(input_ids, torch.long), _to_gpu(positions, torch.long), meta, sample_rows

    def _decode_host_inputs(self, batch: ScheduledBatch, width: int):
        bs = self.bs
        input_ids, positions, slots, ctx_lens, tables = [], [], [], [], []
        for seq in batch.seqs:
            pos = seq.num_tokens - 1
            input_ids.append(seq.token_ids[-1])
            positions.append(pos)
            slots.append(seq.block_table[pos // bs] * bs + pos % bs)
            ctx_lens.append(seq.num_tokens)
            tables.append(seq.block_table + [0] * (width - len(seq.block_table)))
        return input_ids, positions, slots, ctx_lens, tables

    # ---- CUDA graphs for decode --------------------------------------------------------------------------------
    @torch.inference_mode()
    def _capture_cuda_graphs(self):
        m = self.mcfg
        max_bs = max(self.cfg.cuda_graph_batch_sizes)
        self.num_graph_partitions = ops.num_decode_partitions(self.cfg.max_model_len)
        P = self.num_graph_partitions
        self.g_input_ids = torch.zeros(max_bs, dtype=torch.long, device="cuda")
        self.g_positions = torch.zeros(max_bs, dtype=torch.long, device="cuda")
        self.g_slots = torch.full((max_bs,), -1, dtype=torch.long, device="cuda")
        self.g_ctx = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        self.g_tables = torch.zeros(max_bs, self.max_blocks_per_seq, dtype=torch.int32, device="cuda")
        self.g_tmp_out = torch.empty(max_bs, m.num_attention_heads, P, m.head_dim, dtype=torch.float32, device="cuda")
        self.g_tmp_ml = torch.empty(max_bs, m.num_attention_heads, P, 2, dtype=torch.float32, device="cuda")
        pool = None
        for b in sorted(self.cfg.cuda_graph_batch_sizes, reverse=True):
            meta = AttentionMetadata(
                is_prefill=False, slot_mapping=self.g_slots[:b], block_tables=self.g_tables[:b],
                context_lens=self.g_ctx[:b], num_partitions=P, decode_tmp_out=self.g_tmp_out[:b],
                decode_tmp_ml=self.g_tmp_ml[:b],
            )
            self.model(self.g_input_ids[:b], self.g_positions[:b], meta)  # warm up (cuBLAS handles, workspaces)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                out = self.model(self.g_input_ids[:b], self.g_positions[:b], meta)
            pool = g.pool()
            self.graphs[b] = (g, out)
        torch.cuda.synchronize()
        log.info("captured %d decode CUDA graphs (batch sizes %s)", len(self.graphs), self.cfg.cuda_graph_batch_sizes)

    def _run_decode(self, batch: ScheduledBatch) -> torch.Tensor:
        n = len(batch.seqs)
        graph_bs = next((b for b in sorted(self.graphs) if b >= n), None)
        if graph_bs is None:  # eager fallback
            width = max(len(s.block_table) for s in batch.seqs)
            ids, pos, slots, ctx, tables = self._decode_host_inputs(batch, width)
            meta = AttentionMetadata(
                is_prefill=False, slot_mapping=_to_gpu(slots, torch.long), block_tables=_to_gpu(tables, torch.int32),
                context_lens=_to_gpu(ctx, torch.int32), num_partitions=ops.num_decode_partitions(max(ctx)),
            )
            return self.model(_to_gpu(ids, torch.long), _to_gpu(pos, torch.long), meta)

        ids, pos, slots, ctx, tables = self._decode_host_inputs(batch, self.max_blocks_per_seq)
        pad = graph_bs - n
        self.g_input_ids[:graph_bs].copy_(_to_gpu(ids + [0] * pad, torch.long))
        self.g_positions[:graph_bs].copy_(_to_gpu(pos + [0] * pad, torch.long))
        self.g_slots[:graph_bs].copy_(_to_gpu(slots + [-1] * pad, torch.long))
        self.g_ctx[:graph_bs].copy_(_to_gpu(ctx + [0] * pad, torch.int32))
        self.g_tables[:n].copy_(_to_gpu(tables, torch.int32))
        g, out = self.graphs[graph_bs]
        g.replay()
        return out[:n]

    # ---- step --------------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def run(self, batch: ScheduledBatch) -> list[int]:
        """Returns one sampled token per sequence whose computation reached its last token this step."""
        if batch.is_prefill:
            ids, pos, meta, rows = self._prepare_prefill(batch)
            hidden = self.model(ids, pos, meta)
            if not rows:
                return []
            hidden = hidden[torch.tensor(rows, device="cuda")]
            seqs = [s for s, n in zip(batch.seqs, batch.num_tokens) if s.num_computed_tokens + n == s.num_tokens]
        else:
            hidden = self._run_decode(batch)
            seqs = batch.seqs
        logits = self.model.compute_logits(hidden)
        return self.sampler(logits, [s.params for s in seqs]).tolist()
