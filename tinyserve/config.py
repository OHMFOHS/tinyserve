import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_id: list[int]
    use_qk_norm: bool  # Qwen3 normalizes q/k per head before RoPE; Llama-style models do not

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "ModelConfig":
        cfg = json.loads((Path(path) / "config.json").read_text())
        arch = cfg.get("architectures", ["?"])[0]
        if arch not in ("Qwen3ForCausalLM", "LlamaForCausalLM"):
            raise ValueError(f"unsupported architecture {arch}")
        if cfg.get("rope_scaling"):
            raise ValueError("rope_scaling is not supported")
        eos = cfg.get("eos_token_id", [])
        heads = cfg["num_attention_heads"]
        return cls(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=heads,
            num_key_value_heads=cfg.get("num_key_value_heads", heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // heads,
            vocab_size=cfg["vocab_size"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg.get("rope_theta", 10000.0),
            max_position_embeddings=cfg["max_position_embeddings"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            eos_token_id=eos if isinstance(eos, list) else [eos],
            use_qk_norm=arch == "Qwen3ForCausalLM",
        )


@dataclass
class EngineConfig:
    model: str
    max_model_len: int = 4096
    max_num_seqs: int = 128  # max sequences in a running batch
    max_num_batched_tokens: int = 8192  # token budget per prefill step (longer prompts are chunked)
    block_size: int = 16
    gpu_memory_utilization: float = 0.85  # fraction of total GPU memory the engine may use
    enable_prefix_caching: bool = True
    enforce_eager: bool = False  # disable CUDA graphs for decode
    enable_overlap_schedule: bool = True  # schedule step N+1 on the CPU while the GPU runs step N
    num_kv_blocks: int | None = None  # override the automatic KV cache sizing
    cuda_graph_batch_sizes: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.cuda_graph_batch_sizes:
            sizes = [1, 2, 4, 8] + list(range(16, self.max_num_seqs + 1, 16))
            self.cuda_graph_batch_sizes = sorted({s for s in sizes if s <= self.max_num_seqs} | {self.max_num_seqs})
