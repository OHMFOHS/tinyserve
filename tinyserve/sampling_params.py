from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    temperature: float = 1.0  # 0 => greedy
    top_p: float = 1.0
    top_k: int = -1  # -1 => disabled
    max_tokens: int = 256
    ignore_eos: bool = False  # for benchmarks: always generate max_tokens
    stop_token_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
