import torch

from tinyserve.sampling_params import SamplingParams


def _h2d(data, device) -> torch.Tensor:
    """Small host-to-device copy that does not wait for the GPU to drain (pageable copies would)."""
    return torch.tensor(data, pin_memory=True).to(device, non_blocking=True)


class Sampler:
    """Batched greedy / temperature / top-k / top-p sampling with per-request parameters."""

    @torch.inference_mode()
    def __call__(self, logits: torch.Tensor, params: list[SamplingParams]) -> torch.Tensor:
        greedy_ids = logits.argmax(dim=-1)
        if all(p.temperature == 0 for p in params):
            return greedy_ids
        dev = logits.device
        temps = _h2d([p.temperature if p.temperature > 0 else 1.0 for p in params], dev)
        logits = logits / temps[:, None]
        if any(p.top_k > 0 or p.top_p < 1.0 for p in params):
            vocab = logits.shape[-1]
            top_k = _h2d([p.top_k if p.top_k > 0 else vocab for p in params], dev)
            top_p = _h2d([p.top_p for p in params], dev)
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            rank = torch.arange(vocab, device=dev)[None, :]
            sorted_logits = sorted_logits.masked_fill(rank >= top_k[:, None], float("-inf"))
            probs = sorted_logits.softmax(dim=-1)
            # drop a token once the mass of strictly-better tokens already exceeds top_p (always keeps the best)
            drop = probs.cumsum(dim=-1) - probs > top_p[:, None]
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)
        probs = logits.softmax(dim=-1)
        # Exponential-race sampling: argmax(p / Exp(1)) ~ Categorical(p), without a host sync.
        sampled = (probs / torch.empty_like(probs).exponential_()).argmax(dim=-1)
        is_greedy = _h2d([p.temperature == 0 for p in params], dev)
        return torch.where(is_greedy, greedy_ids, sampled)
