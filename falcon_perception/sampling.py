# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Token sampling utilities shared by batch and paged inference engines."""

import torch
import torch.nn.functional as F
from torch import Tensor


def sample_token(
    logits_BV: Tensor,
    rng: torch.Generator | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample one token per batch element from logit scores.

    Args:
        logits_BV: (B, V) raw logits.
        rng: Optional CUDA generator for reproducibility.
        temperature: 0.0 = greedy (argmax).
        top_k: If set, restrict sampling to the top-k logits.

    Returns:
        indices: (B, 1) int64 — sampled token ids.
        logits:  (B, 1) float32 — logit score of the sampled token.
        probs:   (B, 1) float32 — probability of the sampled token.
    """
    B, V = logits_BV.shape

    if temperature == 0.0:
        indices = logits_BV.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits_BV.float(), dim=-1)
        probs = torch.gather(probs, dim=-1, index=indices)
    else:
        scaled = logits_BV / temperature

        if top_k is not None:
            k = min(top_k, V)
            top_vals, top_idx = scaled.topk(k, dim=-1)
            probs = F.softmax(top_vals, dim=-1)
            sample_in_k = torch.multinomial(probs, num_samples=1, generator=rng)
            indices = torch.gather(top_idx, dim=-1, index=sample_in_k)
            probs = torch.gather(probs, dim=-1, index=sample_in_k)
        else:
            probs = F.softmax(scaled, dim=-1)
            indices = torch.multinomial(probs, num_samples=1, generator=rng)
            probs = torch.gather(probs, dim=-1, index=indices)

    logits = torch.gather(logits_BV, dim=-1, index=indices)
    return indices, logits.float(), probs
