# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Token sampling utilities for MLX."""

import mlx.core as mx


def sample_token(
    logits_BV,
    key=None,
    temperature: float = 0.0,
    top_k: int | None = None,
):
    """Sample one token per batch element from logit scores.

    Args:
        logits_BV: (B, V) raw logits.
        key: Optional MLX random key for reproducibility.
        temperature: 0.0 = greedy (argmax).
        top_k: If set, restrict sampling to the top-k logits.

    Returns:
        indices: (B, 1) int32 -- sampled token ids.
        logits:  (B, 1) float32 -- logit score of the sampled token.
        probs:   (B, 1) float32 -- probability of the sampled token.
    """
    B, V = logits_BV.shape

    if temperature == 0.0:
        indices = mx.argmax(logits_BV, axis=-1, keepdims=True)
        probs = mx.softmax(logits_BV.astype(mx.float32), axis=-1)
        probs = mx.take_along_axis(probs, indices, axis=-1)
    else:
        scaled = logits_BV / temperature

        if top_k is not None:
            k = min(top_k, V)
            top_vals = mx.topk(scaled, k, axis=-1)
            threshold = top_vals[:, -1:]
            scaled = mx.where(scaled >= threshold, scaled, mx.finfo(scaled.dtype).min)

        probs = mx.softmax(scaled.astype(mx.float32), axis=-1)
        if key is not None:
            indices = mx.random.categorical(mx.log(probs), key=key)
        else:
            indices = mx.random.categorical(mx.log(probs))
        indices = indices[:, None]
        probs = mx.take_along_axis(probs, indices, axis=-1)

    logits_out = mx.take_along_axis(logits_BV, indices, axis=-1)
    return indices, logits_out.astype(mx.float32), probs
