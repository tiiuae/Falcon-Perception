# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Dense boolean attention masks for MLX batch inference.

Replaces PyTorch's FlexAttention ``BlockMask`` with a materialised
``(B, 1, S, S)`` boolean mask passed to ``mx.fast.scaled_dot_product_attention``.
"""

import mlx.core as mx


def create_batch_attention_mask(
    input_batch,
    *,
    pad_token_id: int,
    eos_token_id: int,
    soi_token_id: int,
    eoi_token_id: int,
    max_len: int | None = None,
):
    """Build a combined dense boolean mask for the batch engine.

    Composes:
      - causal mask (q_idx >= kv_idx)
      - document mask (same document, separated by eos)
      - non-left-pad mask (kv is not a left-pad token)
      - image-prefix mask (bidirectional within same image span)

    Final mask = image_prefix OR (causal AND document AND non_left_pad).

    Args:
        input_batch: (B, S) int32 token ids.

    Returns:
        Boolean mask of shape ``(B, 1, S, S)`` where ``True`` = attend.
    """
    B, S = input_batch.shape
    S_mask = max_len or S

    if S_mask > S:
        pad_cols = mx.full((B, S_mask - S), pad_token_id, dtype=input_batch.dtype)
        padded = mx.concatenate([input_batch, pad_cols], axis=1)
    else:
        padded = input_batch

    q_idx = mx.arange(S_mask).reshape(1, S_mask, 1)   # (1, S, 1)
    kv_idx = mx.arange(S_mask).reshape(1, 1, S_mask)  # (1, 1, S)

    # 1. Causal: q_idx >= kv_idx
    causal = q_idx >= kv_idx  # (1, S, S)

    # 2. Document: same segment between EOS boundaries
    eos_mask = padded == eos_token_id  # (B, S)
    eos_mask = eos_mask.at[:, -1].add(~eos_mask[:, -1])  # force last position True
    cumulative = mx.cumsum(mx.where(eos_mask, 1, 0).astype(mx.int32), axis=1)
    seg_indices = mx.zeros_like(cumulative)
    seg_indices = seg_indices.at[:, 1:].add(cumulative[:, :-1])
    seg_q = seg_indices[:, :, None]  # (B, S, 1)
    seg_kv = seg_indices[:, None, :]  # (B, 1, S)
    document = seg_q == seg_kv  # (B, S, S)

    # 3. Non-left-pad: kv position is not a left-pad token
    non_pad_cum = mx.cumsum((padded != pad_token_id).astype(mx.int32), axis=1)
    non_pad_kv = non_pad_cum[:, None, :] > 0  # (B, 1, S)

    # 4. Image prefix: bidirectional within same image span
    soi_mask = (padded == soi_token_id).astype(mx.int32)
    eoi_mask = (padded == eoi_token_id).astype(mx.int32)
    acc_soi = mx.cumsum(soi_mask, axis=1)
    acc_eoi = mx.cumsum(eoi_mask, axis=1)
    img_mask = (acc_soi - acc_eoi) > 0  # tokens between SOI and EOI
    img_indices = acc_soi * img_mask.astype(mx.int32)

    img_q = img_mask[:, :, None]       # (B, S, 1)
    img_kv = img_mask[:, None, :]      # (B, 1, S)
    idx_q = img_indices[:, :, None]    # (B, S, 1)
    idx_kv = img_indices[:, None, :]   # (B, 1, S)
    image_prefix = img_q & img_kv & (idx_q == idx_kv)  # (B, S, S)

    # Compose: image_prefix OR (causal AND document AND non_left_pad)
    block_causal = causal & document & non_pad_kv
    mask = image_prefix | block_causal  # (B, S, S)

    return mask[:, None, :, :]  # (B, 1, S, S)


def create_decode_mask(kv_len: int, pad_positions=None):
    """Simple causal decode mask: attend to all KV positions.

    For decode (S_q == 1), the query attends to all past KV positions.
    Optionally masks out left-pad positions.

    Returns:
        Boolean mask of shape ``(B, 1, 1, kv_len)`` or ``(1, 1, 1, kv_len)``.
    """
    mask = mx.ones((1, 1, 1, kv_len), dtype=mx.bool_)
    if pad_positions is not None:
        # pad_positions: (B,) int -- number of left-pad tokens per batch element
        B = pad_positions.shape[0]
        kv_idx = mx.arange(kv_len).reshape(1, 1, 1, kv_len)
        pad_end = pad_positions.reshape(B, 1, 1, 1)
        mask = kv_idx >= pad_end
    return mask
