# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

import torch
from torch import Tensor as T
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)

# Compiled flex_attention variants:
# - _decode: fullgraph=True, static shapes — for decode (S_q==1), CUDA-graph safe.
# - _prefill: dynamic=True, symbolic shapes — for prefill (S_q>1), one graph handles all lengths.
compiled_flex_attn_decode = torch.compile(flex_attention, fullgraph=True)
compiled_flex_attn_prefill = torch.compile(flex_attention, dynamic=True)
_compiled_create_block_mask = torch.compile(create_block_mask, dynamic=True)


def offset_mask_mod(mask_mod: _mask_mod_signature, offset: int):
    """Get a mask mod function with an offset applied to the query positions."""

    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)

    return _mask_mod


def get_causal_mask_mod() -> _mask_mod_signature:
    """Causal mask that prevents attention to future tokens."""

    def _causal_mask(b: T, h: T, q_idx: T, kv_idx: T) -> T:
        return q_idx >= kv_idx

    return _causal_mask


def get_document_mask_mod(batch: T, eos_id: int) -> _mask_mod_signature:
    """Creates a document mask that prevents attention across document boundaries.

    Args:
        batch: Input batch tensor with shape [b, s, h, d]
        eos_id: End-of-sequence token ID that marks document boundaries

    Returns:
        A mask modifier function that implements document-level masking.
    """
    # batch is [b, s, h, d] shape
    eos_mask = batch == eos_id
    eos_mask[:, -1] = True
    cumulative_mask = torch.cumsum(torch.where(eos_mask, 1, 0), dim=1)
    sequence_indices = torch.zeros_like(cumulative_mask, dtype=torch.int32)
    sequence_indices[:, 1:] = cumulative_mask[:, :-1]

    def document_mask(b: T, h: T, q_idx: T, kv_idx: T) -> T:
        return sequence_indices[b, q_idx] == sequence_indices[b, kv_idx]

    return document_mask


def get_non_left_pad_mask_mod(batch: T, pad_id: int) -> _mask_mod_signature:
    """Prevent model from attending to the left-padded token required for correct batch inference."""

    non_pad_mask_id = torch.cumsum(batch != pad_id, dim=1)

    # Left-most pad tokens have cumulative id == 0.
    def mask_mod(b, h, q_idx, kv_idx):
        return non_pad_mask_id[b, kv_idx] > 0

    return mask_mod


def get_image_prefix_mask_mod(
    batch: T, soi_id: int, eoi_id: int
) -> _mask_mod_signature:
    # batch is [b, s, h, d] shape
    soi_mask = batch == soi_id
    eoi_mask = batch == eoi_id
    acc_soi_mask = torch.cumsum(soi_mask, dim=1)
    acc_eoi_mask = torch.cumsum(eoi_mask, dim=1)
    # Get every tokens between two soi_id and eoi_id exclusive of eoi_id
    img_mask = (acc_soi_mask - acc_eoi_mask) > 0

    # Create a tensor that assigns each token to its image number
    # Each image starts with SOI token, so we can use acc_soi_mask to track image numbers
    img_indices = acc_soi_mask * img_mask

    def image_prefix_mask_mod(b, h, q_idx, kv_idx):
        # Check if both tokens are image tokens and belong to the same image
        is_img_tokens = img_mask[b, q_idx] & img_mask[b, kv_idx]
        is_same_image = img_indices[b, q_idx] == img_indices[b, kv_idx]
        return is_img_tokens & is_same_image

    return image_prefix_mask_mod


@torch.inference_mode()
def create_attention_mask(*args, **kwargs) -> BlockMask:
    """Compiled for performance; always runs under inference_mode to avoid grad_mode recompiles."""
    return _compiled_create_block_mask(*args, **kwargs)


def create_batch_attention_mask(
    input_batch: T,
    *,
    pad_token_id: int,
    eos_token_id: int,
    soi_token_id: int,
    eoi_token_id: int,
    max_len: int | None = None,
) -> BlockMask:
    """Build the combined FlexAttention mask for the batch engine.

    Composes causal + document + non-left-pad + image-prefix masks.
    """
    B, S = input_batch.size()
    block_causal_mask_mod = and_masks(
        get_causal_mask_mod(),
        get_document_mask_mod(input_batch, eos_token_id),
        get_non_left_pad_mask_mod(input_batch, pad_token_id),
    )
    image_prefix_mask_mod = get_image_prefix_mask_mod(
        batch=input_batch,
        soi_id=soi_token_id,
        eoi_id=eoi_token_id,
    )
    mask_mod = or_masks(image_prefix_mask_mod, block_causal_mask_mod)
    max_len = max_len or S
    return create_attention_mask(mask_mod, B, None, max_len, max_len)
