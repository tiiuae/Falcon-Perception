# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""RoPE (Rotary Position Embeddings) for MLX -- sin/cos form, no complex numbers."""

import einops as E
import mlx.core as mx


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute cos/sin frequency tables for 1D RoPE.

    Returns (cos, sin) each of shape ``(end, dim // 2)``.
    The PyTorch version returns complex64; here we store the real components.
    """
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    t = mx.arange(end).astype(mx.float32)
    angles = mx.outer(t, freqs)  # (end, dim//2)
    return mx.cos(angles), mx.sin(angles)


def apply_rotary_emb(
    xq,
    xk,
    freqs_cos_sin,
):
    """Apply 1D rotary embedding using precomputed (cos, sin) tables.

    ``freqs_cos_sin`` is a tuple ``(cos, sin)`` already indexed by position,
    each with shape ``(B, S, D//2)`` (broadcast-ready after indexing).
    """
    cos, sin = freqs_cos_sin  # each (B, S, D//2)
    cos = E.rearrange(cos, "b s d -> b s 1 d")
    sin = E.rearrange(sin, "b s d -> b s 1 d")

    xq_f = xq.astype(mx.float32)
    xk_f = xk.astype(mx.float32)

    xq_even = xq_f[..., 0::2]
    xq_odd = xq_f[..., 1::2]
    xk_even = xk_f[..., 0::2]
    xk_odd = xk_f[..., 1::2]

    # Rotary application: [even, odd] * [cos, -sin; sin, cos]
    oq_even = xq_even * cos - xq_odd * sin
    oq_odd = xq_even * sin + xq_odd * cos
    ok_even = xk_even * cos - xk_odd * sin
    ok_odd = xk_even * sin + xk_odd * cos

    # Interleave: stack pairs then flatten last two dims -> [e0,o0,e1,o1,...]
    xq_out = mx.stack([oq_even, oq_odd], axis=-1).reshape(*oq_even.shape[:-1], -1)
    xk_out = mx.stack([ok_even, ok_odd], axis=-1).reshape(*ok_even.shape[:-1], -1)

    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


# ── 2D Golden RoPE ────────────────────────────────────────────────────


def apply_golden_freqs_cis_to_visual_pos(freqs_hFP, pos_BSP):
    """Compute golden-gate 2D RoPE cos/sin for every token.

    Text tokens have pos=0, giving angle=0 -> identity rotation.

    Args:
        freqs_hFP: (n_heads, num_freqs, pos_dim=2) -- learned frequencies.
        pos_BSP:   (B, S, pos_dim=2)               -- per-token positions.

    Returns:
        (cos, sin) each of shape (B, S, H, F).
    """
    theta = mx.einsum("bsp,hfp->bshf", pos_BSP.astype(mx.float32), freqs_hFP.astype(mx.float32))
    return mx.cos(theta), mx.sin(theta)


def apply_golden_rotary_emb(input_BShd, cos_sin_BShF):
    """Apply golden-gate 2D rotary embedding to all tokens.

    Text tokens have identity entries (cos=1, sin=0) so they pass through unchanged.
    """
    cos, sin = cos_sin_BShF
    x = input_BShd.astype(mx.float32)
    x_even = x[..., 0::2]  # (B, S, H, F)
    x_odd = x[..., 1::2]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = mx.stack([out_even, out_odd], axis=-1).reshape(*out_even.shape[:-1], -1)
    return out.astype(input_BShd.dtype)


def apply_3d_rotary_emb(
    xq,  # (B, S, H, D)
    xk,  # (B, S, H, D)
    freqs_cos_sin,
    freqs_cos_sin_2d=None,
):
    """Apply 3D rotary: 1D temporal on first half, 2D golden on second half."""
    D = xq.shape[-1]
    half = D // 2
    xq_t, xq_hw = xq[..., :half], xq[..., half:]
    xk_t, xk_hw = xk[..., :half], xk[..., half:]

    xq_t, xk_t = apply_rotary_emb(xq_t, xk_t, freqs_cos_sin)
    if freqs_cos_sin_2d is not None:
        xq_hw = apply_golden_rotary_emb(xq_hw, freqs_cos_sin_2d)
        xk_hw = apply_golden_rotary_emb(xk_hw, freqs_cos_sin_2d)

    xq_out = mx.concatenate([xq_t, xq_hw], axis=-1).astype(xq.dtype)
    xk_out = mx.concatenate([xk_t, xk_hw], axis=-1).astype(xk.dtype)
    return xq_out, xk_out
