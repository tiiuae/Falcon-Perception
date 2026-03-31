# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

import einops as E
import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis  # [S, D//2]


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1D rotary embedding"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    assert freqs_cis.ndim == 3, (
        "Freqs_cis must be indexed by position ids already and has shape (B,S,D)"
    )
    freqs_cis = E.rearrange(freqs_cis, "b s d -> b s 1 d")
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# 2D golden RoPE
"""
Dimension key:
    B: batch size
    S: number of tokens per sample, Seqlen
    P: pos_dim
    h / H: n_heads
    d: head_dim
    F: num_freqs == head_dim // 2
"""


def apply_golden_freqs_cis_to_visual_pos(freqs_hFP, pos_BSP) -> torch.Tensor:
    """
    Compute golden-gate 2D RoPE frequencies for every token in the batch.
    Text tokens have pos=0, giving θ=0 → identity rotation (1+0j).
    """
    theta_BShF = torch.einsum("bsp,hfp->bshf", pos_BSP.float(), freqs_hFP.float())
    freqs_cis_BShF = torch.polar(torch.ones_like(theta_BShF), theta_BShF)
    return freqs_cis_BShF  # (B, S, H, F)


def apply_golden_rotary_emb(input_BShd, freqs_cis_BShF) -> torch.Tensor:
    """
    Apply golden-gate 2D rotary embedding to all tokens.  Text tokens have
    identity entries (1+0j) in freqs_cis_BShF so they pass through unchanged.
    No nonzero / data-dependent shapes → zero CUDA syncs.
    """
    x = input_BShd.float()
    x_even = x[..., 0::2]  # (B, S, H, F)
    x_odd = x[..., 1::2]   # (B, S, H, F)

    cos = freqs_cis_BShF.real
    sin = freqs_cis_BShF.imag

    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out.type_as(input_BShd)


def apply_3d_rotary_emb(
    xq: torch.Tensor,  # (B, S, H, D)
    xk: torch.Tensor,  # (B, S, H, D)
    freqs_cis: torch.Tensor,
    freqs_cis_2d: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_t, xq_hw = xq.chunk(chunks=2, dim=-1)
    xk_t, xk_hw = xk.chunk(chunks=2, dim=-1)

    xq_t, xk_t = apply_rotary_emb(xq_t, xk_t, freqs_cis)
    if freqs_cis_2d is not None:
        xq_hw = apply_golden_rotary_emb(xq_hw, freqs_cis_2d)
        xk_hw = apply_golden_rotary_emb(xk_hw, freqs_cis_2d)

    xq_out = torch.concat([xq_t, xq_hw], dim=-1).type_as(xq)
    xk_out = torch.concat([xk_t, xk_hw], dim=-1).type_as(xk)
    return xq_out, xk_out
