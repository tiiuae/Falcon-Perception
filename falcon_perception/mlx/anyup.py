# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""AnyUp image-feature upsampler for MLX.

Mirrors the PyTorch ``falcon_perception/anyup.py`` using MLX ops and
``mx.fast.scaled_dot_product_attention`` for windowed cross-attention.
"""

from functools import lru_cache

import einops as E
import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────


def _reflect_pad(x, p):
    """Reflect-pad spatial dims of an NHWC tensor."""
    if p <= 0:
        return x
    # Pad height: reflect rows
    top = x[:, 1 : p + 1, :, :][:, ::-1, :, :]
    bottom = x[:, -p - 1 : -1, :, :][:, ::-1, :, :]
    x = mx.concatenate([top, x, bottom], axis=1)
    # Pad width: reflect columns
    left = x[:, :, 1 : p + 1, :][:, :, ::-1, :]
    right = x[:, :, -p - 1 : -1, :][:, :, ::-1, :]
    return mx.concatenate([left, x, right], axis=2)


# ── ResBlock ──────────────────────────────────────────────────────────


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        num_groups=8,
        use_norm=True,
        use_conv_shortcut=False,
    ):
        super().__init__()
        p = kernel_size // 2
        layers = []
        if use_norm:
            layers.append(nn.GroupNorm(num_groups, in_channels, pytorch_compatible=True))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=p, bias=False))
        if use_norm:
            layers.append(nn.GroupNorm(num_groups, out_channels, pytorch_compatible=True))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=p, bias=False))
        self.block = layers

        if use_conv_shortcut or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = None

    def __call__(self, x):
        h = x
        for layer in self.block:
            h = layer(h)
        sc = self.shortcut(x) if self.shortcut is not None else x
        return h + sc


# ── LearnedFeatureUnification ────────────────────────────────────────


class LearnedFeatureUnification(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.basis = mx.random.normal((out_channels, kernel_size, kernel_size, 1))

    def __call__(self, features):
        # features: (B, H, W, C) -- MLX NHWC
        b, h, w, c = features.shape
        k = self.kernel_size
        p = k // 2
        x = mx.pad(features, [(0, 0), (p, p), (p, p), (0, 0)])

        # Per-channel conv with shared basis, matching PyTorch groups=C depthwise.
        # Each channel produces out_ch outputs; concatenate in channel order to
        # replicate PyTorch's flat layout: [ch0_b0..ch0_b127, ch1_b0..ch1_b127, ...]
        parts = []
        for ci in range(c):
            parts.append(mx.conv2d(x[..., ci : ci + 1], self.basis))
        x_flat = mx.concatenate(parts, axis=-1)  # (B, H, W, out_ch * C)

        # Border normalisation (same as PyTorch)
        mask = mx.ones((1, h, w, 1))
        mask_padded = mx.pad(mask, [(0, 0), (p, p), (p, p), (0, 0)])
        ones_kern = mx.ones((1, k, k, 1))
        denom = mx.conv2d(mask_padded, ones_kern)  # (1, H, W, 1)
        x_flat = x_flat / denom

        # Reshape to match PyTorch's .view(B, out_ch, C, H, W) — in NHWC that
        # becomes (B, H, W, out_ch, C).
        x_5d = x_flat.reshape(b, h, w, self.out_channels, c)
        attn = mx.softmax(x_5d, axis=3)
        return attn.mean(axis=4)  # (B, H, W, out_ch)


# ── AnyUp RoPE ────────────────────────────────────────────────────────


def _rotate_half(x):
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


class AnyUpRoPE(nn.Module):
    def __init__(self, dim: int, theta: int = 100):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs = mx.zeros((2, dim))

    def __call__(self, x, coords):
        angle = coords @ self.freqs
        return x * mx.cos(angle) + _rotate_half(x) * mx.sin(angle)


# ── Window mask ───────────────────────────────────────────────────────


def _window2d(low_res, high_res, ratio, device=None):
    if isinstance(high_res, int):
        H = W = high_res
    else:
        H, W = high_res
    if isinstance(low_res, int):
        Lh = Lw = low_res
    else:
        Lh, Lw = low_res

    r_pos = (mx.arange(H).astype(mx.float32) + 0.5) / H
    c_pos = (mx.arange(W).astype(mx.float32) + 0.5) / W
    pos_r = mx.repeat(r_pos[:, None], W, axis=1)  # (H, W)
    pos_c = mx.repeat(c_pos[None, :], H, axis=0)  # (H, W)

    r_lo = mx.clip(pos_r - ratio, 0.0, 1.0)
    r_hi = mx.clip(pos_r + ratio, 0.0, 1.0)
    c_lo = mx.clip(pos_c - ratio, 0.0, 1.0)
    c_hi = mx.clip(pos_c + ratio, 0.0, 1.0)

    r0 = mx.floor(r_lo * Lh).astype(mx.int32)
    r1 = mx.ceil(r_hi * Lh).astype(mx.int32)
    c0 = mx.floor(c_lo * Lw).astype(mx.int32)
    c1 = mx.ceil(c_hi * Lw).astype(mx.int32)

    return r0, r1, c0, c1



# ── Cross attention ──────────────────────────────────────────────────


class AttentionWrapper(nn.Module):
    def __init__(self, qk_dim: int):
        super().__init__()
        self.in_proj_weight = mx.zeros((qk_dim * 3, qk_dim))
        self.in_proj_bias = mx.zeros((qk_dim * 3,))

    def __call__(self, x_q, x_k, x_v):
        w_q, w_k, _ = mx.split(self.in_proj_weight, 3, axis=0)
        b_q, b_k, _ = mx.split(self.in_proj_bias, 3)
        x_q = x_q @ w_q.T + b_q
        x_k = x_k @ w_k.T + b_k
        return x_q, x_k, x_v


class FlexCrossAttention(nn.Module):
    def __init__(self, qk_dim: int, num_heads: int):
        super().__init__()
        self.dim = qk_dim
        self.num_head = num_heads
        self.norm_q = nn.RMSNorm(qk_dim)
        self.norm_k = nn.RMSNorm(qk_dim)
        self.attention = AttentionWrapper(qk_dim)

    def __call__(self, query, key, value, mask=None):
        x_q = self.norm_q(query)
        x_k = self.norm_k(key)
        x_q, x_k, x_v = self.attention(x_q, x_k, value)

        x_q = E.rearrange(x_q, "b HW (h d) -> b h HW d", h=self.num_head)
        x_k = E.rearrange(x_k, "b hw (h d) -> b h hw d", h=self.num_head)
        x_v = E.rearrange(value, "b hw (h d) -> b h hw d", h=self.num_head)

        scale = x_q.shape[-1] ** -0.5
        output = mx.fast.scaled_dot_product_attention(
            x_q, x_k, x_v, scale=scale, mask=mask,
        )
        return E.rearrange(output, "b h hw d -> b hw (h d)")


class CrossAttentionBlock(nn.Module):
    def __init__(self, qk_dim, num_heads, window_ratio=0.1):
        super().__init__()
        self.cross_attn = FlexCrossAttention(qk_dim, num_heads)
        self.window_ratio = window_ratio
        self.conv2d = nn.Conv2d(qk_dim, qk_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def __call__(self, q, k, v, block_mask=None):
        b, h_out, w_out, c_qk = q.shape
        _, h_kv, w_kv, c_v = v.shape

        q = self.conv2d(q)

        num_heads = self.cross_attn.num_head
        d_qk = c_qk // num_heads
        d_v = c_v // num_heads
        ratio = self.window_ratio
        scale = d_qk ** -0.5

        q_flat = q.reshape(b, h_out * w_out, c_qk)
        k_flat = k.reshape(b, h_kv * w_kv, c_qk)
        v_flat = v.reshape(b, h_kv * w_kv, c_v)

        q_n = self.cross_attn.norm_q(q_flat)
        k_n = self.cross_attn.norm_k(k_flat)
        q_n, k_n, _ = self.cross_attn.attention(q_n, k_n, v_flat)

        q_mh = q_n.reshape(b, -1, num_heads, d_qk).transpose(0, 2, 1, 3)
        k_mh = k_n.reshape(b, -1, num_heads, d_qk).transpose(0, 2, 1, 3)
        v_mh = v_flat.reshape(b, -1, num_heads, d_v).transpose(0, 2, 1, 3)

        approx_kv = int((2 * ratio * h_kv + 2) * w_kv)
        target_bytes = 256 * 1024 * 1024
        max_q = max(w_out, target_bytes // max(1, approx_kv * num_heads * 4))
        tile_rows = max(1, max_q // w_out)

        outputs = []
        for rs in range(0, h_out, tile_rows):
            re = min(rs + tile_rows, h_out)
            tile_h = re - rs

            qt = q_mh[:, :, rs * w_out : re * w_out, :]

            r_lo = max(0.0, (rs + 0.5) / h_out - ratio)
            r_hi = min(1.0, (re - 0.5) / h_out + ratio)
            krs = max(0, int(np.floor(r_lo * h_kv)))
            kre = min(h_kv, int(np.ceil(r_hi * h_kv)))

            kt = k_mh[:, :, krs * w_kv : kre * w_kv, :]
            vt = v_mh[:, :, krs * w_kv : kre * w_kv, :]

            mask = _tile_mask(
                rs, re, w_out, h_out, krs, kre, w_kv, h_kv, ratio,
            )
            out = mx.fast.scaled_dot_product_attention(
                qt, kt, vt, scale=scale, mask=mask,
            )
            outputs.append(out)

        out_mh = mx.concatenate(outputs, axis=2)
        out_seq = out_mh.transpose(0, 2, 1, 3).reshape(b, h_out * w_out, c_v)
        return out_seq.reshape(b, h_out, w_out, c_v)


@lru_cache(maxsize=128)
def _tile_mask(rs, re, w_out, h_out, krs, kre, w_kv, h_kv, ratio):
    """Build a small boolean mask for one query-row tile."""
    tile_h = re - rs
    kvt_h = kre - krs

    q_r = (np.arange(rs, re) + 0.5) / h_out
    q_c = (np.arange(w_out) + 0.5) / w_out
    k_r_idx = np.arange(krs, kre)
    k_c_idx = np.arange(w_kv)

    qr_r0 = np.floor(np.clip(q_r - ratio, 0, 1) * h_kv).astype(int)
    qr_r1 = np.ceil(np.clip(q_r + ratio, 0, 1) * h_kv).astype(int)
    r_ok = (k_r_idx[None, :] >= qr_r0[:, None]) & (k_r_idx[None, :] < qr_r1[:, None])

    qc_c0 = np.floor(np.clip(q_c - ratio, 0, 1) * w_kv).astype(int)
    qc_c1 = np.ceil(np.clip(q_c + ratio, 0, 1) * w_kv).astype(int)
    c_ok = (k_c_idx[None, :] >= qc_c0[:, None]) & (k_c_idx[None, :] < qc_c1[:, None])

    mask_np = (r_ok[:, None, :, None] & c_ok[None, :, None, :]).reshape(
        tile_h * w_out, kvt_h * w_kv,
    )
    return mx.array(mask_np).reshape(1, 1, tile_h * w_out, kvt_h * w_kv)


# ── Helpers ───────────────────────────────────────────────────────────


def _pool_to(x, size):
    """Reshape-based area pooling. x: (B, H, W, C) in NHWC."""
    b, H, W, c = x.shape
    oh, ow = size
    if H == oh and W == ow:
        return x
    return x.reshape(b, oh, H // oh, ow, W // ow, c).mean(axis=(2, 4))


def _create_coordinate(h, w):
    x = mx.linspace(0.0, 1.0, h)
    y = mx.linspace(0.0, 1.0, w)
    xx = mx.repeat(x[:, None], w, axis=1)
    yy = mx.repeat(y[None, :], h, axis=0)
    return mx.stack([xx.flatten(), yy.flatten()], axis=-1).reshape(1, h * w, 2)


IMAGENET_MEAN = mx.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
IMAGENET_STD = mx.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)


# ── AnyUp ─────────────────────────────────────────────────────────────


class AnyUp(nn.Module):
    def __init__(
        self,
        input_dim=3,
        qk_dim=128,
        kernel_size=1,
        kernel_size_lfu=5,
        window_ratio=0.1,
        num_heads=4,
    ):
        super().__init__()
        self.qk_dim = qk_dim
        self.window_ratio = window_ratio

        self.image_encoder = self._make_encoder(input_dim, kernel_size)
        self._image_encoder_pad = kernel_size // 2
        self.key_encoder = self._make_encoder(qk_dim, 1)
        self.query_encoder = self._make_encoder(qk_dim, 1)
        self.key_features_encoder = self._make_encoder(
            None, 1, first_layer_k=kernel_size_lfu,
        )

        self.cross_decode = CrossAttentionBlock(
            qk_dim=qk_dim, num_heads=num_heads, window_ratio=window_ratio,
        )
        self.aggregation = self._make_encoder(2 * qk_dim, 3)
        self._aggregation_pad = 3 // 2
        self.rope = AnyUpRoPE(qk_dim)

    def _make_encoder(self, in_ch, k, layers=2, first_layer_k=0):
        parts = []
        if first_layer_k == 0:
            parts.append(nn.Conv2d(in_ch, self.qk_dim, k, padding=0, bias=False))
        else:
            parts.append(LearnedFeatureUnification(self.qk_dim, first_layer_k))
        for _ in range(layers):
            parts.append(
                ResBlock(self.qk_dim, self.qk_dim, kernel_size=1, num_groups=8)
            )
        return parts

    def _run_encoder(self, encoder, x, reflect_pad=0):
        for i, layer in enumerate(encoder):
            if i == 0 and reflect_pad > 0:
                x = _reflect_pad(x, reflect_pad)
            x = layer(x)
        return x

    def _normalize(self, x):
        """L2 normalize along channel (last) axis."""
        return x / (mx.sqrt((x * x).sum(axis=-1, keepdims=True)) + 1e-8)

    def upsample(self, enc_img, feats, out_size):
        b, h, w, c = feats.shape

        q = _pool_to(self._run_encoder(self.query_encoder, enc_img), out_size)
        k = _pool_to(self._run_encoder(self.key_encoder, enc_img), (h, w))
        k = mx.concatenate(
            [k, self._run_encoder(self.key_features_encoder, self._normalize(feats))],
            axis=-1,
        )
        k = self._run_encoder(self.aggregation, k, reflect_pad=self._aggregation_pad)
        v = feats

        return self.cross_decode(q, k, v)

    def __call__(self, images, features, output_size=None):
        # Both images and features arrive in NHWC from the caller
        output_size = output_size if output_size is not None else (images.shape[1], images.shape[2])

        images = images * 0.5 + 0.5
        images = (images - IMAGENET_MEAN) / IMAGENET_STD
        images = images.astype(features.dtype)

        enc = self._run_encoder(self.image_encoder, images, reflect_pad=self._image_encoder_pad)
        h_enc = enc.shape[1]
        coords = _create_coordinate(h_enc, enc.shape[2])
        coords = coords.astype(enc.dtype)
        # (B, H, W, C) -> (B, H*W, C) for RoPE
        enc_flat = enc.reshape(enc.shape[0], -1, enc.shape[-1])
        enc_flat = self.rope(enc_flat, coords)
        enc = enc_flat.reshape(enc.shape[0], h_enc, -1, enc.shape[-1])

        result = self.upsample(enc, features, output_size)
        # Return in NCHW for compatibility with downstream einsum
        return E.rearrange(result, "b h w c -> b c h w")
