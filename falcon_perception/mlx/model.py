# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""FalconPerception model for MLX.

Mirrors ``falcon_perception/model.py`` using MLX ops.
Key differences from the PyTorch version:
  - ``mx.fast.scaled_dot_product_attention`` with native ``sinks`` parameter
    replaces FlexAttention + manual LSE post-attention.
  - Pure MLX ``relu^2 * up`` replaces the Triton FFN kernel.
  - Sin/cos RoPE (no complex numbers).
  - NHWC layout for Conv2d.
"""

import math
from typing import NamedTuple

import einops as E
import mlx.core as mx
import mlx.nn as nn

from falcon_perception import ModelArgs
from falcon_perception.mlx.anyup import AnyUp
from falcon_perception.mlx.kv_cache import KVCache
from falcon_perception.mlx.rope import (
    apply_3d_rotary_emb,
    apply_golden_freqs_cis_to_visual_pos,
    precompute_freqs_cis,
)


# ── Heads ─────────────────────────────────────────────────────────────


class FourierEncoder(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int, out_dim: int):
        super().__init__()
        self.embed = nn.Linear(in_dim, feat_dim // 2, bias=False)
        self.transform = nn.Linear(feat_dim, out_dim, bias=False)

    def __call__(self, x):
        f = 2 * math.pi * self.embed(x)
        f = mx.concatenate([mx.cos(f), mx.sin(f)], axis=-1)
        return self.transform(f)


class BboxDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def __call__(self, x):
        x = mx.maximum(self.w1(x), 0.0)
        x = x * x
        return self.w2(x)


class SegmDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.layers = [nn.Linear(in_dim, in_dim) for _ in range(num_layers - 1)]
        self.pixel_layer = nn.Linear(in_dim, out_dim, bias=False)

    def __call__(self, x):
        for layer in self.layers:
            x = mx.maximum(layer(x), 0.0)
            x = x * x
        return self.pixel_layer(x)


# ── Attention ─────────────────────────────────────────────────────────


def repeat_kv(x, n_rep: int):
    """Repeat KV heads for GQA: (B, S, H, D) -> (B, S, H*n_rep, D)."""
    if n_rep == 1:
        return x
    B, S, H, D = x.shape
    return mx.repeat(x[:, :, :, None, :].reshape(B, S, H, 1, D), n_rep, axis=3).reshape(B, S, H * n_rep, D)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_kv_heads = args.n_kv_heads or args.n_heads
        self.n_rep = args.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.q_dim = args.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.n_heads = args.n_heads

        self.wqkv = nn.Linear(args.dim, self.q_dim + 2 * self.kv_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.sinks = mx.zeros((args.n_heads,))
        self.scale = self.head_dim ** -0.5

        self._norm_w_in = mx.ones((args.dim,))
        self._norm_w_qk = mx.ones((self.head_dim,))

    def __call__(
        self,
        x,
        attention_mask,
        kv_cache: KVCache,
        freqs_cos_sin,
        freqs_cos_sin_2d=None,
    ):
        h = mx.fast.rms_norm(x, self._norm_w_in, eps=1e-5)
        qkv = self.wqkv(h)
        xq, xk, xv = mx.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)

        xq = E.rearrange(xq, "b s (h d) -> b s h d", d=self.head_dim)
        xk = E.rearrange(xk, "b s (h d) -> b s h d", d=self.head_dim)
        xv = E.rearrange(xv, "b s (h d) -> b s h d", d=self.head_dim)

        xq = mx.fast.rms_norm(xq, self._norm_w_qk, eps=1e-5)
        xk = mx.fast.rms_norm(xk, self._norm_w_qk, eps=1e-5)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq, xk = apply_3d_rotary_emb(xq, xk, freqs_cos_sin, freqs_cos_sin_2d)

        # (B, S, H, D) -> (B, H, S, D) for SDPA
        xq = xq.transpose(0, 2, 1, 3)
        xk = xk.transpose(0, 2, 1, 3)
        xv = xv.transpose(0, 2, 1, 3)

        # KV cache insert
        xk, xv = kv_cache.insert_kv(self.layer_id, xk, xv)

        # SDPA with native sinks
        output = mx.fast.scaled_dot_product_attention(
            xq, xk, xv,
            scale=self.scale,
            mask=attention_mask,
            sinks=self.sinks,
        )

        # (B, H, S, D) -> (B, S, H*D)
        output = output.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.wo(output)


# ── FeedForward ───────────────────────────────────────────────────────


class FeedForward(nn.Module):
    """Squared-ReLU gated FFN.

    ``w13`` stores gate and up weights interleaved as ``[g0, u0, g1, u1, ...]``.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.hidden_dim = hidden_dim
        self._norm_w = mx.ones((dim,))

    def __call__(self, x):
        x = mx.fast.rms_norm(x, self._norm_w, eps=1e-5)
        w13_out = self.w13(x)  # (B, S, 2*hidden_dim)
        # Deinterleave: [g0,u0,g1,u1,...] -> gate, up
        gate = w13_out[..., 0::2]
        up = w13_out[..., 1::2]
        return self.w2(mx.maximum(gate, 0.0) ** 2 * up)


# ── TransformerBlock ──────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: ModelArgs):
        super().__init__()
        self.attention = Attention(model_args, layer_id)
        self.feed_forward = FeedForward(model_args.dim, model_args.ffn_dim)

    def __call__(
        self,
        x,
        kv_cache,
        freqs_cos_sin,
        freqs_cos_sin_2d=None,
        attention_mask=None,
    ):
        B, S, D = x.shape
        x = x + self.attention(
            x,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            freqs_cos_sin=freqs_cos_sin,
            freqs_cos_sin_2d=freqs_cos_sin_2d,
        )
        out = x + self.feed_forward(x)
        return out.reshape(B, S, D)


# ── ImgScatterEntry ──────────────────────────────────────────────────


class ImgScatterEntry(NamedTuple):
    batch_idx: int
    token_start: int
    n_tokens: int
    h_valid_patches: int
    w_valid_patches: int


# ── FalconPerception ──────────────────────────────────────────────────


class FalconPerception(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        img_in_dim = (
            args.temporal_patch_size * args.spatial_patch_size ** 2 * args.channel_size
        )
        self.img_projector = nn.Linear(img_in_dim, args.dim, bias=False)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = [TransformerBlock(i, args) for i in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        if args.perception_heads:
            self.coord_encoder = FourierEncoder(2, args.coord_enc_dim, args.dim)
            self.coord_decoder = BboxDecoder(args.dim, args.coord_dec_dim, args.coord_out_dim)
            self.size_encoder = FourierEncoder(2, args.size_enc_dim, args.dim)
            self.size_decoder = BboxDecoder(args.dim, args.size_dec_dim, args.size_out_dim)
            self.itok_upsampler = AnyUp()
            self.proj_segm = SegmDecoder(args.dim, args.segm_out_dim, args.num_segm_layers)
            self.conv_segm = nn.Conv2d(args.dim, args.segm_out_dim, kernel_size=3, padding=1)

        # RoPE
        rope_dim = args.head_dim // 2
        cos_table, sin_table = precompute_freqs_cis(rope_dim, args.max_seq_len, args.rope_theta)
        self.freqs_cos = cos_table
        self.freqs_sin = sin_table
        self.freqs_cis_golden = mx.zeros((args.n_heads, rope_dim // 2, 2))

    @property
    def device(self):
        return mx.default_device()

    @property
    def dtype(self):
        return self.tok_embeddings.weight.dtype

    def _scatter_img_tokens_with_projector(
        self,
        h_BSD,
        pixel_values_NTHWC,
        img_scatter_info: list[ImgScatterEntry],
    ):
        pt = self.args.temporal_patch_size
        ps = self.args.spatial_patch_size

        valid_parts = []
        for img, (_, _, _, h_v, w_v) in zip(pixel_values_NTHWC, img_scatter_info):
            img = img.astype(self.dtype)
            patches = E.rearrange(
                img,
                "(t pt) (h ph) (w pw) c -> (t h w) (pt ph pw c)",
                pt=pt, ph=ps, pw=ps,
            )
            T_i, H_i, W_i, _ = img.shape
            grid = patches.reshape(T_i // pt, H_i // ps, W_i // ps, patches.shape[-1])
            valid_parts.append(grid[:, :h_v, :w_v, :].reshape(-1, patches.shape[-1]))
        valid_patches = mx.concatenate(valid_parts, axis=0)
        valid_feats = self.img_projector(valid_patches)

        offset = 0
        for b, tok_start, n_tok, _, _ in img_scatter_info:
            h_BSD = h_BSD.at[b, tok_start:tok_start + n_tok, :].add(
                valid_feats[offset:offset + n_tok] - h_BSD[b, tok_start:tok_start + n_tok, :]
            )
            offset += n_tok
        return h_BSD

    def _extract_coords(self, coords_BO):
        all_xy, all_hw = [], []
        for coords_O in coords_BO:
            if not coords_O:
                continue
            for coords in coords_O:
                for k, v in coords.items():
                    if k.startswith(("x", "y")):
                        all_xy.append(v)
                    elif k.startswith(("h", "w")):
                        all_hw.append(v)
        all_xy = mx.array(all_xy) if all_xy else mx.zeros((0,))
        all_hw = mx.array(all_hw) if all_hw else mx.zeros((0,))
        return all_xy, all_hw

    def _encode_coords(self, h_BSD, tokens_BS, all_xy):
        coord_tokens_mask = tokens_BS == self.args.coord_token_id
        if all_xy.size == 0:
            return h_BSD
        coord_tokens = self.coord_encoder(all_xy.reshape(-1, 2))
        if coord_tokens.shape[0] == h_BSD.shape[0]:
            h_BSD = mx.where(
                mx.expand_dims(coord_tokens_mask, -1),
                coord_tokens.reshape(h_BSD.shape[0], -1, h_BSD.shape[-1]),
                h_BSD,
            )
        else:
            # Packed prefill: scatter coord tokens into matching positions
            indices = mx.argwhere(coord_tokens_mask.flatten())
            for i, idx in enumerate(indices):
                b = int(idx) // h_BSD.shape[1]
                s = int(idx) % h_BSD.shape[1]
                h_BSD = h_BSD.at[b, s, :].add(coord_tokens[i] - h_BSD[b, s, :])
        return h_BSD

    def _encode_sizes(self, h_BSD, tokens_BS, all_hw):
        size_tokens_mask = tokens_BS == self.args.size_token_id
        if all_hw.size == 0:
            return h_BSD
        size_tokens = self.size_encoder(all_hw.reshape(-1, 2))
        if size_tokens.shape[0] == h_BSD.shape[0]:
            h_BSD = mx.where(
                mx.expand_dims(size_tokens_mask, -1),
                size_tokens.reshape(h_BSD.shape[0], -1, h_BSD.shape[-1]),
                h_BSD,
            )
        else:
            indices = mx.argwhere(size_tokens_mask.flatten())
            for i, idx in enumerate(indices):
                b = int(idx) // h_BSD.shape[1]
                s = int(idx) % h_BSD.shape[1]
                h_BSD = h_BSD.at[b, s, :].add(size_tokens[i] - h_BSD[b, s, :])
        return h_BSD

    def decode_coords(self, h_BSD, labels):
        B, S, D = h_BSD.shape
        coord_masks = labels == self.args.coord_token_id
        coord_indices = mx.argwhere(coord_masks.flatten())
        if coord_indices.size == 0:
            return mx.zeros((0, 2, self.args.coord_out_dim // 2))
        coord_tokens = []
        for idx in coord_indices:
            b = int(idx) // S
            s = int(idx) % S
            coord_tokens.append(h_BSD[b, s])
        coord_tokens = mx.stack(coord_tokens)
        coord_logits = self.coord_decoder(coord_tokens)
        return E.rearrange(coord_logits, "b (two dim) -> b two dim", two=2)

    def decode_sizes(self, h_BSD, labels):
        B, S, D = h_BSD.shape
        size_masks = labels == self.args.size_token_id
        size_indices = mx.argwhere(size_masks.flatten())
        if size_indices.size == 0:
            return mx.zeros((0, 2, self.args.size_out_dim // 2))
        size_tokens = []
        for idx in size_indices:
            b = int(idx) // S
            s = int(idx) % S
            size_tokens.append(h_BSD[b, s])
        size_tokens = mx.stack(size_tokens)
        size_logits = self.size_decoder(size_tokens)
        return E.rearrange(size_logits, "b (two dim) -> b two dim", two=2)

    def process_sizes(self, logits):
        num_bins = logits.shape[-1]
        pred = mx.argmax(logits, axis=-1).astype(mx.float32) / (num_bins - 1)
        min_size = math.log2(1 / num_bins)
        max_size = 0.0
        pred = pred * (max_size - min_size) + min_size
        return mx.power(2.0, pred)

    def upsample_single_img_features(
        self,
        h_SD,
        pixel_values_THWC,
        img_token_start: int,
        h_valid: int,
        w_valid: int,
        output_size=None,
    ):
        ps = self.args.spatial_patch_size
        _, H, W, _ = pixel_values_THWC.shape
        h_patch, w_patch = H // ps, W // ps
        image = pixel_values_THWC[None] if pixel_values_THWC.ndim == 3 else pixel_values_THWC
        # image is (1, H, W, C) -- NHWC throughout

        D = h_SD.shape[-1]
        n_img = h_valid * w_valid
        img_feats = h_SD[img_token_start:img_token_start + n_img, :]
        lr_img_features = img_feats.reshape(1, h_valid, w_valid, D)
        if h_valid < h_patch or w_valid < w_patch:
            lr_img_features = mx.pad(
                lr_img_features,
                [(0, 0), (0, h_patch - h_valid), (0, w_patch - w_valid), (0, 0)],
            )
        # conv_segm: MLX Conv2d operates on NHWC natively
        lr_img_features = self.conv_segm(lr_img_features)  # (1, h, w, segm_out_dim)

        if output_size == (h_patch, w_patch):
            # Return as (C, H, W) for downstream einsum
            return E.rearrange(lr_img_features[0], "h w c -> c h w")
        if output_size is None:
            output_size = (H, W)

        hr_img_features = self.itok_upsampler(
            images=image,
            features=lr_img_features,
            output_size=output_size,
        )
        return hr_img_features[0]  # (C, out_H, out_W) -- AnyUp returns NCHW

    def upsample_img_features(
        self,
        h_BSD,
        pixel_values_NTHWC,
        img_scatter_info: list[ImgScatterEntry],
        output_size=None,
    ):
        hr_parts = []
        for i, entry in enumerate(img_scatter_info):
            hr_i = self.upsample_single_img_features(
                h_BSD[entry.batch_idx],
                pixel_values_NTHWC[i],
                img_token_start=entry.token_start,
                h_valid=entry.h_valid_patches,
                w_valid=entry.w_valid_patches,
                output_size=output_size,
            )
            hr_parts.append(hr_i)
        return mx.stack(hr_parts, axis=0)

    def __call__(
        self,
        tokens,
        attention_mask,
        kv_cache: KVCache,
        rope_pos_t=None,
        rope_pos_hw=None,
        pixel_values=None,
        coord_xy=None,
        size_hw=None,
        img_scatter_info=None,
    ):
        B, S = tokens.shape

        T_pos = kv_cache.get_pos()
        is_prefill = S != 1

        if is_prefill:
            assert rope_pos_t is not None and rope_pos_hw is not None
            pos_t = rope_pos_t[:, T_pos:T_pos + S].astype(mx.int32)
            kv_cache.pos_t = pos_t[:, -1:]
            freqs_cos_sin = (self.freqs_cos[pos_t], self.freqs_sin[pos_t])
            rope_pos_hw_slice = rope_pos_hw[:, T_pos:T_pos + S]
            freqs_cos_sin_2d = apply_golden_freqs_cis_to_visual_pos(
                self.freqs_cis_golden, rope_pos_hw_slice,
            )
        else:
            pos_t = kv_cache.increment_and_get_pos_t()
            freqs_cos_sin = (self.freqs_cos[pos_t], self.freqs_sin[pos_t])
            freqs_cos_sin_2d = None

        h_BSD = self.tok_embeddings(tokens)

        if self.args.perception_heads:
            coord_xy = coord_xy if coord_xy is not None else mx.zeros((0,))
            size_hw = size_hw if size_hw is not None else mx.zeros((0,))
            h_BSD = self._encode_coords(h_BSD, tokens, coord_xy)
            h_BSD = self._encode_sizes(h_BSD, tokens, size_hw)

        if pixel_values is not None:
            assert img_scatter_info is not None
            h_BSD = self._scatter_img_tokens_with_projector(
                h_BSD, pixel_values, img_scatter_info,
            )

        # Slice the mask for this step
        if is_prefill:
            step_mask = attention_mask[:, :, :S, :T_pos + S]
        else:
            step_mask = attention_mask[:, :, T_pos:T_pos + 1, :T_pos + 1]

        for layer in self.layers:
            h_BSD = layer(
                h_BSD,
                attention_mask=step_mask,
                kv_cache=kv_cache,
                freqs_cos_sin=freqs_cos_sin,
                freqs_cos_sin_2d=freqs_cos_sin_2d,
            )

        h_BSD = self.norm(h_BSD)
        logits_BSV = self.output(h_BSD)
        return logits_BSV, h_BSD

    def sample_bbox(self, h_BD, tokens_B):
        B = h_BD.shape[0]
        if not self.args.perception_heads:
            zeros = mx.zeros((B, 2))
            false = mx.zeros((B,), dtype=mx.bool_)
            return zeros, zeros, false, false, zeros

        is_coord_B = tokens_B == self.args.coord_token_id
        is_size_B = tokens_B == self.args.size_token_id

        coord_logits = self.coord_decoder(h_BD).reshape(B, 2, -1)
        num_bins = coord_logits.shape[-1]
        xy_B2 = mx.argmax(coord_logits, axis=-1).astype(mx.float32) / num_bins

        size_logits = self.size_decoder(h_BD).reshape(B, 2, -1)
        hw_B2 = self.process_sizes(size_logits)

        return xy_B2, hw_B2, is_coord_B, is_size_B, coord_logits

    def get_segm_tokens(self, h_BD, tokens_B):
        B = h_BD.shape[0]
        is_segm_B = tokens_B.reshape(B) == self.args.seg_token_id
        if self.args.perception_heads:
            segm_BD = self.proj_segm(h_BD)
        else:
            segm_BD = mx.zeros((B, 0))
        return segm_BD, is_segm_B
