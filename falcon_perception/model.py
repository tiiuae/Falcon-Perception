# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

import math
from typing import NamedTuple

import einops as E
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor as T
from torch import nn
from torch.nn.attention.flex_attention import (
    AuxRequest,
    BlockMask,
)

from falcon_perception import ModelArgs
from falcon_perception.anyup import AnyUp, build_upsampler_block_mask
from falcon_perception.attention import (
    compiled_flex_attn_decode,
    compiled_flex_attn_prefill,
    offset_mask_mod,
)
from falcon_perception.kv_cache import KVCacheBase
from falcon_perception.rope import (
    apply_3d_rotary_emb,
    apply_golden_freqs_cis_to_visual_pos,
    precompute_freqs_cis,
)


# Heads
class FourierEncoder(nn.Module):
    """Based on https://bmild.github.io/fourfeat/"""

    def __init__(self, in_dim: int, feat_dim: int, out_dim: int):
        super().__init__()
        self.embed = nn.Linear(in_dim, feat_dim // 2, bias=False)
        self.transform = nn.Linear(feat_dim, out_dim, bias=False)

    def forward(self, x):
        f = 2 * math.pi * self.embed(x)
        f = torch.cat([f.cos(), f.sin()], dim=-1)
        return self.transform(f)


class BboxDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: T) -> T:
        x = self.w2(F.relu(self.w1(x)).square())
        return x


class SegmDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, in_dim) for _ in range(num_layers - 1)]
        )
        self.pixel_layer = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x)).square()
        return self.pixel_layer(x)


# Attention
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    B, S, H, D = x.shape  # batch, seqlen, num head, head dim
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(B, S, H, n_rep, D)
        .reshape(B, S, H * n_rep, D)
    )


class Attention(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_id: int,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_kv_heads = args.n_kv_heads or args.n_heads
        self.n_rep = args.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.q_dim = args.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        self.wqkv = nn.Linear(args.dim, self.q_dim + 2 * self.kv_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.sinks = nn.Parameter(torch.empty((args.n_heads,)))

    def _pre_attention(self, x, freqs_cis, freqs_cis_2d) -> tuple[T, T, T]:
        """RMSNorm → QKV → reshape → QK-norm → GQA expand → RoPE → (B,H,S,D)."""
        qkv = self.wqkv(F.rms_norm(x, (x.size(-1),)))
        xq, xk, xv = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        xq = E.rearrange(xq, "b s (h d) -> b s h d", d=self.head_dim)
        xk = E.rearrange(xk, "b s (h d) -> b s h d", d=self.head_dim)
        xv = E.rearrange(xv, "b s (h d) -> b s h d", d=self.head_dim)
        xq = F.rms_norm(xq, (xq.size(-1),))
        xk = F.rms_norm(xk, (xk.size(-1),))
        xk = repeat_kv(xk, n_rep=self.n_rep)
        xv = repeat_kv(xv, n_rep=self.n_rep)
        xq, xk = apply_3d_rotary_emb(xq, xk, freqs_cis, freqs_cis_2d)
        xq = xq.permute(0, 2, 1, 3)  # (B,S,H,D) → (B,H,S,D)
        xk = xk.permute(0, 2, 1, 3)
        xv = xv.permute(0, 2, 1, 3)
        return xq, xk, xv

    def _post_attention(self, output: T, lse: T) -> T:
        """Post-attention: sigmoid(lse - sinks) * output → transpose to (B,S,HD)."""
        sinks_BHS = self.sinks.view(1, -1, 1)          # (1, H, 1)
        sink_scale = torch.sigmoid(lse - sinks_BHS)     # (B, H, S)
        output = (output * sink_scale.unsqueeze(-1)).to(output.dtype)  # (B, H, S, D)
        output = output.permute(0, 2, 1, 3).contiguous().flatten(2)     # (B, S, H*D)
        return self.wo(output)

    def compile(self, *, dynamic: bool = True, mode: str = "default"):
        """Compile the pure pointwise substeps of attention (not KV cache or FlexAttention).

        Uses ``dynamic=True`` so that varying prefill sequence lengths
        do not trigger recompilations (one symbolic graph handles all sizes).
        """
        self._pre_attention = torch.compile(self._pre_attention, dynamic=dynamic, mode=mode)
        self._post_attention = torch.compile(self._post_attention, dynamic=dynamic, mode=mode)

    def forward(
        self,
        x: T,
        attention_masks: BlockMask,
        kv_cache: KVCacheBase,
        freqs_cis: T,
        freqs_cis_2d: T | None = None,
        input_pos: T | None = None,
        batch_idx: T | None = None,
        flex_attn_kernel_options=None,
    ):
        xq, xk, xv = self._pre_attention(x, freqs_cis, freqs_cis_2d)
        xk, xv = kv_cache.insert_kv(
            self.layer_id, xk, xv, input_pos=input_pos, batch_idx=batch_idx
        )
        # Decode (S_q == 1): static-shape compiled, CUDA-graph-safe.
        # Prefill (S_q > 1): dynamic-shape compiled, no recompilations.
        flex_fn = compiled_flex_attn_decode if xq.shape[2] == 1 else compiled_flex_attn_prefill
        output, aux_output = flex_fn(
            xq, xk, xv,
            block_mask=attention_masks,
            return_aux=AuxRequest(lse=True),
            kernel_options=flex_attn_kernel_options,
        )
        output = self._post_attention(output, aux_output.lse)
        return output


# FeedForward
@triton.jit
def _squared_relu_gate_kernel(
    packed_ptr, out_ptr,
    n_rows, n_cols,
    in_row_stride, in_col_stride,
    out_row_stride, out_col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_elements = n_rows * n_cols
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    rows = offsets // n_cols
    cols = offsets % n_cols

    gate_idx = rows * in_row_stride + (2 * cols) * in_col_stride
    up_idx = rows * in_row_stride + (2 * cols + 1) * in_col_stride
    out_idx = rows * out_row_stride + cols * out_col_stride

    gate = tl.load(packed_ptr + gate_idx, mask=mask)
    up = tl.load(packed_ptr + up_idx, mask=mask)
    gate = tl.where(gate > 0, gate, 0.0)
    out = gate * gate * up
    tl.store(out_ptr + out_idx, out, mask=mask)


def squared_relu_gate(packed: T, hidden_dim: int) -> T:
    """Fused relu(gate)^2 * up for interleaved packed [g0,u0,g1,u1,...] input."""
    packed_2d = packed.flatten(0, -2)
    n_rows = packed_2d.shape[0]
    n_cols = hidden_dim
    out_2d = torch.empty((n_rows, n_cols), device=packed.device, dtype=packed.dtype)
    n = n_rows * n_cols
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _squared_relu_gate_kernel[grid](
        packed_2d, out_2d,
        n_rows, n_cols,
        packed_2d.stride(0), packed_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        BLOCK_SIZE=1024,
    )
    return out_2d.view(*packed.shape[:-1], hidden_dim)


class FeedForward(nn.Module):
    """Squared-ReLU gated FFN with pre-fused gate+up projection.

    ``w13`` stores the gate (w1) and up (w3) weights interleaved as
    ``[g0, u0, g1, u1, ...]``.  The gate weights include a pre-baked
    ``sqrt(2)`` scaling factor, so inference needs only one matmul.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.rms_norm(x, (x.size(-1),))
        # equal to w13(x) = F.relu(w1(x)).square() * w3(x)
        w13_out = self.w13(x)  # (B, S, 2*hidden_dim)
        return self.w2(squared_relu_gate(w13_out, self.hidden_dim))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        model_args: ModelArgs,
    ):
        super().__init__()
        self.attention = Attention(model_args, layer_id)
        self.feed_forward = FeedForward(model_args.dim, model_args.ffn_dim)

    def compile(self, *, dynamic: bool = True, mode: str = "default"):
        """
        Compile the block's *FFN substep* and *Attention substeps* (pure functions).

        ``dynamic=True`` avoids recompilations when the prefill sequence
        length varies across requests.
        """
        self.feed_forward = torch.compile(self.feed_forward, dynamic=dynamic, mode=mode)
        self.attention.compile(dynamic=dynamic, mode=mode)
        return self

    def forward(
        self,
        x: T,
        kv_cache: KVCacheBase,
        freqs_cis: T,
        freqs_cis_2d: T | None = None,
        attention_masks=None,
        input_pos: T | None = None,
        batch_idx: T | None = None,
        flex_attn_kernel_options=None,
    ):
        B, S, D = x.shape
        x = x + self.attention(
            x,
            attention_masks=attention_masks,
            kv_cache=kv_cache,
            freqs_cis=freqs_cis,
            freqs_cis_2d=freqs_cis_2d,
            input_pos=input_pos,
            batch_idx=batch_idx,
            flex_attn_kernel_options=flex_attn_kernel_options,
        )
        out = x + self.feed_forward(x)
        return out.reshape(B, S, D)


class ImgScatterEntry(NamedTuple):
    """Metadata for placing one image's projected features into token embeddings.

    Pre-computed on CPU so that ``_scatter_img_tokens_with_projector`` can use
    pure slice indexing — no boolean index / masked_scatter, hence no
    ``aten::nonzero`` and no ``cudaStreamSynchronize``.
    """
    batch_idx: int        # Index into the batch dimension of h_BSD (0 for packed prefill).
    token_start: int      # Start position of img_id tokens in the token sequence.
    n_tokens: int         # Number of contiguous img_id placeholder tokens.
    h_valid_patches: int  # Unpadded image height in patch units (H_pixels // patch_size).
    w_valid_patches: int  # Unpadded image width in patch units  (W_pixels // patch_size).


class FalconPerception(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        img_in_dim = (
            args.temporal_patch_size * args.spatial_patch_size**2 * args.channel_size
        )
        self.img_projector = nn.Linear(img_in_dim, args.dim, bias=False)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, args)
        # LM head
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        if args.perception_heads:
            # BBox center head
            self.coord_encoder = FourierEncoder(2, args.coord_enc_dim, args.dim)
            self.coord_decoder = BboxDecoder(
                args.dim, args.coord_dec_dim, args.coord_out_dim
            )
            # Bbox size head
            self.size_encoder = FourierEncoder(2, args.size_enc_dim, args.dim)
            self.size_decoder = BboxDecoder(args.dim, args.size_dec_dim, args.size_out_dim)
            # Segmentation head
            self.itok_upsampler = AnyUp()
            self.proj_segm = SegmDecoder(args.dim, args.segm_out_dim, args.num_segm_layers)
            self.conv_segm = nn.Conv2d(
                args.dim, args.segm_out_dim, kernel_size=3, padding=1
            )

        # Rope
        rope_dim = self.args.head_dim // 2  # 1D+2D rope on each half of head_dim
        freqs_cis = precompute_freqs_cis(rope_dim, args.max_seq_len, args.rope_theta)
        freqs_cis_golden = torch.empty(
            (self.args.n_heads, rope_dim // 2, 2), dtype=torch.float,
        )  # To-be loaded from checkpoint
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.register_buffer("freqs_cis_golden", freqs_cis_golden, persistent=True)

    def compile(self, *, mode: str = "default"):
        """
        One-stop torch.compile configuration for inference.

        Uses mode="default" (not "reduce-overhead") because the decode loop
        is already captured by an external CUDA graph. "reduce-overhead" adds
        Inductor-level CUDA graphs that conflict with the outer capture.

        What we compile:
        - Pre-Attention and Post-Attention substeps
        - FFN block inside each TransformerBlock (prefill + decode). Safe & pure.
        - Segmentation upsampler (`AnyUp`) forward if segmentation is enabled.
        - Coord/size encoder & decoder heads.

        What we do NOT compile:
        - KV cache insertion (mutation -- not functionalization-safe).
        - FlexAttention (module-level compiled_flex_attn_decode / compiled_flex_attn_prefill).
        """
        print("Piecewise torch.compile of FalconPerception...")
        for layer in self.layers.values():
            if hasattr(layer, "compile"):
                layer.compile(dynamic=True, mode=mode)

        if self.args.perception_heads:
            self.coord_encoder = torch.compile(self.coord_encoder, dynamic=True, mode=mode)
            self.coord_decoder = torch.compile(self.coord_decoder, dynamic=True, mode=mode)
            self.size_encoder = torch.compile(self.size_encoder, dynamic=True, mode=mode)
            self.size_decoder = torch.compile(self.size_decoder, dynamic=True, mode=mode)
            self.itok_upsampler.compile(mode=mode, dynamic=True)
        return self

    @property
    def device(self):
        return self.tok_embeddings.weight.device

    @property
    def dtype(self):
        return self.tok_embeddings.weight.dtype

    def to(self, *args, **kwargs):
        # RoPE buffers are complex64; casting to a real dtype like bfloat16 is
        # unsupported.  Temporarily detach them, cast everything else, then
        # re-register on the target device only.
        freqs_cis = self.freqs_cis
        freqs_cis_golden = self.freqs_cis_golden
        del self.freqs_cis
        del self.freqs_cis_golden
        super().to(*args, **kwargs)
        device = self.tok_embeddings.weight.device
        self.register_buffer("freqs_cis", freqs_cis.to(device=device), persistent=False)
        self.register_buffer("freqs_cis_golden", freqs_cis_golden.to(device=device), persistent=True)
        build_upsampler_block_mask.cache_clear()
        return self


    def _scatter_img_tokens_with_projector(
        self,
        h_BSD,
        pixel_values_NTHWC: list[T] | T,
        img_scatter_info: list[ImgScatterEntry],
    ):
        """Replace img_id placeholder embeddings with projected visual features.

        ``pixel_values_NTHWC`` may be a list of native resolution for paged path 
        or a batch of N per-image padded resolution

        Uses slice indexing only — no boolean index / masked_scatter, so no
        aten::nonzero and no cudaStreamSynchronize.
        """
        pt = self.args.temporal_patch_size
        ps = self.args.spatial_patch_size

        valid_parts: list[T] = []
        for img, (_, _, _, h_v, w_v) in zip(pixel_values_NTHWC, img_scatter_info):
            img = img.to(self.dtype)
            patches = E.rearrange(
                img,
                "(t pt) (h ph) (w pw) c -> (t h w) (pt ph pw c)",
                pt=pt, ph=ps, pw=ps,
            )
            T_i, H_i, W_i, _ = img.shape
            grid = patches.reshape(T_i // pt, H_i // ps, W_i // ps, patches.shape[-1])
            valid_parts.append(grid[:, :h_v, :w_v, :].reshape(-1, patches.shape[-1]))
        valid_patches = torch.cat(valid_parts, dim=0)
        valid_feats = self.img_projector(valid_patches)

        offset = 0
        for b, tok_start, n_tok, _, _ in img_scatter_info:
            h_BSD[b, tok_start : tok_start + n_tok, :] = valid_feats[offset : offset + n_tok]
            offset += n_tok
        return h_BSD

    def _extract_coords(self, coords_BO: list[list]):
        all_xy, all_hw = [], []
        for coords_O in coords_BO:  # loop over batch
            if not coords_O:  # Empty sample
                continue
            for coords in coords_O:  # loop over objects
                for k, v in coords.items():
                    if k.startswith(("x", "y")):
                        all_xy.append(v)
                    elif k.startswith(("h", "w")):
                        all_hw.append(v)
        all_xy = torch.tensor(all_xy)
        all_hw = torch.tensor(all_hw)
        return all_xy, all_hw

    def _encode_coords(self, h_BSD: T, tokens_BS: T, all_xy: T):
        coord_tokens_mask = tokens_BS == self.args.coord_token_id
        if all_xy.numel() == 0:
            return h_BSD
        coord_tokens = self.coord_encoder(all_xy.reshape(-1, 2))
        if coord_tokens.shape[0] == h_BSD.shape[0]:
            # Per-element (decode): coord_tokens[b] corresponds to h_BSD[b].
            # torch.where maps each batch position to its own encoded value.
            h_BSD = torch.where(
                coord_tokens_mask.unsqueeze(-1),
                coord_tokens.view(h_BSD.shape[0], -1, h_BSD.shape[-1]),
                h_BSD,
            )
        else:
            # Packed (prefill): coord_tokens are dense for the K <coord> positions.
            h_BSD = h_BSD.masked_scatter_(coord_tokens_mask.unsqueeze(-1), coord_tokens)
        return h_BSD

    def _encode_sizes(self, h_BSD, tokens_BS, all_hw: T):
        size_tokens_mask = tokens_BS == self.args.size_token_id
        if all_hw.numel() == 0:
            return h_BSD
        size_tokens = self.size_encoder(all_hw.reshape(-1, 2))
        if size_tokens.shape[0] == h_BSD.shape[0]:
            # Per-element (decode): size_tokens[b] corresponds to h_BSD[b].
            h_BSD = torch.where(
                size_tokens_mask.unsqueeze(-1),
                size_tokens.view(h_BSD.shape[0], -1, h_BSD.shape[-1]),
                h_BSD,
            )
        else:
            # Packed (prefill): size_tokens are dense for the K <size> positions.
            h_BSD = h_BSD.masked_scatter_(size_tokens_mask.unsqueeze(-1), size_tokens)
        return h_BSD

    def decode_coords(self, h_BSD, labels):
        B, S, D = h_BSD.shape
        coord_masks = labels == self.args.coord_token_id
        coord_tokens = torch.masked_select(h_BSD, coord_masks.unsqueeze(-1))
        coord_logits = self.coord_decoder(coord_tokens.reshape(-1, D))
        return E.rearrange(coord_logits, "b (two dim) -> b two dim", two=2)

    def decode_sizes(self, h_BSD, labels):
        B, S, D = h_BSD.shape
        size_masks = labels == self.args.size_token_id
        size_tokens = torch.masked_select(h_BSD, size_masks.unsqueeze(-1))
        size_logits = self.size_decoder(size_tokens.reshape(-1, D))
        return E.rearrange(size_logits, "b (two dim) -> b two dim", two=2)

    def gather_img_tokens(
        self,
        h_BSD: T,
        tokens_BS: T,
        itok_masks_NTHW: T,
    ):
        B, S, D = h_BSD.shape

        itok_masks_BSD = E.repeat(tokens_BS == self.args.img_id, "b s -> b s d", d=D)
        itok_flatten = torch.masked_select(h_BSD, itok_masks_BSD)

        itok_masks_NTHWD = E.repeat(itok_masks_NTHW, "n t h w -> n t h w d", d=D)
        # Pre-allocate output tensors
        itok_NTHWD = torch.zeros_like(
            itok_masks_NTHWD, dtype=h_BSD.dtype, device=h_BSD.device
        )
        itok_NTHWD = itok_NTHWD.masked_scatter_(itok_masks_NTHWD, itok_flatten)
        return itok_NTHWD

    def process_sizes(self, logits):
        # log2 pred
        num_bins = logits.shape[-1]
        pred = torch.argmax(logits, dim=-1).float() / (num_bins - 1)
        min_size = torch.log2(torch.tensor(1 / num_bins))
        max_size = 0.0  # log2(1)
        pred = pred * (max_size - min_size) + min_size
        return torch.pow(2.0, pred)

    def upsample_single_img_features(
        self,
        h_SD: T,
        pixel_values_THWC: T,
        img_token_start: int,
        h_valid: int,
        w_valid: int,
        output_size: tuple[int, int] | None = None,
    ) -> T:
        """Upsample image features for a single image.

        Args:
            h_SD: hidden states for one sequence, shape ``(S, D)``.
            pixel_values_THWC: pixel values, shape ``(T, H, W, C)`` (T=1).
            img_token_start: start index of image tokens in ``h_SD``.
            h_valid, w_valid: valid patch grid size (before padding).
            output_size: ``(out_H, out_W)``.  *None* = full pixel resolution.
                ``(h_patch, w_patch)`` skips AnyUp entirely.
        Returns: ``(D, H, W)`` high-res features.
        """
        ps = self.args.spatial_patch_size
        _, H, W, _ = pixel_values_THWC.shape
        h_patch, w_patch = H // ps, W // ps
        image = E.rearrange(pixel_values_THWC, "1 h w c -> 1 c h w")

        D = h_SD.shape[-1]
        n_img = h_valid * w_valid
        img_feats = h_SD[img_token_start : img_token_start + n_img, :]
        lr_img_features = img_feats.reshape(1, h_valid, w_valid, D)
        if h_valid < h_patch or w_valid < w_patch:
            lr_img_features = F.pad(
                lr_img_features,
                (0, 0, 0, w_patch - w_valid, 0, h_patch - h_valid),
            )
        lr_img_features = lr_img_features.permute(0, 3, 1, 2)  # (1, D, h, w)
        lr_img_features = self.conv_segm(lr_img_features)

        if output_size == (h_patch, w_patch):
            return lr_img_features[0]
        if output_size is None:
            output_size = (H, W)

        out_H, out_W = output_size
        upsampler_attn_mask = build_upsampler_block_mask(
            out_H, out_W, h_patch, w_patch, device=image.device,
        )
        hr_img_features = self.itok_upsampler(
            images=image,
            features=lr_img_features,
            attn_mask=upsampler_attn_mask,
            output_size=output_size,
        )
        return hr_img_features[0] # (D, out_H, out_W)

    def upsample_img_features(
        self,
        h_BSD: T,
        pixel_values_NTHWC: list[T] | T,
        img_scatter_info: list[ImgScatterEntry],
        output_size: tuple[int, int] | None = None,
    ) -> T:
        """Upsample per-image, then stack.  Keeps peak memory bounded.

        ``pixel_values_NTHWC`` may be a list of N per-image ``(T, H_i, W_i, C)``
        tensors (paged path) or a single batched ``(N, T, H, W, C)`` tensor
        (batch inference path).  Both yield ``(T, H, W, C)`` when indexed.

        Returns: ``(N, segm_out_dim, out_H, out_W)`` high-res features.
        """
        hr_parts: list[T] = []
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
        return torch.stack(hr_parts, dim=0)

    def forward(
        self,
        tokens,
        attention_mask: BlockMask,  # Only support FlexAttention
        kv_cache: KVCacheBase,
        rope_pos_t: T | None = None,
        rope_pos_hw: T | None = None,
        pixel_values: list[T] | T | None = None,  # list (paged) or batched Tensor (batch)
        coord_xy: T | None = None,  # Coord values (flat or (B,2)); empty tensor = no-op
        size_hw: T | None = None,   # Size values (flat or (B,2)); empty tensor = no-op
        # Paged attention metadata (optional; when provided, we treat this as the paged path)
        input_pos: T | None = None,
        batch_idx: T | None = None,
        # Misc
        flex_attn_kernel_options=None,
        img_scatter_info: list[ImgScatterEntry] | None = None,
    ):
        B, S = tokens.size()
        block_mask = attention_mask  # keep ref to original attention_mask
        is_paged = (input_pos is not None) or (batch_idx is not None)

        if is_paged:
            # NOTE: Paged path: caller provides exact RoPE positions + an already-correct block mask,
            # based on prefill or decode step scheduled by the Paged Engine
            assert input_pos is not None and batch_idx is not None, (
                "Paged Attention requires both input_pos and batch_idx (got only one)."
            )
            assert rope_pos_t is not None, "Paged Attention requires explicit pos_t"
            assert input_pos is not None and batch_idx is not None  # for type narrowing
            freqs_cis = self.freqs_cis[rope_pos_t]
            freqs_cis_golden = None  # decode, no image
            if rope_pos_hw is not None:  # prefill  with image
                freqs_cis_golden = apply_golden_freqs_cis_to_visual_pos(
                    self.freqs_cis_golden, rope_pos_hw
                )
        else:
            # NOTE: Standard batching path: KV cache manages physical position & incremental RoPE.
            assert input_pos is None and batch_idx is None, (
                "Batch Inference does not accept input_pos/batch_idx."
            )
            T_pos = kv_cache.get_pos()  # Physical pos in KV cache / mask space, start at 0
            is_prefill = S != 1
            if is_prefill:
                assert rope_pos_t is not None and rope_pos_hw is not None
                # Index freqs_cis by logical pos_t (adjust for left padding via cache pos).
                pos_t = rope_pos_t[:, T_pos : T_pos + S].long()
                kv_cache.pos_t = pos_t[:, -1:]  # Cache last position per batch
                freqs_cis = self.freqs_cis[pos_t]
                rope_pos_hw = rope_pos_hw[:, T_pos : T_pos + S]
                freqs_cis_golden = apply_golden_freqs_cis_to_visual_pos(
                    self.freqs_cis_golden, rope_pos_hw
                )
                block_mask.seq_lengths = (S, S)  # adjust mask to prompt length
            else:
                pos_t = kv_cache.increment_and_get_pos_t()
                freqs_cis = self.freqs_cis[pos_t]
                freqs_cis_golden = None  # No image in decoding
                # Decoding: slice to current query block and offset logical mask_mod by cache pos.
                block_idx = T_pos // block_mask.BLOCK_SIZE[0]
                block_mask = block_mask[:, :, block_idx]
                block_mask.seq_lengths = (S, T_pos + S)
                block_mask.mask_mod = offset_mask_mod(attention_mask.mask_mod, offset=T_pos)

        h_BSD = self.tok_embeddings(tokens)

        if self.args.perception_heads:
            # Encode coord/size special tokens (no-op when tensors are empty).
            coord_xy = coord_xy if coord_xy is not None else h_BSD.new_empty(0)
            size_hw = size_hw if size_hw is not None else h_BSD.new_empty(0)
            h_BSD = self._encode_coords(h_BSD, tokens, coord_xy)
            h_BSD = self._encode_sizes(h_BSD, tokens, size_hw)

        if pixel_values is not None:
            assert img_scatter_info is not None, (
                "img_scatter_info is required when pixel_values is provided"
            )
            h_BSD = self._scatter_img_tokens_with_projector(
                h_BSD, pixel_values, img_scatter_info,
            )

        for i, layer in enumerate(self.layers.values()):
            h_BSD = layer(
                h_BSD,
                attention_masks=block_mask,
                kv_cache=kv_cache,
                freqs_cis=freqs_cis,
                freqs_cis_2d=freqs_cis_golden,
                input_pos=input_pos,
                batch_idx=batch_idx,
                flex_attn_kernel_options=flex_attn_kernel_options,
            )

        h_BSD = self.norm(h_BSD)
        logits_BSV = self.output(h_BSD)
        return logits_BSV, h_BSD

    def sample_bbox(self, h_BD: T, tokens_B: T):
        """Sync-free bbox decoding.  Returns GPU tensors, never calls .item().

        Runs coord_decoder / size_decoder on every batch element when
        perception_heads are present. 

        When perception_heads is False (e.g. OCR mode), returns zero tensors
        and always-False masks.

        Inputs
        ------
        h_BD      : (B, D=1024)   model-dtype GPU  — last hidden state
        tokens_B  : (B,)     int64 GPU         — sampled token ids

        Outputs (all GPU, no sync)
        -------
        xy_B2      : (B, 2)  float32  — predicted (x, y) normalised coords
        hw_B2      : (B, 2)  float32  — predicted (h, w) normalised sizes
        is_coord_B : (B,)    bool     — True where token == <coord>
        is_size_B  : (B,)    bool     — True where token == <size>
        """
        B = h_BD.shape[0]

        if not self.args.perception_heads:
            zeros = torch.zeros(B, 2, device=h_BD.device, dtype=torch.float32)
            false = torch.zeros(B, device=h_BD.device, dtype=torch.bool)
            return zeros, zeros, false, false, zeros

        # Bool masks — fixed shape, no sync
        is_coord_B = tokens_B == self.args.coord_token_id   # (B,) bool
        is_size_B  = tokens_B == self.args.size_token_id    # (B,) bool

        # Coord: h_BD (B,D=1024) → coord_decoder → (B,4096) → view (B,2,2048)
        coord_logits = self.coord_decoder(h_BD).view(B, 2, -1)
        num_bins = coord_logits.size(-1)                                    # 2048
        xy_B2 = torch.argmax(coord_logits, dim=-1).float() / num_bins      # (B,2)

        # Size: h_BD (B,D=1024) → size_decoder → (B,4096) → view (B,2,2048)
        size_logits = self.size_decoder(h_BD).view(B, 2, -1)
        hw_B2 = self.process_sizes(size_logits)                             # (B,2)

        return xy_B2, hw_B2, is_coord_B, is_size_B, coord_logits

    @staticmethod
    def dedup_single_coord(
        xy_2: T,
        is_coord: T,
        all_xy_S2: T,
        is_coord_mask_S: T,
        coord_logits_2N: T,
        threshold: float = 0.01,
        max_attempts: int = 10,
    ) -> None:
        """Replace a duplicate coordinate prediction for one sample in-place.

        Quick-exits (one small GPU sync) when the current prediction is not
        a duplicate.  Only enters the retry loop in the rare case that dedup
        is actually needed.

        Uses the same iterative bin-masking strategy as the original
        ``dedup_coords``: mask one x-bin and one y-bin per retry, then
        re-argmax.  This keeps replacements close to the model's best
        prediction.  If no non-duplicate is found within ``max_attempts``,
        the original prediction is kept (no update).

        Args:
            xy_2: (2,) float32 predicted normalised coords — modified in-place.
            is_coord: scalar bool GPU — True if the current token is ``<coord>``.
            all_xy_S2: (S, 2) all past xy predictions (unfiltered).
            is_coord_mask_S: (S,) bool mask — True where the step was a coord.
            coord_logits_2N: (2, N) pre-computed coord logits from sample_bbox.
            threshold: Normalised distance below which two coords are duplicates.
            max_attempts: Maximum retries before accepting the duplicate.
        """
        # Early exit: check if current prediction is a dup (tiny GPU op + sync).
        diffs = (all_xy_S2 - xy_2.unsqueeze(0)).abs()       # (S, 2)
        is_close = (diffs.amax(dim=-1) < threshold) & is_coord_mask_S  # (S,)
        if not is_close.any():
            return

        num_bins = coord_logits_2N.size(-1)
        logits = coord_logits_2N.clone()  # (2, N) — will be mutated

        for _ in range(max_attempts):
            pred_bins = torch.argmax(logits, dim=-1)          # (2,)
            pred_xy = pred_bins.float() / num_bins            # (2,)

            diffs = (all_xy_S2 - pred_xy.unsqueeze(0)).abs()  # (S, 2)
            is_repeat = ((diffs.amax(dim=-1) < threshold) & is_coord_mask_S).any().item()

            if not is_repeat:
                xy_2.copy_(torch.where(is_coord, pred_xy, xy_2))
                return

            logits[0, pred_bins[0]] = float("-inf")
            logits[1, pred_bins[1]] = float("-inf")

    def get_segm_tokens(self, h_BD: T, tokens_B: T):
        """CUDA Sync-free segm projection.  Returns GPU tensors, never iterates.
        Runs proj_segm unconditionally (SegmDecoder: a few Linear layers,
        in_dim=1024 -> out_dim=256).
        """
        B = h_BD.shape[0]
        is_segm_B = tokens_B.view(B) == self.args.seg_token_id   # (B,) bool

        if self.args.perception_heads:
            segm_BD = self.proj_segm(h_BD)          # (B, 256)
        else:  # OCR mode
            segm_BD = torch.empty(B, 0, device=h_BD.device, dtype=h_BD.dtype)

        return segm_BD, is_segm_B



