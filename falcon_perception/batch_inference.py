# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

from __future__ import annotations

import torch
from torch import Tensor

from falcon_perception.attention import create_batch_attention_mask
from falcon_perception.aux_output import AuxOutput
from falcon_perception.data import ImageProcessor, load_images, tokenize_inputs, get_pos_thw, pad_sequences_left
from falcon_perception.kv_cache import KVCacheBase
from falcon_perception.model import FalconPerception, ImgScatterEntry
from falcon_perception.sampling import sample_token


def process_batch_and_generate(
    tokenizer,
    image_prompt_pairs,
    max_length,
    min_dimension,
    max_dimension,
    patch_size=16,
    merge_size=1,
):
    """Tokenize and pad a batch of (image, prompt) pairs.

    Uses left-padding so all sequences are right-aligned for KV-cache
    batch generation with causal models.

    Returns a dict with ``tokens``, ``pixel_values``, ``pixel_mask``,
    ``pos_t``, and ``pos_hw`` — ready to be passed to the batch engine.
    """
    # Process each image individually first
    all_input_ids = []
    all_selected_images = []

    processor_local = ImageProcessor(patch_size, merge_size)

    for img_path, prompt in image_prompt_pairs:
        images = load_images(
            [img_path],
            min_dimension=min_dimension,
            max_dimension=max_dimension,
        )

        images = processor_local.preprocess(images=images)

        input_ids, selected_images = tokenize_inputs(
            prompt, images, tokenizer, patch_size, merge_size, max_length
        )

        all_input_ids.append(input_ids)
        all_selected_images.extend(selected_images)

    # LEFT PADDING for KV-cache batch generation (numpy, then convert to torch)
    assert tokenizer.pad_token_id is not None, "tokenizer.pad_token_id must be set for batching"
    padded_np = pad_sequences_left(all_input_ids, tokenizer.pad_token_id)

    processed = processor_local.batch_images_with_mask(
        all_selected_images, max_dimension, max_dimension
    )
    assert processed is not None
    pos_t_np, pos_hw_np = get_pos_thw(
        padded_np,
        processed["padding_mask"],
        tokenizer,
        patch_size,
        pad_token_id=tokenizer.pad_token_id,
    )

    return {
        "tokens": torch.from_numpy(padded_np),
        "pixel_values": torch.from_numpy(processed["pixel_values"]),
        "pixel_mask": torch.from_numpy(processed["padding_mask"]),
        "pos_t": torch.from_numpy(pos_t_np),
        "pos_hw": torch.from_numpy(pos_hw_np),
    }

class KVCache(KVCacheBase):
    """
    KV Cache for efficient generation.

    Args:
        max_batch_size (int): Maximum batch size for the cache.
        max_seq_length (int): Maximum sequence length for the cache.
        n_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        dtype (torch.dtype): Data type for the cache tensors.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        num_layers: int,
    ):
        self.kv_shape = (
            num_layers,
            2,  # kv
            max_batch_size,
            n_heads,
            max_seq_length,
            head_dim,
        )
        self.kv_cache = None
        self.pos = 0  # Current position in the sequence in the cache
        self.pos_t: Tensor | None = None  # Current position for freqs_cis

    def reset(self):
        self.pos = 0
        self.pos_t = None

    def get_pos(self):
        return self.pos

    def set_pos_t(self, pos_t):
        self.pos_t = pos_t

    def increment_and_get_pos_t(self):
        assert self.pos_t is not None, "pos_t for rope is not initialized."
        self.pos_t += 1
        return self.pos_t

    def insert_kv(self, layer_id: int, k: Tensor, v: Tensor, **kwargs):
        del kwargs
        assert self.pos_t is not None, "pos_t for rope is not initialized."
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)

        # Insert new keys/values to the cache and return the full cache so far
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # Insert k, v into the cache
        self.kv_cache[layer_id, 0, :, :, t0:t1] = k
        self.kv_cache[layer_id, 1, :, :, t0:t1] = v
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_id, 0, :, :, :t1]
        value_view = self.kv_cache[layer_id, 1, :, :, :t1]
        # Increment pos after the last layer forward
        if layer_id == self.kv_cache.size(0) - 1:
            self.pos = t1

        return key_view, value_view


class BatchInferenceEngine:
    def __init__(self, model: FalconPerception, tokenizer, kernel_options: dict | None = None):
        self.model = model
        self.model_args = model.args
        self.tokenizer = tokenizer
        self.kernel_options = kernel_options or {}

    def pad_input_to_max_length(self, tokens_BS: Tensor, max_length: int):
        B, S = tokens_BS.size()
        padding_BL = torch.full(
            (B, max_length - S),
            self.tokenizer.pad_token_id,
            dtype=tokens_BS.dtype,
            device=tokens_BS.device,
        )
        return torch.cat([tokens_BS, padding_BL], dim=-1).contiguous()

    @torch.inference_mode()
    def generate(
        self,
        # inputs
        tokens: Tensor,
        pos_t: Tensor,
        pos_hw: Tensor,
        pixel_values: Tensor,
        pixel_mask: Tensor,
        coords: list | None = None,
        # Sampling params
        max_new_tokens: int = 100,
        temperature: float = 0.0,  # Deterministic by default
        top_k: int | None = None,
        block_size: int = 128,
        stop_token_ids: Tensor | list[int] | None = None,
        seed: int | None = None,
        coord_dedup_threshold: float = 0.01,
        task: str = "segmentation",
    ):
        device = tokens.device
        rng = torch.Generator(device).manual_seed(seed) if seed is not None else None
        B, L = tokens.size()  # batch x prompts' length
        # Round up max seqlen to multiple of block_size for better performance
        S = (L + max_new_tokens + block_size - 1) // block_size * block_size
        assert S <= self.model_args.max_seq_len, (
            f"max generation length: {S} > Model's MAX_SEQ_LEN: {self.model_args.max_seq_len}"
        )
        # KV cache is dynamically initialized and current position is updated automatically
        kv_cache = KVCache(
            max_batch_size=B,
            max_seq_length=S,
            n_heads=self.model_args.n_heads,
            head_dim=self.model_args.head_dim,
            num_layers=self.model_args.n_layers,
        )

        padded_tokens_BS = self.pad_input_to_max_length(tokens, max_length=S)
        attention_mask = create_batch_attention_mask(
            padded_tokens_BS,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            soi_token_id=self.tokenizer.image_cls_token_id,
            eoi_token_id=self.tokenizer.end_of_image_token_id,
            max_len=S,
        )

        # Extract tensor coords from dict format (empty tensors when no coords)
        all_xy, all_hw = self.model._extract_coords(coords or [[]])
        coord_xy = all_xy.to(device=device, dtype=self.model.dtype)
        size_hw = all_hw.to(device=device, dtype=self.model.dtype)

        # Pre-compute image scatter info on CPU so the model uses slice
        # indexing (no boolean mask / nonzero / GPU sync).
        img_scatter_info: list[ImgScatterEntry] = []
        ps = self.model_args.spatial_patch_size
        tokens_cpu = tokens if not tokens.is_cuda else tokens.cpu()
        pmask_cpu = pixel_mask if not pixel_mask.is_cuda else pixel_mask.cpu()
        for b in range(B):
            img_pos = (tokens_cpu[b] == self.model_args.img_id).nonzero(as_tuple=True)[0]
            if len(img_pos) > 0:
                mask_b = pmask_cpu[b]
                h_v = int(mask_b.sum(dim=-2).max()) // ps
                w_v = int(mask_b.sum(dim=-1).max()) // ps
                img_scatter_info.append(ImgScatterEntry(b, int(img_pos[0]), len(img_pos), h_v, w_v))

        # Prefill
        logits_BSV, h_BSD = self.model(
            tokens=tokens,  # Original sequence, no padding
            rope_pos_t=pos_t,
            rope_pos_hw=pos_hw,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            pixel_values=pixel_values,
            coord_xy=coord_xy,
            size_hw=size_hw,
            img_scatter_info=img_scatter_info or None,
            flex_attn_kernel_options=self.kernel_options or None,
        )

        hr_image_features = None
        if task == "segmentation":
            hr_image_features = self.model.upsample_img_features(
                h_BSD, pixel_values, img_scatter_info,
            )

        aux_outputs: list[AuxOutput] = [AuxOutput() for _ in range(B)]
        stop_token_ids = stop_token_ids or [self.tokenizer.eos_token_id]
        stop_ids = torch.tensor(stop_token_ids).to(device)
        should_stop_B = torch.full((B,), False, dtype=torch.bool, device=tokens.device)

        seg_token_id = self.tokenizer.seg_token_id

        while not torch.all(should_stop_B) and (pos := kv_cache.get_pos()) < S:
            tokens_B1, _, _ = sample_token(logits_BSV[:, -1], rng, temperature, top_k)
            if torch.any(should_stop_B):
                tokens_B1 = tokens_B1.clone()
                tokens_B1[should_stop_B, :] = self.tokenizer.pad_token_id
            padded_tokens_BS[:, pos] = tokens_B1[:, -1]

            h_last = h_BSD[:, -1, :]  # (B, D)
            xy_B2, hw_B2, is_coord_B, is_size_B, coord_logits = self.model.sample_bbox(h_last, tokens_B1.squeeze(-1))

            if coord_dedup_threshold > 0:
                for b in range(B):
                    raw = aux_outputs[b].coord_history_raw()
                    if raw is not None:
                        self.model.dedup_single_coord(
                            xy_B2[b], is_coord_B[b],
                            raw[0], raw[1], coord_logits[b],
                            threshold=coord_dedup_threshold,
                        )

            for b in range(B):
                aux_outputs[b].append_bbox(xy_B2[b], hw_B2[b], is_coord_B[b], is_size_B[b])

            sample_w_coord = torch.where(is_coord_B)[0]
            sample_w_size = torch.where(is_size_B)[0]
            xy_b2 = xy_B2[sample_w_coord] if sample_w_coord.numel() > 0 else xy_B2.new_empty(0, 2)
            hw_b2 = hw_B2[sample_w_size] if sample_w_size.numel() > 0 else hw_B2.new_empty(0, 2)

            # Project segmentation embeddings (skip when detection-only)
            if task == "segmentation":
                sample_w_segm = torch.where(tokens_B1 == seg_token_id)[0]
                if sample_w_segm.numel() > 0:
                    segm_embeds = self.model.proj_segm(h_BSD[sample_w_segm, -1, :])
                    for i, b in enumerate(sample_w_segm.tolist()):
                        aux_outputs[b].append_segm(segm_embeds[i])

            logits_BSV, h_BSD = self.model(
                tokens=tokens_B1,
                attention_mask=attention_mask,
                coord_xy=xy_b2.to(self.model.dtype),
                size_hw=hw_b2.to(self.model.dtype),
                kv_cache=kv_cache,
                flex_attn_kernel_options=self.kernel_options or None,
            )

            hit_stop_B = torch.isin(tokens_B1, stop_ids).any(dim=-1)
            should_stop_B = should_stop_B.logical_or(hit_stop_B)

        # Batch-finalize segmentation masks per image.
        # hr_image_features are at the padded canvas size (max_dim x max_dim).
        # Crop each to the actual image extent (remove batch padding) so masks
        # are produced at the model's processing resolution, not the padded size.
        for b in range(B):
            hr_feat_b = None
            if hr_image_features is not None:
                hr_feat_b = hr_image_features[b]  # (D, H_pad, W_pad)
                mask_b = pixel_mask[b, 0]        # (H_pad, W_pad)
                h_actual = int(mask_b.any(dim=1).sum())
                w_actual = int(mask_b.any(dim=0).sum())
                if h_actual > 0 and w_actual > 0:
                    hr_feat_b = hr_feat_b[:, :h_actual, :w_actual]
            aux_outputs[b].finalize(
                hr_image_features=hr_feat_b,
                task=task,
            )

        return padded_tokens_BS, aux_outputs
