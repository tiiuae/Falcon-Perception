# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Batch inference engine for MLX.

Mirrors ``falcon_perception/batch_inference.py`` using MLX ops.
Data preprocessing is framework-agnostic (numpy/PIL); conversion to
``mx.array`` happens at the boundary in this module.
"""

from __future__ import annotations

import os
import mlx.core as mx
import numpy as np

# Periodic cache clear: default to 50% of Metal recommended working set.
# Adapts automatically to machine size.  Override with FALCON_CACHE_LIMIT_GB.
def _get_cache_clear_threshold() -> int:
    override = os.environ.get("FALCON_CACHE_LIMIT_GB")
    if override is not None:
        return int(override) * 1024**3
    try:
        total = mx.device_info()["memory_size"]
    except Exception:
        total = 16 * 1024**3
    return min(int(total * 0.5), 16 * 1024**3)

_CACHE_CLEAR_BYTES = _get_cache_clear_threshold()
_CACHE_CHECK_INTERVAL = 10

from falcon_perception.mlx.attention import create_batch_attention_mask
from falcon_perception.mlx.kv_cache import KVCache
from falcon_perception.mlx.model import FalconPerception, ImgScatterEntry
from falcon_perception.mlx.sampling import sample_token
from falcon_perception.data import (
    ImageProcessor, load_images, tokenize_inputs, get_pos_thw, pad_sequences_left,
)


def _to_mx(x) -> mx.array:
    """Convert numpy array to mx.array."""
    if isinstance(x, mx.array):
        return x
    if isinstance(x, np.ndarray):
        return mx.array(x)
    return mx.array(np.asarray(x))


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

    Returns a dict of mx.arrays ready for the MLX batch engine.
    """
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
            prompt, images, tokenizer, patch_size, merge_size, max_length,
        )
        all_input_ids.append(input_ids)
        all_selected_images.extend(selected_images)

    padded_np = pad_sequences_left(all_input_ids, tokenizer.pad_token_id)

    processed = processor_local.batch_images_with_mask(
        all_selected_images, max_dimension, max_dimension,
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
        "tokens": _to_mx(padded_np),
        "pixel_values": _to_mx(processed["pixel_values"]),
        "pixel_mask": _to_mx(processed["padding_mask"]),
        "pos_t": _to_mx(pos_t_np),
        "pos_hw": _to_mx(pos_hw_np),
    }


def _dedup_single_coord(
    xy_B2,
    b: int,
    is_coord,
    all_xy_S2,
    is_coord_mask_S,
    coord_logits_2N,
    threshold: float = 0.01,
    max_attempts: int = 10,
):
    """Replace a duplicate coordinate prediction in-place (MLX version).

    Matches the iterative bin-masking strategy from the PyTorch
    ``FalconPerception.dedup_single_coord``.
    """
    if not bool(np.array(is_coord)):
        return

    xy_2 = xy_B2[b]
    diffs = mx.abs(all_xy_S2 - mx.expand_dims(xy_2, 0))
    is_close = (mx.max(diffs, axis=-1) < threshold) & is_coord_mask_S
    if not bool(mx.any(is_close).item()):
        return

    num_bins = coord_logits_2N.shape[-1]
    logits = mx.array(coord_logits_2N)  # copy

    for _ in range(max_attempts):
        pred_bins = mx.argmax(logits, axis=-1)
        pred_xy = pred_bins.astype(mx.float32) / num_bins

        diffs = mx.abs(all_xy_S2 - mx.expand_dims(pred_xy, 0))
        is_repeat = bool(mx.any((mx.max(diffs, axis=-1) < threshold) & is_coord_mask_S).item())

        if not is_repeat:
            xy_B2 = xy_B2.at[b].add(
                mx.where(is_coord, pred_xy, xy_B2[b]) - xy_B2[b]
            )
            return

        b0, b1 = int(pred_bins[0].item()), int(pred_bins[1].item())
        logits = logits.at[0, b0].add(float("-inf") - logits[0, b0])
        logits = logits.at[1, b1].add(float("-inf") - logits[1, b1])


class AuxOutput:
    """Lightweight aux output accumulator for MLX batch inference.

    Stores bbox predictions as numpy arrays (tiny: 2 floats each) to avoid
    holding MLX lazy-graph references that prevent memory reclamation.
    Segmentation embeddings stay as evaluated mx.arrays for ``mx.einsum``
    at finalization.
    """

    def __init__(self):
        self._coord_xy: list[np.ndarray] = []
        self._size_hw: list[np.ndarray] = []
        self._is_coord: list[bool] = []
        self._is_size: list[bool] = []
        self._segm_embeds: list[mx.array] = []
        self.bboxes_raw: list[dict] = []
        self.masks_rle: list[dict] = []
        # Incremental history for coord_history_raw (mirrors PyTorch pattern).
        self._xy_cat: np.ndarray | None = None
        self._is_coord_cat: np.ndarray | None = None

    def append_bbox(self, xy, hw, is_coord, is_size):
        xy_np = np.array(xy)
        self._coord_xy.append(xy_np)
        self._size_hw.append(np.array(hw))
        ic = bool(np.array(is_coord).item())
        self._is_coord.append(ic)
        self._is_size.append(bool(np.array(is_size).item()))
        # Incremental concat for coord_history_raw (avoids O(N²) re-stacking).
        xy_row = xy_np[np.newaxis, :]
        ic_row = np.array([ic])
        if self._xy_cat is None:
            self._xy_cat = xy_row
            self._is_coord_cat = ic_row
        else:
            self._xy_cat = np.concatenate([self._xy_cat, xy_row])
            self._is_coord_cat = np.concatenate([self._is_coord_cat, ic_row])

    def append_segm(self, embed):
        mx.eval(embed)
        self._segm_embeds.append(embed)

    def coord_history_raw(self):
        if self._xy_cat is None:
            return None
        return mx.array(self._xy_cat), mx.array(self._is_coord_cat)

    def materialize_bboxes(self) -> list[dict]:
        if not self._coord_xy:
            return self.bboxes_raw
        xy = np.stack(self._coord_xy)
        hw = np.stack(self._size_hw)
        is_coord = np.array(self._is_coord)
        is_size = np.array(self._is_size)

        result = []
        for i in range(len(xy)):
            if is_coord[i]:
                result.append({"x": float(xy[i, 0]), "y": float(xy[i, 1])})
            if is_size[i]:
                result.append({"h": float(hw[i, 0]), "w": float(hw[i, 1])})
        return result

    def finalize_masks(self, hr_image_features, threshold: float = 0.3):
        if not self._segm_embeds or hr_image_features is None:
            return []

        tokens = mx.stack(self._segm_embeds)
        masks = mx.einsum("dhw,kd->khw", hr_image_features, tokens)
        binary = mx.sigmoid(masks.astype(mx.float32)) > threshold

        # Convert to numpy for COCO RLE encoding
        binary_np = np.array(binary)
        from pycocotools import mask as mask_utils

        results = []
        for i in range(binary_np.shape[0]):
            if not binary_np[i].any():
                continue
            # pycocotools expects Fortran-order uint8
            rle = mask_utils.encode(np.asfortranarray(binary_np[i].astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            results.append(rle)
        return results

    def finalize(self, hr_image_features=None, threshold=0.3, task="segmentation"):
        self.bboxes_raw = self.materialize_bboxes()
        if task == "segmentation":
            self.masks_rle = self.finalize_masks(hr_image_features, threshold)


class BatchInferenceEngine:
    def __init__(self, model: FalconPerception, tokenizer):
        self.model = model
        self.model_args = model.args
        self.tokenizer = tokenizer

    def pad_input_to_max_length(self, tokens_BS, max_length: int):
        B, S = tokens_BS.shape
        padding = mx.full(
            (B, max_length - S),
            self.tokenizer.pad_token_id,
            dtype=tokens_BS.dtype,
        )
        return mx.concatenate([tokens_BS, padding], axis=-1)

    def generate(
        self,
        tokens,
        pos_t,
        pos_hw,
        pixel_values,
        pixel_mask,
        coords=None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_k: int | None = None,
        block_size: int = 128,
        stop_token_ids=None,
        seed: int | None = None,
        coord_dedup_threshold: float = 0.01,
        task: str = "segmentation",
    ):
        if seed is not None:
            mx.random.seed(seed)

        B, L = tokens.shape
        S = (L + max_new_tokens + block_size - 1) // block_size * block_size
        assert S <= self.model_args.max_seq_len, (
            f"max generation length: {S} > Model's MAX_SEQ_LEN: {self.model_args.max_seq_len}"
        )

        kv_cache = KVCache(
            max_batch_size=B,
            max_seq_length=S,
            n_heads=self.model_args.n_heads,
            head_dim=self.model_args.head_dim,
            num_layers=self.model_args.n_layers,
            dtype=self.model.dtype,
        )

        padded_tokens_BS = self.pad_input_to_max_length(tokens, max_length=S).astype(mx.int32)
        attention_mask = create_batch_attention_mask(
            padded_tokens_BS,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            soi_token_id=self.tokenizer.image_cls_token_id,
            eoi_token_id=self.tokenizer.end_of_image_token_id,
            max_len=S,
        )

        all_xy, all_hw = self.model._extract_coords(coords or [[]])
        coord_xy = all_xy.astype(self.model.dtype)
        size_hw = all_hw.astype(self.model.dtype)

        # Pre-compute image scatter info
        img_scatter_info: list[ImgScatterEntry] = []
        ps = self.model_args.spatial_patch_size
        tokens_np = np.array(tokens)
        pmask_np = np.array(pixel_mask)
        for b in range(B):
            img_pos = np.where(tokens_np[b] == self.model_args.img_id)[0]
            if len(img_pos) > 0:
                mask_b = pmask_np[b]
                h_v = int(mask_b.sum(axis=-2).max()) // ps
                w_v = int(mask_b.sum(axis=-1).max()) // ps
                img_scatter_info.append(
                    ImgScatterEntry(b, int(img_pos[0]), len(img_pos), h_v, w_v)
                )

        # Prefill
        logits_BSV, h_BSD = self.model(
            tokens=tokens,
            rope_pos_t=pos_t,
            rope_pos_hw=pos_hw,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            pixel_values=pixel_values,
            coord_xy=coord_xy,
            size_hw=size_hw,
            img_scatter_info=img_scatter_info or None,
        )
        mx.eval(logits_BSV, h_BSD)

        hr_image_features = None
        if task == "segmentation" and img_scatter_info and self.model_args.perception_heads:
            hr_image_features = self.model.upsample_img_features(
                h_BSD, pixel_values, img_scatter_info,
            )

        aux_outputs = [AuxOutput() for _ in range(B)]
        stop_token_ids = stop_token_ids or [self.tokenizer.eos_token_id]
        stop_ids_set = set(stop_token_ids)
        should_stop = [False] * B
        has_perception = self.model_args.perception_heads

        seg_token_id = self.tokenizer.seg_token_id
        pad_id = self.tokenizer.pad_token_id
        generated_tokens: list[list[int]] = [[] for _ in range(B)]

        _decode_step = 0
        while not all(should_stop) and (pos := kv_cache.get_pos()) < S:
            _decode_step += 1
            tokens_B1, _, _ = sample_token(logits_BSV[:, -1], temperature=temperature, top_k=top_k)
            tokens_B1 = tokens_B1.astype(mx.int32)
            mx.eval(tokens_B1)
            tokens_flat = tokens_B1[:, 0].tolist()

            for b in range(B):
                if should_stop[b]:
                    tokens_flat[b] = pad_id
                generated_tokens[b].append(tokens_flat[b])

            if any(should_stop):
                tokens_B1 = mx.array([[t] for t in tokens_flat], dtype=mx.int32)

            if has_perception:
                h_last = h_BSD[:, -1, :]
                xy_B2, hw_B2, is_coord_B, is_size_B, coord_logits = self.model.sample_bbox(
                    h_last, tokens_B1.squeeze(-1),
                )
                # Materialize bbox predictions so stored slices don't hold
                # the full model computation graph alive in MLX's lazy evaluator.
                mx.eval(xy_B2, hw_B2, is_coord_B, is_size_B, coord_logits)

                if coord_dedup_threshold > 0:
                    for b in range(B):
                        raw = aux_outputs[b].coord_history_raw()
                        if raw is not None:
                            _dedup_single_coord(
                                xy_B2, b, is_coord_B[b],
                                raw[0], raw[1], coord_logits[b],
                                threshold=coord_dedup_threshold,
                            )

                for b in range(B):
                    aux_outputs[b].append_bbox(xy_B2[b], hw_B2[b], is_coord_B[b], is_size_B[b])

                is_coord_np = np.array(is_coord_B)
                is_size_np = np.array(is_size_B)
                sample_w_coord = np.where(is_coord_np)[0]
                sample_w_size = np.where(is_size_np)[0]
                xy_b2 = xy_B2[mx.array(sample_w_coord)] if len(sample_w_coord) > 0 else mx.zeros((0, 2))
                hw_b2 = hw_B2[mx.array(sample_w_size)] if len(sample_w_size) > 0 else mx.zeros((0, 2))

                if task == "segmentation":
                    segm_indices = [b for b, t in enumerate(tokens_flat) if t == seg_token_id]
                    if segm_indices:
                        segm_h = h_BSD[mx.array(segm_indices), -1, :]
                        segm_embeds = self.model.proj_segm(segm_h)
                        mx.eval(segm_embeds)
                        for i, b in enumerate(segm_indices):
                            aux_outputs[b].append_segm(segm_embeds[i])
            else:
                xy_b2 = mx.zeros((0, 2))
                hw_b2 = mx.zeros((0, 2))

            logits_BSV, h_BSD = self.model(
                tokens=tokens_B1,
                attention_mask=attention_mask,
                coord_xy=xy_b2.astype(self.model.dtype),
                size_hw=hw_b2.astype(self.model.dtype),
                kv_cache=kv_cache,
            )

            mx.eval(logits_BSV, h_BSD)

            if (_decode_step % _CACHE_CHECK_INTERVAL == 0
                    and mx.get_cache_memory() > _CACHE_CLEAR_BYTES):
                mx.clear_cache()

            for b, t in enumerate(tokens_flat):
                if t in stop_ids_set:
                    should_stop[b] = True

        # Reconstruct padded_tokens_BS with generated tokens
        prefill_np = np.array(padded_tokens_BS)
        for b in range(B):
            gen = generated_tokens[b]
            prefill_np[b, L:L + len(gen)] = gen
        padded_tokens_BS = mx.array(prefill_np)

        # Finalize
        for b in range(B):
            hr_feat_b = None
            if hr_image_features is not None:
                hr_feat_b = hr_image_features[b]
                mask_b = pixel_mask[b, 0]
                h_actual = int(np.array(mx.any(mask_b, axis=1).astype(mx.int32).sum()))
                w_actual = int(np.array(mx.any(mask_b, axis=0).astype(mx.int32).sum()))
                if h_actual > 0 and w_actual > 0:
                    hr_feat_b = hr_feat_b[:, :h_actual, :w_actual]
            aux_outputs[b].finalize(
                hr_image_features=hr_feat_b,
                task=task,
            )

        return padded_tokens_BS, aux_outputs
