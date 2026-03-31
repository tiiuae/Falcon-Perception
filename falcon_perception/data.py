# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data loading and preprocessing utilities.

All functions operate on numpy arrays and PIL images. Framework-specific
tensor types (torch.Tensor, mx.array) are created at the call site boundary
— this module never imports torch or mlx.
"""

import io
import math

import einops as E
import numpy as np
import requests
from tqdm import tqdm
from PIL import Image

IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]


# ── Image I/O ──────────────────────────────────────────────────────────


def load_image(image) -> Image.Image | None:
    """Convert *image* (path, URL, PIL Image, ndarray, or bytes) to a PIL Image."""
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            response = requests.get(image, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        if image.endswith(".npy"):
            img_array = io.BytesIO(np.load(image))
            return Image.open(img_array)
        return Image.open(image)
    if isinstance(image, np.bytes_):
        return Image.open(io.BytesIO(image))
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    err = f"Unknown image format {image}"
    raise TypeError(err)


def load_images(
    images_input: list | None,
    min_dimension: int = 256,
    max_dimension: int = 1024,
) -> list[Image.Image]:
    """Load and resize a list of images."""
    if images_input is None:
        return []
    out: list[Image.Image] = []
    for inp in images_input:
        img = load_image(inp)
        img = resize_image_if_necessary(img, min_dimension, max_dimension)
        out.append(img)
    return out


def resize_image_if_necessary(
    image,
    min_dimension: int = 256,
    max_dimension: int = 1024,
):
    """Resize *image* so both sides are in [min_dimension, max_dimension].

    Aspect ratio is preserved.  If the image already fits, it is returned
    unchanged.
    """
    assert min_dimension <= max_dimension, (
        f"min_dimension ({min_dimension}) must be <= max_dimension ({max_dimension})"
    )
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if (
        min_dimension <= original_width <= max_dimension
        and min_dimension <= original_height <= max_dimension
    ):
        return image

    is_vertical_image = original_width < original_height
    if original_width < min_dimension or original_height < min_dimension:
        if is_vertical_image:
            new_width = min_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min_dimension
            new_width = int(new_height * aspect_ratio)
    else:
        if is_vertical_image:
            new_width = max_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_dimension
            new_width = int(new_height * aspect_ratio)

    if new_width > max_dimension:
        new_width = max_dimension
        new_height = int(new_width / aspect_ratio)
    if new_height > max_dimension:
        new_height = max_dimension
        new_width = int(new_height * aspect_ratio)

    return image.resize((new_width, new_height))


# ── Image processing (pure numpy/PIL) ─────────────────────────────────


def _convert_to_rgb(image):
    """Ensure a PIL image is in RGB mode."""
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
    return image


def _to_numpy_array(image) -> np.ndarray:
    """Convert PIL image or array to numpy (H, W, C) float or uint8."""
    if isinstance(image, Image.Image):
        return np.array(image)
    if isinstance(image, np.ndarray):
        return image
    return np.asarray(image)


def _get_image_size(image) -> tuple[int, int]:
    """Return (height, width) from a numpy array or PIL image.

    Handles 2D (H, W), 3D (H, W, C) / (C, H, W), and 4D (T, H, W, C) arrays.
    """
    if isinstance(image, Image.Image):
        w, h = image.size
        return h, w
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image.shape[0], image.shape[1]
        if image.ndim == 3:
            if image.shape[0] in (1, 3, 4) and image.shape[2] not in (1, 3, 4):
                return image.shape[1], image.shape[2]
            return image.shape[0], image.shape[1]
        if image.ndim >= 4:
            # (…, H, W, C) — channel-last with leading batch/temporal dims
            return image.shape[-3], image.shape[-2]
    raise ValueError(f"Cannot get size from {type(image)}")


def _infer_channel_dim(image: np.ndarray) -> str:
    """Return 'first' or 'last' for the channel dimension."""
    if image.ndim == 3:
        if image.shape[0] in (1, 3, 4) and image.shape[2] not in (1, 3, 4):
            return "first"
    return "last"


def _resize_image(image: np.ndarray, size: tuple[int, int], resample, input_data_format: str) -> np.ndarray:
    """Resize numpy image to (h, w) using PIL."""
    h_target, w_target = size
    if input_data_format == "first":
        # (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))

    pil_img = Image.fromarray(image.astype(np.uint8) if image.dtype != np.uint8 else image)
    pil_img = pil_img.resize((w_target, h_target), resample)
    result = np.array(pil_img)

    if input_data_format == "first":
        result = np.transpose(result, (2, 0, 1))
    return result


def _rescale(image: np.ndarray, scale: float) -> np.ndarray:
    """Rescale pixel values by a factor."""
    return image.astype(np.float32) * scale


def _normalize(image: np.ndarray, mean: list[float], std: list[float], input_data_format: str) -> np.ndarray:
    """Normalize an image with mean and std."""
    image = image.astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    if input_data_format == "first":
        # (C, H, W)
        mean = mean[:, None, None]
        std = std[:, None, None]
    else:
        # (H, W, C)
        mean = mean[None, None, :]
        std = std[None, None, :]
    return (image - mean) / std


def smart_resize(
    image,
    factor: int,
    resample,
    input_data_format,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
):
    height, width = _get_image_size(image)
    if height < factor or width < factor:
        err = f"{height=} or {width=} must be larger than {factor=}"
        raise ValueError(err)
    if max(height, width) / min(height, width) > 200:
        err = f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        raise ValueError(err)
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    image = _resize_image(image, (h_bar, w_bar), resample, input_data_format)
    return image


class ImageProcessor:
    def __init__(
        self,
        patch_size,
        merge_size,
        do_resize: bool = True,
        resample: Image.Resampling = Image.Resampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        **kwargs,
    ) -> None:
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or IMAGE_MEAN
        self.image_std = image_std or IMAGE_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.do_convert_rgb = do_convert_rgb

    def _preprocess(self, image, do_rescale=None, do_normalize=None):
        if self.do_convert_rgb:
            image = _convert_to_rgb(image)
        image = _to_numpy_array(image)

        input_data_format = _infer_channel_dim(image)
        if self.do_resize:
            image = smart_resize(
                image,
                factor=self.patch_size * self.merge_size,
                resample=self.resample,
                input_data_format=input_data_format,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        if do_rescale or self.do_rescale:
            image = _rescale(image, self.rescale_factor)
        if do_normalize or self.do_normalize:
            image = _normalize(image, self.image_mean, self.image_std, input_data_format)
        return image

    def preprocess(self, images=None, do_rescale=None, do_normalize=None, **kwargs):
        if images is None:
            return []
        images = [item for item in images if item is not None]
        pixel_values = []
        for image in images:
            processed_image = self._preprocess(image, do_rescale, do_normalize)
            processed_image = processed_image[None, ...]
            pixel_values.append(processed_image)
        return pixel_values

    def batch_images_with_mask(self, pixel_values, max_image_height, max_image_width):
        """Batch images into padded arrays with a boolean mask.

        Returns dict with numpy arrays: ``pixel_values`` (N,T,H,W,C) and
        ``padding_mask`` (N,T,H,W).
        """
        if pixel_values is None:
            return None
        pixel_values = [
            item for item in pixel_values if item is not None and len(item) != 0
        ]
        if len(pixel_values) == 0:
            return None

        N = len(pixel_values)
        max_temporal = max(img.shape[0] for img in pixel_values)
        C = pixel_values[0].shape[-1]

        batched = np.zeros(
            (N, max_temporal, max_image_height, max_image_width, C),
            dtype=np.float32,
        )
        masks = np.zeros(
            (N, max_temporal, max_image_height, max_image_width),
            dtype=bool,
        )
        for i, img in enumerate(pixel_values):
            img = np.asarray(img)
            t, h, w = img.shape[0], img.shape[1], img.shape[2]
            batched[i, :t, :h, :w, :] = img
            masks[i, :t, :h, :w] = True

        return {"pixel_values": batched, "padding_mask": masks}


# ── RoPE position computation (numpy) ──────────────────────────────────


def _compute_image_spatial_positions(
    pixel_mask_THW: np.ndarray,
    spatial_patch_size: int,
    temporal_patch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute flattened (h, w) spatial rope positions for one image.

    Returns:
        hpos: (num_patches,) float
        wpos: (num_patches,) float
    """
    mask_thw = E.reduce(
        pixel_mask_THW,
        "(t tp) (h hp) (w wp) -> t h w",
        reduction="any",
        tp=temporal_patch_size,
        hp=spatial_patch_size,
        wp=spatial_patch_size,
    )
    width = int(E.reduce(mask_thw.sum(axis=-1).astype(int), "t h -> ", reduction="max"))
    height = int(E.reduce(mask_thw.sum(axis=-2).astype(int), "t w -> ", reduction="max"))

    xlim = np.sqrt(width / height)
    ylim = np.sqrt(height / width)
    xpos = np.linspace(-xlim, xlim, width)
    ypos = np.linspace(-ylim, ylim, height)
    wpos, hpos = np.meshgrid(xpos, ypos, indexing="xy")
    return hpos.flatten(), wpos.flatten()


def _get_image_token_masks(tokens: np.ndarray, tokenizer):
    """Build masks for image-related tokens."""
    spatial_mask = tokens == tokenizer.image_token_id
    no_increase_mask = (
        spatial_mask
        | (tokens == tokenizer.image_reg_1_token_id)
        | (tokens == tokenizer.image_reg_2_token_id)
        | (tokens == tokenizer.image_reg_3_token_id)
        | (tokens == tokenizer.image_reg_4_token_id)
        | (tokens == tokenizer.end_of_image_token_id)
    )
    return spatial_mask, no_increase_mask


def get_pos_thw_single(
    tokens: np.ndarray,
    pixel_mask_THW: np.ndarray | None,
    tokenizer,
    spatial_patch_size: int,
    temporal_patch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rope positions for a single unpadded sequence.

    Returns:
        tpos:   (S,) int64   — temporal positions
        hw_pos: (S, 2) float — spatial (h, w); 0 for non-image tokens
    """
    S = tokens.shape[0]
    spatial_img_mask, no_increase_mask = _get_image_token_masks(tokens, tokenizer)

    hpos = np.zeros(S, dtype=np.float32)
    wpos = np.zeros(S, dtype=np.float32)

    if pixel_mask_THW is not None and spatial_img_mask.any():
        img_hpos, img_wpos = _compute_image_spatial_positions(
            pixel_mask_THW, spatial_patch_size, temporal_patch_size,
        )
        hpos[spatial_img_mask] = img_hpos
        wpos[spatial_img_mask] = img_wpos

    tpos = np.ones(S, dtype=np.float32)
    tpos[no_increase_mask] = 0
    tpos = np.cumsum(tpos) - 1

    return tpos.astype(np.int64), np.stack([hpos, wpos], axis=-1)


def get_pos_thw(
    tokens: np.ndarray,
    pixel_masks_NTHW: np.ndarray,
    tokenizer,
    spatial_patch_size: int,
    temporal_patch_size: int = 1,
    *,
    pad_token_id: int,
):
    """Batch version: compute rope positions for padded batch.

    All inputs and outputs are numpy arrays.
    """
    assert tokens.ndim == 2, f"expected tokens shape (B,S), got {tuple(tokens.shape)=}"
    assert pixel_masks_NTHW.ndim == 4, (
        f"expected pixel_masks_NTHW shape (N,T,H,W), got {tuple(pixel_masks_NTHW.shape)=}"
    )

    spatial_img_token_mask_BS, no_increase_idx_img_token_mask_BS = (
        _get_image_token_masks(tokens, tokenizer)
    )

    hpos_parts, wpos_parts = [], []
    for i in range(pixel_masks_NTHW.shape[0]):
        h, w = _compute_image_spatial_positions(
            pixel_masks_NTHW[i], spatial_patch_size, temporal_patch_size,
        )
        hpos_parts.append(h)
        wpos_parts.append(w)

    hpos_N = np.concatenate(hpos_parts) if hpos_parts else np.empty(0)
    wpos_N = np.concatenate(wpos_parts) if wpos_parts else np.empty(0)

    expected_tokens = spatial_img_token_mask_BS.sum()
    actual_tokens = hpos_N.size
    assert actual_tokens == expected_tokens, (
        f"Mismatch between spatial image tokens ({expected_tokens}) and generated positions ({actual_tokens}). "
        "Check patch_size/merge_size alignment and prompt/image blocks."
    )

    hpos_BS = np.zeros_like(tokens, dtype=np.float32)
    wpos_BS = np.zeros_like(tokens, dtype=np.float32)
    hpos_BS[spatial_img_token_mask_BS] = hpos_N
    wpos_BS[spatial_img_token_mask_BS] = wpos_N

    tpos_BS = np.ones_like(tokens, dtype=np.float32)
    tpos_BS[no_increase_idx_img_token_mask_BS] = 0
    tpos_BS = np.cumsum(tpos_BS, axis=1) - 1
    tpos_BS[tokens == pad_token_id] = 0

    hw_pos_BS2 = np.stack([hpos_BS, wpos_BS], axis=-1)
    return tpos_BS.astype(np.int64), hw_pos_BS2


# ── Tokenization ───────────────────────────────────────────────────────


def calculate_image_tokens(image, patch_size, merge_size):
    height, width = _get_image_size(image)
    return int((height * width) / (patch_size * patch_size * merge_size * merge_size))


def tokenize_inputs(
    prompt,
    images,
    tokenizer,
    patch_size,
    merge_size,
    max_length,
):
    """Tokenize prompt with image tokens.

    Returns:
        input_ids: np.ndarray (int64) — token ids
        selected_images: list — images that were actually inserted
    """
    img_reg_ids = [
        tokenizer.image_reg_1_token_id,
        tokenizer.image_reg_2_token_id,
        tokenizer.image_reg_3_token_id,
        tokenizer.image_reg_4_token_id,
    ]

    if images is not None and len(images) > 0:
        image_token_counts = [
            calculate_image_tokens(image, patch_size, merge_size) for image in images
        ]
    else:
        image_token_counts = []
    prompt_chunks = [
        tokenizer.encode(chunk) for chunk in prompt.split(tokenizer.image_token)
    ]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, sep) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and (tokenizer.bos_id is not None and prompt_chunks[0][0] == tokenizer.bos_id)
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    separators = []
    for count in image_token_counts:
        tokens = [tokenizer.image_token_id] * count
        image_block = [
            tokenizer.image_cls_token_id,
            *img_reg_ids,
            *tokens,
            tokenizer.end_of_image_token_id,
        ]
        separators.append(image_block)

    if len(separators) != 0 and len(separators) != len(prompt_chunks):
        separators.append(separators[-1])

    selected_images = []
    if len(separators) == 0:
        input_ids = prompt_chunks[0]
    else:
        for index, x in enumerate(insert_separator(prompt_chunks, separators)):
            if index % 2 != 0:
                if (len(input_ids) + len(x)) < max_length:
                    input_ids.extend(x)
                    selected_images.append(images[index // 2])
            elif index % 2 == 0:
                input_ids.extend(x[offset:])

    return np.array(input_ids, dtype=np.int64), selected_images


def pad_sequences_left(sequences: list[np.ndarray], pad_value: int) -> np.ndarray:
    """Left-pad a list of variable-length int sequences into a 2D numpy array."""
    max_len = max(len(s) for s in sequences)
    B = len(sequences)
    padded = np.full((B, max_len), pad_value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        padded[i, max_len - len(seq):] = seq
    return padded


# ── Demo / sample utilities ────────────────────────────────────────────


def stream_samples_from_hf_dataset(
    dataset_id: str,
    split: str = "test",
    limit: int = 1,
) -> list[dict]:
    """Stream up to *limit* samples from a HuggingFace dataset."""
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_id} / {split} (limit={limit})")
    ds = load_dataset(dataset_id, split=split, streaming=True)
    samples: list[dict] = []
    for sample in tqdm(ds, desc="Streaming samples"):
        samples.append(sample)
        if 0 < limit <= len(samples):
            break
    return samples
