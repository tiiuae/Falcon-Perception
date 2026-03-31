# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Visualization helpers for the Falcon Perception Agent.

Provides:
- Set-of-Marks (SoM) rendering: colored mask overlays with numbered labels
- Final mask rendering: clean overlay for the agent's selected answer masks
- Crop extraction: padded bounding-box crops for ``get_crop`` tool calls
- PIL ↔ base64 conversion for the OpenAI API
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Palette: each mask gets a distinct color (cycles after 10)
_PALETTE: list[tuple[int, int, int]] = [
    (255,  80,  80),   # red
    ( 80, 200,  80),   # green
    ( 80, 120, 255),   # blue
    (255, 220,  50),   # yellow
    (220,  80, 220),   # magenta
    ( 50, 210, 210),   # cyan
    (255, 150,  40),   # orange
    (160,  80, 255),   # purple
    ( 50, 210, 140),   # spring-green
    (255,  80, 160),   # deep-pink
]


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_som(
    image: Image.Image,
    masks: dict[int, dict],
    *,
    interior_opacity: float = 0.40,
    label_radius: int = 13,
) -> Image.Image:
    """Render a Set-of-Marks overlay on *image*.

    Each mask in *masks* (key = 1-indexed mask ID, value = metadata dict with
    ``rle``, ``centroid_norm``, ``bbox_norm``) is drawn as a semi-transparent
    colored fill with a numbered white-circle label at its centroid.

    Returns a new PIL RGB image; the original is not modified.
    """
    from falcon_perception.visualization_utils import decode_coco_rle

    img_rgb = image.convert("RGB")
    W, H = img_rgb.size
    base_np = np.array(img_rgb, dtype=np.uint8)

    if not masks:
        return img_rgb.copy()

    sorted_ids = sorted(masks.keys())

    # ── 1. Composite colored fills ──────────────────────────────────────────
    # Build a per-pixel index map (smallest mask wins on overlap)
    idx_map = np.full((H, W), -1, dtype=np.int32)
    binary_masks: list[np.ndarray] = []
    for rank, mask_id in enumerate(sorted_ids):
        raw_rle = masks[mask_id].get("rle")
        if raw_rle is None:
            binary_masks.append(np.zeros((H, W), dtype=np.uint8))
            continue
        m = decode_coco_rle(raw_rle)
        if m is None:
            binary_masks.append(np.zeros((H, W), dtype=np.uint8))
            continue
        if m.shape != (H, W):
            m = np.array(
                Image.fromarray(m).resize((W, H), Image.NEAREST)
            ).astype(np.uint8)
        binary_masks.append(m)

    # Sort: largest area first so smaller masks render on top
    areas = [m.sum() for m in binary_masks]
    draw_order = np.argsort(areas)[::-1]

    for rank_in_order in draw_order:
        m = binary_masks[rank_in_order]
        idx_map[m > 0] = rank_in_order

    has_mask = idx_map >= 0
    if has_mask.any():
        palette_np = np.array(_PALETTE, dtype=np.uint8)
        P = len(palette_np)
        ordered_colors = palette_np[
            np.array([int(draw_order[i]) % P for i in range(len(sorted_ids))], dtype=np.intp)
        ]
        clamped = np.where(has_mask, idx_map, 0)
        fill_rgb = ordered_colors[clamped]   # (H, W, 3)

        composite = base_np.copy().astype(np.float32)
        mask_3d = has_mask[:, :, np.newaxis]
        composite = np.where(
            mask_3d,
            interior_opacity * fill_rgb.astype(np.float32)
            + (1.0 - interior_opacity) * composite,
            composite,
        )
        # Contour borders (single-pixel edges from index-map)
        border = np.zeros((H, W), dtype=np.bool_)
        border[:, 1:] |= idx_map[:, 1:] != idx_map[:, :-1]
        border[:, :-1] |= idx_map[:, 1:] != idx_map[:, :-1]
        border[1:, :] |= idx_map[1:, :] != idx_map[:-1, :]
        border[:-1, :] |= idx_map[1:, :] != idx_map[:-1, :]
        border &= has_mask
        if border.any():
            bright = np.clip(
                0.65 * ordered_colors.astype(np.float32) + 89.25, 0, 255
            ).astype(np.uint8)
            composite[border] = bright[np.where(border, idx_map, 0)][border]

        result_np = np.clip(composite, 0, 255).astype(np.uint8)
    else:
        result_np = base_np.copy()

    # ── 2. Draw numbered circle labels ──────────────────────────────────────
    result_pil = Image.fromarray(result_np)
    draw = ImageDraw.Draw(result_pil)
    font = _load_font(size=max(12, label_radius))

    for mask_id in sorted_ids:
        meta = masks[mask_id]
        cx_norm = meta.get("centroid_norm", {}).get("x", 0.5)
        cy_norm = meta.get("centroid_norm", {}).get("y", 0.5)
        cx_px = int(cx_norm * W)
        cy_px = int(cy_norm * H)
        r = label_radius

        # White filled circle
        draw.ellipse(
            [cx_px - r, cy_px - r, cx_px + r, cy_px + r],
            fill="white",
            outline="black",
            width=2,
        )
        # Centered number label
        draw.text((cx_px, cy_px), str(mask_id), fill="black", font=font, anchor="mm")

    return result_pil


def render_final(
    image: Image.Image,
    masks: dict[int, dict],
    selected_ids: list[int],
) -> Image.Image:
    """Render only the *selected_ids* masks on *image* for the final answer display."""
    selected = {k: v for k, v in masks.items() if k in selected_ids}
    return render_som(image, selected)


def get_crop(
    image: Image.Image,
    mask_info: dict,
    *,
    padding_frac: float = 0.15,
) -> Image.Image:
    """Return a padded bounding-box crop of *image* for the given mask.

    ``mask_info`` must have a ``bbox_norm`` key with ``x1, y1, x2, y2``
    normalised to [0, 1].  The crop is padded by ``padding_frac`` of the
    bounding-box dimensions on each side.
    """
    W, H = image.size
    bbox = mask_info.get("bbox_norm", {"x1": 0, "y1": 0, "x2": 1, "y2": 1})
    x1 = bbox["x1"] * W
    y1 = bbox["y1"] * H
    x2 = bbox["x2"] * W
    y2 = bbox["y2"] * H

    pad_x = (x2 - x1) * padding_frac
    pad_y = (y2 - y1) * padding_frac
    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(W), x2 + pad_x)
    y2 = min(float(H), y2 + pad_y)

    return image.convert("RGB").crop((int(x1), int(y1), int(x2), int(y2)))


def pil_to_base64_url(image: Image.Image, *, max_side: int = 1536) -> str:
    """Encode a PIL image as a JPEG base64 data-URL for the OpenAI API.

    Images larger than *max_side* on their longest edge are downscaled with
    LANCZOS to reduce API token cost while preserving readability.
    """
    img = image.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"
