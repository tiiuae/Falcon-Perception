# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""
mask_renderer.py
Backend equivalent of buildCompositeImageData() from usePerception.jsx.

Takes a list of segmentation objects (each with a COCO RLE mask, color, bbox, label)
and composites them into a single RGBA overlay image — identical blending logic
to the frontend canvas renderer.

Dependencies
------------
    pip install numpy pillow pycocotools
"""

from __future__ import annotations

import io
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


# ---------------------------------------------------------------------------
# Palette — mirrors MASK_COLORS in colors.js
# ---------------------------------------------------------------------------

MASK_COLORS = [
    {"r": 239, "g": 68,  "b": 100},  # rose
    {"r": 59,  "g": 180, "b": 246},  # sky
    {"r": 16,  "g": 200, "b": 129},  # emerald
    {"r": 168, "g": 85,  "b": 247},  # violet
    {"r": 245, "g": 158, "b": 11 },  # amber
    {"r": 236, "g": 72,  "b": 153},  # pink
    {"r": 6,   "g": 205, "b": 212},  # cyan
    {"r": 249, "g": 115, "b": 22 },  # orange
    {"r": 132, "g": 204, "b": 22 },  # lime
    {"r": 99,  "g": 102, "b": 241},  # indigo
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_rle(rle: dict) -> np.ndarray:
    """
    Decode a COCO compressed RLE dict to a 2-D uint8 mask (H, W).

    rle = {"counts": "abc123...", "size": [H, W]}
    """
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    decoded = coco_mask.decode({"counts": counts, "size": [h, w]})
    return decoded.astype(np.uint8)   # shape (H, W)


def resize_mask(
    raw: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int
) -> np.ndarray:
    """
    Nearest-neighbour resize of a binary mask — mirrors resizeMask() in JS.
    """
    scale_x = src_w / dst_w
    scale_y = src_h / dst_h
    xs = np.minimum(np.floor(np.arange(dst_w) * scale_x).astype(int), src_w - 1)
    ys = np.minimum(np.floor(np.arange(dst_h) * scale_y).astype(int), src_h - 1)
    return raw[np.ix_(ys, xs)]


def detect_edges(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Return a boolean array (H, W) that is True where a foreground pixel
    has at least one background neighbour within `radius` pixels —
    mirrors the edge-detection loop in buildCompositeImageData().
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    # A pixel is an edge if the eroded mask differs from the original
    from scipy.ndimage import binary_erosion
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    eroded = binary_erosion(mask, structure=struct)
    return (mask.astype(bool)) & (~eroded)


def alpha_composite_pixel(
    bg_rgba: np.ndarray, fg_rgb: tuple[int, int, int], fg_alpha: int
) -> np.ndarray:
    """
    Porter-Duff 'source over' blend — mirrors the JS canvas compositing.
    bg_rgba : (H, W, 4) uint8 array
    Returns updated (H, W, 4) uint8 array.
    """
    a1 = fg_alpha / 255.0
    a0 = bg_rgba[:, :, 3] / 255.0
    a_out = a1 + a0 * (1.0 - a1)

    out = bg_rgba.copy().astype(np.float32)

    valid = a_out > 0
    for ch, val in enumerate(fg_rgb):
        out[:, :, ch] = np.where(
            valid,
            (val * a1 + bg_rgba[:, :, ch] * a0 * (1.0 - a1)) / np.where(valid, a_out, 1),
            bg_rgba[:, :, ch],
        )
    out[:, :, 3] = a_out * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Internal: render one mask onto a canvas tile
# ---------------------------------------------------------------------------

def _composite_mask_onto_canvas(
    canvas: np.ndarray,
    mask: np.ndarray,
    c: dict,
    img_w: int,
    img_h: int,
) -> None:
    """
    Mutates canvas in-place — mirrors the JS Porter-Duff blend exactly:

        if (ea === 0) {
            pixels[idx] = c.r; pixels[idx+3] = alpha;
        } else {
            aOut = a1 + a0*(1-a1)
            pixels[idx] = (c.r*a1 + pixels[idx]*a0*(1-a1)) / aOut
        }
    """
    edges = detect_edges(mask, radius=3)

    # Per-pixel alpha: 220 on edge, 89 interior (matches JS)
    alpha_layer = np.zeros((img_h, img_w), dtype=np.uint8)
    alpha_layer[mask == 1] = 89
    alpha_layer[edges]     = 220

    fg = mask.astype(bool)
    if not fg.any():
        return

    # Work in float32 for precision
    cv = canvas.astype(np.float32)          # (H, W, 4)
    a1 = alpha_layer.astype(np.float32) / 255.0   # foreground alpha
    a0 = cv[:, :, 3] / 255.0                       # background alpha
    a_out = a1 + a0 * (1.0 - a1)                   # combined alpha

    empty = fg & (canvas[:, :, 3] == 0)    # pixels with no existing color
    blend = fg & (canvas[:, :, 3] != 0)    # pixels with existing color

    for ch, val in enumerate((c["r"], c["g"], c["b"])):
        # Empty pixel — just write the color directly (JS: ea === 0 branch)
        cv[:, :, ch] = np.where(empty, val, cv[:, :, ch])
        # Blend pixel — Porter-Duff source-over (JS: else branch)
        cv[:, :, ch] = np.where(
            blend,
            np.where(
                a_out > 0,
                (val * a1 + cv[:, :, ch] * a0 * (1.0 - a1)) / a_out,
                cv[:, :, ch],
            ),
            cv[:, :, ch],
        )

    # Alpha channel
    cv[:, :, 3] = np.where(empty, alpha_layer.astype(np.float32),
                  np.where(blend, a_out * 255.0, cv[:, :, 3]))

    canvas[:] = np.clip(cv, 0, 255).astype(np.uint8)


def _ndarray_to_rgba_base64(rgba: np.ndarray) -> str:
    """
    Encode a (H, W, 4) uint8 RGBA array as base64 raw bytes.
    Frontend decodes this directly into an ImageData object:

        const raw = Uint8ClampedArray.from(atob(b64), c => c.charCodeAt(0));
        const imageData = new ImageData(raw, width, height);
    """
    import base64
    # Ensure row-major (C order) flat RGBA bytes — same memory layout as ImageData.data
    flat = rgba.astype(np.uint8).tobytes()
    return base64.b64encode(flat).decode("utf-8")


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_masks(
    objects: list[dict],
    img_w: int,
    img_h: int,
) -> dict:
    """
    Process a list of COCO RLE masks and return an enriched response.

    Parameters
    ----------
    objects : list of dicts — any shape, must contain "counts" and "size".
              All existing keys are preserved untouched.
    img_w, img_h : display canvas size (original image dimensions)

    Returns
    -------
    {
        "masks": [
            { ...all original keys..., "color": {"r": 239, "g": 68, "b": 100} },
            ...
        ],
        "combined_mask": {
            "data":   "<base64 raw RGBA bytes>",  # Uint8ClampedArray-ready
            "width":  img_w,
            "height": img_h,
        }
    }

    Frontend usage (mirrors buildCompositeImageData return value):
        const { data: b64, width, height } = res.combined_mask;
        const raw = Uint8ClampedArray.from(atob(b64), c => c.charCodeAt(0));
        const maskPixels = new ImageData(raw, width, height);
    """
    enriched = []
    composite_canvas = np.zeros((img_h, img_w, 4), dtype=np.uint8)

    for i, obj in enumerate(objects):
        src_h, src_w = obj["size"]
        c = MASK_COLORS[i % len(MASK_COLORS)]

        # Decode + resize
        raw  = decode_rle({"counts": obj["counts"], "size": [src_h, src_w]})
        mask = resize_mask(raw, src_w, src_h, img_w, img_h)

        # Composite canvas (all masks together)
        _composite_mask_onto_canvas(composite_canvas, mask, c, img_w, img_h)

        # All original keys preserved, color added
        enriched.append({**obj, "color": c})

    combined_mask = {
        "data":   _ndarray_to_rgba_base64(composite_canvas),
        "width":  img_w,
        "height": img_h,
    } if enriched else None

    return enriched, combined_mask


# ---------------------------------------------------------------------------
# FastAPI endpoint example
# ---------------------------------------------------------------------------
#
# from fastapi import FastAPI
# from pydantic import BaseModel
#
# app = FastAPI()
#
# class RenderRequest(BaseModel):
#     objects: list[dict]   # list of {"counts": str, "size": [int, int], ...}
#     img_w: int
#     img_h: int
#
# @app.post("/render-mask")
# def render_mask(req: RenderRequest):
#     return render_masks(req.objects, req.img_w, req.img_h)
#
# Response shape:
# {
#   "masks": [
#     { ...all original keys..., "color": {"r":239,"g":68,"b":100} },
#     ...
#   ],
#   "composite": "data:image/png;base64,..."
# }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pycocotools import mask as coco_mask

    m = np.zeros((100, 100), dtype=np.uint8)
    m[20:80, 20:80] = 1
    rle = coco_mask.encode(np.asfortranarray(m))
    rle["counts"] = rle["counts"].decode("utf-8")

    m2 = np.zeros((100, 100), dtype=np.uint8)
    m2[10:50, 40:90] = 1
    rle2 = coco_mask.encode(np.asfortranarray(m2))
    rle2["counts"] = rle2["counts"].decode("utf-8")

    result = render_masks([rle, rle2], img_w=200, img_h=200)

    for i, mask in enumerate(result["masks"]):
        print(f"Mask {i}: color={mask['color']}")

    print(f"combined_mask base64 length: {len(result['combined_mask'])}")