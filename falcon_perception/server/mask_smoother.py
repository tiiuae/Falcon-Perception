# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""
mask_smoothing.py
Post-processing utilities to smooth noisy segmentation masks.

Two-step pipeline:
  1. Morphological opening  (erode → dilate) — removes noise & thin spurs
  2. Morphological closing  (dilate → erode) — fills small interior holes

Dependencies
------------
- numpy        (always required)
- scipy        (required)
- pycocotools  (required for compressed RLE)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, maximum_filter, minimum_filter, uniform_filter


# ---------------------------------------------------------------------------
# RLE helpers
# ---------------------------------------------------------------------------

def decode_rle(rle: dict) -> np.ndarray:
    """Decode a COCO compressed RLE string dict to a 2-D uint8 mask (H, W)."""
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, (bytes, str)):
        from pycocotools import mask as coco_mask
        if isinstance(counts, str):
            counts = counts.encode("utf-8")
        return coco_mask.decode({"counts": counts, "size": [h, w]}).astype(np.uint8)
    # Uncompressed list
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, run in enumerate(counts):
        if i % 2 == 1:
            flat[pos: pos + run] = 1
        pos += run
    return flat.reshape(h, w, order="F").astype(np.uint8)


def encode_rle_compressed(mask: np.ndarray) -> dict:
    """Encode a 2-D binary mask to a compressed COCO RLE dict (string counts)."""
    from pycocotools import mask as coco_mask
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def smooth_mask_rle(
    rle: dict,
    *,
    morph_radius: int = 2,
    hole_radius: int = 1,
    contour_radius: int = 0,
) -> dict:
    """
    Smooth a COCO RLE mask and return a new compressed RLE dict.

    Pipeline
    --------
    1. Opening  (erode → dilate, radius=morph_radius) — removes noise & thin spurs
    2. Closing  (dilate → erode, radius=hole_radius)  — fills small interior holes
    3. Optional contour re-binarise (off by default). vis.py's pretty look comes from
       *soft alpha compositing* (mask_combiner + frontend), not from altering RLE.

    Parameters
    ----------
    rle            : COCO RLE dict {"counts": "...", "size": [H, W]}
    morph_radius   : Kernel radius for opening — controls noise removal (default 2).
    hole_radius    : Kernel radius for closing — controls hole filling (default 1).
    contour_radius : If > 0, optional morphological contour step. Default 0.

    Returns
    -------
    dict  Compressed COCO RLE {"counts": "...", "size": [H, W]}
    """
    m = decode_rle(rle).astype(bool)

    # 1. Opening: remove noise speckles & thin spurs
    open_struct = np.ones((2 * morph_radius + 1, 2 * morph_radius + 1), dtype=bool)
    m = binary_erosion(m, structure=open_struct)
    m = binary_dilation(m, structure=open_struct)

    # 2. Closing: fill small interior holes
    close_struct = np.ones((2 * hole_radius + 1, 2 * hole_radius + 1), dtype=bool)
    m = binary_dilation(m, structure=close_struct)
    m = binary_erosion(m, structure=close_struct)

    # 3. Boundary smoothing — direct numpy port of vis.py:
    #    F.max_pool2d  → maximum_filter
    #    F.max_pool2d(-x) → minimum_filter
    #    F.avg_pool2d(kernel=3) → uniform_filter(size=3)
    if contour_radius > 0:
        k = 2 * contour_radius + 1
        m_f = m.astype(np.float32)
        dil = maximum_filter(m_f, size=k)           # dilation
        ero = minimum_filter(m_f, size=k)           # erosion
        contour = np.clip(dil - ero, 0.0, 1.0)     # boundary band
        contour = uniform_filter(contour, size=3)   # box blur (avg_pool2d k=3)
        m = (ero + contour) >= 0.5                  # re-binarise

    return encode_rle_compressed(m.astype(np.uint8))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pycocotools import mask as coco_mask

    raw = np.zeros((64, 64), dtype=np.uint8)
    raw[16:48, 16:48] = 1
    rng = np.random.default_rng(42)
    noise = (rng.random(raw.shape) < 0.05).astype(np.uint8)
    noisy = np.clip(raw + noise, 0, 1).astype(np.uint8)

    rle = coco_mask.encode(np.asfortranarray(noisy))
    rle["counts"] = rle["counts"].decode("utf-8")

    smoothed = smooth_mask_rle(rle, morph_radius=2, hole_radius=2, contour_radius=0)
    print("Input  foreground px:", decode_rle(rle).sum())
    print("Output foreground px:", decode_rle(smoothed).sum())
    print("Output counts type  :", type(smoothed["counts"]))