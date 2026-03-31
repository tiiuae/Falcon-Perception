# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Falcon Perception tool implementations for the agent loop.

Public API
----------
run_ground_expression(engine, tokenizer, image, expression, **kw) → dict[int, dict]
    Run FP inference and return per-mask metadata keyed by 1-indexed mask ID.

compute_relations(masks, mask_ids) → dict
    Compute pairwise spatial relationships between a set of masks using
    pycocotools and centroid arithmetic.

masks_to_vlm_json(masks) → list[dict]
    Strip the internal ``rle`` field before sending metadata to the VLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# Ensure the Falcon-Perception root is importable when the demo/ tree is run
# as a script (e.g. ``python demo/perception_agent.py``).
_ROOT = Path(__file__).resolve().parents[2]   # Falcon-Perception/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from falcon_perception import build_prompt_for_task
from falcon_perception.paged_inference import SamplingParams, Sequence
from falcon_perception.visualization_utils import decode_coco_rle, pair_bbox_entries


# ---------------------------------------------------------------------------
# Internal RLE helpers (inlined to keep agent/ self-contained)
# ---------------------------------------------------------------------------

def _to_bytes_rle(rle: dict) -> dict:
    """Return *rle* with counts as bytes (pycocotools requirement)."""
    out = rle.copy()
    if isinstance(out.get("counts"), str):
        out["counts"] = out["counts"].encode("utf-8")
    return out


def _resize_rle(rle: dict, target_h: int, target_w: int) -> dict:
    """Nearest-neighbour resize of a COCO RLE mask to ``(target_h, target_w)``."""
    cur_h, cur_w = rle["size"]
    if cur_h == target_h and cur_w == target_w:
        return rle
    binary = mask_utils.decode(_to_bytes_rle(rle))
    resized = np.asfortranarray(
        np.array(
            Image.fromarray(binary).resize((target_w, target_h), Image.NEAREST)
        ).astype(np.uint8)
    )
    new_rle = mask_utils.encode(resized)
    if isinstance(new_rle.get("counts"), bytes):
        new_rle["counts"] = new_rle["counts"].decode("utf-8")
    return new_rle


# ---------------------------------------------------------------------------
# Mask metadata computation
# ---------------------------------------------------------------------------

def _image_region_label(cx_norm: float, cy_norm: float) -> str:
    """Map a normalised centroid to a human-readable image-region string."""
    if cx_norm < 0.33:
        h = "left"
    elif cx_norm < 0.67:
        h = "center"
    else:
        h = "right"

    if cy_norm < 0.33:
        v = "top"
    elif cy_norm < 0.67:
        v = "middle"
    else:
        v = "bottom"

    if v == "middle" and h == "center":
        return "center"
    if v == "middle":
        return h
    return f"{v}-{h}"


def _compute_mask_metadata(
    rle: dict,
    img_w: int,
    img_h: int,
    mask_id: int,
) -> dict | None:
    """Decode *rle* and compute spatial metadata.

    Returns a dict with keys:
        id, area_fraction, centroid_norm, bbox_norm, image_region, rle

    Returns *None* if the mask is empty or cannot be decoded.
    """
    binary = decode_coco_rle(rle)
    if binary is None or not binary.any():
        return None

    area = int(binary.sum())
    area_fraction = round(area / (img_h * img_w), 4)

    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
        cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
    else:
        rmin, rmax = 0, img_h - 1
        cmin, cmax = 0, img_w - 1

    cx_norm = round((cmin + cmax + 1) / 2.0 / img_w, 4)
    cy_norm = round((rmin + rmax + 1) / 2.0 / img_h, 4)

    return {
        "id": mask_id,
        "area_fraction": area_fraction,
        "centroid_norm": {"x": cx_norm, "y": cy_norm},
        "bbox_norm": {
            "x1": round(cmin / img_w, 4),
            "y1": round(rmin / img_h, 4),
            "x2": round((cmax + 1) / img_w, 4),
            "y2": round((rmax + 1) / img_h, 4),
        },
        "image_region": _image_region_label(cx_norm, cy_norm),
        "rle": rle,   # kept internally for compute_relations / get_crop
    }


# ---------------------------------------------------------------------------
# Core tool: ground_expression
# ---------------------------------------------------------------------------

def run_ground_expression(
    engine,
    tokenizer,
    image: Image.Image,
    expression: str,
    *,
    max_new_tokens: int = 2048,
    hr_upsample_ratio: int = 8,
    max_dimension: int = 1024,
    min_dimension: int = 256,
) -> dict[int, dict]:
    """Run Falcon Perception on *image* with a full referring *expression*.

    Returns a dict mapping 1-indexed mask IDs to metadata dicts.  Masks that
    are empty after decoding are silently dropped.

    The returned dict is ``{mask_id: {id, area_fraction, centroid_norm,
    bbox_norm, image_region, rle}}``.  The ``rle`` field is at the original
    image resolution and is needed by :func:`compute_relations` and
    :func:`~demo.agent.viz.get_crop`.
    """
    pil_image = image.convert("RGB")
    orig_w, orig_h = pil_image.size

    prompt = build_prompt_for_task(expression, "segmentation")

    stop_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "end_of_query_token_id"):
        stop_ids.append(tokenizer.end_of_query_token_id)

    seq = Sequence(
        text=prompt,
        image=pil_image,
        min_image_size=min_dimension,
        max_image_size=max_dimension,
        request_idx=0,
        task="segmentation",
    )

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_ids,
        hr_upsample_ratio=hr_upsample_ratio,
    )

    engine.generate(
        [seq],
        sampling_params=sampling_params,
        use_tqdm=False,
        print_stats=False,
    )

    masks_rle = (seq.output_aux.masks_rle if seq.output_aux else []) or []

    masks: dict[int, dict] = {}
    assigned_id = 1
    for raw_rle in masks_rle:
        if not isinstance(raw_rle, dict) or "counts" not in raw_rle:
            continue
        # Resize from inference resolution → original image resolution
        rle_orig = _resize_rle(raw_rle, orig_h, orig_w)
        meta = _compute_mask_metadata(rle_orig, orig_w, orig_h, mask_id=assigned_id)
        if meta is not None:
            masks[assigned_id] = meta
            assigned_id += 1

    return masks


# ---------------------------------------------------------------------------
# Core tool: compute_relations
# ---------------------------------------------------------------------------

def compute_relations(
    masks: dict[int, dict],
    mask_ids: list[int],
) -> dict:
    """Compute pairwise spatial relationships between the given *mask_ids*.

    Uses pycocotools IoU and centroid arithmetic.  Returns a dict:

    .. code-block:: json

        {
          "1_vs_2": {
            "iou": 0.02,
            "1_left_of_2": true,
            "1_above_2": false,
            "1_larger_than_2": true,
            "size_ratio_1_over_2": 2.1,
            "centroid_distance_norm": 0.38
          }
        }
    """
    valid_ids = [mid for mid in mask_ids if mid in masks]
    if len(valid_ids) < 2:
        return {
            "note": (
                "Need at least 2 valid mask IDs for pairwise relations. "
                f"Requested: {mask_ids}, available: {sorted(masks.keys())}"
            )
        }

    # Pre-convert RLEs to bytes for pycocotools
    prepped: dict[int, dict] = {}
    for mid in valid_ids:
        prepped[mid] = _to_bytes_rle(masks[mid]["rle"])

    pairs: dict[str, dict] = {}
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            a_id = valid_ids[i]
            b_id = valid_ids[j]

            iou_mat = np.asarray(
                mask_utils.iou([prepped[a_id]], [prepped[b_id]], [False])
            )
            iou = round(float(iou_mat[0][0]), 4)

            a = masks[a_id]
            b = masks[b_id]

            cx_a = a["centroid_norm"]["x"]
            cy_a = a["centroid_norm"]["y"]
            cx_b = b["centroid_norm"]["x"]
            cy_b = b["centroid_norm"]["y"]
            dist = round(((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5, 4)

            area_a = a["area_fraction"]
            area_b = b["area_fraction"]
            size_ratio = round(area_a / area_b, 3) if area_b > 0 else None

            key = f"{a_id}_vs_{b_id}"
            pairs[key] = {
                "iou": iou,
                f"{a_id}_left_of_{b_id}": cx_a < cx_b,
                f"{a_id}_above_{b_id}": cy_a < cy_b,
                f"{a_id}_larger_than_{b_id}": area_a > area_b,
                f"size_ratio_{a_id}_over_{b_id}": size_ratio,
                "centroid_distance_norm": dist,
            }

    return {"pairs": pairs}


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------

def masks_to_vlm_json(masks: dict[int, dict]) -> list[dict]:
    """Return a JSON-serialisable list of mask metadata, omitting the ``rle`` field."""
    out = []
    for mask_id in sorted(masks.keys()):
        m = masks[mask_id]
        out.append({
            "id": m["id"],
            "area_fraction": m["area_fraction"],
            "centroid_norm": m["centroid_norm"],
            "bbox_norm": m["bbox_norm"],
            "image_region": m["image_region"],
        })
    return out
