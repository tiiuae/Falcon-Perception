# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""PBench segmentation metrics — F1 evaluation helpers.

Pure Python / NumPy / pycocotools — no PyTorch dependency.
Import and use directly from notebooks or other scripts::

    from eval.metrics import sample_f1, aggregate, IOU_THRESHOLDS
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

# Standard PBench IoU threshold grids
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
IOU_THRESHOLDS_DENSE = [0.5]


# ---------------------------------------------------------------------------
# RLE helpers
# ---------------------------------------------------------------------------

def _to_bytes_rle(rle: dict) -> dict:
    """Return a copy of *rle* with counts as bytes (pycocotools requirement)."""
    out = rle.copy()
    if isinstance(out.get("counts"), str):
        out["counts"] = out["counts"].encode("utf-8")
    return out


def resize_rle(rle: dict, target_h: int, target_w: int) -> dict:
    """Resize a COCO RLE mask to ``(target_h, target_w)`` via nearest-neighbor.

    Returns *rle* unchanged if the dimensions already match.
    """
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
# NMS
# ---------------------------------------------------------------------------

def nms(rles: list[dict], iou_threshold: float) -> list[dict]:
    """Greedy NMS on COCO-RLE masks, suppressing smaller overlapping masks.

    Masks are sorted by area (largest first).  Any mask whose IoU with an
    already-kept mask exceeds ``iou_threshold`` is removed.
    """
    if len(rles) <= 1:
        return rles
    prepped = [_to_bytes_rle(r) for r in rles]
    areas = [int(mask_utils.area(r)) for r in prepped]
    order = list(np.argsort(areas)[::-1])
    iou_mat = np.asarray(mask_utils.iou(prepped, prepped, [False] * len(prepped)))
    keep: list[int] = []
    suppressed: set[int] = set()
    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        for other in order:
            if other != idx and other not in suppressed:
                if iou_mat[idx, other] > iou_threshold:
                    suppressed.add(other)
    return [rles[i] for i in keep]


# ---------------------------------------------------------------------------
# IoU matrix
# ---------------------------------------------------------------------------

def _iou_matrix(pred_rles: list[dict], gt_rles: list[dict]) -> np.ndarray:
    p = [_to_bytes_rle(r) for r in pred_rles]
    g = [_to_bytes_rle(r) for r in gt_rles]
    return np.asarray(mask_utils.iou(p, g, [False] * len(g)), dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-sample F1
# ---------------------------------------------------------------------------

def sample_f1(
    pred_rles: list[dict],
    gt_rles: list[dict],
    iou_thresholds: list[float],
) -> dict:
    """Compute per-sample F1 and image-level classification signals.

    Uses Hungarian matching to find the optimal assignment between predicted
    and ground-truth masks, then computes F1 at each IoU threshold.

    Returns
    -------
    dict with keys:
        ``f1``         – mean F1 across thresholds (``-1.0`` for true-negative
                         samples where both GT and predictions are empty).
        ``il_tp/tn/fp/fn`` – image-level classification signal (0.0 or 1.0).
        ``agg_tp/fp/fn``   – np.ndarray of TP / FP / FN counts at each threshold.
    """
    n_pred = len(pred_rles)
    n_gt = len(gt_rles)
    n_thresh = len(iou_thresholds)
    is_pos = n_gt > 0
    has_pred = n_pred > 0

    result: dict = {
        "f1": -1.0,
        "il_tp": float(is_pos and has_pred),
        "il_tn": float(not is_pos and not has_pred),
        "il_fp": float(not is_pos and has_pred),
        "il_fn": float(is_pos and not has_pred),
        "agg_tp": np.zeros(n_thresh),
        "agg_fp": np.zeros(n_thresh),
        "agg_fn": np.zeros(n_thresh),
    }

    if is_pos and has_pred:
        cost = -_iou_matrix(pred_rles, gt_rles)
        ri, ci = linear_sum_assignment(cost)
        matched_ious = -cost[ri, ci]
        f1s = []
        for t, thresh in enumerate(iou_thresholds):
            tp = int((matched_ious >= thresh).sum())
            fp = n_pred - tp
            fn = n_gt - tp
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1s.append(2.0 * prec * rec / (prec + rec + 1e-8))
            result["agg_tp"][t] = tp
            result["agg_fp"][t] = fp
            result["agg_fn"][t] = fn
        result["f1"] = float(np.mean(f1s))

    elif is_pos and not has_pred:
        result["f1"] = 0.0
        result["agg_fn"] = np.full(n_thresh, float(n_gt))

    return result


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------

NMS_THRESHOLD = 0.5


def aggregate(per_sample: list[dict], iou_thresholds: list[float]) -> dict:
    """Aggregate a list of :func:`sample_f1` results into dataset-level metrics.

    Parameters
    ----------
    per_sample:
        One dict per sample as returned by :func:`sample_f1`.
    iou_thresholds:
        The same threshold list used when calling :func:`sample_f1`.

    Returns
    -------
    dict with:
        ``f1``             – macro F1: mean of per-sample F1s (positive GT only).
        ``il_tp/tn/fp/fn`` – summed image-level classification counts.
        ``n_samples``      – total sample count.
        ``n_valid_f1``     – samples contributing to F1.
    """
    il_tp = il_tn = il_fp = il_fn = 0.0
    valid_f1s: list[float] = []

    for r in per_sample:
        il_tp += r["il_tp"]
        il_tn += r["il_tn"]
        il_fp += r["il_fp"]
        il_fn += r["il_fn"]
        if r["f1"] >= 0.0:
            valid_f1s.append(r["f1"])

    return {
        "f1": float(np.mean(valid_f1s)) if valid_f1s else 0.0,
        "il_tp": int(il_tp),
        "il_tn": int(il_tn),
        "il_fp": int(il_fp),
        "il_fn": int(il_fn),
        "n_samples": len(per_sample),
        "n_valid_f1": len(valid_f1s),
    }
