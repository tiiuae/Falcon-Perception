# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import utils as tv_utils


# Mask-based NMS

def mask_nms(
    detections: list[dict],
    iou_threshold: float = 0.5,
    nms_max_side: int = 256,
) -> tuple[list[dict], int]:
    """Greedy mask-based NMS for binary masks.

    Scores each detection by mask area (larger = kept first), then
    iteratively suppresses masks whose pairwise IoU with a kept mask
    exceeds *iou_threshold*.

    All IoU computation is batched on CPU using torch for speed.
    Masks are optionally downscaled to *nms_max_side* before IoU
    computation to save memory when mask counts are high.

    Parameters
    ----------
    detections : list[dict]
        Each dict must have a ``'mask'`` key (numpy uint8 or torch tensor).
    iou_threshold : float
        Masks with pairwise IoU above this are suppressed (default 0.5).
    nms_max_side : int | None
        If set, masks are resized so the longest side is at most this value
        before IoU computation (default 256).

    Returns
    -------
    kept : list[dict]
        Subset of *detections* that survive NMS.
    n_suppressed : int
        Number of detections that were suppressed.
    """
    mask_indices: list[int] = []
    masks: list[torch.Tensor] = []
    for idx, det in enumerate(detections):
        m = det.get("mask")
        if m is None:
            continue
        if not isinstance(m, torch.Tensor):
            m = torch.from_numpy(np.asarray(m)).float()
        else:
            m = m.float()
        mask_indices.append(idx)
        masks.append(m)

    if len(masks) <= 1:
        return detections, 0

    # Determine target resolution for IoU computation
    base_h, base_w = masks[0].shape
    if nms_max_side is not None and nms_max_side > 0:
        scale = min(1.0, float(nms_max_side) / float(max(base_h, base_w)))
        target_h = max(1, int(round(base_h * scale)))
        target_w = max(1, int(round(base_w * scale)))
    else:
        target_h, target_w = base_h, base_w

    # Resize and stack
    stack = []
    for m in masks:
        if m.shape != (target_h, target_w):
            m = F.interpolate(
                m[None, None], size=(target_h, target_w), mode="nearest",
            ).squeeze(0).squeeze(0)
        stack.append(m)

    binary = torch.stack(stack).clamp(0, 1)  # (N, h, w)
    binary = (binary > 0.5).float()

    # Score by area (larger masks kept first)
    areas = binary.view(binary.shape[0], -1).sum(dim=1)  # (N,)
    scores = areas

    # Pairwise IoU — fully batched
    N = binary.shape[0]
    flat = binary.view(N, -1)  # (N, H*W)
    intersection = flat @ flat.T  # (N, N)
    union = areas[:, None] + areas[None, :] - intersection
    iou_matrix = intersection / union.clamp(min=1)

    # Greedy suppression
    order = scores.argsort(descending=True).tolist()
    suppressed: set[int] = set()
    keep_local: list[int] = []
    for i in order:
        if i in suppressed:
            continue
        keep_local.append(i)
        ious = iou_matrix[i]
        for j in order:
            if j not in suppressed and j != i and ious[j].item() > iou_threshold:
                suppressed.add(j)

    n_suppressed = len(suppressed)

    # Map back to original detection indices; also keep mask-less detections
    kept_det_indices = {mask_indices[i] for i in keep_local}
    no_mask_indices = {i for i in range(len(detections)) if i not in mask_indices}
    all_kept = sorted(kept_det_indices | no_mask_indices)

    return [detections[i] for i in all_kept], n_suppressed


_PALETTE_UINT8: list[tuple[int, int, int]] = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (255, 128, 0),   # Orange
    (128, 0, 255),   # Purple
    (0, 255, 128),   # Spring Green
    (255, 0, 128),   # Deep Pink
]

_PALETTE = torch.tensor(_PALETTE_UINT8, dtype=torch.float32) / 255.0  # (P, 3)
_PALETTE_NP = np.array(_PALETTE_UINT8, dtype=np.uint8)  # (P, 3)


def _resize_mask(m: torch.Tensor, H: int, W: int, mode: str) -> torch.Tensor:
    """Resize a single mask to (H, W) if needed."""
    if m.shape == (H, W):
        return m
    if mode == "nearest":
        return F.interpolate(m[None, None], (H, W), mode="nearest").squeeze(0).squeeze(0)
    return F.interpolate(
        m[None, None], (H, W), mode="bilinear", align_corners=False
    ).squeeze(0).squeeze(0)


def _draw_bboxes(overlay: torch.Tensor, detections: list[dict]) -> torch.Tensor:
    """Draw bounding boxes on a (3, H, W) float overlay, return updated overlay."""
    _COLOR_NAMES = [
        "red", "green", "blue", "yellow", "magenta",
        "cyan", "orange", "purple", "springgreen", "deeppink",
    ]
    _, H, W = overlay.shape
    boxes, box_colors = [], []
    for det_idx, det in enumerate(detections):
        xy = det.get("xy")
        hw = det.get("hw")
        if xy is None:
            continue
        cx, cy = xy["x"] * W, xy["y"] * H
        if hw and "w" in hw and "h" in hw:
            bw, bh = hw["w"] * W, hw["h"] * H
        else:
            bw, bh = 10, 10
        xmin = max(0, int(round(cx - bw / 2.0)))
        ymin = max(0, int(round(cy - bh / 2.0)))
        xmax = min(W - 1, int(round(cx + bw / 2.0)))
        ymax = min(H - 1, int(round(cy + bh / 2.0)))
        boxes.append([xmin, ymin, xmax, ymax])
        box_colors.append(_COLOR_NAMES[det_idx % len(_COLOR_NAMES)])
    if boxes:
        overlay_uint8 = (overlay * 255).clamp(0, 255).to(torch.uint8)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        overlay_uint8 = tv_utils.draw_bounding_boxes(
            overlay_uint8, boxes=boxes_tensor, colors=box_colors, width=2,
        )
        overlay = overlay_uint8.float() / 255.0
    return overlay


def _collect_masks_and_colors(detections):
    """Extract torch masks, colour indices and sort keys from detection dicts."""
    raw_masks = []
    color_indices = []
    sort_keys = []
    for det_idx, det in enumerate(detections):
        mask = det.get("mask")
        if mask is not None and isinstance(mask, torch.Tensor):
            raw_masks.append(mask)
            color_indices.append(det_idx % len(_PALETTE))
            hw = det.get("hw")
            if hw and "w" in hw and "h" in hw:
                sort_keys.append(float(hw["w"]) * float(hw["h"]))
            else:
                sort_keys.append(float(mask.numel()))
    return raw_masks, color_indices, sort_keys


def _composite_binary_masks(
    overlay: torch.Tensor,
    raw_masks: list[torch.Tensor],
    order: list[int],
    N: int,
    H: int,
    W: int,
    pixels: int,
    device: torch.device,
    ordered_colors: torch.Tensor,
    ordered_border_colors: torch.Tensor,
    interior_opacity: float,
    border_opacity: float,
    k: int,
    pad: int,
) -> torch.Tensor:
    """Index-map compositing: build a per-pixel top-mask map, then composite
    fill and borders each in one vectorised O(H*W) pass."""
    chunk_size = max(1, min(256, 128_000_000 // pixels))
    mask_idx_HW = torch.full((H, W), -1, dtype=torch.long, device=device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_ordered = [raw_masks[order[ri]] for ri in range(start, end)]
        needs_resize = any(m.shape != (H, W) for m in chunk_ordered)
        if needs_resize:
            chunk_stacked = torch.stack([
                _resize_mask(m.to(device=device, dtype=torch.float32), H, W, mode="nearest")
                for m in chunk_ordered
            ]).clamp_(0.0, 1.0)
        else:
            chunk_stacked = torch.stack(chunk_ordered).to(
                device=device, dtype=torch.float32
            ).clamp_(0.0, 1.0)

        chunk_ris = torch.arange(start, end, device=device, dtype=torch.long)
        chunk_binary = chunk_stacked > 0.5
        ri_map = torch.where(
            chunk_binary,
            chunk_ris.view(-1, 1, 1).expand_as(chunk_binary),
            torch.tensor(-1, device=device, dtype=torch.long),
        )
        mask_idx_HW = torch.maximum(mask_idx_HW, ri_map.amax(dim=0))
        del chunk_stacked, ri_map, chunk_binary

    has_mask = mask_idx_HW >= 0
    if has_mask.any():
        fill_color_3HW = ordered_colors[mask_idx_HW.clamp(min=0)].permute(2, 0, 1)
        fill_alpha = has_mask.float().unsqueeze(0) * interior_opacity
        overlay = overlay * (1.0 - fill_alpha) + fill_color_3HW * fill_alpha
        del fill_color_3HW, fill_alpha

    if has_mask.any():
        idx = mask_idx_HW
        is_boundary = (
            (idx != torch.roll(idx, 1, 1))
            | (idx != torch.roll(idx, -1, 1))
            | (idx != torch.roll(idx, 1, 0))
            | (idx != torch.roll(idx, -1, 0))
        ) & has_mask

        if k > 1:
            is_boundary = (
                F.max_pool2d(
                    is_boundary.float()[None, None], k, stride=1, padding=pad,
                )[0, 0]
                > 0.5
            )

        border_alpha_HW = F.avg_pool2d(
            is_boundary.float()[None, None], kernel_size=3, stride=1, padding=1,
        )[0, 0].clamp_(0.0, 1.0)

        if border_alpha_HW.any():
            border_color_3HW = ordered_border_colors[
                mask_idx_HW.clamp(min=0)
            ].permute(2, 0, 1)
            border_a = (border_alpha_HW * border_opacity).clamp_(0, 1).unsqueeze(0)
            overlay = overlay * (1.0 - border_a) + border_color_3HW * border_a
            del border_color_3HW, border_a
        del border_alpha_HW

    return overlay


def _composite_soft_masks(
    overlay: torch.Tensor,
    raw_masks: list[torch.Tensor],
    order: list[int],
    N: int,
    H: int,
    W: int,
    chunk_size: int,
    device: torch.device,
    palette: torch.Tensor,
    color_indices: list[int],
    interior_opacity: float,
    border_opacity: float,
    k: int,
    pad: int,
) -> torch.Tensor:
    """Sequential per-mask alpha compositing with sigmoid probabilities,
    anti-aliased contours, and layered blending."""
    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32
    fill_gamma = 0.9
    all_cidx = torch.tensor(color_indices, device=device, dtype=torch.long)
    colours_N3 = palette.float()[all_cidx]
    border_colours_N3 = (0.65 * colours_N3 + 0.35).clamp_(0.0, 1.0)

    for start in range(0, N, chunk_size):
        chunk_order = order[start : start + chunk_size]
        chunk_masks = []
        for gi in chunk_order:
            m = raw_masks[gi].to(device=device, dtype=compute_dtype)
            m = _resize_mask(m, H, W, mode="bilinear")
            chunk_masks.append(m)

        logits_B1HW = torch.stack(chunk_masks).unsqueeze(1)
        prob_BHW = torch.sigmoid(logits_B1HW.squeeze(1))
        del logits_B1HW

        fill_alpha_BHW = (prob_BHW.pow(fill_gamma) * interior_opacity).clamp_(0.0, 1.0)
        prob_B1HW = prob_BHW.unsqueeze(1)
        dil_BHW = F.max_pool2d(prob_B1HW, k, stride=1, padding=pad).squeeze(1)
        ero_BHW = -F.max_pool2d(-prob_B1HW, k, stride=1, padding=pad).squeeze(1)
        contour_BHW = (dil_BHW - ero_BHW).clamp_(0.0, 1.0)
        contour_BHW = F.avg_pool2d(
            contour_BHW.unsqueeze(1), kernel_size=3, stride=1, padding=1
        ).squeeze(1)
        border_alpha_BHW = (contour_BHW * border_opacity).clamp_(0.0, 1.0)
        del prob_B1HW, dil_BHW, ero_BHW, contour_BHW, prob_BHW

        for local_idx, global_idx in enumerate(chunk_order):
            fill_a = fill_alpha_BHW[local_idx].float()[None]
            border_a = border_alpha_BHW[local_idx].float()[None]
            fill_c = colours_N3[global_idx].float()[:, None, None]
            edge_c = border_colours_N3[global_idx].float()[:, None, None]
            overlay = overlay * (1.0 - fill_a) + fill_c * fill_a
            overlay = overlay * (1.0 - border_a) + edge_c * border_a

        del fill_alpha_BHW, border_alpha_BHW

    return overlay


def make_overlay_single(
    img_tensor: torch.Tensor | np.ndarray,
    detections: list[dict],
    draw_bbox: bool = False,
    interior_opacity: float = 0.35,
    border_opacity: float = 1.0,
    border_thickness: int = 3,
    masks_are_binary: bool = False,
) -> np.ndarray:
    """
    Overlay segmentation masks on an image.

    Two rendering paths:
      - **Binary** (``masks_are_binary=True``): single-pass index-map
        compositing.  Builds a per-pixel "top mask" map, then composites
        fill and contour borders each in one vectorised operation.
        Scales to hundreds of masks with minimal overhead.
      - **Soft**: sequential per-mask alpha compositing with sigmoid
        probabilities, anti-aliased contours, and layered blending.

    Args:
        img_tensor: (C, H, W) float32 torch tensor in [0, 1].
        detections: list of dicts with 'mask' (torch.Tensor), 'xy', 'hw'.
        draw_bbox: whether to draw bounding boxes.
        interior_opacity: opacity for the mask fill.
        border_opacity: opacity for the mask contour border.
        border_thickness: thickness of the contour border in pixels.
        masks_are_binary: if True, skip sigmoid (masks are already 0/1).
    Returns:
        (C, H, W) float32 torch tensor in [0, 1].
    """
    assert img_tensor.ndim == 3, "img must be CxHxW"
    C, H, W = img_tensor.shape
    device = img_tensor.device

    raw_masks, color_indices, sort_keys = _collect_masks_and_colors(detections)
    if not raw_masks:
        overlay = img_tensor.clone()
        if draw_bbox:
            overlay = _draw_bboxes(overlay, detections)
        return overlay

    N = len(raw_masks)
    # Largest masks first so smaller masks render on top at overlaps.
    order = sorted(range(N), key=lambda i: sort_keys[i], reverse=True)

    palette = _PALETTE.to(device=device, dtype=torch.float32)
    ordered_cidx = torch.tensor(
        [color_indices[order[i]] for i in range(N)], device=device, dtype=torch.long
    )
    ordered_colors = palette[ordered_cidx]                              # (N, 3)
    ordered_border_colors = (0.65 * ordered_colors + 0.35).clamp_(0, 1)  # (N, 3)

    k = max(1, int(border_thickness))
    if k % 2 == 0:
        k += 1
    pad = k // 2

    pixels = max(1, H * W)
    chunk_size = max(1, min(64, 32_000_000 // pixels))

    overlay = img_tensor.to(torch.float32).clone()

    if masks_are_binary:
        overlay = _composite_binary_masks(
            overlay, raw_masks, order, N, H, W, pixels, device,
            ordered_colors, ordered_border_colors,
            interior_opacity, border_opacity, k, pad,
        )
    else:
        overlay = _composite_soft_masks(
            overlay, raw_masks, order, N, H, W, chunk_size, device,
            palette, color_indices,
            interior_opacity, border_opacity, k, pad,
        )

    overlay.clamp_(0.0, 1.0)

    if draw_bbox:
        overlay = _draw_bboxes(overlay, detections)

    return overlay


def _overlay_binary_masks_numpy(
    base_img: np.ndarray,
    detections: list[dict],
    draw_bbox: bool = True,
    interior_opacity: float = 0.35,
    border_thickness: int = 3,
) -> np.ndarray:
    """Pure-numpy binary mask overlay via per-pixel index map.

    Fill and border compositing are each a single vectorised pass over the
    image (O(H*W)), independent of the number of masks.
    """
    if base_img.dtype != np.uint8:
        base_img = (np.clip(base_img, 0, 1) * 255).astype(np.uint8)
    H, W = base_img.shape[:2]
    overlay = base_img[..., :3].copy()

    palette = _PALETTE_NP
    P = len(palette)
    masks: list[np.ndarray] = []
    color_indices: list[int] = []
    sort_keys: list[float] = []
    for det_idx, det in enumerate(detections):
        raw = det.get("mask")
        if raw is None:
            continue
        m = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
        if m.shape != (H, W):
            m = np.array(Image.fromarray(m.astype(np.uint8)).resize(
                (W, H), resample=Image.Resampling.NEAREST,
            ))
        masks.append(np.ascontiguousarray(m))
        color_indices.append(det_idx % P)
        hw = det.get("hw")
        if hw and "w" in hw and "h" in hw:
            sort_keys.append(float(hw["w"]) * float(hw["h"]))
        else:
            sort_keys.append(float(m.size))

    if masks:
        N = len(masks)
        order = sorted(range(N), key=lambda i: sort_keys[i], reverse=True)

        # Build per-pixel index map (last/smallest mask wins)
        mask_idx = np.full((H, W), -1, dtype=np.int32)
        for ri, oi in enumerate(order):
            mask_idx[masks[oi] > 0] = ri

        has_mask = mask_idx >= 0

        # Vectorised fill composite
        if has_mask.any():
            ordered_colors = palette[
                np.array([color_indices[order[i]] for i in range(N)], dtype=np.intp)
            ]
            clamped = np.where(has_mask, mask_idx, 0)
            fill_rgb = ordered_colors[clamped]
            alpha = interior_opacity
            mask_3d = has_mask[:, :, np.newaxis]
            overlay = np.where(
                mask_3d,
                (alpha * fill_rgb.astype(np.float32)
                 + (1.0 - alpha) * overlay.astype(np.float32)
                 + 0.5).astype(np.uint8),
                overlay,
            )

        # Border from index-map edges
        if has_mask.any():
            border = np.zeros((H, W), dtype=np.bool_)
            border[:, 1:] |= (mask_idx[:, 1:] != mask_idx[:, :-1])
            border[:, :-1] |= (mask_idx[:, 1:] != mask_idx[:, :-1])
            border[1:, :] |= (mask_idx[1:, :] != mask_idx[:-1, :])
            border[:-1, :] |= (mask_idx[1:, :] != mask_idx[:-1, :])
            border &= has_mask

            if border_thickness > 1:
                from PIL import ImageFilter
                border_pil = Image.fromarray(border.view(np.uint8) * 255)
                for _ in range(max(1, border_thickness // 2)):
                    border_pil = border_pil.filter(ImageFilter.MaxFilter(3))
                border = np.asarray(border_pil) > 127

            if border.any():
                ordered_border = np.clip(
                    0.65 * ordered_colors.astype(np.float32) + 89.25, 0, 255,
                ).astype(np.uint8)
                border_rgb = ordered_border[np.where(has_mask & border, mask_idx, 0)]
                overlay[border] = border_rgb[border]

    if draw_bbox:
        from PIL import ImageDraw
        pil_overlay = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil_overlay)
        _COLOR_NAMES = [
            "red", "green", "blue", "yellow", "magenta",
            "cyan", "orange", "purple", "springgreen", "deeppink",
        ]
        for det_idx, det in enumerate(detections):
            xy = det.get("xy")
            hw_d = det.get("hw")
            if xy is None:
                continue
            cx, cy = xy["x"] * W, xy["y"] * H
            if hw_d and "w" in hw_d and "h" in hw_d:
                bw, bh = hw_d["w"] * W, hw_d["h"] * H
            else:
                bw, bh = 10.0, 10.0
            x0 = max(0, int(round(cx - bw / 2)))
            y0 = max(0, int(round(cy - bh / 2)))
            x1 = min(W - 1, int(round(cx + bw / 2)))
            y1 = min(H - 1, int(round(cy + bh / 2)))
            draw.rectangle([x0, y0, x1, y1],
                           outline=_COLOR_NAMES[det_idx % len(_COLOR_NAMES)], width=2)
        overlay = np.asarray(pil_overlay)

    return overlay


def overlay_detections_on_image_v2(
    image,
    detections: list[dict],
    draw_bbox: bool = True,
    masks_are_binary: bool = False,
):
    """
    Overlay detections on an image.

    When *masks_are_binary* is True a fast pure-numpy path is used that
    avoids all torch overhead (no float32 conversion, no tensor pooling).
    Falls back to the torch soft-alpha renderer for soft/logit masks.

    Accepts PIL Image, numpy array, torch tensor or image path.
    Returns an uint8 numpy array (H, W, 3).
    """
    base_img = load_frame(image)

    if masks_are_binary:
        return _overlay_binary_masks_numpy(base_img, detections, draw_bbox=draw_bbox)

    if base_img.dtype == np.uint8:
        base_img = base_img.astype(np.float32) / 255.0
    elif base_img.max() > 1.0:
        base_img = base_img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(base_img[..., :3]).permute(2, 0, 1).float()

    converted_dets = [dict(det) for det in detections]
    np_mask_indices: list[int] = []
    np_masks: list[np.ndarray] = []
    for i, d in enumerate(converted_dets):
        mask = d.get("mask")
        if mask is not None and not isinstance(mask, torch.Tensor):
            np_mask_indices.append(i)
            np_masks.append(np.asarray(mask))
    if np_masks:
        stacked_torch = torch.from_numpy(np.stack(np_masks)).float()
        for j, det_idx in enumerate(np_mask_indices):
            converted_dets[det_idx]["mask"] = stacked_torch[j]

    overlay_chw = make_overlay_single(
        img_tensor,
        converted_dets,
        draw_bbox=draw_bbox,
        masks_are_binary=False,
    )
    overlay_np = (
        (overlay_chw.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    )
    return overlay_np


def pack_detections(raw_list: list, segmentation: bool = True) -> list[dict]:
    """
    Convert a flat list of aux outputs into structured detections.
    Each detection is a dict with 'xy', 'hw', and optional 'mask'.

    Tolerates inconsistent model output patterns (e.g. repeated <size>
    tokens without a matching <seg>) by consuming elements based on type
    rather than a fixed stride.
    """
    dets = []
    xy = hw = mask = None
    for item in raw_list:
        if isinstance(item, dict):
            if "x" in item or "y" in item:
                if xy is not None and hw is not None:
                    dets.append({"xy": xy, "hw": hw, "mask": mask})
                    mask = None
                xy = item
                hw = None
            elif "h" in item or "w" in item:
                hw = item
        elif isinstance(item, torch.Tensor):
            if segmentation:
                mask = (torch.sigmoid(item) > 0.5).float()
                if xy is not None and hw is not None:
                    dets.append({"xy": xy, "hw": hw, "mask": mask})
                    xy = hw = mask = None
    if xy is not None and hw is not None:
        dets.append({"xy": xy, "hw": hw, "mask": mask})
    return dets


def normalize_aux_outputs(
    aux_outputs,
    pixel_mask_1hw: torch.Tensor | None,
    orig_hw: tuple[int, int],
    segmentation: bool = True,
):
    """
    Normalize model aux outputs into a list ready for pack_detections.
    - Crops masks to the active image region using pixel_mask if provided.
    - Resizes masks to the original image size (orig_hw: (H, W)).
    """
    processed_aux = []
    orig_h, orig_w = orig_hw
    min_h = min_w = 0
    act_h = act_w = None
    if pixel_mask_1hw is not None:
        nonzero = torch.nonzero(pixel_mask_1hw, as_tuple=False)
        if len(nonzero) > 0:
            min_h, min_w = nonzero.min(dim=0)[0]
            max_h, max_w = nonzero.max(dim=0)[0]
            act_h, act_w = (max_h - min_h + 1).item(), (max_w - min_w + 1).item()
    for item in aux_outputs:
        if segmentation and isinstance(item, torch.Tensor):
            mask = item
            if act_h is not None and act_w is not None:
                mask = mask[min_h : min_h + act_h, min_w : min_w + act_w]
            mask = mask.unsqueeze(0).unsqueeze(0).float()
            mask = F.interpolate(mask, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            processed_aux.append(mask.squeeze(0).squeeze(0).cpu())
        else:
            processed_aux.append(item)
    return processed_aux


def detections_from_batch_aux(
    aux_outputs,
    *,
    pixel_mask_1hw: torch.Tensor | None = None,
    orig_hw: tuple[int, int] | None = None,
    segmentation: bool = True,
) -> list[dict]:
    """Convert one sample's aux output to detection dicts.

    Accepts both the legacy ``list[dict | Tensor]`` format and the new
    :class:`AuxOutput` from either engine.

    Returns ``[{"xy": {...}, "hw": {...}, "mask": <H,W> | None}, ...]``.
    """
    from falcon_perception.aux_output import AuxOutput

    if isinstance(aux_outputs, AuxOutput):
        bboxes = pair_bbox_entries(aux_outputs.bboxes_raw)
        masks_rle = aux_outputs.masks_rle if segmentation else []
        decoded_masks: list[np.ndarray] = []
        for rle in masks_rle:
            m = decode_coco_rle(rle)
            if m is not None:
                if orig_hw is not None and m.shape != orig_hw:
                    from PIL import Image as _PILImage
                    m = np.array(_PILImage.fromarray(m).resize(
                        (orig_hw[1], orig_hw[0]), _PILImage.NEAREST,
                    ))
                decoded_masks.append(m)
        dets: list[dict] = []
        for idx, b in enumerate(bboxes):
            mask_arr = decoded_masks[idx] if idx < len(decoded_masks) else None
            mask_t = torch.from_numpy(mask_arr).float() if mask_arr is not None else None
            dets.append({
                "xy": {"x": b.get("x", 0), "y": b.get("y", 0)},
                "hw": {"h": b.get("h", 0), "w": b.get("w", 0)},
                "mask": mask_t,
            })
        return dets

    # Legacy path: list[dict | Tensor]
    if orig_hw is None:
        orig_hw = (512, 512)
    processed_aux = normalize_aux_outputs(
        aux_outputs,
        pixel_mask_1hw=pixel_mask_1hw,
        orig_hw=orig_hw,
        segmentation=segmentation,
    )
    return pack_detections(processed_aux, segmentation=segmentation)


def pair_bbox_entries(raw: list[dict]) -> list[dict]:
    """Pair [{x,y}, {h,w}, ...] into [{x,y,h,w}, ...].

    Coordinate and size predictions are expected to be normalised to [0, 1].
    """
    bboxes: list[dict] = []
    current: dict = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        current.update(entry)
        if all(k in current for k in ("x", "y", "h", "w")):
            bboxes.append(dict(current))
            current = {}
    return bboxes


def _mask_to_bbox_xywh(mask: np.ndarray, img_w: int, img_h: int):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        cx = cy = 0.5
        w = h = 10.0 / max(img_w, img_h)
        return {"x": cx, "y": cy}, {"w": w, "h": h}
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cx = (cmin + cmax + 1) / 2 / img_w
    cy = (rmin + rmax + 1) / 2 / img_h
    w = (cmax - cmin + 1) / img_w
    h = (rmax - rmin + 1) / img_h
    return {"x": cx, "y": cy}, {"w": w, "h": h}


def _to_display_image(img_tensor, image_processor, target_hw: tuple[int, int] | None):
    if img_tensor is None:
        return None
    img = img_tensor
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    else:
        img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]  # remove temporal dim
    if img.ndim != 3 or img.shape[-1] != 3:
        return None
    if image_processor.do_normalize:
        mean = np.array(image_processor.image_mean, dtype=np.float32)
        std = np.array(image_processor.image_std, dtype=np.float32)
        img = img * std + mean
    if image_processor.do_rescale:
        img = img / image_processor.rescale_factor
    img = np.clip(img, 0, 255).astype(np.uint8)
    if target_hw and img.shape[:2] != target_hw:
        try:
            img = np.array(Image.fromarray(img).resize((target_hw[1], target_hw[0])))
        except Exception as exc:  # pragma: no cover - visualization only
            print(f"[plot] Resize failed for display image: {exc}")
    return img


def _img_tensor_hw(img_tensor) -> tuple[int, int] | None:
    if img_tensor is None:
        return None
    img = img_tensor
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    else:
        img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[-1] == 3:
        return img.shape[0], img.shape[1]
    return None


def decode_coco_rle(rle_dict: dict) -> np.ndarray | None:
    """Decode a COCO RLE dict ``{"counts": str, "size": [H, W]}`` to a binary mask.

    Handles both string and bytes ``counts`` (pycocotools expects bytes
    internally, so string counts are encoded to UTF-8 automatically).

    Returns (H, W) uint8 numpy array, or *None* on failure.
    """
    from pycocotools import mask as mask_utils

    try:
        rle = rle_dict
        if isinstance(rle.get("counts"), str):
            rle = {**rle_dict, "counts": rle_dict["counts"].encode("utf-8")}
        return mask_utils.decode(rle).astype(np.uint8)
    except Exception:
        return None


def _sequence_masks_to_array(seq):
    """
    Paged inference: decode COCO RLE masks from ``seq.output_aux.masks_rle``.

    Returns a (K, H, W) uint8 numpy array, or *None* if no decodable masks.
    """
    aux = getattr(seq, "output_aux", None)
    masks_rle = (aux.masks_rle if aux else None) or getattr(seq, "all_masks_rle", None) or []
    if not masks_rle:
        return None

    decoded_masks = []
    for item in masks_rle:
        if not isinstance(item, dict):
            continue
        if "counts" in item and "size" in item:
            mask = decode_coco_rle(item)
            if mask is not None and mask.any():
                decoded_masks.append(mask)
    if not decoded_masks:
        return None
    return np.stack(decoded_masks, axis=0)


def _sequence_target_hw(seq) -> tuple[int, int] | None:
    """
    Determine the (H,W) that sequence masks should be decoded/resized to.
    Priority:
      1) seq.original_image_size
      2) seq.pil_image size
      3) image_tensor spatial shape
    """
    target_hw = getattr(seq, "original_image_size", None)
    if target_hw is not None:
        return int(target_hw[0]), int(target_hw[1])
    pil_img = getattr(seq, "pil_image", None)
    if pil_img is not None and hasattr(pil_img, "height") and hasattr(pil_img, "width"):
        return int(pil_img.height), int(pil_img.width)
    return _img_tensor_hw(getattr(seq, "image_tensor", None))


def detections_from_sequence(
    seq,
    *,
    target_hw: tuple[int, int] | None = None,
) -> list[dict]:
    """
    Paged inference helper: convert a Sequence into the same detection dict list
    shape used by batch visualization.

    For detection tasks (no masks), bbox-only detections are returned so that
    boxes can still be drawn on the image.
    """
    target_hw = target_hw or _sequence_target_hw(seq)
    if target_hw is None:
        return []

    masks_np = _sequence_masks_to_array(seq)
    aux = getattr(seq, "output_aux", None)
    bboxes = pair_bbox_entries(aux.bboxes_raw if aux else getattr(seq, "all_bbox", []))

    if masks_np is None:
        if not bboxes:
            return []
        detections: list[dict] = []
        for box in bboxes:
            detections.append(
                {
                    "xy": {"x": box.get("x", 0.5), "y": box.get("y", 0.5)},
                    "hw": {"w": box.get("w", 0.0), "h": box.get("h", 0.0)},
                    "mask": None,
                }
            )
        return detections

    target_h, target_w = target_hw
    if masks_np.shape[1:] != (target_h, target_w):
        resized_masks = []
        for m in masks_np:
            m_uint8 = (m > 0).astype(np.uint8)
            m_resized = np.array(
                Image.fromarray(m_uint8).resize(
                    (target_w, target_h), resample=Image.Resampling.NEAREST
                )
            )
            resized_masks.append(m_resized)
        masks_np = np.stack(resized_masks, axis=0)

    detections: list[dict] = []
    for idx, mask in enumerate(masks_np):
        if idx < len(bboxes):
            box = bboxes[idx]
            detections.append(
                {
                    "xy": {"x": box.get("x", 0.5), "y": box.get("y", 0.5)},
                    "hw": {"w": box.get("w", 0.0), "h": box.get("h", 0.0)},
                    "mask": mask,
                }
            )
        else:
            xy, hw = _mask_to_bbox_xywh(mask, target_w, target_h)
            detections.append({"xy": xy, "hw": hw, "mask": mask})
    return detections


def _get_sequence_base_image(seq, image_processor, target_hw: tuple[int, int]):
    base_img = _to_display_image(
        getattr(seq, "image_tensor", None), image_processor, target_hw
    )
    if base_img is None:
        try:
            pil_img = getattr(seq, "pil_image", None)
            if pil_img is None or not hasattr(pil_img, "convert"):
                img_ref = getattr(seq, "_image_raw", None)
                if isinstance(img_ref, str) and os.path.isfile(img_ref):
                    pil_img = Image.open(img_ref)
                elif hasattr(img_ref, "convert"):
                    pil_img = img_ref
            if pil_img is not None and hasattr(pil_img, "convert"):
                pil_img = pil_img.convert("RGB")  # type: ignore[union-attr]
                base_img = np.array(
                    pil_img.resize((target_hw[1], target_hw[0]), resample=Image.Resampling.BILINEAR)
                )
        except Exception as exc:  # pragma: no cover - visualization only
            print(f"[plot] Fallback to white canvas: {exc}")
            base_img = None
    if base_img is None:
        base_img = np.ones((target_hw[0], target_hw[1], 3), dtype=np.uint8) * 255
    return base_img


def _cap_hw(hw: tuple[int, int], max_side: int) -> tuple[int, int]:
    """Downscale (H, W) so that the longest side is at most *max_side*."""
    h, w = hw
    if max(h, w) <= max_side:
        return hw
    scale = max_side / max(h, w)
    return int(h * scale), int(w * scale)


def render_sequence_overlay(
    seq,
    image_processor,
    draw_bbox: bool = True,
    max_vis_size: int = 2048,
):
    """
    Render a sequence's masks and/or bboxes as a numpy uint8 overlay (or None).

    Uses the vectorised soft-alpha renderer (``make_overlay_single``)
    with anti-aliased contours and fast index-map compositing for binary
    masks.  For detection tasks (no masks), renders bounding boxes only.

    ``max_vis_size`` caps the longest side of the output visualization so
    that very large original images (e.g. 4608x4608) are not rendered at
    full resolution.
    """
    target_hw = _sequence_target_hw(seq)
    if target_hw is None:
        return None
    target_hw = _cap_hw(target_hw, max_vis_size)
    detections = detections_from_sequence(seq, target_hw=target_hw)
    if not detections:
        return None
    base_img = _get_sequence_base_image(seq, image_processor, target_hw)
    return overlay_detections_on_image_v2(
        base_img, detections, draw_bbox=draw_bbox, masks_are_binary=True,
    )


# Comparison panel visualization

_COMPARISON_COLORS = _PALETTE_UINT8[:8]


def _resize_masks_to(masks: list[np.ndarray], H: int, W: int) -> list[np.ndarray]:
    """Resize a list of binary masks to (H, W) using nearest-neighbour.

    Skips masks that are already the target size.
    """
    out: list[np.ndarray] = []
    for m in masks:
        if m is None:
            continue
        if m.shape[0] == H and m.shape[1] == W:
            out.append(m)
        else:
            out.append(np.array(
                Image.fromarray(m.astype(np.uint8)).resize(
                    (W, H), resample=Image.Resampling.NEAREST),
            ))
    return out


def _overlay_masks_indexed(
    masks: list[np.ndarray],
    background: np.ndarray,
    opacity: float = 0.35,
) -> np.ndarray:
    """Fast mask overlay using a per-pixel index map — no per-mask resize.

    Caller must ensure all masks are already at the background resolution.
    Complexity: O(N*H*W) for building the index map, O(H*W) for compositing.
    """
    H, W = background.shape[:2]
    overlay = background[..., :3].copy()
    N = len(masks)
    if N == 0:
        return overlay

    palette = _PALETTE_NP
    P = len(palette)

    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    order = np.argsort(-areas)  # largest first → smallest wins (overwrites)

    mask_idx = np.full((H, W), -1, dtype=np.int32)
    for ri, oi in enumerate(order):
        mask_idx[masks[oi] > 0] = ri

    has_mask = mask_idx >= 0
    if not has_mask.any():
        return overlay

    ordered_colors = palette[np.array([int(order[i]) % P for i in range(N)], dtype=np.intp)]
    clamped = np.where(has_mask, mask_idx, 0)
    fill_rgb = ordered_colors[clamped]
    overlay = np.where(
        has_mask[:, :, np.newaxis],
        (opacity * fill_rgb.astype(np.float32)
         + (1.0 - opacity) * overlay.astype(np.float32)
         + 0.5).astype(np.uint8),
        overlay,
    )

    # Lightweight border from index-map edges (single-pixel, no dilation).
    border = np.zeros((H, W), dtype=np.bool_)
    border[:, 1:] |= mask_idx[:, 1:] != mask_idx[:, :-1]
    border[:, :-1] |= mask_idx[:, 1:] != mask_idx[:, :-1]
    border[1:, :] |= mask_idx[1:, :] != mask_idx[:-1, :]
    border[:-1, :] |= mask_idx[1:, :] != mask_idx[:-1, :]
    border &= has_mask
    if border.any():
        bright = np.clip(
            0.65 * ordered_colors.astype(np.float32) + 89.25, 0, 255,
        ).astype(np.uint8)
        overlay[border] = bright[np.where(border, mask_idx, 0)][border]

    return overlay


def save_comparison_vis(
    image_pil: Image.Image,
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    expression: str,
    best_iou: float,
    save_path: str | os.PathLike,
    max_side: int = 640,
):
    """Save a three-panel comparison image: Original | GT masks | Pred masks.

    Args:
        image_pil: Original PIL image.
        gt_masks: List of (H, W) uint8 ground-truth binary masks.
        pred_masks: List of (H, W) uint8 predicted binary masks.
        expression: Text expression / query string shown in the header.
        best_iou: Best matched IoU score shown in the header.
        save_path: Output file path (JPEG recommended).
        max_side: Cap each panel to this many pixels on the longest side.
    """
    from PIL import ImageDraw

    orig_w, orig_h = image_pil.size
    scale = min(1.0, max_side / max(orig_w, orig_h))
    w = int(orig_w * scale)
    h = int(orig_h * scale)

    bg = np.array(image_pil.resize((w, h)).convert("RGB"))

    gt_resized = _resize_masks_to(gt_masks, h, w)
    pred_resized = _resize_masks_to(pred_masks, h, w)

    header_h = 40
    canvas = Image.new("RGB", (w * 3, h + header_h), (255, 255, 255))
    canvas.paste(Image.fromarray(bg), (0, header_h))
    canvas.paste(Image.fromarray(_overlay_masks_indexed(gt_resized, bg)), (w, header_h))
    canvas.paste(Image.fromarray(_overlay_masks_indexed(pred_resized, bg)), (w * 2, header_h))

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (5, 5),
        f"\"{expression}\"  IoU={best_iou:.3f}  pred={len(pred_masks)} gt={len(gt_masks)}",
        fill=(0, 0, 0),
    )
    draw.text((w + 5, 5), "GT masks", fill=(0, 128, 0))
    draw.text((w * 2 + 5, 5), "Pred masks", fill=(255, 0, 0))

    canvas.save(str(save_path), quality=90)


def load_frame(frame):
    if isinstance(frame, np.ndarray):
        img = frame
    elif isinstance(frame, Image.Image):
        img = np.array(frame)
    elif isinstance(frame, str) and os.path.isfile(frame):
        img = np.array(Image.open(frame).convert("RGB"))
    else:
        raise ValueError(f"Invalid frame type: {type(frame)=}")
    return img


def _safe_filename_stem(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return "sample"
    # Avoid path separators and other problematic characters.
    n = n.replace(os.sep, "_")
    if os.altsep:
        n = n.replace(os.altsep, "_")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ ")
    cleaned = "".join(ch if ch in allowed else "_" for ch in n)
    cleaned = cleaned.strip().replace(" ", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("._-")
    return cleaned or "sample"


def render_batch_inference_outputs(
    run_name: str,
    inputs: dict,
    aux_outputs: list,
    image_paths: list[str],
    task: str,
    out_dir: str | Path = "inference_outputs",
    queries: list[str] | None = None,
) -> None:
    """Render detections for a batch and save overlays."""
    out_dir = Path(out_dir)
    mask_out_dir = out_dir / "masks"
    mask_out_dir.mkdir(exist_ok=True, parents=True)
    pixel_mask_batch = inputs["pixel_mask"][:, 0]
    has_masks = task == "segmentation"

    from falcon_perception.aux_output import AuxOutput

    for i, aux in enumerate(aux_outputs):
        is_aux_output = isinstance(aux, AuxOutput)
        if not is_aux_output and not aux:
            print(f"Sample {i}: No detections found.")
            continue
        if is_aux_output and not aux.bboxes_raw and not aux.masks_rle:
            print(f"Sample {i}: No detections found.")
            continue

        orig_img = inputs.get("__orig_images__", None)
        if orig_img is None:
            from falcon_perception.data import load_image

            orig_img = load_image(image_paths[i])
            if orig_img is not None and hasattr(orig_img, "convert"):
                orig_img = orig_img.convert("RGB")  # type: ignore[union-attr]
        else:
            orig_img = orig_img[i]
        h_attr = getattr(orig_img, "height", None)
        w_attr = getattr(orig_img, "width", None)
        if isinstance(h_attr, (int, float)) and isinstance(w_attr, (int, float)):
            orig_h, orig_w = int(h_attr), int(w_attr)
        else:
            arr = np.asarray(orig_img)
            orig_h, orig_w = int(arr.shape[0]), int(arr.shape[1])

        detections = detections_from_batch_aux(
            aux,
            pixel_mask_1hw=pixel_mask_batch[i] if not is_aux_output else None,
            orig_hw=(orig_h, orig_w),
            segmentation=has_masks,
        )
        overlay = overlay_detections_on_image_v2(orig_img, detections, draw_bbox=True, masks_are_binary=True)
        overlay = Image.fromarray(overlay)
        if queries and i < len(queries):
            stem = _safe_filename_stem(queries[i])
        else:
            stem = _safe_filename_stem(f"{run_name}_{i}")
        out_path = mask_out_dir / f"{stem}.jpg"
        overlay.save(out_path)
        print(f"[plot] Saved {out_path}")


def render_paged_inference_outputs(
    sequences: list,
    image_processor,
    output_dir: str | Path = "mask_plots",
    task: str = "segmentation",
) -> None:
    """
    Visualize per-sequence masks (shape: [M, H, W]) and/or predicted boxes.
    """
    from falcon_perception.data import ImageProcessor  # lazy import

    assert isinstance(image_processor, ImageProcessor)
    base_out_dir = Path(output_dir)
    mask_out_dir = base_out_dir / "masks" if task == "segmentation" else base_out_dir / "boxes"
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    for seq_idx, seq in enumerate(sequences):
        overlay = render_sequence_overlay(seq, image_processor, draw_bbox=True)
        if overlay is None:
            print(f"[plot] No detections for sequence {seq_idx}, skipping.")
            continue
        query = seq.text.split("start_of_query|>")[-1].split("<|")[0]
        stem = _safe_filename_stem(query)
        out_path = mask_out_dir / f"{seq_idx:03d}_{stem}.jpg"
        Image.fromarray(overlay).save(str(out_path))
        print(f"[plot] Saved {out_path}")
