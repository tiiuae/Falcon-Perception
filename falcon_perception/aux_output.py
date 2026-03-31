# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Unified per-sequence auxiliary output accumulated during decode.

Both the batch and paged inference engines populate an `AuxOutput` per
image/sequence during autoregressive decoding.  All data stays on GPU as
tensors until explicit materialization, avoiding host-device sync in the
hot loop.

Lifecycle
---------
1. **Decode** — ``append_bbox()`` every step, ``append_segm()`` on <seg> steps.
2. **Finalize** — ``finalize_masks()`` batches all seg tokens for one image
   into a single einsum + threshold + RLE encode pass.
3. **Read** — ``bboxes_raw`` / ``masks_rle`` are plain Python objects ready
   for serialization or visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import einops as E
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from torch import Tensor


def _rle_gpu_prep(binary_masks: torch.Tensor) -> tuple:
    """GPU phase: compute diff/nonzero for COCO RLE.  No CPU sync."""
    C, H, W = binary_masks.shape
    has_any = E.reduce(binary_masks, "c h w -> c", reduction="any")
    binary_col = E.rearrange(binary_masks, "c h w -> c (w h)")
    diffs = binary_col[:, 1:] != binary_col[:, :-1]
    nz = torch.nonzero(diffs, as_tuple=False)
    first_vals = binary_col[:, 0]
    return nz, has_any, first_vals, C, H, W


def _rle_cpu_encode(prep: tuple) -> list[dict]:
    """CPU phase: transfer to CPU and encode COCO RLE strings."""
    nz, has_any, first_vals, C, H, W = prep
    nz_cpu = nz.cpu().numpy()
    has_any_cpu = has_any.cpu().numpy()
    first_vals_cpu = first_vals.cpu().numpy()

    N = H * W
    if nz_cpu.shape[0] > 0:
        mask_ids = nz_cpu[:, 0]
        change_cols = nz_cpu[:, 1]
        uniq, grp_starts = np.unique(mask_ids, return_index=True)
        grp_ends = np.append(grp_starts[1:], len(mask_ids))
        mask_to_grp = {
            int(m): (int(gs), int(ge))
            for m, gs, ge in zip(uniq, grp_starts, grp_ends)
        }
    else:
        change_cols = np.array([], dtype=np.intp)
        mask_to_grp = {}

    results = []
    for i in range(C):
        if not has_any_cpu[i]:
            continue

        if i in mask_to_grp:
            gs, ge = mask_to_grp[i]
            cidx = change_cols[gs:ge]
        else:
            cidx = np.array([], dtype=np.intp)

        num_runs = len(cidx) + 1
        starts = np.empty(num_runs, dtype=np.intp)
        starts[0] = 0
        if len(cidx) > 0:
            starts[1:] = cidx + 1

        counts = np.empty(num_runs, dtype=np.uint32)
        if num_runs > 1:
            counts[:-1] = np.diff(starts)
        counts[-1] = N - starts[-1]

        # COCO RLE expects counts starting from 0-valued run.
        if first_vals_cpu[i]:
            counts = np.concatenate([[0], counts])

        rle = {"counts": counts.tolist(), "size": [H, W]}
        rle = mask_utils.frPyObjects(rle, H, W)
        rle["counts"] = rle["counts"].decode("utf-8")
        results.append(rle)

    return results


@dataclass
class AuxOutput:
    """Per-sequence auxiliary predictions accumulated during decode.

    All list fields grow by one element per decode step (bbox) or per
    <seg> token (segm).  Tensors are kept on the original device until
    :meth:`materialize_bboxes` or :meth:`finalize_masks` is called.
    """

    # Per-step bbox data (always appended, even when is_coord/is_size=False)
    _coord_xy: list[Tensor] = field(default_factory=list)  # (2,) float32 GPU
    _size_hw: list[Tensor] = field(default_factory=list)    # (2,) float32 GPU
    _is_coord: list[Tensor] = field(default_factory=list)   # () bool GPU
    _is_size: list[Tensor] = field(default_factory=list)    # () bool GPU

    # Incrementally maintained unfiltered coord data for dedup
    _xy_cat: Tensor | None = field(default=None, repr=False)        # (S, 2) GPU
    _is_coord_cat: Tensor | None = field(default=None, repr=False)  # (S,) GPU bool

    # Segmentation embeddings + masks (always appended; filtered at finalization)
    _segm_embeds: list[Tensor] = field(default_factory=list)  # (D_segm,) GPU
    _segm_masks: list[Tensor] = field(default_factory=list)   # () bool GPU

    # Populated at finalization
    bboxes_raw: list[dict] = field(default_factory=list)
    masks_rle: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Decode-time helpers (GPU-only, sync-free)
    # ------------------------------------------------------------------

    def append_bbox(
        self,
        xy: Tensor,
        hw: Tensor,
        is_coord: Tensor,
        is_size: Tensor,
    ) -> None:
        """Record one decode step's bbox prediction (all GPU tensors)."""
        self._coord_xy.append(xy)
        self._size_hw.append(hw)
        self._is_coord.append(is_coord)
        self._is_size.append(is_size)
        # Incremental cat to avoid rebuilding history from scratch each step.
        xy_row = xy.unsqueeze(0)             # (1, 2)
        ic_row = is_coord.unsqueeze(0)       # (1,)
        if self._xy_cat is None:
            self._xy_cat = xy_row
            self._is_coord_cat = ic_row
        else:
            self._xy_cat = torch.cat([self._xy_cat, xy_row])         # type: ignore[list-item]
            self._is_coord_cat = torch.cat([self._is_coord_cat, ic_row])  # type: ignore[list-item]

    def append_segm(self, embed: Tensor, is_segm: Tensor | None = None) -> None:
        """Record segm embedding + optional GPU bool mask.

        When ``is_segm`` is provided (paged path), it is stored as a GPU
        bool and used to filter at finalization — never evaluated on CPU.
        When omitted (batch path pre-filters callers), all embeds are valid.
        """
        self._segm_embeds.append(embed)
        if is_segm is not None:
            self._segm_masks.append(is_segm)

    @property
    def segm_embeds(self) -> list[Tensor]:
        """Return only the embeddings where is_segm was True.

        If no masks were recorded (batch inference path), returns all.
        """
        if not self._segm_masks:
            return self._segm_embeds
        mask_cpu = torch.stack(self._segm_masks).cpu().tolist()
        return [e for e, m in zip(self._segm_embeds, mask_cpu) if m]

    def coord_history_raw(self) -> tuple[Tensor, Tensor] | None:
        """Return ``(all_xy, is_coord_mask)`` or None if no history.

        Returns the **unfiltered** (S, 2) xy tensor and (S,) bool mask,
        maintained incrementally via ``append_bbox``.  The caller
        (``dedup_single_coord``) incorporates the mask into its check
        without boolean indexing (avoids dynamic shapes and GPU sync).
        """
        if self._xy_cat is None:
            return None
        return self._xy_cat, self._is_coord_cat

    # ------------------------------------------------------------------
    # Materialization (GPU → CPU, called once per sequence at completion)
    # ------------------------------------------------------------------

    def materialize_bboxes(self) -> list[dict]:
        """Convert GPU bbox tensors to a CPU list of dicts.

        Returns interleaved ``[{x, y}, {h, w}, ...]`` matching the
        legacy format expected by ``pair_bbox_entries``.
        """
        if not self._coord_xy:
            return self.bboxes_raw

        xy_N2 = torch.stack(self._coord_xy).cpu()
        hw_N2 = torch.stack(self._size_hw).cpu()
        is_coord_N = torch.stack(self._is_coord).cpu()
        is_size_N = torch.stack(self._is_size).cpu()

        result: list[dict] = []
        for i in range(len(xy_N2)):
            if is_coord_N[i]:
                result.append({"x": xy_N2[i, 0].item(), "y": xy_N2[i, 1].item()})
            if is_size_N[i]:
                result.append({"h": hw_N2[i, 0].item(), "w": hw_N2[i, 1].item()})
        return result

    def finalize_masks(
        self,
        hr_image_features: Tensor | None,
        threshold: float = 0.3,
        original_image_size: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Compute segmentation masks from stored embeddings.

        ``einsum → bilinear upsample → sigmoid → threshold → COCO RLE``.

        When *original_image_size* ``(H, W)`` is provided the logit masks
        are bilinear-upsampled to that resolution **before** thresholding,
        so the binary boundary is placed with sub-pixel precision at the
        display resolution instead of at the (smaller) feature-map size.
        """
        embeds = self.segm_embeds
        if not embeds or hr_image_features is None:
            return []
        tokens = torch.cat([e.unsqueeze(0) for e in embeds], dim=0)
        masks = torch.einsum("dhw,kd->khw", hr_image_features, tokens)

        if original_image_size is not None:
            tgt_h, tgt_w = original_image_size
            if (masks.shape[-2], masks.shape[-1]) != (tgt_h, tgt_w):
                masks = F.interpolate(
                    masks.float().unsqueeze(1),
                    size=(tgt_h, tgt_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

        binary = torch.sigmoid(masks.float()) > threshold
        return _rle_cpu_encode(_rle_gpu_prep(binary))

    def finalize(
        self,
        hr_image_features: Tensor | None = None,
        threshold: float = 0.3,
        task: str = "segmentation",
        original_image_size: tuple[int, int] | None = None,
    ) -> None:
        """One-shot materialization of both bboxes and masks.

        After this call, ``bboxes_raw`` and ``masks_rle`` are populated
        and GPU tensor lists can be freed.
        """
        self.bboxes_raw = self.materialize_bboxes()
        if task == "segmentation":
            self.masks_rle = self.finalize_masks(
                hr_image_features, threshold, original_image_size,
            )
        self._free_gpu()

    def _free_gpu(self) -> None:
        """Release GPU tensor lists after materialization."""
        self._coord_xy.clear()
        self._size_hw.clear()
        self._is_coord.clear()
        self._is_size.clear()
        self._xy_cat = None
        self._is_coord_cat = None
        self._segm_embeds.clear()
        self._segm_masks.clear()
