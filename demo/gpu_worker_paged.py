"""Falcon Perception GPU worker using PagedInferenceEngine.

Drop-in replacement for gpu_worker.py.  Same queue protocol —
receives ``(frames_rgb, queries, task)`` and returns
``("result", xyxy_list, masks_list)`` — but uses the paged inference
engine which batches decode steps across all queries with CUDA graphs.

For *N* queries on *F* frames the engine sees *F×N* Sequence objects.
All decode in parallel, giving much higher GPU utilisation than the
batch-engine's sequential per-query loop.

When ``task="segmentation"`` the worker also returns per-detection binary
masks (resized to original frame resolution) alongside bounding boxes.
"""

import sys


import torch
import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def _decode_and_resize_masks(
    masks_rle: list[dict],
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Decode COCO RLE masks and resize to (N, target_h, target_w)."""
    if not masks_rle:
        return np.empty((0, target_h, target_w), dtype=np.uint8)
    decoded = []
    for rle in masks_rle:
        m = mask_utils.decode(rle)  # (rle_h, rle_w) uint8
        if m.shape[0] != target_h or m.shape[1] != target_w:
            m = cv2.resize(
                m, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )
        decoded.append(m)
    return np.stack(decoded, axis=0).astype(np.uint8)


def _rle_to_xyxy(masks_rle: list[dict], target_h: int, target_w: int) -> np.ndarray:
    """Convert COCO RLE masks to xyxy boxes in target image coordinates.

    Uses pycocotools.mask.toBbox (returns xywh) then converts to xyxy,
    scaling from the RLE resolution to (target_h, target_w).
    """
    if not masks_rle:
        return np.empty((0, 4), dtype=np.float32)
    rle_h, rle_w = masks_rle[0]["size"]
    xywh = mask_utils.toBbox(masks_rle).astype(np.float32)  # (N, 4)
    xywh[:, 0] *= target_w / rle_w
    xywh[:, 1] *= target_h / rle_h
    xywh[:, 2] *= target_w / rle_w
    xywh[:, 3] *= target_h / rle_h
    xyxy = xywh.copy()
    xyxy[:, 2] += xyxy[:, 0]
    xyxy[:, 3] += xyxy[:, 1]
    return xyxy


def _bboxes_to_xyxy(
    bboxes: list[dict], img_w: int, img_h: int
) -> np.ndarray:
    """Convert paired bbox dicts (cx/cy/w/h, normalised) to pixel xyxy."""
    return np.array(
        [
            [
                (b["x"] - b["w"] / 2) * img_w,
                (b["y"] - b["h"] / 2) * img_h,
                (b["x"] + b["w"] / 2) * img_w,
                (b["y"] + b["h"] / 2) * img_h,
            ]
            for b in bboxes
        ],
        dtype=np.float32,
    )


def run(gpu_id, in_queue, out_queue, config):
    """Load model + paged engine on *gpu_id*, then loop on the queues."""
    torch.cuda.set_device(gpu_id)

    from falcon_perception import (
        build_prompt_for_task,
        load_and_prepare_model,
        setup_torch_config,
    )
    from falcon_perception.data import ImageProcessor
    from falcon_perception.paged_inference import (
        PagedInferenceEngine,
        SamplingParams,
        Sequence,
        engine_config_for_gpu,
    )
    from falcon_perception.visualization_utils import pair_bbox_entries

    setup_torch_config()

    model, tokenizer, _ = load_and_prepare_model(
        hf_model_id=config["hf_model_id"],
        hf_revision=config.get("hf_revision", "main"),
        device=f"cuda:{gpu_id}",
        dtype=config.get("dtype", "float32"),
        compile=config.get("compile", True),
    )
    device = f"cuda:{gpu_id}"

    # --- Build paged engine with auto-detected GPU preset ---
    image_processor = ImageProcessor(patch_size=model.args.spatial_patch_size, merge_size=1)
    ecfg = engine_config_for_gpu(
        max_image_size=config.get("max_dimension", 1024),
        device=device,
        dtype=model.dtype,
    )
    page_size = ecfg.get("page_size", 128)
    max_seq_length = page_size * 32  # 4096 — ample for detection

    engine = PagedInferenceEngine(
        model,
        tokenizer,
        image_processor,
        max_batch_size=ecfg["max_batch_size"],
        max_seq_length=max_seq_length,
        n_pages=ecfg["n_pages"],
        page_size=page_size,
        prefill_length_limit=ecfg.get("prefill_length_limit", -1),
        seed=42,
        enable_hr_cache=True,
        capture_cudagraph=True,
    )

    stop_ids = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]

    out_queue.put(("ready", gpu_id))

    while True:
        msg = in_queue.get()
        if msg is None:
            break

        frames_rgb, queries, task = msg
        is_segm = task == "segmentation"
        n_queries = len(queries)
        pil_images = [Image.fromarray(f) for f in frames_rgb]

        # Build one Sequence per (frame, query) pair
        sequences: list[Sequence] = []
        for fi, pil_img in enumerate(pil_images):
            for qi, query in enumerate(queries):
                prompt = build_prompt_for_task(query, task)
                seq = Sequence(
                    text=prompt,
                    image=pil_img,
                    min_image_size=config.get("min_dimension", 512),
                    max_image_size=config.get("max_dimension", 1024),
                    request_idx=fi * n_queries + qi,
                    task=task,
                )
                sequences.append(seq)

        sampling = SamplingParams(
            max_new_tokens=config.get("max_new_tokens", 2048),
            stop_token_ids=stop_ids,
            coord_dedup_threshold=0.01,
        )

        done = engine.generate(sequences, sampling_params=sampling, temperature=0.0)

        # Group results back to per-frame xyxy arrays (and masks if segm)
        xyxy_results: list[np.ndarray] = []
        mask_results: list[np.ndarray | None] = []
        for fi, pil_img in enumerate(pil_images):
            w, h = pil_img.size
            frame_xyxy: list[np.ndarray] = []
            frame_masks: list[np.ndarray] = []
            for qi in range(n_queries):
                seq = done[fi * n_queries + qi]
                bboxes = pair_bbox_entries(seq.output_aux.bboxes_raw)
                if not bboxes:
                    continue

                n_det = len(bboxes)

                if is_segm and len(seq.output_aux.masks_rle) == n_det:
                    # Segmentation: derive tight xyxy from RLE directly
                    frame_xyxy.append(
                        _rle_to_xyxy(seq.output_aux.masks_rle, h, w)
                    )
                    masks = _decode_and_resize_masks(
                        seq.output_aux.masks_rle, h, w
                    )
                    frame_masks.append(masks)
                elif is_segm:
                    # Segmentation but mask count mismatch: fall back to
                    # model bboxes and zero masks
                    frame_xyxy.append(_bboxes_to_xyxy(bboxes, w, h))
                    frame_masks.append(
                        np.zeros((n_det, h, w), dtype=np.uint8)
                    )
                else:
                    # Detection: use model bboxes directly
                    frame_xyxy.append(_bboxes_to_xyxy(bboxes, w, h))

            if frame_xyxy:
                xyxy_results.append(np.concatenate(frame_xyxy, axis=0))
                if is_segm and frame_masks:
                    mask_results.append(np.concatenate(frame_masks, axis=0))
                else:
                    mask_results.append(None)
            else:
                xyxy_results.append(np.empty((0, 4), dtype=np.float32))
                mask_results.append(None)

        out_queue.put(("result", xyxy_results, mask_results))
