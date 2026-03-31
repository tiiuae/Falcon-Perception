# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Falcon Perception — PBench benchmark evaluation.

Streams PBench from HuggingFace (or a local copy), runs the paged inference
engine, and computes F1 metrics at 10 IoU thresholds (0.5 → 0.95 in steps of
0.05).  The ``dense`` split uses only IoU=0.5.

Evaluation protocol
-------------------
1. Each image is **force-resized** so its longest edge equals ``max_dimension``
   (default 1024) before being fed to the model.  This is the canonical PBench
   resolution and differs from the soft clamp used in the demo scripts.
2. Predicted masks are output at the upsampled inference resolution.  They are
   **resized back to the original image resolution** (nearest-neighbor) before
   being compared against GT masks, which are always at the original resolution.
3. Optional greedy NMS can be applied to predicted masks before scoring
   (``--nms-threshold``, default disabled).

Usage
-----
    # Default: level_0, first 100 samples
    python eval/pbench.py

    # Full split
    python eval/pbench.py --split level_0 --limit 0

    # All 6 splits in one run → prints a summary table
    python eval/pbench.py --split all --limit 0

    # Different resolution ablation
    python eval/pbench.py --split level_0 --max-dimension 768

    # Local model
    python eval/pbench.py --hf-local-dir /path/to/export --split level_1

    # Save results JSON
    python eval/pbench.py --split all --out-dir ./results/pbench/
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from PIL import Image

# Allow `import metrics` when this file is run as a script (eval/ dir on path).
sys.path.insert(0, str(Path(__file__).parent))
import metrics  # noqa: E402  (eval/metrics.py)

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from falcon_perception import (
    PERCEPTION_MODEL_ID,
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

setup_torch_config()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PBENCH_DATASET = "tiiuae/PBench"
SPLITS = ["level_0", "level_1", "level_2", "level_3", "level_4", "dense"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_split(dataset: str, split: str, limit: int) -> list[dict]:
    """Return a list of samples for *split* from *dataset*.

    *dataset* can be a HuggingFace Hub ID (e.g. ``"tiiuae/PBench"``) or a
    path to a locally saved HF dataset directory.  When *limit* is 0, all
    samples are returned.
    """
    if os.path.isdir(dataset):
        from datasets import load_from_disk, DatasetDict, Dataset
        print(f"Loading local dataset: {dataset} / {split}")
        raw = load_from_disk(dataset)
        ds = raw[split] if isinstance(raw, DatasetDict) else raw
    else:
        from datasets import load_dataset
        print(f"Streaming dataset: {dataset} / {split}")
        ds = load_dataset(dataset, split=split, streaming=True)

    samples: list[dict] = []
    for sample in ds:
        samples.append(sample)
        if 0 < limit <= len(samples):
            break

    print(f"  → {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# Image force-resize
# ---------------------------------------------------------------------------

def _force_resize(img: Image.Image, max_dimension: int) -> Image.Image:
    """Scale *img* so its longest edge is exactly *max_dimension* (LANCZOS).

    Unlike ``resize_image_if_necessary`` (which soft-clamps), this always
    produces an image whose longest edge equals *max_dimension*.
    """
    w, h = img.size
    longest = max(w, h)
    if longest == max_dimension:
        return img
    scale = max_dimension / longest
    return img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.LANCZOS,
    )


# ---------------------------------------------------------------------------
# Single-split inference + evaluation
# ---------------------------------------------------------------------------

def _run_split(
    engine: PagedInferenceEngine,
    tokenizer,
    samples: list[dict],
    *,
    split: str,
    max_dimension: int,
    min_dimension: int,
    max_new_tokens: int,
    hr_upsample_ratio: int,
) -> dict:
    """Run inference + F1 evaluation for one PBench split.

    Returns the aggregated metrics dict produced by :func:`metrics.aggregate`.
    """
    iou_thresholds = (
        metrics.IOU_THRESHOLDS_DENSE if split == "dense" else metrics.IOU_THRESHOLDS
    )

    # ── Build sequences ──────────────────────────────────────────────────────
    sequences: list[Sequence] = []
    sample_data: list[dict] = []
    original_sizes: list[tuple[int, int]] = []  # (orig_w, orig_h) in PIL order

    for i, sample in enumerate(samples):
        pil_image = sample["image"]
        if not isinstance(pil_image, Image.Image):
            pil_image = Image.fromarray(pil_image)
        pil_image = pil_image.convert("RGB")

        orig_w, orig_h = pil_image.size
        original_sizes.append((orig_w, orig_h))

        pil_image = _force_resize(pil_image, max_dimension)
        prompt = build_prompt_for_task(sample["expression"], "segmentation")

        seq = Sequence(
            text=prompt,
            image=pil_image,
            min_image_size=min_dimension,
            max_image_size=max_dimension,
            request_idx=i,
            task="segmentation",
        )
        sequences.append(seq)
        sample_data.append(sample)

    print(f"\nBuilt {len(sequences)} sequences "
          f"(force-resized max_edge={max_dimension}, NMS IoU={metrics.NMS_THRESHOLD})")

    # ── Sampling params ───────────────────────────────────────────────────────
    stop_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "end_of_query_token_id"):
        stop_ids.append(tokenizer.end_of_query_token_id)

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_ids,
        coord_dedup_threshold=0.01,
        hr_upsample_ratio=hr_upsample_ratio,
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    engine.generate(
        sequences,
        sampling_params=sampling_params,
        use_tqdm=True,
        print_stats=True,
    )

    wall_time = time.perf_counter() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0.0
    )
    print(
        f"\n[perf] wall={wall_time:.1f}s  "
        f"({len(sequences) / wall_time:.1f} samples/s)  "
        f"peak={peak_gb:.2f} GiB"
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    per_sample: list[dict] = []
    for seq, sample, (orig_w, orig_h) in zip(sequences, sample_data, original_sizes):
        # Ground-truth masks are at the original image resolution
        gt_rles: list[dict] = []
        for rle in sample.get("masks", []):
            if isinstance(rle, str):
                rle = json.loads(rle)
            if isinstance(rle, dict) and "counts" in rle and "size" in rle:
                gt_rles.append(rle)

        # Predicted masks come out at the upsampled inference resolution;
        # resize them to the original image resolution before scoring.
        pred_rles: list[dict] = []
        for rle in seq.output_aux.masks_rle:
            if isinstance(rle, dict) and "counts" in rle:
                pred_rles.append(metrics.resize_rle(rle, orig_h, orig_w))

        if pred_rles:
            pred_rles = metrics.nms(pred_rles, metrics.NMS_THRESHOLD)

        per_sample.append(metrics.sample_f1(pred_rles, gt_rles, iou_thresholds))

    result = metrics.aggregate(per_sample, iou_thresholds)
    result.update(
        split=split,
        max_dimension=max_dimension,
        iou_thresholds=iou_thresholds,
        wall_time_s=round(wall_time, 2),
        peak_gpu_gib=round(peak_gb, 3),
    )
    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary_table(all_results: list[dict]) -> None:
    w = 64
    print(f"\n{'=' * w}")
    print("PBENCH EVALUATION SUMMARY")
    print(f"{'=' * w}")
    print(
        f"{'Split':<12} {'Samples':>8} {'F1':>10}"
        f" {'IL TP':>8} {'IL TN':>8} {'IL FP':>8} {'IL FN':>8}"
    )
    print("-" * w)
    for r in all_results:
        print(
            f"{r['split']:<12} {r['n_samples']:>8}"
            f" {r['f1'] * 100:>10.2f}"
            f" {r['il_tp']:>8} {r['il_tn']:>8} {r['il_fp']:>8} {r['il_fn']:>8}"
        )
    if len(all_results) > 1:
        avg_f1 = float(np.mean([r["f1"] for r in all_results]))
        print("-" * w)
        print(f"{'AVERAGE':<12} {'':>8} {avg_f1 * 100:>10.2f}")
    print(f"{'=' * w}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@torch.inference_mode()
def main(
    hf_model_id: str | None = None,
    hf_revision: str = "main",
    hf_local_dir: str | None = None,
    dataset: str = PBENCH_DATASET,
    split: str = "level_0",
    limit: int = 100,
    device: str = "cuda",
    dtype: Literal["bfloat16", "float32"] = "bfloat16",
    compile: bool = True,
    cudagraph: bool = True,
    max_new_tokens: int = 2048,
    min_dimension: int = 256,
    max_dimension: int = 1024,
    hr_upsample_ratio: int = 8,
    out_dir: str = "./eval_results/pbench/",
):
    """Evaluate Falcon Perception on the PBench segmentation benchmark.

    Args:
        hf_model_id:      HuggingFace model ID (default: tiiuae/Falcon-Perception).
        hf_revision:      Model revision / branch (default: main).
        hf_local_dir:     Path to a locally downloaded model export.
        dataset:          HF dataset ID or path to a local HF-format directory.
        split:            Split to evaluate. Pass 'all' to run all 6 splits.
        limit:            Max samples per split (0 = all samples).
        device:           Compute device (default: cuda).
        dtype:            Model dtype — bfloat16 recommended for eval.
        compile:          Enable torch.compile (default: True).
        cudagraph:        Capture CUDA graphs for decode (default: True).
        max_new_tokens:   Token budget per sequence.
        min_dimension:    Minimum image edge before force-resize.
        max_dimension:    Target longest edge for force-resize (default: 1024).
        hr_upsample_ratio: High-resolution upsampling ratio for segmentation.
        out_dir:          Directory for per-split JSON results. '' disables saving.
    """
    splits_to_run = SPLITS if split == "all" else [split]

    # ── Load model once (shared across all splits) ───────────────────────────
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=hf_model_id or PERCEPTION_MODEL_ID,
        hf_revision=hf_revision,
        hf_local_dir=hf_local_dir,
        device=device,
        dtype=dtype,
        compile=compile,
    )

    image_processor = ImageProcessor(patch_size=16, merge_size=1)
    cfg = engine_config_for_gpu(max_image_size=max_dimension, dtype=model.dtype)
    print(f"Engine config: {cfg}")

    engine = PagedInferenceEngine(
        model,
        tokenizer,
        image_processor,
        max_seq_length=model_args.max_seq_len,
        capture_cudagraph=cudagraph,
        **cfg,
    )

    # ── Run each split ───────────────────────────────────────────────────────
    all_results: list[dict] = []

    for sp in splits_to_run:
        print(f"\n{'─' * 60}")
        print(f"  Split: {sp}")
        print(f"{'─' * 60}")

        samples = _load_split(dataset, sp, limit)
        if not samples:
            print(f"  [SKIP] No samples found for split '{sp}'.")
            continue

        result = _run_split(
            engine,
            tokenizer,
            samples,
            split=sp,
            max_dimension=max_dimension,
            min_dimension=min_dimension,
            max_new_tokens=max_new_tokens,
            hr_upsample_ratio=hr_upsample_ratio,
        )
        all_results.append(result)

        # Per-split console summary
        print(f"\n  {sp}  F1: {result['f1'] * 100:.2f}%")

        # Per-split JSON save
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            out_path = Path(out_dir) / f"{sp}_results.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {out_path}")

    # ── Final summary table ──────────────────────────────────────────────────
    if all_results:
        _print_summary_table(all_results)

        if out_dir and len(all_results) > 1:
            summary_path = Path(out_dir) / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nFull summary saved: {summary_path}")


if __name__ == "__main__":
    tyro.cli(main)
