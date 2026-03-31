"""Falcon Perception — PBench benchmark evaluation.

Runs the paged inference engine on a subset of PBench, computes IoU metrics
against ground-truth masks, and reports runtime / GPU statistics.

Usage
-----
    # Default: PBench level_1, first 50 samples
    python run_perception_benchmark.py

    # Full level_1
    python run_perception_benchmark.py --limit -1

    # Different level
    python run_perception_benchmark.py --split level_0 --limit 20

    # Local model export
    python run_perception_benchmark.py --hf-local-dir ./my_export/ --limit 10
"""

import json
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from falcon_perception import (
    PERCEPTION_MODEL_ID,
    cuda_timed,
    load_and_prepare_model,
    setup_torch_config,
)
from falcon_perception.data import ImageProcessor, stream_samples_from_hf_dataset
from falcon_perception.paged_inference import (
    PagedInferenceEngine,
    SamplingParams,
    Sequence,
    engine_config_for_gpu,
)
from falcon_perception.visualization_utils import decode_coco_rle, save_comparison_vis

setup_torch_config()


def matched_iou(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    eval_size: int = 256,
) -> float:
    """Bipartite-matched mean IoU via vectorised matrix multiply.

    All masks are resized to ``(eval_size, eval_size)`` once, then the full
    N×M IoU matrix is computed with a single ``P @ G.T`` — no Python loop
    over pairs.
    """
    if not pred_masks or not gt_masks:
        return 0.0

    from PIL import Image
    from scipy.optimize import linear_sum_assignment

    def _flatten(masks: list[np.ndarray]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for m in masks:
            if m is None:
                continue
            if m.shape[0] != eval_size or m.shape[1] != eval_size:
                m = np.array(Image.fromarray((m > 0).astype(np.uint8) * 255).resize(
                    (eval_size, eval_size), Image.Resampling.NEAREST))
            rows.append((m > 0).ravel())
        return np.stack(rows).astype(np.float32) if rows else np.empty((0, eval_size * eval_size), dtype=np.float32)

    P = _flatten(pred_masks)  # (N, eval_size^2)
    G = _flatten(gt_masks)    # (M, eval_size^2)
    if P.shape[0] == 0 or G.shape[0] == 0:
        return 0.0

    inter = P @ G.T                                    # (N, M)
    p_sum = P.sum(axis=1, keepdims=True)               # (N, 1)
    g_sum = G.sum(axis=1, keepdims=True)               # (M, 1)
    union = p_sum + g_sum.T - inter                    # (N, M)
    iou_matrix = np.where(union > 0, inter / union, 0.0)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched = iou_matrix[row_ind, col_ind]
    return float(matched.mean()) if len(matched) > 0 else 0.0


@torch.inference_mode()
def main(
    hf_model_id: str | None = None,
    hf_revision: str = "main",
    hf_local_dir: str | None = None,
    dataset: str = "tiiuae/PBench",
    split: str = "level_1",
    limit: int = 50,
    device: str = "cuda",
    dtype: Literal["bfloat16", "float", "float32"] = "float32",
    compile: bool = True,
    cudagraph: bool = True,
    max_new_tokens: int = 2048,
    min_dimension: int = 256,
    max_dimension: int = 1024,
    n_pages: int = 512,
    max_decode_steps_between_prefills: int = 8,
    hr_upsample_ratio: int = 8,
    profile: bool = False,
    profile_steps: int = 10,
    out_dir: str = "./outputs_dense/",
):
    """Evaluate Falcon Perception on PBench (or similar HF datasets).

    Loads the dataset, runs paged inference on each sample, computes IoU
    against ground-truth masks, and prints aggregate metrics + runtime stats.
    """
    os.makedirs(out_dir, exist_ok=True)

    ds = stream_samples_from_hf_dataset(dataset, split=split, limit=limit)

    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=hf_model_id or PERCEPTION_MODEL_ID,
        hf_revision=hf_revision,
        hf_local_dir=hf_local_dir,
        device=device,
        dtype=dtype,
        compile=compile,
    )

    image_processor = ImageProcessor(patch_size=16, merge_size=1)
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]

    cfg = engine_config_for_gpu(max_image_size=max_dimension, dtype=model.dtype)
    print(f"Auto-config: {cfg}")
    engine = PagedInferenceEngine(
        model, tokenizer, image_processor,
        max_seq_length=8192,
        capture_cudagraph=cudagraph,
        max_decode_steps_between_prefills=max_decode_steps_between_prefills,
        **cfg,
    )

    sequences: list[Sequence] = []
    sample_data: list[dict] = []

    for i, sample in enumerate(ds):
        expression = sample["expression"]
        prefix = "Segment these expressions in the image:"
        prompt = f"<|image|>{prefix}<|start_of_query|>{expression}<|REF_SEG|>"

        seq = Sequence(
            text=prompt,
            image=sample["image"],
            min_image_size=min_dimension,
            max_image_size=max_dimension,
            request_idx=i,
        )
        sequences.append(seq)
        sample_data.append(sample)

    print(f"Built {len(sequences)} sequences")

    sampling_params = SamplingParams(
        max_new_tokens, stop_token_ids=stop_token_ids,
        coord_dedup_threshold=0.01, hr_upsample_ratio=hr_upsample_ratio,
    )

    # Warmup absorbs torch.compile JIT cost so the benchmark measures steady-state.
    print("Warmup run ...")
    warmup_seq = sequences[0].copy()
    warmup_seq.request_idx = 0
    with cuda_timed(reset_peak_memory=False) as warmup_timer:
        engine.generate([warmup_seq], sampling_params=sampling_params)
    print(f"Warmup done in {warmup_timer.elapsed:.1f}s")

    profiler = None
    if profile:
        from torch.profiler import ProfilerActivity, profile as torch_profile, schedule

        profile_dir = os.path.join(out_dir, "profiler")
        os.makedirs(profile_dir, exist_ok=True)
        trace_path = os.path.join(profile_dir, "trace.json")
        profiler = torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=0,
                warmup=1,
                active=profile_steps,
                repeat=1,
            ),
            on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
            record_shapes=True,
            with_stack=True,
        )
        profiler.start()

    torch.cuda.reset_peak_memory_stats()
    with cuda_timed() as timer:
        engine.generate(
            sequences,
            sampling_params=sampling_params,
            use_tqdm=True,
            print_stats=True,
            profiler=profiler,
        )

    if profiler is not None:
        profiler.stop()
        print(f"\nProfiler trace saved to {trace_path}")
        print("View with: chrome://tracing or https://ui.perfetto.dev/")

    wall_time = timer.elapsed

    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    all_ious: list[float] = []
    all_pred_counts: list[int] = []
    all_gt_counts: list[int] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_masks = 0

    print(f"\n{'=' * 80}")
    print("PER-SAMPLE RESULTS")
    print("=" * 80)

    for i, (seq, sample) in enumerate(zip(sequences, sample_data)):
        expression = sample["expression"]
        gt_rles = sample["masks"]
        pil_image = sample["image"].convert("RGB")

        gt_masks = []
        for rle in gt_rles:
            if isinstance(rle, dict):
                gt_masks.append(decode_coco_rle(rle))
            elif isinstance(rle, str):
                gt_masks.append(decode_coco_rle(json.loads(rle)))

        pred_masks = []
        for rle in seq.output_aux.masks_rle:
            if isinstance(rle, dict) and "counts" in rle:
                pred_masks.append(decode_coco_rle(rle))

        iou = matched_iou(pred_masks, gt_masks)
        decoded = tokenizer.decode(seq.output_ids)

        all_ious.append(iou)
        all_pred_counts.append(len(pred_masks))
        all_gt_counts.append(len(gt_masks))
        total_input_tokens += seq.input_length
        total_output_tokens += len(seq._output_ids)
        total_masks += len(seq.output_aux.masks_rle)

        print(f"\n[{i:3d}] expr={expression!r}")
        print(f"      GT masks: {len(gt_masks)}  |  Pred masks: {len(pred_masks)}  |  IoU: {iou:.3f}")
        print(f"      decoded: {decoded[:120]}{'...' if len(decoded) > 120 else ''}")

        safe_expr = "".join(c if c.isalnum() or c in " _-" else "_" for c in expression)[:30].strip()
        vis_path = os.path.join(vis_dir, f"{i:04d}_iou{iou:.2f}_{safe_expr}.jpg")
        save_comparison_vis(pil_image, gt_masks, pred_masks, expression, iou, vis_path)

    total_tokens = total_input_tokens + total_output_tokens
    total_tps = total_tokens / wall_time if wall_time > 0 else 0
    images_per_sec = len(sequences) / wall_time if wall_time > 0 else 0

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"  Dataset          : {dataset} / {split}")
    print(f"  Samples          : {len(sequences)}")
    print(f"  Wall time        : {wall_time:.1f}s")
    print(f"  Total tok/s      : {total_tps:.1f}  (prefill + decode)")
    print(f"  Images/s         : {images_per_sec:.2f}")
    print(f"  Input tokens     : {total_input_tokens}")
    print(f"  Output tokens    : {total_output_tokens}")
    print(f"  Total masks      : {total_masks}")
    print()
    print(f"  Mean IoU         : {np.mean(all_ious):.4f}")
    print(f"  Mean pred count  : {np.mean(all_pred_counts):.1f}")
    print(f"  Mean GT count    : {np.mean(all_gt_counts):.1f}")
    print(f"  Count match rate : {sum(1 for p, g in zip(all_pred_counts, all_gt_counts) if p == g)}/{len(sequences)}")
    print(f"  Zero-pred samples: {sum(1 for p in all_pred_counts if p == 0)}/{len(sequences)}")

    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print()
        print(f"  GPU              : {torch.cuda.get_device_name()}")
        print(f"  Peak VRAM alloc  : {peak_alloc:.2f} GiB")
        print(f"  Peak VRAM reserv : {peak_reserved:.2f} GiB")

    print(f"\n  Visualizations   : {vis_dir}")


if __name__ == "__main__":
    tyro.cli(main)
