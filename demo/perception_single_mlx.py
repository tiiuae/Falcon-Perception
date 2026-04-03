#!/usr/bin/env python3
# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Single-image perception demo using the MLX backend.

Usage:
    python demo/perception_single_mlx.py
    python demo/perception_single_mlx.py --image photo.jpg --query "cat"
    python demo/perception_single_mlx.py --image photo.jpg --query "cat" --task detection
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from falcon_perception import (
    PERCEPTION_MODEL_ID,
    build_prompt_for_task,
    load_and_prepare_model,
)
from falcon_perception.data import load_image, stream_samples_from_hf_dataset

_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (192, 64, 64), (64, 192, 64),
]


def pair_bbox_entries(raw: list[dict]) -> list[dict]:
    """Pair [{x,y}, {h,w}, ...] into [{x,y,h,w}, ...]."""
    bboxes, current = [], {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        current.update(entry)
        if all(k in current for k in ("x", "y", "h", "w")):
            bboxes.append(dict(current))
            current = {}
    return bboxes


def decode_rle_mask(rle: dict) -> np.ndarray | None:
    try:
        from pycocotools import mask as mask_utils
        return mask_utils.decode(rle).astype(np.uint8)
    except Exception:
        return None


def visualize(
    image: Image.Image,
    bboxes: list[dict],
    masks_rle: list[dict],
    out_path: str,
    interior_opacity: float = 0.35,
    border_thickness: int = 3,
):
    img = image.convert("RGB")
    W, H = img.size
    overlay = np.array(img, dtype=np.float32)

    masks = []
    for rle in masks_rle:
        m = decode_rle_mask(rle)
        if m is not None:
            if m.shape != (H, W):
                m = np.array(Image.fromarray(m).resize((W, H), Image.NEAREST))
            masks.append(m)

    n_det = min(len(bboxes), len(masks)) if masks else len(bboxes)
    P = len(_PALETTE)

    for i in range(min(n_det, len(masks))):
        m = masks[i]
        color = np.array(_PALETTE[i % P], dtype=np.float32)
        region = m > 0
        overlay[region] = overlay[region] * (1 - interior_opacity) + color * interior_opacity

        from scipy.ndimage import binary_dilation
        border = binary_dilation(region, iterations=border_thickness) & ~region
        overlay[border] = color

    result = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)

    for i, bbox in enumerate(bboxes[:n_det]):
        cx, cy = bbox["x"] * W, bbox["y"] * H
        bw, bh = bbox["w"] * W, bbox["h"] * H
        x0, y0 = cx - bw / 2, cy - bh / 2
        x1, y1 = cx + bw / 2, cy + bh / 2
        color = _PALETTE[i % P]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

    result.save(out_path)
    print(f"Saved visualization to {out_path}")


def _fmt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def main():
    parser = argparse.ArgumentParser(description="MLX Perception single-image demo")
    parser.add_argument("--image", type=str, default=None, help="Path or URL to image")
    parser.add_argument("--query", type=str, default=None, help="Query text")
    parser.add_argument("--task", type=str, default="segmentation", choices=["segmentation", "detection"])
    parser.add_argument("--model-id", type=str, default=PERCEPTION_MODEL_ID)
    parser.add_argument("--local-dir", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-dim", type=int, default=256)
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--metal-cache-limit-gb", type=int, default=0,
                        help="Metal buffer cache limit in GB (0=unlimited, 4-8 recommended for bounded memory)")
    parser.add_argument("--out-dir", type=str, default="./outputs/mlx")
    args = parser.parse_args()

    timings: dict[str, float] = {}

    # ── Model loading ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=args.model_id,
        hf_local_dir=args.local_dir,
        dtype=args.dtype,
        backend="mlx",
    )
    timings["model_load"] = time.perf_counter() - t0

    # ── Image loading ─────────────────────────────────────────────────
    if args.image is not None:
        pil_image = load_image(args.image).convert("RGB")
    else:
        print("No --image provided, loading a demo sample ...")
        sample = stream_samples_from_hf_dataset("tiiuae/PBench", split="level_1")[0]
        pil_image = sample["image"]
        sample_query = sample.get("expression") or sample.get("expressions") or "all objects"
        if isinstance(sample_query, list):
            sample_query = ", ".join(str(q) for q in sample_query) if sample_query else "all objects"
        print(f"  Sample query: {sample_query!r}")
        if args.query is None:
            args.query = str(sample_query)

    if args.query is None:
        args.query = "all objects"

    if hasattr(pil_image, "convert"):
        pil_image = pil_image.convert("RGB")

    w, h = pil_image.size
    print(f"  Task  : {args.task}")
    print(f"  Query : {args.query!r}")
    print(f"  Image : {w} x {h}")
    print()

    from falcon_perception.mlx.batch_inference import (
        BatchInferenceEngine,
        process_batch_and_generate,
    )

    engine = BatchInferenceEngine(model, tokenizer)

    # ── Preprocessing ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    prompt = build_prompt_for_task(args.query, args.task)
    batch = process_batch_and_generate(
        tokenizer,
        [(pil_image, prompt)],
        max_length=model_args.max_seq_len,
        min_dimension=args.min_dim,
        max_dimension=args.max_dim,
    )
    timings["preprocess"] = time.perf_counter() - t0

    print(f"Input tokens: {batch['tokens'].shape}")
    print("Running MLX batch inference...")

    # ── Generation ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    output_tokens, aux_outputs = engine.generate(
        tokens=batch["tokens"],
        pos_t=batch["pos_t"],
        pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        task=args.task,
        metal_cache_limit_gb=args.metal_cache_limit_gb,
    )
    timings["generation"] = time.perf_counter() - t0

    # ── Decode & parse ────────────────────────────────────────────────
    t0 = time.perf_counter()
    decoded = tokenizer.decode(np.array(output_tokens[0]).tolist(), skip_special_tokens=False)
    aux = aux_outputs[0]
    bboxes = pair_bbox_entries(aux.bboxes_raw)
    timings["decode"] = time.perf_counter() - t0

    print(f"\nOutput:\n{decoded}")
    print(f"\nDetections: {len(bboxes)} bboxes, {len(aux.masks_rle)} masks")
    for i, bbox in enumerate(bboxes):
        mask_status = "with mask" if i < len(aux.masks_rle) else "no mask"
        print(f"  [{i}] cx={bbox['x']:.3f} cy={bbox['y']:.3f} "
              f"h={bbox['h']:.4f} w={bbox['w']:.4f}  ({mask_status})")

    # ── Visualization ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_image_path = out_dir / "perception_input.jpg"
    pil_image.save(input_image_path)
    stem = Path(args.image).stem if args.image else "demo"
    safe_query = "".join(c if c.isalnum() or c in " _-" else "_" for c in args.query)[:30].strip()
    out_path = out_dir / f"{stem}_{safe_query}.jpg"
    visualize(pil_image, bboxes, aux.masks_rle, str(out_path))
    timings["visualize"] = time.perf_counter() - t0

    # ── Timing summary ────────────────────────────────────────────────
    all_toks = np.array(output_tokens[0]).flatten()
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    n_prefill = batch["tokens"].shape[1]
    # Decoded tokens = everything after prefill, up to (and including) the first EOS/pad
    decoded_toks = all_toks[n_prefill:]
    eos_positions = np.where((decoded_toks == eos_id) | (decoded_toks == pad_id))[0]
    n_decoded = int(eos_positions[0] + 1) if len(eos_positions) > 0 else len(decoded_toks)
    gen_time = timings["generation"]
    decode_tok_per_sec = n_decoded / gen_time if gen_time > 0 else 0
    total_tok_per_sec = (n_prefill + n_decoded) / gen_time if gen_time > 0 else 0

    print("\n" + "=" * 52)
    print("  Timing Benchmark")
    print("=" * 52)
    print(f"  Model loading ......... {_fmt(timings['model_load']):>10}")
    print(f"  Preprocessing ......... {_fmt(timings['preprocess']):>10}")
    print(f"  Generation ............ {_fmt(timings['generation']):>10}")
    print(f"    Prefill tokens ...... {n_prefill:>10}")
    print(f"    Decoded tokens ...... {n_decoded:>10}")
    print(f"    Decode tok/s ........ {decode_tok_per_sec:>10.1f}")
    print(f"    Total tok/s ......... {total_tok_per_sec:>10.1f}")
    print(f"  Decode + parse ........ {_fmt(timings['decode']):>10}")
    print(f"  Visualization ......... {_fmt(timings['visualize']):>10}")
    print("-" * 52)
    total = sum(timings.values())
    print(f"  Total ................. {_fmt(total):>10}")
    print("=" * 52)

    print(f"\n  Input image : {input_image_path}")
    print(f"  Output dir  : {out_dir}")


if __name__ == "__main__":
    main()
