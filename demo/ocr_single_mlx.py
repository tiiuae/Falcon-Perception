#!/usr/bin/env python3
# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Single-image OCR demo using the MLX backend.

Usage:
    python demo/ocr_single_mlx.py --image document.png
    python demo/ocr_single_mlx.py  # loads a demo sample from OCRBench-v2
"""

import argparse
import time
from pathlib import Path

import numpy as np

from falcon_perception import (
    OCR_MODEL_ID,
    build_prompt_for_task,
    load_and_prepare_model,
)
from falcon_perception.data import load_image, stream_samples_from_hf_dataset


def _fmt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def main():
    parser = argparse.ArgumentParser(description="MLX OCR single-image demo")
    parser.add_argument("--image", type=str, default=None, help="Path or URL to image")
    parser.add_argument("--model-id", type=str, default=OCR_MODEL_ID)
    parser.add_argument("--local-dir", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-dim", type=int, default=64)
    parser.add_argument("--max-dim", type=int, default=1024)
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
        sample = stream_samples_from_hf_dataset(
            "lmms-lab/OCRBench-v2", split="test",
        )[0]
        pil_image = sample["image"]
        print(f"  Sample question: {sample.get('question', '')!r}")

    if hasattr(pil_image, "convert"):
        pil_image = pil_image.convert("RGB")

    w, h = pil_image.size
    print(f"  Image : {w} x {h}")

    from falcon_perception.mlx.batch_inference import (
        BatchInferenceEngine,
        process_batch_and_generate,
    )

    engine = BatchInferenceEngine(model, tokenizer)

    # ── Preprocessing ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    prompt = build_prompt_for_task("", "ocr_plain")
    batch = process_batch_and_generate(
        tokenizer,
        [(pil_image, prompt)],
        max_length=model_args.max_seq_len,
        min_dimension=args.min_dim,
        max_dimension=args.max_dim,
    )
    timings["preprocess"] = time.perf_counter() - t0

    print(f"  Input tokens: {batch['tokens'].shape}")
    print("Running MLX OCR inference ...")

    # ── Generation ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]
    output_tokens, _ = engine.generate(
        tokens=batch["tokens"],
        pos_t=batch["pos_t"],
        pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        stop_token_ids=stop_token_ids,
        task="detection",
    )
    timings["generation"] = time.perf_counter() - t0

    # ── Decode ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    all_toks = np.array(output_tokens[0]).flatten()
    n_prefill = batch["tokens"].shape[1]
    decoded_toks = all_toks[n_prefill:]

    stop_set = set(stop_token_ids + [tokenizer.pad_token_id])
    eos_positions = np.where(np.isin(decoded_toks, list(stop_set)))[0]
    n_decoded = int(eos_positions[0]) if len(eos_positions) > 0 else len(decoded_toks)
    text_ids = decoded_toks[:n_decoded].tolist()
    text = tokenizer.decode(text_ids, skip_special_tokens=True)
    timings["decode"] = time.perf_counter() - t0

    gen_time = timings["generation"]
    decode_tok_per_sec = n_decoded / gen_time if gen_time > 0 else 0
    total_tok_per_sec = (n_prefill + n_decoded) / gen_time if gen_time > 0 else 0

    print(f"\n{'=' * 60}")
    print("  OCR Result")
    print("=" * 60)
    print(text)
    print("=" * 60)

    # ── Save outputs ───────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_image_path = out_dir / "ocr_input.jpg"
    pil_image.save(input_image_path)

    print(f"\n{'=' * 52}")
    print("  Timing Benchmark")
    print("=" * 52)
    print(f"  Model loading ......... {_fmt(timings['model_load']):>10}")
    print(f"  Preprocessing ......... {_fmt(timings['preprocess']):>10}")
    print(f"  Generation ............ {_fmt(timings['generation']):>10}")
    print(f"    Prefill tokens ...... {n_prefill:>10}")
    print(f"    Decoded tokens ...... {n_decoded:>10}")
    print(f"    Decode tok/s ........ {decode_tok_per_sec:>10.1f}")
    print(f"    Total tok/s ......... {total_tok_per_sec:>10.1f}")
    print(f"  Decode ................ {_fmt(timings['decode']):>10}")
    print("-" * 52)
    total = sum(timings.values())
    print(f"  Total ................. {_fmt(total):>10}")
    print("=" * 52)
    print(f"\n  Input image : {input_image_path} \n")


if __name__ == "__main__":
    main()
