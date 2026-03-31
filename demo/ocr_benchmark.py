"""Falcon OCR — OCRBench-v2 benchmark run.

Runs plain OCR on samples from lmms-lab/OCRBench-v2 and reports runtime
statistics with a preview of the first few outputs.

Usage
-----
    # Default: first 50 samples (streamed, no full download)
    python run_ocr_benchmark.py

    # More samples
    python run_ocr_benchmark.py --limit 200

    # Local model export
    python run_ocr_benchmark.py --hf-local-dir ./my_ocr_export/ --limit 10

    # With profiler
    python run_ocr_benchmark.py --profile --profile-steps 20
"""

import logging
import os
from typing import Literal

import torch
import tyro

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from falcon_perception import OCR_MODEL_ID, cuda_timed, load_and_prepare_model, setup_torch_config
from falcon_perception.data import ImageProcessor, stream_samples_from_hf_dataset
from falcon_perception.paged_inference import engine_config_for_gpu
from falcon_perception.paged_ocr_inference import OCRInferenceEngine

setup_torch_config()


@torch.inference_mode()
def main(
    hf_model_id: str | None = None,
    hf_revision: str = "main",
    hf_local_dir: str | None = None,
    dataset: str = "lmms-lab/OCRBench-v2",
    split: str = "test",
    limit: int = 50,
    device: str = "cuda",
    dtype: Literal["bfloat16", "float", "float32"] = "float32",
    compile: bool = True,
    cudagraph: bool = True,
    max_new_tokens: int = 4096,
    profile: bool = False,
    profile_steps: int = 10,
    out_dir: str = "./outputs_ocr/",
):
    """Run Falcon OCR on OCRBench-v2 samples.

    Streams up to --limit samples (default 50) from the dataset, runs plain
    OCR, and prints runtime stats with a preview of the first outputs.
    """
    os.makedirs(out_dir, exist_ok=True)

    samples = stream_samples_from_hf_dataset(dataset, split=split, limit=limit)

    images = []
    questions = []
    for sample in samples:
        pil = sample["image"]
        if hasattr(pil, "convert"):
            pil = pil.convert("RGB")
        images.append(pil)
        questions.append(sample.get("question", ""))

    print(f"  {len(images)} samples loaded")

    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=hf_model_id or OCR_MODEL_ID,
        hf_revision=hf_revision,
        hf_local_dir=hf_local_dir,
        device=device,
        dtype=dtype,
        compile=compile,
    )

    image_processor = ImageProcessor(patch_size=16, merge_size=1)

    cfg = engine_config_for_gpu(max_image_size=1024, dtype=model.dtype)
    cfg.pop("max_hr_cache_entries", None)
    cfg.pop("max_image_size", None)
    print(f"Auto-config: {cfg}")
    engine = OCRInferenceEngine(
        model, tokenizer, image_processor,
        max_seq_length=4096,
        capture_cudagraph=cudagraph,
        **cfg,
    )

    # Warmup absorbs torch.compile JIT cost so the benchmark measures steady-state.
    print("Warmup run ...")
    with cuda_timed(reset_peak_memory=False) as warmup_timer:
        engine.generate_plain(images[:1], max_new_tokens=max_new_tokens, use_tqdm=False)
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

    print(f"Running OCR on {len(images)} images ...")

    torch.cuda.reset_peak_memory_stats()
    with cuda_timed() as timer:
        predictions = engine.generate_plain(
            images,
            max_new_tokens=max_new_tokens,
            use_tqdm=True,
            print_stats=True,
            profiler=profiler,
        )

    if profiler is not None:
        profiler.stop()
        print(f"\nProfiler trace saved to {trace_path}")
        print("View with: chrome://tracing or https://ui.perfetto.dev/")

    wall_time = timer.elapsed

    n_preview = min(5, len(predictions))
    print(f"\n{'=' * 80}")
    print(f"SAMPLE OUTPUTS (first {n_preview})")
    print("=" * 80)

    for i in range(n_preview):
        w, h = images[i].size
        preview = predictions[i][:200]
        if len(predictions[i]) > 200:
            preview += "..."
        print(f"\n[{i}] {w}x{h}  Q: {questions[i][:60]}")
        print(f"    {preview}")

    n = len(predictions)
    images_per_sec = n / wall_time if wall_time > 0 else 0

    total_output_tokens = sum(len(tokenizer.encode(p)) for p in predictions)
    ps = model_args.spatial_patch_size
    total_input_tokens = sum(
        len(tokenizer.encode(questions[i])) + (images[i].size[0] // ps) * (images[i].size[1] // ps)
        for i in range(n)
    )
    total_tokens = total_input_tokens + total_output_tokens
    total_tps = total_tokens / wall_time if wall_time > 0 else 0

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"  Dataset          : {dataset} / {split}")
    print(f"  Samples          : {n}")
    print(f"  Wall time        : {wall_time:.1f}s")
    print(f"  Total tok/s      : {total_tps:.1f}  (prefill + decode)")
    print(f"  Images/s         : {images_per_sec:.2f}")
    print(f"  Input tokens     : {total_input_tokens}")
    print(f"  Output tokens    : {total_output_tokens}")

    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print()
        print(f"  GPU              : {torch.cuda.get_device_name()}")
        print(f"  Peak VRAM alloc  : {peak_alloc:.2f} GiB")
        print(f"  Peak VRAM reserv : {peak_reserved:.2f} GiB")

    print()


if __name__ == "__main__":
    tyro.cli(main)
