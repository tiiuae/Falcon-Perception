"""Falcon OCR — single-image inference.

Run OCR on one image using either plain (full-page) or layout (region-level)
mode.  If no image is provided, a sample is loaded from OCRBench-v2.

Usage
-----
    python run_ocr_single.py
    python run_ocr_single.py --image document.png
    python run_ocr_single.py --image document.png --task ocr_layout
    python run_ocr_single.py --hf-local-dir ./my_ocr_export/
"""

from pathlib import Path
from typing import Literal

import torch
import tyro

from falcon_perception import OCR_MODEL_ID, cuda_timed, load_and_prepare_model, setup_torch_config
from falcon_perception.data import load_image, stream_samples_from_hf_dataset

setup_torch_config()


@torch.inference_mode()
def main(
    image: str | None = None,
    task: Literal["ocr_plain", "ocr_layout"] = "ocr_plain",
    hf_model_id: str | None = None,
    hf_revision: str = "main",
    hf_local_dir: str | None = None,
    device: str | None = None,
    dtype: Literal["bfloat16", "float", "float32"] = "float32",
    flex_attn_safe: bool = False,
    out_dir: str = "./outputs/",
    compile: bool = True,
    cudagraph: bool = True,
):
    """Run Falcon OCR on a single image.

    If --image is omitted, a sample is loaded from OCRBench-v2.

    Use --flex-attn-safe on GPUs with limited per-SM shared memory
    (A40, RTX 3090/4090, L40) to avoid FlexAttention Triton OOM. See README.
    """
    kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1} if flex_attn_safe else None
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=hf_model_id or OCR_MODEL_ID,
        hf_revision=hf_revision,
        hf_local_dir=hf_local_dir,
        device=device,
        dtype=dtype,
        compile=compile,
    )

    if image is not None:
        pil_image = load_image(image).convert("RGB")
    else:
        print("No --image provided, loading a demo sample ...")
        sample = stream_samples_from_hf_dataset("lmms-lab/OCRBench-v2", split="test")[0]
        pil_image = sample["image"]
        print(f"  Sample question: {sample.get('question', '')!r}")

    if hasattr(pil_image, "convert"):
        pil_image = pil_image.convert("RGB")

    w, h = pil_image.size
    print(f"  Task  : {task}")
    print(f"  Image : {w} x {h}")
    print()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    input_image_path = out_path / "ocr_input.jpg"
    pil_image.save(input_image_path)

    from falcon_perception.data import ImageProcessor
    from falcon_perception.paged_ocr_inference import OCRInferenceEngine

    image_processor = ImageProcessor(patch_size=16, merge_size=1)

    engine = OCRInferenceEngine(model, tokenizer, image_processor, capture_cudagraph=cudagraph, kernel_options=kernel_options)

    print("Running inference ...")

    if task == "ocr_layout":
        with cuda_timed(reset_peak_memory=False) as timer:
            results = engine.generate_with_layout(
                images=[pil_image],
                use_tqdm=True,
            )

        elements = results[0]
        print(f"\n{'=' * 60}")
        print(f"OCR Layout Results  ({len(elements)} regions, {timer.elapsed:.1f}s)")
        print("=" * 60)
        for elem in elements:
            preview = (elem["text"][:100] + "...") if len(elem["text"]) > 100 else elem["text"]
            print(f"  [{elem['category']}] score={elem['score']:.3f}  {preview}")

        full_text = "\n".join(e["text"] for e in elements if e["text"])
        print(f"\n--- Extracted text ---\n{full_text}")

        print(f"\n  Input image : {input_image_path}")

    else:  # ocr_plain
        with cuda_timed(reset_peak_memory=False) as timer:
            texts = engine.generate_plain(
                images=[pil_image],
                use_tqdm=True,
            )

        text = texts[0] if texts else ""
        print(f"\n{'=' * 60}")
        print(f"OCR Plain Result  ({timer.elapsed:.1f}s)")
        print("=" * 60)
        print(text)

        print(f"\n  Input image : {input_image_path}")


if __name__ == "__main__":
    tyro.cli(main)
