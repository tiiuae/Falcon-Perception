# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""
``python -m falcon_perception`` — print package info and available model variants.
"""

from falcon_perception import ModelArgs, get_model_args


def _fmt_row(variant: str, args: ModelArgs) -> str:
    return (
        f"  {variant:<12s}  layers={args.n_layers:>2d}  dim={args.dim:>5d}  "
        f"heads={args.n_heads:>2d}  kv_heads={args.n_kv_heads}  "
        f"head_dim={args.head_dim:>3d}  ffn_dim={args.ffn_dim:>5d}  "
        f"perception_heads={args.perception_heads}"
    )


def main():
    print("Falcon Perception")
    print("=" * 60)
    print()
    print("Available model variants:")
    for variant in ("perception", "ocr"):
        print(_fmt_row(variant, get_model_args(variant)))
    print()
    print("Usage:")
    print("  python run_perception_single.py                       # Quick demo (auto-downloads sample)")
    print("  python run_perception_single.py --image photo.jpg --query dog")
    print("  python run_perception_benchmark.py                    # PBench evaluation")
    print("  python run_ocr_single.py                              # OCR demo")
    print("  python -m falcon_perception.server                    # Inference server")
    print()


main()
