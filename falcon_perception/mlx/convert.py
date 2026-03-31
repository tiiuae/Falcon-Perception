# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""On-the-fly weight conversion from PyTorch safetensors to MLX.

Handles:
  - Conv2d weight transpose: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
  - RoPE complex buffers: complex64 freqs_cis -> (cos, sin) real tables
  - Weight key renaming: ModuleDict keys, layer numbering
"""

import mlx.core as mx
import numpy as np


def _is_conv_weight(key: str, value) -> bool:
    """All 4D tensors in this model are Conv2d-like weights (OIHW) that need
    transposing to OHWI for MLX.  This includes Conv2d layers and the
    LearnedFeatureUnification basis."""
    return value.ndim == 4


def _transpose_conv_weight(w):
    """PyTorch (O, I, H, W) -> MLX (O, H, W, I)."""
    return w.transpose(0, 2, 3, 1)


def _convert_complex_freqs(freqs_complex):
    """Convert complex64 RoPE frequencies to real (cos, sin) pair.

    Input:  (S, D//2) complex64  (stored as (S, D//2, 2) float32 in safetensors)
    Output: cos (S, D//2), sin (S, D//2)
    """
    if freqs_complex.ndim == 3 and freqs_complex.shape[-1] == 2:
        # safetensors stores complex as (real, imag) pairs
        real = freqs_complex[..., 0]
        imag = freqs_complex[..., 1]
        return real, imag
    raise ValueError(f"Unexpected freqs_cis shape: {freqs_complex.shape}")


def _remap_layer_keys(weights: dict) -> dict:
    """Remap PyTorch nn.ModuleDict layer keys to list indices.

    PyTorch: ``layers.0.attention.wqkv.weight``
    MLX:     ``layers.0.attention.wqkv.weight``  (same, MLX uses list indices)

    No actual renaming needed since ModuleDict keys are already numeric strings,
    and MLX list indexing uses the same numeric pattern.
    """
    return weights


def convert_weights(raw_weights: dict) -> list[tuple[str, mx.array]]:
    """Convert a raw safetensors weight dict to MLX-compatible (key, array) pairs.

    Steps:
      1. Transpose Conv2d weights from OIHW to OHWI.
      2. Convert complex RoPE buffers to (cos, sin) tables.
      3. Return as list of (key, mx.array) for ``nn.Module.load_weights()``.
    """
    converted = {}

    for key, value in raw_weights.items():
        if not isinstance(value, mx.array):
            value = mx.array(value)

        if key == "freqs_cis":
            cos, sin = _convert_complex_freqs(value)
            converted["freqs_cos"] = cos
            converted["freqs_sin"] = sin
            continue

        if _is_conv_weight(key, value):
            value = _transpose_conv_weight(value)

        converted[key] = value

    return list(converted.items())


def load_mlx_weights(safetensors_path: str) -> list[tuple[str, mx.array]]:
    """Load and convert weights from a safetensors file.

    Uses ``mx.load`` (no torch dependency for this step).
    """
    raw = mx.load(safetensors_path)
    return convert_weights(raw)
