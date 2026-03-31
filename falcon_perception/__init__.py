# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

__all__ = [
    "ModelArgs",
    "get_model_args",
    "get_tokenizer",
    "setup_torch_config",
    "cuda_timed",
    "PERCEPTION_MODEL_ID",
    "OCR_MODEL_ID",
    "load_from_hf_export",
    "load_and_prepare_model",
    "build_prompt_for_task",
]


@dataclass
class ModelArgs:
    max_seq_len: int = 8192
    rope_theta: int = 10000
    # base
    dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    head_dim: int = 128
    n_kv_heads: int = 8
    vocab_size: int = 65536
    ffn_dim: int = 3072
    norm_eps: float = 1e-5
    # vision
    channel_size: int = 3
    spatial_patch_size: int = 16
    temporal_patch_size: int = 1
    # heads
    coord_enc_dim: int = 512
    coord_dec_dim: int = 8192
    coord_out_dim: int = 2048
    size_enc_dim: int = 512
    size_dec_dim: int = 8192
    size_out_dim: int = 2048
    perception_heads: bool = True
    segm_out_dim: int = 256
    num_segm_layers: int = 3

    def update(self, tokenizer):
        self.eos_id = tokenizer.eos_token_id
        self.img_id = tokenizer.image_token_id
        self.img_start_id = tokenizer.start_of_image_token_id
        self.img_end_id = tokenizer.end_of_image_token_id
        self.img_row_sep_id = tokenizer.image_row_sep_token_id
        self.image_cls_token_id = tokenizer.image_cls_token_id
        self.image_reg_1_token_id = tokenizer.image_reg_1_token_id
        self.image_reg_2_token_id = tokenizer.image_reg_2_token_id
        self.image_reg_3_token_id = tokenizer.image_reg_3_token_id
        self.image_reg_4_token_id = tokenizer.image_reg_4_token_id
        self.seg_token_id = tokenizer.seg_token_id
        self.coord_token_id = tokenizer.coord_token_id
        self.size_token_id = tokenizer.size_token_id


def get_model_args(variant: str):
    if variant == "perception":
        return ModelArgs(perception_heads=True)
    if variant == "ocr":
        return ModelArgs(n_layers=22, head_dim=64, dim=768, ffn_dim=2304, perception_heads=False)
    raise ValueError("Unknown model variant")


class _FalconTokenizer:
    """Lightweight tokenizer backed by the ``tokenizers`` (Rust) library.
    Replaces the previous ``transformers.AutoTokenizer`` dependency.
    """

    def __init__(self, path: str):
        from tokenizers import Tokenizer

        tok_file = os.path.join(path, "tokenizer.json")
        self._tok = Tokenizer.from_file(tok_file)

        config_file = os.path.join(path, "tokenizer_config.json")
        stm_file = os.path.join(path, "special_tokens_map.json")

        config: dict = {}
        if os.path.isfile(config_file):
            config = json.loads(Path(config_file).read_text(encoding="utf-8"))
        stm: dict = {}
        if os.path.isfile(stm_file):
            stm = json.loads(Path(stm_file).read_text(encoding="utf-8"))

        self.special_tokens_map: dict[str, str] = {}
        for token_name, token_val in stm.items():
            if isinstance(token_val, str):
                self.special_tokens_map[token_name] = token_val
        for token_name, token_val in config.get("model_specific_special_tokens", {}).items():
            if isinstance(token_val, str):
                self.special_tokens_map[token_name] = token_val

        for token_name, token_str in self.special_tokens_map.items():
            setattr(self, token_name, token_str)
            tid = self._tok.token_to_id(token_str)
            setattr(self, token_name + "_id", tid)

        self.eos_token_id: int = self._tok.token_to_id(
            config.get("eos_token", stm.get("eos_token", "<|end_of_text|>"))
        )
        self.bos_token_id: int | None = None
        bos_str = config.get("bos_token") or stm.get("bos_token")
        if bos_str:
            self.bos_token_id = self._tok.token_to_id(bos_str)
        self.bos_id = self.bos_token_id

        self.pad_token_id: int = self._tok.token_to_id(
            config.get("pad_token", stm.get("pad_token", "<|pad|>"))
        )

        self.padding_side: str = "left"

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if not isinstance(ids, list):
            ids = list(ids)
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._tok.token_to_id(token)


def get_tokenizer(path):
    return _FalconTokenizer(path)


# ── Torch configuration ───────────────────────────────────────────────


def setup_torch_config():
    """Configure torch / inductor / dynamo / CUDA allocator flags for inference.

    Call once at process startup, before any CUDA operations.  Safe to call
    multiple times (env vars use ``setdefault``).
    """
    import torch

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    torch._inductor.config.triton.unique_kernel_names = True
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    torch._inductor.config.triton.cudagraphs = False
    torch._inductor.config.fx_graph_cache = True
    torch.set_float32_matmul_precision("high")


# ── Timing utility ────────────────────────────────────────────────────


class cuda_timed:
    """Context manager that times a block with proper CUDA synchronisation.

    Usage::

        with cuda_timed() as t:
            engine.generate(...)
        print(f"Took {{t.elapsed:.1f}}s")

    If *reset_peak_memory* is True (default), ``torch.cuda.reset_peak_memory_stats``
    is called on entry so that peak VRAM reflects only the timed block.

    Assumes torch is already imported by the caller (all torch inference
    scripts import it at the top level).
    """

    __slots__ = ("elapsed", "_reset_peak", "_t0")

    def __init__(self, *, reset_peak_memory: bool = True):
        self._reset_peak = reset_peak_memory
        self.elapsed: float = 0.0

    def __enter__(self):
        import torch
        if torch.cuda.is_available():
            if self._reset_peak:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self._t0 = __import__("time").perf_counter()
        return self

    def __exit__(self, *exc):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = __import__("time").perf_counter() - self._t0
        return False


# ── Model loading & prompt utilities ───────────────────────────────────

PERCEPTION_MODEL_ID = "tiiuae/Falcon-Perception"
OCR_MODEL_ID = "tiiuae/Falcon-OCR"


def load_from_hf_export(
    *,
    hf_model_id: str | None = None,
    hf_revision: str = "main",
    hf_local_dir: str | None = None,
) -> tuple[Any, Any, ModelArgs]:
    """Load model, tokenizer, and args from a HuggingFace export.

    Accepts a local directory or a Hub model id (downloaded via
    ``snapshot_download``).  The variant (perception vs ocr) is
    auto-detected from ``config.json``'s ``architectures`` field.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as safetensors_load_file

    from falcon_perception.model import FalconPerception

    if not (hf_local_dir or hf_model_id):
        raise ValueError("Provide hf_local_dir or hf_model_id")

    print("Loading model from Hugging Face Hub ...")
    export_dir = Path(hf_local_dir) if hf_local_dir else Path(
        snapshot_download(
            repo_id=hf_model_id or PERCEPTION_MODEL_ID,
            repo_type="model",
            revision=hf_revision,
        )
    )

    tokenizer = get_tokenizer(str(export_dir))

    variant = "perception"
    config_path = export_dir / "config.json"
    if config_path.exists():
        archs = json.loads(config_path.read_text(encoding="utf-8")).get("architectures", [])
        if "FalconOCRForCausalLM" in archs:
            variant = "ocr"

    model_args = get_model_args(variant)
    model_args.update(tokenizer)

    state = safetensors_load_file(str(export_dir / "model.safetensors"))
    model = FalconPerception(model_args).eval()
    model.load_state_dict(state, strict=True)
    print(f"  Variant: {variant} (perception_heads={model_args.perception_heads})")
    return model, tokenizer, model_args


def build_prompt_for_task(query: str, task: str) -> str:
    """Build the model prompt for a given task and query text."""
    if task in ("segmentation", "detection"):
        prefix = "Segment these expressions in the image:"
        return f"<|image|>{prefix}<|start_of_query|>{query}<|REF_SEG|>"
    elif task == "ocr_plain":
        return "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>"
    elif task == "ocr_layout":
        return ""  # layout crops build their own prompts internally
    else:
        return f"<|image|>{query}"


def load_from_hf_export_mlx(
    *,
    hf_model_id: str | None = None,
    hf_revision: str = "main",
    hf_local_dir: str | None = None,
    dtype: str = "float16",
) -> tuple[Any, Any, "ModelArgs"]:
    """Load model for the MLX backend from a HuggingFace export.

    Uses ``mx.load`` for safetensors + on-the-fly weight conversion
    (Conv2d transpose, RoPE complex->sin/cos).

    Returns ``(model, tokenizer, model_args)``.
    """
    import mlx.core as mx
    from huggingface_hub import snapshot_download

    from falcon_perception.mlx.convert import load_mlx_weights
    from falcon_perception.mlx.model import FalconPerception as FalconPerceptionMLX

    if not (hf_local_dir or hf_model_id):
        raise ValueError("Provide hf_local_dir or hf_model_id")

    print("Loading MLX model from Hugging Face Hub ...")
    export_dir = Path(hf_local_dir) if hf_local_dir else Path(
        snapshot_download(
            repo_id=hf_model_id or PERCEPTION_MODEL_ID,
            repo_type="model",
            revision=hf_revision,
        )
    )

    tokenizer = get_tokenizer(str(export_dir))

    variant = "perception"
    config_path = export_dir / "config.json"
    if config_path.exists():
        archs = json.loads(config_path.read_text(encoding="utf-8")).get("architectures", [])
        if "FalconOCRForCausalLM" in archs:
            variant = "ocr"

    model_args = get_model_args(variant)
    model_args.update(tokenizer)

    weights = load_mlx_weights(str(export_dir / "model.safetensors"))
    model = FalconPerceptionMLX(model_args)
    model.load_weights(weights, strict=False)

    dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
    mx_dtype = dtype_map.get(dtype, mx.float16)
    from mlx.utils import tree_map
    casted = tree_map(lambda x: x.astype(mx_dtype), model.parameters())
    model.update(casted)

    print(f"  Variant: {variant} (perception_heads={model_args.perception_heads})")
    print(f"  MLX model ready (dtype={dtype}).\n")
    return model, tokenizer, model_args


def load_and_prepare_model(
    *,
    hf_model_id: str | None = None,
    hf_revision: str = "main",
    hf_local_dir: str | None = None,
    device: str | None = None,
    dtype: str = "float32",
    compile: bool = True,
    backend: str = "torch",
) -> tuple[Any, Any, "ModelArgs"]:
    """Load, move to device, and optionally compile the model.

    Combines :func:`load_from_hf_export`, ``model.to()``, and
    ``model.compile()`` — the shared boilerplate used by every ``run_*.py``
    script.

    Args:
        backend: ``"torch"`` (default) or ``"mlx"`` for Apple Silicon.

    Returns ``(model, tokenizer, model_args)``.
    """
    if backend == "mlx":
        dtype_str = dtype if isinstance(dtype, str) else "float16"
        return load_from_hf_export_mlx(
            hf_model_id=hf_model_id,
            hf_revision=hf_revision,
            hf_local_dir=hf_local_dir,
            dtype=dtype_str,
        )

    import torch

    model, tokenizer, model_args = load_from_hf_export(
        hf_model_id=hf_model_id,
        hf_revision=hf_revision,
        hf_local_dir=hf_local_dir,
    )

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype_resolved = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    model = model.to(device=resolved_device, dtype=dtype_resolved)

    if compile:
        model.compile(mode="default")

    print(f"Model ready (dtype={model.dtype}, device={model.device}).\n")
    return model, tokenizer, model_args
