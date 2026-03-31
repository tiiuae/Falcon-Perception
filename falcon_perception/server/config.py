import os
from dataclasses import dataclass, field
from typing import Literal


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.lower() not in ("0", "false", "no", "off")


def _env_optional(key: str) -> str | None:
    return os.environ.get(key) or None


@dataclass
class ServerConfig:

    # ── Model ──────────────────────────────────────────────────────────
    hf_model_id: str = field(default_factory=lambda: _env("HF_MODEL_ID", "tiiuae/Falcon-Perception"))
    hf_revision: str = field(default_factory=lambda: _env("HF_REVISION", "main"))
    hf_local_dir: str | None = field(default_factory=lambda: _env_optional("HF_LOCAL_DIR"))

    # ── Engine ─────────────────────────────────────────────────────────
    dtype: Literal["bfloat16", "float32"] = field(default_factory=lambda: _env("DTYPE", "float32"))
    num_gpus: int = field(default_factory=lambda: _env_int("NUM_GPUS", -1))
    compile: bool = field(default_factory=lambda: _env_bool("COMPILE", True))
    cudagraph: bool = field(default_factory=lambda: _env_bool("CUDAGRAPH", True))
    max_batch_size: int = field(default_factory=lambda: _env_int("MAX_BATCH_SIZE", 128))
    max_seq_length: int = field(default_factory=lambda: _env_int("MAX_SEQ_LENGTH", 8192))
    n_pages: int = field(default_factory=lambda: _env_int("N_PAGES", 1024))
    page_size: int = field(default_factory=lambda: _env_int("PAGE_SIZE", 128))
    prefill_length_limit: int = field(default_factory=lambda: _env_int("PREFILL_LENGTH_LIMIT", 16384))
    max_decode_steps_between_prefills: int = field(default_factory=lambda: _env_int("MAX_DECODE_STEPS_BETWEEN_PREFILLS", 16))
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.0))
    top_k: int | None = field(default_factory=lambda: _env_int("TOP_K", None) if os.environ.get("TOP_K") else None)

    # ── Image defaults ────────────────────────────────────────────────
    min_image_size: int = field(default_factory=lambda: _env_int("MIN_IMAGE_SIZE", 256))
    max_image_size: int = field(default_factory=lambda: _env_int("MAX_IMAGE_SIZE", 1024))
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 8192))

    # ── Layout detection ──────────────────────────────────────────────
    layout_threshold: float = field(default_factory=lambda: _env_float("LAYOUT_THRESHOLD", 0.3))

    # ── Server ─────────────────────────────────────────────────────────
    host: str = field(default_factory=lambda: _env("HOSTNAME", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 7860))
    startup_timeout: int = field(default_factory=lambda: _env_int("STARTUP_TIMEOUT", 600))
    images_dir: str = field(default_factory=lambda: _env("IMAGES_DIR", "./public/images"))
