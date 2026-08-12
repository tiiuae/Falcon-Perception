# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.
"""Optional NVTX ranges for Nsight profiling.

Enable with the single flag ``FALCON_NVTX=1`` (or call ``set_nvtx_enabled(True)``).
Ranges are pushed only from Python call sites *outside* ``torch.compile`` /
CUDA-graph capture regions, so enabling them does not introduce Dynamo graph
breaks or corrupt captured graphs.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Iterator

__all__ = [
    "is_nvtx_enabled",
    "set_nvtx_enabled",
    "nvtx_range",
]


def _env_enabled() -> bool:
    v = os.environ.get("FALCON_NVTX", "0")
    return v.lower() not in ("0", "false", "no", "off", "")


_NVTX_ENABLED: bool = _env_enabled()


def is_nvtx_enabled() -> bool:
    return _NVTX_ENABLED


def set_nvtx_enabled(enabled: bool) -> None:
    """Toggle NVTX annotations at runtime (overrides ``FALCON_NVTX``)."""
    global _NVTX_ENABLED
    _NVTX_ENABLED = bool(enabled)


@contextmanager
def _nvtx_range_impl(name: str) -> Iterator[None]:
    import torch

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def nvtx_range(name: str):
    """Context manager that emits an NVTX range when annotations are enabled.

    When disabled, returns a no-op ``nullcontext`` (no generator overhead).
    Safe to use around compiled / CUDA-graph call sites — never call this
    *inside* a ``torch.compile``'d function body.
    """
    if not _NVTX_ENABLED:
        return nullcontext()
    return _nvtx_range_impl(name)
