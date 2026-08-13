# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.
"""Optional NVTX ranges for Nsight profiling.

Enable with the single flag ``FALCON_NVTX=1`` (or call ``set_nvtx_enabled(True)``).
Ranges are pushed only from Python call sites *outside* ``torch.compile`` /
CUDA-graph capture regions, so enabling them does not introduce Dynamo graph
breaks or corrupt captured graphs.

``nvtx_range`` is a :class:`~contextlib.ContextDecorator`, so both forms work::

    with nvtx_range("prefill"):
        ...

    @nvtx_range("decode")
    def decode_step(...):
        ...
"""

from __future__ import annotations

import os
from contextlib import ContextDecorator

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


class nvtx_range(ContextDecorator):
    """NVTX range as ``with nvtx_range("name"):`` or ``@nvtx_range("name")``.

    No-op when annotations are disabled. The enabled check runs on enter/exit,
    so ``set_nvtx_enabled`` applies to both forms. Safe to use around compiled
    / CUDA-graph call sites — never call this *inside* a ``torch.compile``'d
    function body.
    """

    def __init__(self, name: str):
        self._name = name
        self._pushed = False

    def __enter__(self):
        if _NVTX_ENABLED:
            import torch
            torch.cuda.nvtx.range_push(self._name)
            self._pushed = True
        return self

    def __exit__(self, *exc):
        if self._pushed:
            import torch
            torch.cuda.nvtx.range_pop()
            self._pushed = False
        return None
