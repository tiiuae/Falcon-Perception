# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Abstract base class for KV caches used by attention layers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class KVCacheBase(ABC):
    """Common interface for :class:`KVCache` (batch) and :class:`PagedKVCache` (paged).

    Every concrete cache must store key/value tensors and expose ``insert_kv``
    which the attention layers call to append new KV pairs and retrieve the
    full cached sequence.
    """

    kv_cache: Tensor | None

    @abstractmethod
    def insert_kv(
        self,
        layer_id: int,
        k: Tensor,
        v: Tensor,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Insert new keys/values and return the full cached KV views."""
        ...
