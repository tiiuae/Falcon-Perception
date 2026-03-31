# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""KV cache for MLX batch inference."""

import mlx.core as mx


class KVCache:
    """Per-layer KV cache using concatenation.

    Each layer stores keys and values as separate arrays that grow via
    ``mx.concatenate`` on each decode step. This avoids the cost of
    scatter-updating a single monolithic tensor.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        num_layers: int,
    ):
        self.num_layers = num_layers
        self.keys: list[mx.array | None] = [None] * num_layers
        self.values: list[mx.array | None] = [None] * num_layers
        self.pos = 0
        self.pos_t = None

    def reset(self):
        self.pos = 0
        self.pos_t = None
        for i in range(self.num_layers):
            self.keys[i] = None
            self.values[i] = None

    def get_pos(self):
        return self.pos

    def set_pos_t(self, pos_t):
        self.pos_t = pos_t

    def increment_and_get_pos_t(self):
        assert self.pos_t is not None, "pos_t for rope is not initialized."
        self.pos_t = self.pos_t + 1
        return self.pos_t

    def insert_kv(self, layer_id: int, k, v, **kwargs):
        """Insert new keys/values and return full cached KV views.

        Args:
            k, v: (B, H, T_add, D) new keys/values to insert.

        Returns:
            (key_view, value_view) each (B, H, T_total, D).
        """
        del kwargs
        assert self.pos_t is not None, "pos_t for rope is not initialized."

        if self.keys[layer_id] is not None:
            self.keys[layer_id] = mx.concatenate([self.keys[layer_id], k], axis=2)
            self.values[layer_id] = mx.concatenate([self.values[layer_id], v], axis=2)
        else:
            self.keys[layer_id] = k
            self.values[layer_id] = v

        if layer_id == self.num_layers - 1:
            self.pos += k.shape[2]

        return self.keys[layer_id], self.values[layer_id]
