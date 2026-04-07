# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""KV cache for MLX batch inference."""

import mlx.core as mx


class KVCache:
    """Pre-allocated KV cache for MLX batch inference.

    Allocates the full (B, H, S_max, D) buffers upfront and writes to them
    via slice assignment each step.  This avoids the concat-and-grow pattern
    that creates unreusable dead Metal buffers every decode step.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        num_layers: int,
        dtype=mx.float16,
    ):
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.keys: list[mx.array] = [mx.zeros(shape, dtype=dtype) for _ in range(num_layers)]
        self.values: list[mx.array] = [mx.zeros(shape, dtype=dtype) for _ in range(num_layers)]
        mx.eval(self.keys, self.values)
        self.pos = 0
        self.pos_t = None

    def reset(self):
        self.pos = 0
        self.pos_t = None

    def get_pos(self):
        return self.pos

    def set_pos_t(self, pos_t):
        self.pos_t = pos_t

    def increment_and_get_pos_t(self):
        assert self.pos_t is not None, "pos_t for rope is not initialized."
        self.pos_t = self.pos_t + 1
        return self.pos_t

    def insert_kv(self, layer_id: int, k, v, **kwargs):
        """Insert new keys/values and return the valid cached KV slice.

        Args:
            k, v: (B, H, T_add, D) new keys/values to insert.

        Returns:
            (key_view, value_view) each (B, H, T_total, D) covering
            positions 0..pos+T_add-1.
        """
        del kwargs
        assert self.pos_t is not None, "pos_t for rope is not initialized."

        t = k.shape[2]
        end = self.pos + t

        self.keys[layer_id][:, :, self.pos:end, :] = k
        self.values[layer_id][:, :, self.pos:end, :] = v

        if layer_id == self.num_layers - 1:
            self.pos = end

        return self.keys[layer_id][:, :, :end, :], self.values[layer_id][:, :, :end, :]
