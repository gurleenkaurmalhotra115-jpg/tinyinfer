"""Key-Value (KV) Cache for efficient autoregressive transformer decoding.

During generation, past Key and Value representations are invariant to newly generated
tokens. Instead of recomputing Q, K, and V across the full sequence at each step O(N^2),
we cache past K and V tensors and only project Q, K, and V for the single newly generated
token O(N).

Tensor Shapes:
    Cached Key for layer i:   (B, n_head, past_len, head_dim)
    Cached Value for layer i: (B, n_head, past_len, head_dim)
    New Key chunk:            (B, n_head, T_new, head_dim)
    New Value chunk:          (B, n_head, T_new, head_dim)
    Updated Key/Value:        (B, n_head, past_len + T_new, head_dim)
"""

from __future__ import annotations
import numpy as np


class KVCache:
    """Manages Key-Value states across all transformer layers for incremental decoding."""

    def __init__(self, num_layers: int = 12) -> None:
        """Initialize empty cache for the specified number of layers.

        Args:
            num_layers: Total number of transformer layers (default 12 for GPT-2 small).
        """
        self.num_layers = num_layers
        # List of cached Key tensors per layer: each (B, n_head, seq_len, head_dim)
        self.k: list[np.ndarray | None] = [None] * num_layers
        # List of cached Value tensors per layer: each (B, n_head, seq_len, head_dim)
        self.v: list[np.ndarray | None] = [None] * num_layers

    def update(
        self,
        layer_idx: int,
        k_new: np.ndarray,
        v_new: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Append newly computed Key and Value states to the cache for the given layer.

        Args:
            layer_idx: Index of the transformer layer (0 <= layer_idx < num_layers).
            k_new: New key tensor of shape (B, n_head, T_new, head_dim).
            v_new: New value tensor of shape (B, n_head, T_new, head_dim).

        Returns:
            Tuple of (k_total, v_total) containing all past + current keys and values,
            each with shape (B, n_head, past_len + T_new, head_dim).
        """
        assert 0 <= layer_idx < self.num_layers, f"Invalid layer_idx: {layer_idx}"

        if self.k[layer_idx] is None:
            # Prefill step: initialize cache with the prompt's keys and values
            self.k[layer_idx] = k_new
            self.v[layer_idx] = v_new
        else:
            # Decoding step: concatenate along sequence dimension (axis 2)
            # Past shape: (B, nh, past_len, hs)
            # New shape:  (B, nh, T_new,    hs)
            # Total:      (B, nh, past_len + T_new, hs)
            self.k[layer_idx] = np.concatenate([self.k[layer_idx], k_new], axis=2)
            self.v[layer_idx] = np.concatenate([self.v[layer_idx], v_new], axis=2)

        return self.k[layer_idx], self.v[layer_idx]

    def get_seq_len(self, layer_idx: int = 0) -> int:
        """Return the current sequence length stored in the cache.

        Args:
            layer_idx: Layer to inspect (default: 0).

        Returns:
            Number of tokens currently cached.
        """
        if self.k[layer_idx] is None:
            return 0
        return self.k[layer_idx].shape[2]

    def truncate(self, length: int) -> None:
        """Roll back the cache to a specified sequence length (used in speculative decoding).

        Args:
            length: Target sequence length to retain. If <= 0, cache is cleared.
        """
        if length <= 0:
            self.clear()
            return

        for i in range(self.num_layers):
            if self.k[i] is not None and self.k[i].shape[2] > length:
                self.k[i] = self.k[i][:, :, :length, :]
                self.v[i] = self.v[i][:, :, :length, :]

    def clear(self) -> None:
        """Reset and wipe all cached states."""
        self.k = [None] * self.num_layers
        self.v = [None] * self.num_layers

    def get_memory_bytes(self) -> int:
        """Compute the total memory in bytes currently occupied by cached tensors.

        Returns:
            Total RAM usage of the cache in bytes.
        """
        total = 0
        for i in range(self.num_layers):
            if self.k[i] is not None:
                total += self.k[i].nbytes
            if self.v[i] is not None:
                total += self.v[i].nbytes
        return total

    def copy(self) -> KVCache:
        """Create an independent deep copy of this KV-cache.

        Returns:
            New KVCache instance with duplicate array buffers.
        """
        new_cache = KVCache(num_layers=self.num_layers)
        for i in range(self.num_layers):
            if self.k[i] is not None:
                new_cache.k[i] = np.copy(self.k[i])
            if self.v[i] is not None:
                new_cache.v[i] = np.copy(self.v[i])
        return new_cache
