"""Multi-Head Causal Self-Attention with optional Key-Value Cache.

Detailed comments document the exact tensor transformations and shapes at every step.
"""

from __future__ import annotations
import numpy as np
from tinyinfer.kv_cache import KVCache
from tinyinfer.tensor_ops import linear, softmax


def causal_self_attention(
    x: np.ndarray,
    c_attn_weight: np.ndarray,
    c_attn_bias: np.ndarray,
    c_proj_weight: np.ndarray,
    c_proj_bias: np.ndarray,
    n_head: int = 12,
    kv_cache: KVCache | None = None,
    layer_idx: int = 0,
) -> np.ndarray:
    """Compute Multi-Head Causal Self-Attention with optional KV-cache.

    Tensor Shape Flow:
        Input:                (B, T_new, C)
        QKV Projection:       (B, T_new, 3 * C)
        Split into Q, K, V:   3 x (B, T_new, C)
        Reshape & Transpose:  3 x (B, nh, T_new, hs) where hs = C // nh
        Update KV-cache:      K, V become (B, nh, T_total, hs) where T_total = past_len + T_new
        Attention Scores:     (B, nh, T_new, T_total) = Q @ K.T / sqrt(hs)
        Apply Causal Mask:    (B, nh, T_new, T_total)
        Softmax:              (B, nh, T_new, T_total)
        Value Multiplication: (B, nh, T_new, hs) = Attention @ V
        Transpose & Reshape:  (B, T_new, C)
        Output Projection:    (B, T_new, C)

    Args:
        x: Input tensor of shape (B, T_new, C) where:
           B = batch size
           T_new = number of new input tokens (e.g. prompt length for prefill, 1 for decode)
           C = embedding dimension (n_embd, e.g. 768 for GPT-2 small)
        c_attn_weight: Fused QKV weight matrix of shape (C, 3 * C).
        c_attn_bias: Fused QKV bias vector of shape (3 * C,).
        c_proj_weight: Output projection weight matrix of shape (C, C).
        c_proj_bias: Output projection bias vector of shape (C,).
        n_head: Number of attention heads (default: 12).
        kv_cache: Optional KVCache instance for caching past keys and values.
        layer_idx: Layer index when reading/writing to kv_cache.

    Returns:
        Attention output tensor of shape (B, T_new, C).
    """
    B, T_new, C = x.shape
    head_dim = C // n_head
    assert C % n_head == 0, f"Embedding dimension {C} must be divisible by n_head {n_head}"

    # Step 1: Fused QKV Linear Projection
    # x: (B, T_new, C) -> qkv: (B, T_new, 3 * C)
    qkv = linear(x, c_attn_weight, c_attn_bias)

    # Step 2: Split into Query, Key, and Value tensors along channel axis
    # Each tensor has shape: (B, T_new, C)
    q, k, v = np.split(qkv, 3, axis=-1)

    # Step 3: Reshape and transpose to multi-head format: (B, n_head, T_new, head_dim)
    q = q.reshape(B, T_new, n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T_new, n_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T_new, n_head, head_dim).transpose(0, 2, 1, 3)

    # Step 4: KV-Cache Retrieval and Concatenation
    if kv_cache is not None:
        past_len = kv_cache.get_seq_len(layer_idx)
        # k, v updated to (B, n_head, past_len + T_new, head_dim)
        k, v = kv_cache.update(layer_idx, k, v)
        T_total = k.shape[2]
    else:
        past_len = 0
        T_total = T_new

    # Step 5: Scaled Dot-Product Attention Scores
    # q: (B, nh, T_new, hs) @ k.T: (B, nh, hs, T_total) -> (B, nh, T_new, T_total)
    scaling = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scaling

    # Step 6: Causal Masking (prevent attending to future positions)
    # Token at index (past_len + i) can only attend to keys at index j <= past_len + i
    if T_new == 1 and past_len > 0:
        # Fast path for single-token autoregressive decoding:
        # All keys in cache correspond to past tokens (j <= past_len), so all are causal.
        pass
    else:
        # General causal masking for arbitrary T_new (prefill or speculative verification)
        q_pos = past_len + np.arange(T_new)[:, None]  # shape: (T_new, 1)
        k_pos = np.arange(T_total)[None, :]           # shape: (1, T_total)
        causal_mask = k_pos <= q_pos                  # shape: (T_new, T_total)
        scores = np.where(causal_mask, scores, -1e4)

    # Step 7: Softmax over the last dimension (across all available keys)
    # attn_weights: (B, nh, T_new, T_total)
    attn_weights = softmax(scores, axis=-1)

    # Step 8: Value aggregation
    # (B, nh, T_new, T_total) @ (B, nh, T_total, hs) -> (B, nh, T_new, hs)
    context = np.matmul(attn_weights, v)

    # Step 9: Concatenate heads back into a single vector of size C
    # (B, nh, T_new, hs) -> (B, T_new, nh, hs) -> (B, T_new, C)
    context = context.transpose(0, 2, 1, 3).reshape(B, T_new, C)

    # Step 10: Final Linear Output Projection
    # (B, T_new, C) -> (B, T_new, C)
    out = linear(context, c_proj_weight, c_proj_bias)
    return out
