"""GPT-2 Transformer Architecture implemented in pure NumPy with optional KV-cache.

Full forward pass from token IDs to output vocabulary logits.
"""

from __future__ import annotations
import numpy as np
from tinyinfer.attention import causal_self_attention
from tinyinfer.kv_cache import KVCache
from tinyinfer.tensor_ops import gelu, layer_norm, linear
from tinyinfer.weights import GPT2Weights


def transformer_block(
    x: np.ndarray,
    block_weights: dict[str, np.ndarray],
    n_head: int = 12,
    kv_cache: KVCache | None = None,
    layer_idx: int = 0,
) -> np.ndarray:
    """Execute a single GPT-2 Transformer Decoder Block with optional KV caching.

    Architecture (Pre-LN formulation):
        Input x:                (B, T_new, C)
        LN1:                    (B, T_new, C)
        Self-Attention:         (B, T_new, C)
        Residual 1:             x = x + Self-Attention
        LN2:                    (B, T_new, C)
        MLP (FC -> GELU -> Proj): (B, T_new, C)
        Residual 2:             x = x + MLP

    Args:
        x: Input activation tensor of shape (B, T_new, C).
        block_weights: Dictionary containing block parameters.
        n_head: Number of attention heads (default: 12).
        kv_cache: Optional KVCache instance.
        layer_idx: Layer index for reading/writing to the cache.

    Returns:
        Output activation tensor of shape (B, T_new, C).
    """
    # 1. Pre-LayerNorm 1
    # a: (B, T_new, C)
    a = layer_norm(x, block_weights["ln_1_g"], block_weights["ln_1_b"])

    # 2. Multi-Head Causal Self-Attention (with KV cache)
    # attn_out: (B, T_new, C)
    attn_out = causal_self_attention(
        a,
        block_weights["c_attn_w"],
        block_weights["c_attn_b"],
        block_weights["c_proj_w"],
        block_weights["c_proj_b"],
        n_head=n_head,
        kv_cache=kv_cache,
        layer_idx=layer_idx,
    )

    # 3. Residual Connection 1
    x = x + attn_out

    # 4. Pre-LayerNorm 2
    # m: (B, T_new, C)
    m = layer_norm(x, block_weights["ln_2_g"], block_weights["ln_2_b"])

    # 5. Feed-Forward MLP: FC (C -> 4*C) -> GELU -> Proj (4*C -> C)
    # fc: (B, T_new, 4 * C)
    fc = linear(m, block_weights["c_fc_w"], block_weights["c_fc_b"])
    # gelu_out: (B, T_new, 4 * C)
    gelu_out = gelu(fc)
    # mlp_out: (B, T_new, C)
    mlp_out = linear(gelu_out, block_weights["c_mlp_proj_w"], block_weights["c_mlp_proj_b"])

    # 6. Residual Connection 2
    x = x + mlp_out
    return x


def gpt2_forward(
    input_ids: np.ndarray,
    weights: GPT2Weights,
    kv_cache: KVCache | None = None,
    pos_offset: int | None = None,
) -> np.ndarray:
    """Execute the GPT-2 forward pass from token IDs to unnormalized vocabulary logits.

    Tensor Flow:
        Input IDs:              (B, T_new)
        Pos Offset p:           derived from kv_cache or explicit pos_offset
        Token Embeddings (wte): (B, T_new, C)
        Pos Embeddings (wpe):   (T_new, C) for positions [p, p+1, ..., p + T_new - 1]
        Fused Input:            x = wte + wpe: (B, T_new, C)
        Pass 12 Blocks:         x = Block_i(x, kv_cache): (B, T_new, C)
        Final LayerNorm:        x = LN_f(x): (B, T_new, C)
        Language Model Head:    logits = x @ lm_head.T: (B, T_new, vocab_size)

    Args:
        input_ids: Array of token integers of shape (B, T_new).
        weights: Structured GPT2Weights parameter container.
        kv_cache: Optional KVCache storing past keys and values.
        pos_offset: Optional manual position index offset. If None and kv_cache is provided,
                    defaults to the cached sequence length.

    Returns:
        Logits tensor of shape (B, T_new, vocab_size).
    """
    # Ensure 2D shape (B, T_new)
    if input_ids.ndim == 1:
        input_ids = np.expand_dims(input_ids, axis=0)

    B, T_new = input_ids.shape

    # Determine position offset (past sequence length)
    if pos_offset is None:
        if kv_cache is not None:
            pos_offset = kv_cache.get_seq_len(0)
        else:
            pos_offset = 0

    total_len = pos_offset + T_new
    assert total_len <= weights.config.get("n_positions", 1024), (
        f"Total sequence length {total_len} exceeds maximum context length of 1024"
    )

    # 1. Token Embeddings: lookup in wte
    # input_ids: (B, T_new) -> token_emb: (B, T_new, C)
    token_emb = weights.wte[input_ids]

    # 2. Position Embeddings: lookup for positions pos_offset, ..., pos_offset + T_new - 1
    # pos_ids: (T_new,) -> pos_emb: (T_new, C)
    pos_ids = np.arange(pos_offset, pos_offset + T_new)
    pos_emb = weights.wpe[pos_ids]

    # 3. Sum embeddings (pos_emb broadcasts across batch B)
    # x: (B, T_new, C)
    x = token_emb + pos_emb

    # 4. Pass through all Transformer Blocks sequentially
    for layer_idx, block in enumerate(weights.blocks):
        x = transformer_block(
            x,
            block,
            n_head=weights.n_head,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
        )

    # 5. Final Layer Normalization
    # x: (B, T_new, C)
    x = layer_norm(x, weights.ln_f_g, weights.ln_f_b)

    # 6. Output Language Modeling Head (logits projection)
    # (B, T_new, C) @ (C, vocab_size) -> (B, T_new, vocab_size)
    if weights.lm_head_w.shape[0] == weights.vocab_size:
        logits = np.matmul(x, weights.lm_head_w.T)
    else:
        logits = np.matmul(x, weights.lm_head_w)

    return logits
