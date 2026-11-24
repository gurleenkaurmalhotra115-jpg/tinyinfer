"""Autoregressive text generation pipeline for tinyinfer supporting KV-caching."""

from __future__ import annotations
import time
import numpy as np
from tinyinfer.kv_cache import KVCache
from tinyinfer.model import gpt2_forward
from tinyinfer.sampling import sample_next_token
from tinyinfer.tokenizer import LocalGPT2Tokenizer
from tinyinfer.weights import GPT2Weights


def generate(
    prompt: str,
    weights: GPT2Weights,
    tokenizer: LocalGPT2Tokenizer,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    top_k: int | None = None,
    use_cache: bool = True,
) -> str:
    """Generate text completion from a prompt using autoregressive generation.

    Args:
        prompt: Initial input prompt string.
        weights: Loaded GPT-2 weights.
        tokenizer: Local tokenizer.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature (0.0 for greedy argmax).
        top_k: Optional top-k filter.
        use_cache: If True, uses incremental KV-caching (O(N) decode complexity);
                   if False, runs full vanilla sequence recomputation at every step (O(N^2)).

    Returns:
        Generated text completion including the initial prompt.
    """
    token_ids = tokenizer.encode(prompt)
    curr_ids = list(token_ids)

    if not use_cache:
        # Vanilla O(N^2) generation without KV-cache
        for _ in range(max_new_tokens):
            input_array = np.array([curr_ids], dtype=np.int64)
            logits = gpt2_forward(input_array, weights)
            next_token_logits = logits[0, -1, :]
            next_id = sample_next_token(next_token_logits, temperature=temperature, top_k=top_k)
            curr_ids.append(next_id)
        return tokenizer.decode(curr_ids)

    # Incremental O(N) generation with KV-cache
    cache = KVCache(num_layers=weights.n_layer)

    # 1. Prefill step: process initial prompt sequence
    input_array = np.array([curr_ids], dtype=np.int64)  # (1, T_prompt)
    logits = gpt2_forward(input_array, weights, kv_cache=cache)
    next_id = sample_next_token(logits[0, -1, :], temperature=temperature, top_k=top_k)
    curr_ids.append(next_id)

    # 2. Incremental decode steps: process single new token (1, 1)
    for _ in range(max_new_tokens - 1):
        single_token_array = np.array([[next_id]], dtype=np.int64)  # (1, 1)
        logits = gpt2_forward(single_token_array, weights, kv_cache=cache)  # (1, 1, vocab_size)
        next_id = sample_next_token(logits[0, -1, :], temperature=temperature, top_k=top_k)
        curr_ids.append(next_id)

    return tokenizer.decode(curr_ids)


def generate_with_metrics(
    prompt: str,
    weights: GPT2Weights,
    tokenizer: LocalGPT2Tokenizer,
    max_new_tokens: int = 32,
    use_cache: bool = True,
) -> dict:
    """Generate tokens and record throughput and memory metrics.

    Args:
        prompt: Input text prompt.
        weights: Loaded GPT-2 weights.
        tokenizer: Local tokenizer.
        max_new_tokens: Number of tokens to generate.
        use_cache: Whether to use KV cache.

    Returns:
        Dictionary with keys: 'text', 'tokens_generated', 'elapsed_sec', 'tokens_per_sec', 'peak_cache_mb'.
    """
    token_ids = tokenizer.encode(prompt)
    curr_ids = list(token_ids)
    
    start_time = time.perf_counter()
    peak_cache_bytes = 0

    if not use_cache:
        for _ in range(max_new_tokens):
            input_array = np.array([curr_ids], dtype=np.int64)
            logits = gpt2_forward(input_array, weights)
            next_id = sample_next_token(logits[0, -1, :], temperature=0.0)
            curr_ids.append(next_id)
    else:
        cache = KVCache(num_layers=weights.n_layer)
        input_array = np.array([curr_ids], dtype=np.int64)
        logits = gpt2_forward(input_array, weights, kv_cache=cache)
        next_id = sample_next_token(logits[0, -1, :], temperature=0.0)
        curr_ids.append(next_id)
        peak_cache_bytes = max(peak_cache_bytes, cache.get_memory_bytes())

        for _ in range(max_new_tokens - 1):
            single_token = np.array([[next_id]], dtype=np.int64)
            logits = gpt2_forward(single_token, weights, kv_cache=cache)
            next_id = sample_next_token(logits[0, -1, :], temperature=0.0)
            curr_ids.append(next_id)
            peak_cache_bytes = max(peak_cache_bytes, cache.get_memory_bytes())

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = max_new_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "text": tokenizer.decode(curr_ids),
        "tokens_generated": max_new_tokens,
        "elapsed_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "peak_cache_mb": peak_cache_bytes / (1024 * 1024),
    }
