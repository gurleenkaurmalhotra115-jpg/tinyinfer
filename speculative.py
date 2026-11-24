"""Exact greedy speculative decoding with KV-cache rollback for tinyinfer."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from tinyinfer.kv_cache import KVCache
from tinyinfer.model import gpt2_forward
from tinyinfer.quantize import QuantizedGPT2Weights
from tinyinfer.sampling import sample_next_token
from tinyinfer.tokenizer import LocalGPT2Tokenizer
from tinyinfer.weights import GPT2Weights


def _resolve_draft_weights(draft_weights: GPT2Weights | QuantizedGPT2Weights) -> GPT2Weights:
    """Convert QuantizedGPT2Weights to executable GPT2Weights if needed."""
    if isinstance(draft_weights, QuantizedGPT2Weights):
        return draft_weights.dequantize_to_fp32()
    return draft_weights


def speculative_generate_with_metrics(
    prompt: str,
    target_weights: GPT2Weights,
    draft_weights: GPT2Weights | QuantizedGPT2Weights,
    tokenizer: LocalGPT2Tokenizer,
    max_new_tokens: int = 32,
    gamma: int = 4,
    draft_forward_fn: Callable[[np.ndarray, Any, KVCache | None], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run lossless greedy speculative decoding with KVCache verification and rollback.

    Drafts up to ``gamma`` candidate tokens using the draft model, verifies all
    candidates in a single batched forward pass of the target model, and truncates
    both KV-caches cleanly on the first mismatch using ``KVCache.truncate()``.

    Args:
        prompt: Input prompt text.
        target_weights: Authoritative target GPT-2 weights.
        draft_weights: Fast draft weights (FP32 or INT8 QuantizedGPT2Weights).
        tokenizer: Local BPE tokenizer.
        max_new_tokens: Total new tokens to generate.
        gamma: Number of speculative tokens proposed per verification step.
        draft_forward_fn: Optional custom draft forward callable (useful for testing).

    Returns:
        Dictionary containing generated text and speculative decoding telemetry.
    """
    if max_new_tokens <= 0:
        return {
            "text": prompt,
            "tokens_generated": 0,
            "elapsed_sec": 0.0,
            "tokens_per_sec": 0.0,
            "drafted_tokens": 0,
            "accepted_tokens": 0,
            "acceptance_rate": 0.0,
            "target_forward_calls": 0,
            "gamma": gamma,
        }

    exec_draft = draft_weights if draft_forward_fn is not None else _resolve_draft_weights(draft_weights)
    draft_fn = draft_forward_fn or gpt2_forward

    prompt_ids = tokenizer.encode(prompt)
    curr_ids = list(prompt_ids)

    target_cache = KVCache(num_layers=target_weights.n_layer)
    draft_cache = KVCache(num_layers=exec_draft.n_layer)

    start_time = time.perf_counter()

    # Prefill both caches on the prompt
    prompt_arr = np.array([prompt_ids], dtype=np.int64)
    target_logits = gpt2_forward(prompt_arr, target_weights, kv_cache=target_cache)
    _ = draft_fn(prompt_arr, exec_draft, kv_cache=draft_cache)
    target_calls = 1

    next_token = sample_next_token(target_logits[0, -1, :], temperature=0.0)
    curr_ids.append(next_token)

    drafted_total = 0
    accepted_total = 0
    target_len = len(prompt_ids) + max_new_tokens

    while len(curr_ids) < target_len:
        remaining = target_len - len(curr_ids)
        k = min(gamma, remaining)
        base_len = target_cache.get_seq_len()

        # 1. Draft k candidate tokens autoregressively
        draft_tokens: list[int] = []
        d_input_tok = next_token
        for _ in range(k):
            d_arr = np.array([[d_input_tok]], dtype=np.int64)
            d_logits = draft_fn(d_arr, exec_draft, kv_cache=draft_cache)
            d_tok = sample_next_token(d_logits[0, -1, :], temperature=0.0)
            draft_tokens.append(d_tok)
            d_input_tok = d_tok
        drafted_total += k

        # 2. Verify all k draft tokens in a single batched target forward pass
        verify_seq = [next_token] + draft_tokens
        verify_arr = np.array([verify_seq], dtype=np.int64)
        verify_logits = gpt2_forward(verify_arr, target_weights, kv_cache=target_cache)
        target_calls += 1

        # 3. Compare target greedy predictions against draft tokens
        mismatch_idx: int | None = None
        corrected_tok: int | None = None
        for j in range(k):
            target_pred = sample_next_token(verify_logits[0, j, :], temperature=0.0)
            if target_pred == draft_tokens[j]:
                accepted_total += 1
            else:
                mismatch_idx = j
                corrected_tok = target_pred
                break

        if mismatch_idx is not None:
            # Accept prefix draft_tokens[:mismatch_idx] + corrected_tok
            curr_ids.extend(draft_tokens[:mismatch_idx])
            curr_ids.append(corrected_tok)  # type: ignore[arg-type]
            keep_len = base_len + 1 + mismatch_idx
            target_cache.truncate(keep_len)
            draft_cache.truncate(keep_len)
            next_token = corrected_tok  # type: ignore[assignment]
        else:
            # All k draft tokens accepted; extract bonus token from target logits[k]
            curr_ids.extend(draft_tokens)
            if len(curr_ids) >= target_len:
                break
            bonus_tok = sample_next_token(verify_logits[0, k, :], temperature=0.0)
            curr_ids.append(bonus_tok)
            # Align draft_cache with target_cache by feeding the last accepted draft token
            _ = draft_fn(np.array([[draft_tokens[-1]]], dtype=np.int64), exec_draft, kv_cache=draft_cache)
            next_token = bonus_tok

    curr_ids = curr_ids[:target_len]
    elapsed = time.perf_counter() - start_time
    tokens_generated = len(curr_ids) - len(prompt_ids)

    return {
        "text": tokenizer.decode(curr_ids),
        "tokens_generated": tokens_generated,
        "elapsed_sec": elapsed,
        "tokens_per_sec": tokens_generated / elapsed if elapsed > 0 else 0.0,
        "drafted_tokens": drafted_total,
        "accepted_tokens": accepted_total,
        "acceptance_rate": accepted_total / drafted_total if drafted_total > 0 else 0.0,
        "target_forward_calls": target_calls,
        "gamma": gamma,
    }


def speculative_generate(
    prompt: str,
    target_weights: GPT2Weights,
    draft_weights: GPT2Weights | QuantizedGPT2Weights,
    tokenizer: LocalGPT2Tokenizer,
    max_new_tokens: int = 32,
    gamma: int = 4,
) -> str:
    """Convenience wrapper returning only the generated string."""
    res = speculative_generate_with_metrics(
        prompt=prompt,
        target_weights=target_weights,
        draft_weights=draft_weights,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        gamma=gamma,
    )
    return str(res["text"])
