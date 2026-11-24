"""Stage 2 tests: Key-Value (KV) Cache Verification and Equivalence.

Tests:
1. KV-cache operations: update, seq_len, truncate, clear, memory calculation, copy.
2. Step-by-step logit equivalence: verifying that logits from cached decoding (1, 1)
   match the full context forward pass (1, T) within floating point tolerance.
3. End-to-end token parity: greedy generation with cache matches generation without cache 100%.
"""

from __future__ import annotations
import os
import numpy as np
import pytest
from tinyinfer.kv_cache import KVCache
from tinyinfer.model import gpt2_forward
from tinyinfer.generate import generate
from tinyinfer.sampling import sample_next_token
from tinyinfer.tokenizer import LocalGPT2Tokenizer
from tinyinfer.weights import load_gpt2_weights

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2")


@pytest.fixture(scope="module")
def weights_and_tokenizer():
    weights = load_gpt2_weights(MODEL_DIR)
    tokenizer = LocalGPT2Tokenizer(MODEL_DIR)
    return weights, tokenizer


def test_kv_cache_operations():
    """Verify core KVCache data structure methods: update, truncate, clear, copy."""
    cache = KVCache(num_layers=4)
    assert cache.get_seq_len() == 0
    assert cache.get_memory_bytes() == 0

    # Simulate layer 0 update with shape (1, 12, 3, 64)
    k1 = np.ones((1, 12, 3, 64), dtype=np.float32)
    v1 = np.ones((1, 12, 3, 64), dtype=np.float32) * 2.0
    k_out, v_out = cache.update(0, k1, v1)

    assert cache.get_seq_len(0) == 3
    assert k_out.shape == (1, 12, 3, 64)
    assert v_out.shape == (1, 12, 3, 64)
    assert cache.get_memory_bytes() == 2 * (1 * 12 * 3 * 64 * 4)

    # Append 1 new token
    k2 = np.ones((1, 12, 1, 64), dtype=np.float32) * 3.0
    v2 = np.ones((1, 12, 1, 64), dtype=np.float32) * 4.0
    k_out2, v_out2 = cache.update(0, k2, v2)

    assert cache.get_seq_len(0) == 4
    assert k_out2.shape == (1, 12, 4, 64)
    assert v_out2.shape == (1, 12, 4, 64)

    # Test copy
    cloned = cache.copy()
    assert cloned.get_seq_len(0) == 4
    np.testing.assert_allclose(cloned.k[0], cache.k[0])

    # Test truncate to 2 tokens
    cache.truncate(2)
    assert cache.get_seq_len(0) == 2
    assert cache.k[0].shape == (1, 12, 2, 64)
    # Ensure clone was unaffected (deep copy)
    assert cloned.get_seq_len(0) == 4

    # Test clear
    cache.clear()
    assert cache.get_seq_len(0) == 0
    assert cache.k[0] is None


@pytest.mark.parametrize(
    "prompt",
    [
        "Artificial intelligence is",
        "def fibonacci(n):",
        "The fastest animal on earth is",
    ],
)
def test_kv_cache_step_by_step_logit_parity(prompt, weights_and_tokenizer):
    """Verify that cached incremental decoding produces the exact same logits as full recompute."""
    weights, tokenizer = weights_and_tokenizer
    prompt_ids = tokenizer.encode(prompt)
    
    # 1. Prefill with cache
    cache = KVCache(num_layers=weights.n_layer)
    prompt_arr = np.array([prompt_ids], dtype=np.int64)
    
    cached_logits = gpt2_forward(prompt_arr, weights, kv_cache=cache)
    full_logits = gpt2_forward(prompt_arr, weights, kv_cache=None)
    
    np.testing.assert_allclose(
        cached_logits,
        full_logits,
        atol=1e-5,
        err_msg="Prefill logits mismatch between cached and non-cached forward pass",
    )

    # 2. Step through 5 incremental decode steps
    curr_tokens = list(prompt_ids)
    for step in range(5):
        next_id = int(np.argmax(cached_logits[0, -1, :]))
        curr_tokens.append(next_id)

        # Full recompute without cache
        full_arr = np.array([curr_tokens], dtype=np.int64)
        full_out = gpt2_forward(full_arr, weights, kv_cache=None)
        expected_next_logits = full_out[0, -1, :]

        # Cached incremental step with single token (1, 1)
        single_arr = np.array([[next_id]], dtype=np.int64)
        cached_out = gpt2_forward(single_arr, weights, kv_cache=cache)
        cached_next_logits = cached_out[0, -1, :]

        max_diff = np.max(np.abs(cached_next_logits - expected_next_logits))
        assert max_diff < 1e-4, (
            f"Step {step}: max logit difference {max_diff:.3e} exceeds 1e-4"
        )
        assert np.argmax(cached_next_logits) == np.argmax(expected_next_logits), (
            f"Step {step}: predicted next token differs between cached and non-cached!"
        )


@pytest.mark.parametrize(
    "prompt,num_tokens",
    [
        ("In a galaxy far, far away", 15),
        ("The scientific method relies on", 15),
    ],
)
def test_kv_cache_generation_end_to_end_parity(prompt, num_tokens, weights_and_tokenizer):
    """Verify that end-to-end greedy generation is 100% token-for-token identical."""
    weights, tokenizer = weights_and_tokenizer

    output_no_cache = generate(
        prompt=prompt,
        weights=weights,
        tokenizer=tokenizer,
        max_new_tokens=num_tokens,
        temperature=0.0,
        use_cache=False,
    )

    output_with_cache = generate(
        prompt=prompt,
        weights=weights,
        tokenizer=tokenizer,
        max_new_tokens=num_tokens,
        temperature=0.0,
        use_cache=True,
    )

    print(f"\nPrompt: {prompt!r}")
    print(f"No-Cache:   {output_no_cache!r}")
    print(f"With-Cache: {output_with_cache!r}")

    assert output_with_cache == output_no_cache, (
        "Generated text differs between cached and non-cached generation!"
    )
