"""Stage 4 tests: Speculative Decoding Exact Parity & KV-Cache Rollback."""

from __future__ import annotations

import os
import numpy as np
import pytest

from tinyinfer.generate import generate
from tinyinfer.kv_cache import KVCache
from tinyinfer.model import gpt2_forward
from tinyinfer.quantize import QuantizedGPT2Weights
from tinyinfer.speculative import speculative_generate_with_metrics
from tinyinfer.tokenizer import LocalGPT2Tokenizer
from tinyinfer.weights import load_gpt2_weights

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2")


@pytest.fixture(scope="module")
def weights_and_tokenizer():
    weights = load_gpt2_weights(MODEL_DIR)
    tokenizer = LocalGPT2Tokenizer(MODEL_DIR)
    return weights, tokenizer


def test_speculative_exact_parity_with_perturbed_draft(weights_and_tokenizer):
    """Verify exact token parity even when draft model intentionally disagrees periodically."""
    target_weights, tokenizer = weights_and_tokenizer
    prompt = "The theory of computation shows that"
    max_tokens = 12

    baseline = generate(
        prompt=prompt,
        weights=target_weights,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=0.0,
        use_cache=True,
    )

    step_counter = 0

    def noisy_draft_forward(input_ids: np.ndarray, w, kv_cache: KVCache | None = None) -> np.ndarray:
        nonlocal step_counter
        logits = gpt2_forward(input_ids, w, kv_cache=kv_cache)
        # Periodically force a wrong draft token during decode to exercise KVCache.truncate rollback
        if input_ids.shape[1] == 1:
            step_counter += 1
            if step_counter % 2 == 0:
                logits = logits.copy()
                logits[0, -1, 0] += 1000.0
        return logits

    res = speculative_generate_with_metrics(
        prompt=prompt,
        target_weights=target_weights,
        draft_weights=target_weights,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        gamma=3,
        draft_forward_fn=noisy_draft_forward,
    )

    assert res["text"] == baseline
    assert res["tokens_generated"] == max_tokens
    assert res["drafted_tokens"] >= 1
    assert res["accepted_tokens"] < res["drafted_tokens"]  # Confirms rollbacks occurred


def test_speculative_exact_parity_with_int8_draft(weights_and_tokenizer):
    """Verify lossless greedy parity when using INT8 QuantizedGPT2Weights as draft."""
    target_weights, tokenizer = weights_and_tokenizer
    prompt = "Neural networks learn representations by"
    max_tokens = 10

    baseline = generate(
        prompt=prompt,
        weights=target_weights,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=0.0,
        use_cache=True,
    )

    q_draft = QuantizedGPT2Weights(target_weights)
    res = speculative_generate_with_metrics(
        prompt=prompt,
        target_weights=target_weights,
        draft_weights=q_draft,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        gamma=4,
    )

    assert res["text"] == baseline
    assert res["tokens_generated"] == max_tokens
    assert res["target_forward_calls"] < max_tokens
    assert res["acceptance_rate"] > 0.5
