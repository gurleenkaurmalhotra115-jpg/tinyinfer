"""Stage 3 tests: INT8 Per-Channel Weight Quantization Verification.

Tests:
1. Mathematical reconstruction accuracy of affine per-channel quantize & dequantize.
2. Memory compression ratio verification (verifying ~3.8x weight size reduction).
3. Logit and generation fidelity between FP32 and INT8 dequantized model.
4. Perplexity comparison ensuring minimal quality degradation.
"""

from __future__ import annotations
import os
import numpy as np
import pytest
from tinyinfer.quantize import (
    QuantizedGPT2Weights,
    compute_cross_entropy_perplexity,
    dequantize_channel,
    quantize_channel,
)
from tinyinfer.model import gpt2_forward
from tinyinfer.generate import generate
from tinyinfer.tokenizer import LocalGPT2Tokenizer
from tinyinfer.weights import load_gpt2_weights

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2")


@pytest.fixture(scope="module")
def weights_and_tokenizer():
    weights = load_gpt2_weights(MODEL_DIR)
    tokenizer = LocalGPT2Tokenizer(MODEL_DIR)
    return weights, tokenizer


def test_quantize_channel_math_reconstruction():
    """Verify that per-channel INT8 quantization error is bounded by scale / 2."""
    rng = np.random.default_rng(42)
    # Simulate a linear projection matrix of shape (768, 768)
    w_orig = rng.normal(loc=0.0, scale=0.02, size=(768, 768)).astype(np.float32)

    q_tensor = quantize_channel(w_orig, axis=0)

    assert q_tensor.q_data.dtype == np.uint8
    assert q_tensor.scale.shape == (1, 768)
    assert q_tensor.zero_point.shape == (1, 768)
    assert q_tensor.q_data.shape == (768, 768)

    w_rec = q_tensor.dequantize()

    # In uniform quantization, round-off error is bounded by 0.5 * scale
    max_error = np.max(np.abs(w_orig - w_rec))
    max_scale = np.max(q_tensor.scale)
    assert max_error <= (0.5 * max_scale + 1e-6), (
        f"Reconstruction error {max_error:.4e} exceeds theoretical bound {0.5 * max_scale:.4e}"
    )

    # Cosine similarity between original and reconstructed weights should be > 0.9999
    cos_sim = np.sum(w_orig * w_rec) / (np.linalg.norm(w_orig) * np.linalg.norm(w_rec))
    assert cos_sim > 0.9999, f"Cosine similarity {cos_sim:.5f} is below 0.9999"


def test_quantized_model_memory_reduction(weights_and_tokenizer):
    """Verify that INT8 quantization reduces block weights by nearly 4x."""
    weights, _ = weights_and_tokenizer
    q_weights = QuantizedGPT2Weights(weights)
    mem_info = q_weights.get_memory_bytes()

    print(f"\nFP32 Memory:      {mem_info['fp32_mb']:.2f} MB")
    print(f"Quantized Memory:  {mem_info['quantized_mb']:.2f} MB")
    print(f"Compression Ratio: {mem_info['compression_ratio']:.2f}x")

    # Compression ratio of the transformer weights should be well above 3x
    assert mem_info["compression_ratio"] > 3.0, (
        f"Expected compression ratio > 3.0x, got {mem_info['compression_ratio']:.2f}x"
    )


def test_quantized_generation_fidelity(weights_and_tokenizer):
    """Verify that the quantized model produces coherent, high-fidelity text."""
    weights, tokenizer = weights_and_tokenizer
    q_weights = QuantizedGPT2Weights(weights)
    deq_weights = q_weights.dequantize_to_fp32()

    prompt = "The theory of general relativity was developed by"

    out_fp32 = generate(
        prompt=prompt,
        weights=weights,
        tokenizer=tokenizer,
        max_new_tokens=15,
        temperature=0.0,
        use_cache=True,
    )

    out_int8 = generate(
        prompt=prompt,
        weights=deq_weights,
        tokenizer=tokenizer,
        max_new_tokens=15,
        temperature=0.0,
        use_cache=True,
    )

    print(f"\nPrompt: {prompt!r}")
    print(f"FP32 Output: {out_fp32!r}")
    print(f"INT8 Output: {out_int8!r}")

    # "Albert Einstein" should appear in both generations
    assert "Einstein" in out_fp32
    assert "Einstein" in out_int8


def test_quantized_perplexity_parity(weights_and_tokenizer):
    """Verify that perplexity degradation on a text sample is under 5%."""
    weights, tokenizer = weights_and_tokenizer
    sample_text = (
        "Alan Turing was a British mathematician, computer scientist, logician, cryptanalyst, "
        "philosopher, and theoretical biologist. Turing was highly influential in the development "
        "of theoretical computer science, providing a formalisation of the concepts of algorithm "
        "and computation with the Turing machine."
    )
    token_ids = tokenizer.encode(sample_text)

    ppl_fp32 = compute_cross_entropy_perplexity(token_ids, weights, max_length=64)

    q_weights = QuantizedGPT2Weights(weights)
    deq_weights = q_weights.dequantize_to_fp32()
    ppl_int8 = compute_cross_entropy_perplexity(token_ids, deq_weights, max_length=64)

    rel_diff_pct = abs(ppl_int8 - ppl_fp32) / ppl_fp32 * 100.0

    print(f"\nFP32 Perplexity: {ppl_fp32:.3f}")
    print(f"INT8 Perplexity: {ppl_int8:.3f}")
    print(f"Relative Degradation: {rel_diff_pct:.2f}%")

    assert rel_diff_pct < 5.0, (
        f"Perplexity degradation {rel_diff_pct:.2f}% exceeds 5% threshold"
    )
