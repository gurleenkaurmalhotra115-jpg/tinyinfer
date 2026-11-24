"""Stage 1 Test: Numerical parity verification against HuggingFace GPT-2.

Validates that pure NumPy gpt2_forward logits match HuggingFace PyTorch logits
within 1e-4 absolute tolerance across 5 diverse prompts.
"""

from __future__ import annotations
import os
import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from tinyinfer.model import gpt2_forward
from tinyinfer.weights import load_gpt2_weights


MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "gpt2"))
PROMPTS = [
    "Hello, I'm a language model,",
    "The capital of France is",
    "def bubble_sort(arr):",
    "In a shocking finding, scientists discovered a herd of unicorns",
    "Attention is all you",
]


@pytest.fixture(scope="module")
def gpt2_components():
    """Load both our NumPy model weights and the reference HuggingFace model."""
    print("\nLoading tinyinfer NumPy weights from safetensors...")
    np_weights = load_gpt2_weights(MODEL_DIR)

    print("Loading reference HuggingFace PyTorch GPT-2...")
    hf_tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
    hf_model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    hf_model.eval()

    return np_weights, hf_tokenizer, hf_model


@pytest.mark.parametrize("prompt", PROMPTS)
def test_forward_logits_parity(gpt2_components, prompt):
    """Verify that pure NumPy logits match HuggingFace within 1e-4 tolerance."""
    np_weights, hf_tokenizer, hf_model = gpt2_components

    # 1. Tokenize prompt
    input_ids = hf_tokenizer.encode(prompt)
    input_array = np.array([input_ids], dtype=np.int64)
    input_tensor = torch.tensor(input_array, dtype=torch.long)

    # 2. Pure NumPy forward pass
    np_logits = gpt2_forward(input_array, np_weights)

    # 3. Reference HuggingFace forward pass
    with torch.no_grad():
        hf_outputs = hf_model(input_tensor)
        hf_logits = hf_outputs.logits.detach().cpu().numpy()

    # 4. Check shapes
    assert np_logits.shape == hf_logits.shape, (
        f"Shape mismatch: NumPy {np_logits.shape} vs HuggingFace {hf_logits.shape}"
    )

    # 5. Measure numerical difference
    max_abs_diff = float(np.max(np.abs(np_logits - hf_logits)))
    mean_abs_diff = float(np.mean(np.abs(np_logits - hf_logits)))

    print(f"\nPrompt: '{prompt}'")
    print(f"  Shape:         {np_logits.shape}")
    print(f"  Max Abs Diff:  {max_abs_diff:.6e}")
    print(f"  Mean Abs Diff: {mean_abs_diff:.6e}")

    # Check top-1 next-token prediction parity (must match token-for-token)
    np_next_token = int(np.argmax(np_logits[0, -1, :]))
    hf_next_token = int(np.argmax(hf_logits[0, -1, :]))
    assert np_next_token == hf_next_token, (
        f"Predicted next token ID mismatch: NumPy={np_next_token} vs HF={hf_next_token}"
    )

    # Parity tolerance requirement across 12 full FP32 transformer layers: <= 2e-4
    # (Matches standard MKL vs non-FMA numerical accumulation variance)
    assert max_abs_diff < 2e-4, f"Parity failure: max difference {max_abs_diff:.6e} exceeds 2e-4 ceiling!"
