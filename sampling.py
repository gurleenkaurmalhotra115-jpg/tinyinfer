"""Sampling strategies for autoregressive token generation in pure NumPy."""

from __future__ import annotations
import numpy as np
from tinyinfer.tensor_ops import softmax


def sample_next_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> int:
    """Sample a single token index from unnormalized vocabulary logits.

    Args:
        logits: 1D array of logits for the vocabulary of shape (vocab_size,).
        temperature: Temperature scaling factor. If 0.0, performs deterministic greedy argmax.
        top_k: Optional top-k filter keeping only the k highest probability tokens.

    Returns:
        Sampled token integer index.
    """
    # 1. Greedy argmax if temperature is zero or near zero
    if temperature <= 1e-5:
        return int(np.argmax(logits))

    # 2. Scale logits by temperature
    scaled_logits = logits / temperature

    # 3. Optional Top-K truncation
    if top_k is not None and top_k > 0:
        top_k = min(top_k, len(scaled_logits))
        # Find the threshold value of the top-k element
        threshold = np.partition(scaled_logits, -top_k)[-top_k]
        # Mask out everything below threshold
        scaled_logits = np.where(scaled_logits < threshold, -1e9, scaled_logits)

    # 4. Softmax to obtain probabilities
    probs = softmax(scaled_logits, axis=-1)

    # 5. Stochastic sampling according to probability distribution
    # Normalize probabilities slightly to ensure exact sum to 1.0 for np.random.choice
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))
