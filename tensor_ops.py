"""Fundamental tensor operations implemented in pure NumPy.

Every function is explicitly documented with input and output tensor shapes.
No PyTorch or external deep learning framework is used.
"""

from __future__ import annotations
import numpy as np


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Apply Layer Normalization over the last dimension (channels).
    
    Formula:
        y = ((x - mean) / sqrt(var + eps)) * gamma + beta

    Args:
        x: Input tensor of shape (..., C), typically (B, T, C).
        gamma: Scale parameter of shape (C,).
        beta: Shift/bias parameter of shape (C,).
        eps: Small epsilon for numerical stability.

    Returns:
        Normalized tensor of shape (..., C), matching input shape.
    """
    # Shape of x: (..., C)
    # Compute mean across channel dimension C, keeping dims for broadcasting: (..., 1)
    mean = np.mean(x, axis=-1, keepdims=True)
    
    # Compute variance across channel dimension C: (..., 1)
    # Note: ddof=0 gives population variance (1/N), matching PyTorch's torch.nn.LayerNorm
    variance = np.var(x, axis=-1, keepdims=True)
    
    # Standardize x: (..., C)
    x_hat = (x - mean) / np.sqrt(variance + eps)
    
    # Scale by gamma (C,) and shift by beta (C,): (..., C)
    return x_hat * gamma + beta


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) with tanh approximation used by GPT-2.

    Formula:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

    Args:
        x: Input tensor of arbitrary shape (...).

    Returns:
        Activated tensor of the same shape (...).
    """
    # Constant sqrt(2 / pi) approx 0.7978845608028654
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    inner = sqrt_2_over_pi * (x + 0.044715 * np.power(x, 3))
    return 0.5 * x * (1.0 + np.tanh(inner))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable Softmax along the specified axis.

    Formula:
        softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Args:
        x: Input tensor of arbitrary shape (...).
        axis: Dimension along which softmax will be computed (default: -1).

    Returns:
        Normalized probability distribution tensor of the same shape (...).
    """
    # Subtract max for numerical stability (prevents overflow in exp): keepdims for broadcasting
    max_val = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_val)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x


def linear(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Affine linear transformation.

    Handles both standard linear shapes (C_out, C_in) and GPT-2 Conv1D shapes (C_in, C_out).

    Args:
        x: Input activations of shape (..., C_in).
        weight: Weight matrix of shape (C_in, C_out) or (C_out, C_in).
        bias: Optional bias vector of shape (C_out,).

    Returns:
        Output tensor of shape (..., C_out).
    """
    c_in = x.shape[-1]

    # GPT-2's Conv1D weights are stored transposed as (C_in, C_out)
    if weight.shape[0] == c_in:
        out = np.matmul(x, weight)
    elif weight.shape[1] == c_in:
        out = np.matmul(x, weight.T)
    else:
        raise ValueError(
            f"Shape mismatch in linear layer: x shape {x.shape} has C_in={c_in}, "
            f"but weight has shape {weight.shape}"
        )

    if bias is not None:
        out = out + bias

    return out
