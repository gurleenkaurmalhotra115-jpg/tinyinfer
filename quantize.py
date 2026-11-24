"""INT8 Per-Channel Weight Quantization with Scale and Zero-Point.

Implements asymmetric (affine) per-channel weight quantization for transformer linear layers:
    W_fp32 ≈ Scale * (W_int8 - ZeroPoint)

Where:
    Scale = (max(W) - min(W)) / 255.0
    ZeroPoint = round(-min(W) / Scale)
    W_int8 = clip(round(W_fp32 / Scale) + ZeroPoint, 0, 255)

This compresses 32-bit floating point weights into 8-bit unsigned integers,
reducing memory footprint by ~4x while maintaining minimal perplexity degradation.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
from tinyinfer.weights import GPT2Weights


@dataclass
class QuantizedTensor:
    """Represents a per-channel INT8 quantized tensor with scale and zero-point."""
    q_data: np.ndarray      # uint8 array of quantized values: shape (C_in, C_out)
    scale: np.ndarray       # float32 scale per channel: shape (C_out,) or (1, C_out)
    zero_point: np.ndarray  # int32 / float32 zero point per channel: shape (C_out,)
    axis: int = 0           # Quantization axis (0 = per-column / output channel)

    def dequantize(self) -> np.ndarray:
        """Dequantize back to float32 representation for arithmetic execution.

        Formula:
            w_rec = scale * (q_data - zero_point)

        Returns:
            Reconstructed float32 tensor of original shape (C_in, C_out).
        """
        # q_data: (C_in, C_out)
        # scale: (1, C_out)
        # zero_point: (1, C_out)
        return self.scale * (self.q_data.astype(np.float32) - self.zero_point)

    @property
    def nbytes(self) -> int:
        """Total memory consumed by quantized data and metadata in bytes."""
        return self.q_data.nbytes + self.scale.nbytes + self.zero_point.nbytes


def dequantize_channel(
    q_data: np.ndarray,
    scale: np.ndarray,
    zero_point: np.ndarray,
) -> np.ndarray:
    """Dequantize uint8 data given scale and zero-point back to float32.

    Formula:
        w_rec = scale * (q_data - zero_point)

    Returns:
        Reconstructed float32 tensor.
    """
    return scale * (q_data.astype(np.float32) - zero_point)



def quantize_channel(
    w: np.ndarray,
    axis: int = 0,
) -> QuantizedTensor:
    """Perform asymmetric INT8 quantization along the specified channel axis.

    For a 2D weight matrix (C_in, C_out), quantizing along axis=0 produces
    per-output-channel scales and zero points of shape (1, C_out).

    Args:
        w: Original float32 tensor of shape (C_in, C_out).
        axis: Dimension along which min/max ranges are computed.

    Returns:
        QuantizedTensor containing uint8 data, float32 scale, and zero-point.
    """
    w_fp32 = w.astype(np.float32)
    w_min = np.min(w_fp32, axis=axis, keepdims=True)
    w_max = np.max(w_fp32, axis=axis, keepdims=True)

    # Avoid zero division on constant weights
    diff = np.maximum(w_max - w_min, 1e-8)
    scale = (diff / 255.0).astype(np.float32)
    zero_point = np.round(-w_min / scale).astype(np.float32)

    # Affine mapping: q = round(w / scale) + zero_point
    q = np.round(w_fp32 / scale) + zero_point
    q_clipped = np.clip(q, 0, 255).astype(np.uint8)

    return QuantizedTensor(
        q_data=q_clipped,
        scale=scale,
        zero_point=zero_point,
        axis=axis,
    )


class QuantizedGPT2Weights:
    """GPT-2 weight container storing INT8 quantized weights for all linear projections."""

    def __init__(self, fp32_weights: GPT2Weights, quantize_embeddings: bool = True) -> None:
        """Quantize all transformer block linear layers from fp32_weights to INT8."""
        self.config = copy.deepcopy(fp32_weights.config)
        self.vocab_size = fp32_weights.vocab_size
        self.n_embd = fp32_weights.n_embd
        self.n_layer = fp32_weights.n_layer
        self.n_head = fp32_weights.n_head
        self.quantize_embeddings = quantize_embeddings

        # Embeddings and output projection
        if quantize_embeddings:
            self.wte = quantize_channel(fp32_weights.wte, axis=0)
            self.lm_head_w = quantize_channel(fp32_weights.lm_head_w, axis=0)
        else:
            self.wte = fp32_weights.wte
            self.lm_head_w = fp32_weights.lm_head_w

        self.wpe = fp32_weights.wpe
        self.ln_f_g = fp32_weights.ln_f_g
        self.ln_f_b = fp32_weights.ln_f_b

        # Quantize blocks
        self.quantized_blocks: list[dict[str, QuantizedTensor | np.ndarray]] = []
        for b in fp32_weights.blocks:
            q_block = {
                # LayerNorms and biases kept in FP32
                "ln_1_g": b["ln_1_g"],
                "ln_1_b": b["ln_1_b"],
                "ln_2_g": b["ln_2_g"],
                "ln_2_b": b["ln_2_b"],
                "c_attn_b": b["c_attn_b"],
                "c_proj_b": b["c_proj_b"],
                "c_fc_b": b["c_fc_b"],
                "c_mlp_proj_b": b["c_mlp_proj_b"],
                # Linear weights quantized to INT8
                "c_attn_w": quantize_channel(b["c_attn_w"], axis=0),
                "c_proj_w": quantize_channel(b["c_proj_w"], axis=0),
                "c_fc_w": quantize_channel(b["c_fc_w"], axis=0),
                "c_mlp_proj_w": quantize_channel(b["c_mlp_proj_w"], axis=0),
            }
            self.quantized_blocks.append(q_block)

    def dequantize_to_fp32(self) -> GPT2Weights:
        """Reconstruct a standard FP32 GPT2Weights instance by dequantizing all INT8 weights.

        Returns:
            Reconstructed GPT2Weights instance ready for execution in existing model pipelines.
        """
        reconstructed_blocks = []
        for qb in self.quantized_blocks:
            rb = {
                "ln_1_g": qb["ln_1_g"],
                "ln_1_b": qb["ln_1_b"],
                "ln_2_g": qb["ln_2_g"],
                "ln_2_b": qb["ln_2_b"],
                "c_attn_b": qb["c_attn_b"],
                "c_proj_b": qb["c_proj_b"],
                "c_fc_b": qb["c_fc_b"],
                "c_mlp_proj_b": qb["c_mlp_proj_b"],
                "c_attn_w": qb["c_attn_w"].dequantize(),
                "c_proj_w": qb["c_proj_w"].dequantize(),
                "c_fc_w": qb["c_fc_w"].dequantize(),
                "c_mlp_proj_w": qb["c_mlp_proj_w"].dequantize(),
            }
            reconstructed_blocks.append(rb)

        new_weights = object.__new__(GPT2Weights)
        new_weights.config = self.config
        new_weights.vocab_size = self.vocab_size
        new_weights.n_embd = self.n_embd
        new_weights.n_layer = self.n_layer
        new_weights.n_head = self.n_head
        new_weights.wte = self.wte.dequantize() if isinstance(self.wte, QuantizedTensor) else self.wte
        new_weights.wpe = self.wpe
        new_weights.ln_f_g = self.ln_f_g
        new_weights.ln_f_b = self.ln_f_b
        new_weights.lm_head_w = self.lm_head_w.dequantize() if isinstance(self.lm_head_w, QuantizedTensor) else self.lm_head_w
        new_weights.blocks = reconstructed_blocks
        return new_weights

    def get_memory_bytes(self) -> dict[str, float]:
        """Compute exact memory consumption in bytes of the quantized model vs FP32 equivalent."""
        q_bytes = 0
        fp32_bytes = 0

        # Embeddings & norms
        if isinstance(self.wte, QuantizedTensor):
            q_bytes += self.wte.nbytes
            fp32_bytes += self.wte.q_data.size * 4
        else:
            q_bytes += self.wte.nbytes
            fp32_bytes += self.wte.nbytes

        q_bytes += self.wpe.nbytes + self.ln_f_g.nbytes + self.ln_f_b.nbytes
        fp32_bytes += self.wpe.nbytes + self.ln_f_g.nbytes + self.ln_f_b.nbytes

        if isinstance(self.lm_head_w, QuantizedTensor):
            q_bytes += self.lm_head_w.nbytes
            fp32_bytes += self.lm_head_w.q_data.size * 4
        else:
            q_bytes += self.lm_head_w.nbytes
            fp32_bytes += self.lm_head_w.nbytes

        block_q_bytes = 0
        block_fp32_bytes = 0

        for qb in self.quantized_blocks:
            for k in ["ln_1_g", "ln_1_b", "ln_2_g", "ln_2_b", "c_attn_b", "c_proj_b", "c_fc_b", "c_mlp_proj_b"]:
                block_q_bytes += qb[k].nbytes
                block_fp32_bytes += qb[k].nbytes
            for k in ["c_attn_w", "c_proj_w", "c_fc_w", "c_mlp_proj_w"]:
                q_tensor: QuantizedTensor = qb[k]
                block_q_bytes += q_tensor.nbytes
                block_fp32_bytes += q_tensor.q_data.size * 4

        q_bytes += block_q_bytes
        fp32_bytes += block_fp32_bytes

        return {
            "quantized_bytes": q_bytes,
            "quantized_mb": q_bytes / (1024 * 1024),
            "fp32_bytes": fp32_bytes,
            "fp32_mb": fp32_bytes / (1024 * 1024),
            "compression_ratio": fp32_bytes / q_bytes if q_bytes > 0 else 1.0,
            "block_quantized_mb": block_q_bytes / (1024 * 1024),
            "block_fp32_mb": block_fp32_bytes / (1024 * 1024),
            "block_compression_ratio": block_fp32_bytes / block_q_bytes if block_q_bytes > 0 else 1.0,
        }



def compute_cross_entropy_perplexity(
    token_ids: list[int],
    weights: GPT2Weights,
    max_length: int = 128,
) -> float:
    """Compute perplexity (exponentiated cross-entropy loss) on a sequence of tokens.

    Loss formula:
        L = - (1 / N) * sum_{t=1}^{N} log P(x_t | x_{<t})
        Perplexity = exp(L)

    Args:
        token_ids: Ground-truth sequence of token integers.
        weights: GPT2Weights parameter container.
        max_length: Maximum sequence slice to evaluate.

    Returns:
        Perplexity value (float).
    """
    from tinyinfer.model import gpt2_forward
    from tinyinfer.tensor_ops import softmax

    eval_tokens = token_ids[:max_length]
    N = len(eval_tokens)
    assert N > 1, "Evaluation sequence must contain at least 2 tokens"

    # Forward pass over full prompt sequence: (1, N) -> (1, N, vocab_size)
    input_arr = np.array([eval_tokens], dtype=np.int64)
    logits = gpt2_forward(input_arr, weights)  # (1, N, vocab_size)

    # For position t in 0 .. N-2, logits[0, t] predicts token eval_tokens[t+1]
    total_neg_log_prob = 0.0
    evaluated_count = 0

    for t in range(N - 1):
        target_token = eval_tokens[t + 1]
        probs = softmax(logits[0, t, :], axis=-1)
        target_prob = float(probs[target_token])
        # Numerical guard
        target_prob = max(target_prob, 1e-12)
        total_neg_log_prob += -np.log(target_prob)
        evaluated_count += 1

    mean_loss = total_neg_log_prob / evaluated_count
    return float(np.exp(mean_loss))
