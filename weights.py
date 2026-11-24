"""Safetensors weight loader for tinyinfer in pure NumPy.

Loads GPT-2 safetensors directly into NumPy float32 arrays without PyTorch.
"""

from __future__ import annotations
import json
import os
from typing import Any
import numpy as np
from safetensors.numpy import load_file


class GPT2Weights:
    """Structured container holding all GPT-2 parameters in NumPy float32 format."""

    def __init__(self, raw_weights: dict[str, np.ndarray], config: dict[str, Any]):
        self.config = config
        self.n_layer = config.get("n_layer", 12)
        self.n_head = config.get("n_head", 12)
        self.n_embd = config.get("n_embd", 768)
        self.vocab_size = config.get("vocab_size", 50257)

        # Token & Position Embeddings
        # wte: (vocab_size, n_embd) = (50257, 768)
        self.wte = raw_weights["wte.weight"]
        # wpe: (max_positions, n_embd) = (1024, 768)
        self.wpe = raw_weights["wpe.weight"]

        # Transformer Blocks (0 to n_layer - 1)
        self.blocks = []
        for i in range(self.n_layer):
            prefix = f"h.{i}."
            block = {
                # LayerNorm 1 (pre-attention)
                "ln_1_g": raw_weights[f"{prefix}ln_1.weight"],  # (768,)
                "ln_1_b": raw_weights[f"{prefix}ln_1.bias"],    # (768,)
                # Attention projections
                "c_attn_w": raw_weights[f"{prefix}attn.c_attn.weight"],  # (768, 2304)
                "c_attn_b": raw_weights[f"{prefix}attn.c_attn.bias"],    # (2304,)
                "c_proj_w": raw_weights[f"{prefix}attn.c_proj.weight"],  # (768, 768)
                "c_proj_b": raw_weights[f"{prefix}attn.c_proj.bias"],    # (768,)
                # LayerNorm 2 (pre-MLP)
                "ln_2_g": raw_weights[f"{prefix}ln_2.weight"],  # (768,)
                "ln_2_b": raw_weights[f"{prefix}ln_2.bias"],    # (768,)
                # MLP projections
                "c_fc_w": raw_weights[f"{prefix}mlp.c_fc.weight"],      # (768, 3072)
                "c_fc_b": raw_weights[f"{prefix}mlp.c_fc.bias"],        # (3072,)
                "c_mlp_proj_w": raw_weights[f"{prefix}mlp.c_proj.weight"],  # (3072, 768)
                "c_mlp_proj_b": raw_weights[f"{prefix}mlp.c_proj.bias"],    # (768,)
            }
            self.blocks.append(block)

        # Final LayerNorm
        self.ln_f_g = raw_weights["ln_f.weight"]  # (768,)
        self.ln_f_b = raw_weights["ln_f.bias"]    # (768,)

        # LM Head (tied with wte in GPT-2)
        if "lm_head.weight" in raw_weights:
            self.lm_head_w = raw_weights["lm_head.weight"]
        else:
            self.lm_head_w = self.wte  # weight tying


def load_gpt2_weights(model_dir: str) -> GPT2Weights:
    """Load GPT-2 weights and configuration from a local safetensors directory.

    Args:
        model_dir: Path to directory containing config.json and model.safetensors.

    Returns:
        GPT2Weights structured instance.
    """
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Safetensors weights not found at {weights_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    raw_tensors = load_file(weights_path)
    return GPT2Weights(raw_tensors, config)
