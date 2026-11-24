"""Benchmark speculative decoding (INT8 draft -> FP32 target) vs standard KV-cached decoding."""

from __future__ import annotations

import argparse
import os

from tinyinfer.generate import generate_with_metrics
from tinyinfer.quantize import QuantizedGPT2Weights
from tinyinfer.speculative import speculative_generate_with_metrics
from tinyinfer.tokenizer import LocalGPT2Tokenizer
from tinyinfer.weights import load_gpt2_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tinyinfer speculative decoding.")
    parser.add_argument("--model-dir", default=os.path.join(os.path.dirname(__file__), "models", "gpt2"))
    parser.add_argument("--prompt", default="Alan Turing was a pioneer of")
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--gamma", type=int, default=4)
    args = parser.parse_args()

    weights = load_gpt2_weights(args.model_dir)
    tokenizer = LocalGPT2Tokenizer(args.model_dir)
    quantized_draft = QuantizedGPT2Weights(weights)

    base = generate_with_metrics(args.prompt, weights, tokenizer, max_new_tokens=args.tokens, use_cache=True)
    spec = speculative_generate_with_metrics(
        args.prompt,
        target_weights=weights,
        draft_weights=quantized_draft,
        tokenizer=tokenizer,
        max_new_tokens=args.tokens,
        gamma=args.gamma,
    )

    print("=== Speculative Decoding Benchmark ===")
    print(f"Exact Token Parity : {base['text'] == spec['text']}")
    print(f"Baseline (KV)      : {base['elapsed_sec']:.2f}s ({base['tokens_per_sec']:.2f} tok/s)")
    print(f"Speculative (g={args.gamma}) : {spec['elapsed_sec']:.2f}s ({spec['tokens_per_sec']:.2f} tok/s)")
    print(f"Acceptance Rate    : {spec['acceptance_rate']:.1%} ({spec['accepted_tokens']}/{spec['drafted_tokens']})")
    print(f"Target Forward Calls: {spec['target_forward_calls']} (vs {args.tokens} naive KV steps)")


if __name__ == "__main__":
    main()
