"""Benchmark script comparing FP32 vs INT8 Quantization: Memory, Perplexity, and Speed.

Evaluates:
- Model weight memory footprint (MB)
- Compression ratio
- Perplexity (PPL) on standard evaluation text
- Decoding speed (tokens/sec)

Outputs:
- results_quantize.csv
"""

from __future__ import annotations
import argparse
import os
import sys
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from tinyinfer.generate import generate_with_metrics
from tinyinfer.quantize import (
    QuantizedGPT2Weights,
    compute_cross_entropy_perplexity,
)
from tinyinfer.tokenizer import LocalGPT2Tokenizer
from tinyinfer.weights import load_gpt2_weights

SAMPLE_EVAL_TEXT = (
    "In mathematics and computer science, an algorithm is a finite sequence of rigorous instructions, "
    "typically used to solve a class of specific problems or to perform a computation. Algorithms are "
    "used as specifications for performing calculations, data processing, automated reasoning, and other tasks. "
    "As an effective method, an algorithm can be expressed within a finite amount of space and time."
)


def run_quantization_benchmark(
    model_dir: str = "tinyinfer/models/gpt2",
    output_dir: str = "tinyinfer",
    eval_tokens: int = 16,
) -> pd.DataFrame:
    """Run comprehensive FP32 vs INT8 quantization comparison benchmark."""
    print("=" * 60)
    print("tinyinfer Stage 3: FP32 vs INT8 Quantization Benchmark")
    print("=" * 60)

    print(f"\n[1/5] Loading FP32 GPT-2 weights from: {model_dir}")
    weights_fp32 = load_gpt2_weights(model_dir)
    tokenizer = LocalGPT2Tokenizer(model_dir)

    # 1. Evaluate FP32
    print("\n[2/5] Evaluating FP32 Model...")
    tokens = tokenizer.encode(SAMPLE_EVAL_TEXT)
    ppl_fp32 = compute_cross_entropy_perplexity(tokens, weights_fp32, max_length=64)
    print(f"      FP32 Perplexity (PPL): {ppl_fp32:.4f}")

    gen_fp32 = generate_with_metrics(
        prompt="Artificial intelligence is transforming",
        weights=weights_fp32,
        tokenizer=tokenizer,
        max_new_tokens=eval_tokens,
        use_cache=True,
    )
    speed_fp32 = gen_fp32["tokens_per_sec"]
    time_fp32 = gen_fp32["elapsed_sec"]
    print(f"      FP32 Decoding Speed:   {speed_fp32:.2f} tok/s ({time_fp32:.2f}s for {eval_tokens} tokens)")

    # 2. Quantize to INT8
    print("\n[3/5] Quantizing linear layers to INT8 per-channel with scale and zero-point...")
    t0 = time.perf_counter()
    q_weights = QuantizedGPT2Weights(weights_fp32)
    quant_time = time.perf_counter() - t0
    mem_info = q_weights.get_memory_bytes()
    print(f"      Quantization finished in {quant_time:.2f}s")
    print(f"      FP32 Weights Size:  {mem_info['fp32_mb']:.2f} MB")
    print(f"      INT8 Weights Size:  {mem_info['quantized_mb']:.2f} MB")
    print(f"      Compression Ratio:  {mem_info['compression_ratio']:.2f}x")

    # 3. Evaluate INT8
    print("\n[4/5] Evaluating INT8 Quantized Model...")
    deq_weights = q_weights.dequantize_to_fp32()
    ppl_int8 = compute_cross_entropy_perplexity(tokens, deq_weights, max_length=64)
    ppl_degradation_pct = (ppl_int8 - ppl_fp32) / ppl_fp32 * 100.0
    print(f"      INT8 Perplexity (PPL): {ppl_int8:.4f} (Degradation: {ppl_degradation_pct:+.2f}%)")

    gen_int8 = generate_with_metrics(
        prompt="Artificial intelligence is transforming",
        weights=deq_weights,
        tokenizer=tokenizer,
        max_new_tokens=eval_tokens,
        use_cache=True,
    )
    speed_int8 = gen_int8["tokens_per_sec"]
    time_int8 = gen_int8["elapsed_sec"]
    print(f"      INT8 Decoding Speed:   {speed_int8:.2f} tok/s ({time_int8:.2f}s for {eval_tokens} tokens)")

    # 4. Generate comparison table
    print("\n[5/5] Compiling results...")
    results = [
        {
            "Precision": "FP32",
            "Weights (MB)": round(mem_info["fp32_mb"], 2),
            "Compression": "1.00x",
            "Perplexity (PPL)": round(ppl_fp32, 4),
            "PPL Delta (%)": "0.00%",
            "Tokens/sec": round(speed_fp32, 2),
            "Latency (s)": round(time_fp32, 2),
        },
        {
            "Precision": "INT8 (Per-Channel)",
            "Weights (MB)": round(mem_info["quantized_mb"], 2),
            "Compression": f"{mem_info['compression_ratio']:.2f}x",
            "Perplexity (PPL)": round(ppl_int8, 4),
            "PPL Delta (%)": f"{ppl_degradation_pct:+.2f}%",
            "Tokens/sec": round(speed_int8, 2),
            "Latency (s)": round(time_int8, 2),
        },
    ]

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "results_quantize.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Results saved to: {csv_path}\n")
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tinyinfer Quantization Benchmark")
    parser.add_argument("--model-dir", default="tinyinfer/models/gpt2", help="Model directory")
    parser.add_argument("--output-dir", default="tinyinfer", help="Output directory")
    parser.add_argument("--eval-tokens", type=int, default=16, help="Tokens to generate for speed test")
    args = parser.parse_args()

    run_quantization_benchmark(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        eval_tokens=args.eval_tokens,
    )
