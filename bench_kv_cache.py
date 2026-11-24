"""Benchmark script comparing decoding throughput and memory: With KV-Cache vs. Without KV-Cache.

Measures:
- Tokens per second (throughput)
- Total elapsed time (latency)
- Peak KV-cache memory usage (MB)
- Speedup factor achieved by KV caching

Outputs:
- results_kv_cache.csv
- results_kv_cache.png
"""

from __future__ import annotations
import argparse
import os
import sys
import tempfile
import time

# Ensure clean imports and writable matplotlib cache
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tinyinfer.generate import generate_with_metrics
from tinyinfer.tokenizer import LocalGPT2Tokenizer
from tinyinfer.weights import load_gpt2_weights


def run_benchmark(
    model_dir: str,
    token_lengths: list[int],
    prompt: str = "The history of modern computing began with",
    output_dir: str = "tinyinfer",
) -> pd.DataFrame:
    """Run decoding benchmark across varying token generation lengths.

    Args:
        model_dir: Path to directory containing GPT-2 safetensors and tokenizer.
        token_lengths: List of token generation counts to benchmark.
        prompt: Initial prompt text.
        output_dir: Destination directory for CSV and plot.

    Returns:
        Pandas DataFrame containing benchmark metrics.
    """
    print(f"Loading GPT-2 weights and tokenizer from: {model_dir}")
    weights = load_gpt2_weights(model_dir)
    tokenizer = LocalGPT2Tokenizer(model_dir)

    print(f"\nInitial Prompt: {prompt!r}")
    print(f"Token counts to benchmark: {token_lengths}")

    # Warmup pass
    print("Warming up inference engine...")
    _ = generate_with_metrics(prompt, weights, tokenizer, max_new_tokens=4, use_cache=True)
    _ = generate_with_metrics(prompt, weights, tokenizer, max_new_tokens=4, use_cache=False)

    records = []

    for n in token_lengths:
        print(f"\n{'='*50}")
        print(f"Benchmarking N = {n} tokens...")

        # 1. With KV-Cache
        print(f"  [1/2] Running WITH KV-cache...")
        res_cache = generate_with_metrics(prompt, weights, tokenizer, max_new_tokens=n, use_cache=True)
        t_cache = res_cache["elapsed_sec"]
        tok_sec_cache = res_cache["tokens_per_sec"]
        mem_cache = res_cache["peak_cache_mb"]
        print(f"        Time: {t_cache:.2f}s | Speed: {tok_sec_cache:.2f} tok/s | Cache RAM: {mem_cache:.2f} MB")

        # 2. Without KV-Cache (skip without-cache if n > 128 on CPU to prevent excessive wait)
        if n > 128:
            print(f"  [2/2] Skipping Without-Cache for N={n} (would take >1hr without cache on CPU).")
            t_nocache = np.nan
            tok_sec_nocache = np.nan
            speedup = np.nan
        else:
            print(f"  [2/2] Running WITHOUT KV-cache...")
            res_nocache = generate_with_metrics(prompt, weights, tokenizer, max_new_tokens=n, use_cache=False)
            t_nocache = res_nocache["elapsed_sec"]
            tok_sec_nocache = res_nocache["tokens_per_sec"]
            speedup = t_nocache / t_cache if t_cache > 0 else np.nan
            print(f"        Time: {t_nocache:.2f}s | Speed: {tok_sec_nocache:.2f} tok/s | Speedup: {speedup:.2f}x")

        records.append({
            "tokens": n,
            "cache_time_sec": t_cache,
            "cache_tokens_per_sec": tok_sec_cache,
            "cache_peak_mb": mem_cache,
            "nocache_time_sec": t_nocache,
            "nocache_tokens_per_sec": tok_sec_nocache,
            "speedup": speedup,
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "results_kv_cache.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Benchmark results saved to: {csv_path}")
    print(df.to_string(index=False))

    # Generate benchmark plots
    plot_path = os.path.join(output_dir, "results_kv_cache.png")
    generate_plot(df, plot_path)
    print(f"[+] Benchmark plot saved to: {plot_path}")

    return df


def generate_plot(df: pd.DataFrame, output_path: str) -> None:
    """Generate high-resolution comparison plots from benchmark data."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Subplot 1: Generation Time vs Number of Tokens
    ax1 = axes[0]
    ax1.plot(df["tokens"], df["cache_time_sec"], marker="o", color="#1f77b4", linewidth=2, label="With KV-Cache (O(N))")
    
    # Filter valid no-cache points
    valid_nocache = df.dropna(subset=["nocache_time_sec"])
    if not valid_nocache.empty:
        ax1.plot(valid_nocache["tokens"], valid_nocache["nocache_time_sec"], marker="s", color="#d62728", linewidth=2, linestyle="--", label="Without KV-Cache (O(N²))")

    ax1.set_title("Decoding Latency vs Generated Tokens", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Generated Tokens (N)", fontsize=11)
    ax1.set_ylabel("Total Latency (seconds)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # Subplot 2: Throughput (Tokens/sec)
    ax2 = axes[1]
    ax2.plot(df["tokens"], df["cache_tokens_per_sec"], marker="o", color="#2ca02c", linewidth=2, label="With KV-Cache")
    if not valid_nocache.empty:
        ax2.plot(valid_nocache["tokens"], valid_nocache["nocache_tokens_per_sec"], marker="^", color="#ff7f0e", linewidth=2, linestyle="--", label="Without KV-Cache")

    ax2.set_title("Decoding Throughput (Tokens/sec)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Generated Tokens (N)", fontsize=11)
    ax2.set_ylabel("Tokens / Second", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tinyinfer KV-Cache Benchmark")
    parser.add_argument("--model-dir", default="tinyinfer/models/gpt2", help="Path to GPT-2 model directory")
    parser.add_argument("--tokens", nargs="+", type=int, default=[16, 32, 64], help="Token lengths to benchmark")
    parser.add_argument("--output-dir", default="tinyinfer", help="Output directory for CSV and plot")
    args = parser.parse_args()

    run_benchmark(
        model_dir=args.model_dir,
        token_lengths=args.tokens,
        output_dir=args.output_dir,
    )
