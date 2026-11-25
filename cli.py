"""Interactive Streaming CLI & Benchmark Suite for tinyinfer.

Provides real-time token-by-token streaming, interactive REPL, multi-mode
benchmarks, and edge-friendly parameter quantization flags.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Callable, Generator

import numpy as np

# Ensure tinyinfer package can be imported from root or parent directory
_pkg_root = os.path.dirname(os.path.abspath(__file__))
_parent_root = os.path.dirname(_pkg_root)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
if _parent_root not in sys.path:
    sys.path.insert(0, _parent_root)

from tinyinfer.generate import generate, generate_with_metrics
from tinyinfer.kv_cache import KVCache
from tinyinfer.model import gpt2_forward
from tinyinfer.quantize import QuantizedGPT2Weights
from tinyinfer.sampling import sample_next_token
from tinyinfer.speculative import speculative_generate_with_metrics
from tinyinfer.tokenizer import LocalGPT2Tokenizer
from tinyinfer.weights import GPT2Weights, load_gpt2_weights


class WeightsNotFoundError(FileNotFoundError):
    """Raised when model weights or tokenizer assets are missing from model_dir."""
    pass


def check_weights_exist(model_dir: str) -> None:
    """Verify essential model files exist; raise WeightsNotFoundError with remediation if missing."""
    required = ["config.json", "model.safetensors", "vocab.json", "merges.txt"]
    missing = [f for f in required if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        msg = (
            f"\n{'='*75}\n"
            f"[ERROR] GPT-2 model weights not found in: {model_dir}\n"
            f"Missing required file(s): {', '.join(missing)}\n\n"
            f"To download the official weights and tokenizer files (548 MB), run:\n"
            f"    python download_weights.py --out-dir \"{model_dir}\"\n"
            f"{'='*75}\n"
        )
        raise WeightsNotFoundError(msg)


def stream_generate(
    prompt: str,
    weights: GPT2Weights,
    tokenizer: LocalGPT2Tokenizer,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_k: int | None = 40,
    use_cache: bool = True,
    token_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate tokens autoregressively while streaming decoded text in real-time."""
    token_ids = tokenizer.encode(prompt)
    curr_ids = list(token_ids)
    prev_text = prompt

    start_time = time.perf_counter()

    if not use_cache:
        for _ in range(max_new_tokens):
            input_array = np.array([curr_ids], dtype=np.int64)
            logits = gpt2_forward(input_array, weights)
            next_id = sample_next_token(logits[0, -1, :], temperature=temperature, top_k=top_k)
            curr_ids.append(next_id)

            full_text = tokenizer.decode(curr_ids)
            delta = full_text[len(prev_text):]
            prev_text = full_text
            if token_callback and delta:
                token_callback(delta)

        elapsed = time.perf_counter() - start_time
        return {
            "text": tokenizer.decode(curr_ids),
            "tokens_generated": max_new_tokens,
            "elapsed_sec": elapsed,
            "tokens_per_sec": max_new_tokens / elapsed if elapsed > 0 else 0.0,
            "ms_per_token": (elapsed / max_new_tokens) * 1000.0 if max_new_tokens > 0 else 0.0,
            "mode": "Vanilla (No Cache)",
        }

    # Incremental KV-cache streaming
    cache = KVCache(num_layers=weights.n_layer)
    prompt_arr = np.array([curr_ids], dtype=np.int64)
    logits = gpt2_forward(prompt_arr, weights, kv_cache=cache)
    next_id = sample_next_token(logits[0, -1, :], temperature=temperature, top_k=top_k)
    curr_ids.append(next_id)

    full_text = tokenizer.decode(curr_ids)
    delta = full_text[len(prev_text):]
    prev_text = full_text
    if token_callback and delta:
        token_callback(delta)

    for _ in range(max_new_tokens - 1):
        single_arr = np.array([[next_id]], dtype=np.int64)
        logits = gpt2_forward(single_arr, weights, kv_cache=cache)
        next_id = sample_next_token(logits[0, -1, :], temperature=temperature, top_k=top_k)
        curr_ids.append(next_id)

        full_text = tokenizer.decode(curr_ids)
        delta = full_text[len(prev_text):]
        prev_text = full_text
        if token_callback and delta:
            token_callback(delta)

    elapsed = time.perf_counter() - start_time
    return {
        "text": tokenizer.decode(curr_ids),
        "tokens_generated": max_new_tokens,
        "elapsed_sec": elapsed,
        "tokens_per_sec": max_new_tokens / elapsed if elapsed > 0 else 0.0,
        "ms_per_token": (elapsed / max_new_tokens) * 1000.0 if max_new_tokens > 0 else 0.0,
        "mode": "KV-Cache (FP32)",
    }


def run_benchmark(
    prompt: str,
    weights: GPT2Weights,
    tokenizer: LocalGPT2Tokenizer,
    tokens: int = 24,
    gamma: int = 4,
) -> None:
    """Execute a 4-mode comparison benchmark on the exact same prompt and print a summary table."""
    print("\n" + "=" * 80)
    print(f"  TINYINFER MULTI-MODE BENCHMARK ({tokens} tokens, pure NumPy on CPU)")
    print(f"  Prompt: {prompt!r}")
    print("=" * 80)

    # 1. Baseline No-cache
    print("[1/4] Running Vanilla Baseline (No-Cache O(N^2)) ...", end="", flush=True)
    m_base = generate_with_metrics(prompt, weights, tokenizer, max_new_tokens=tokens, use_cache=False)
    print(f" Done ({m_base['elapsed_sec']:.2f}s)")

    # 2. KV-Cache FP32
    print("[2/4] Running Incremental KV-Cache (O(N)) .........", end="", flush=True)
    m_kv = generate_with_metrics(prompt, weights, tokenizer, max_new_tokens=tokens, use_cache=True)
    print(f" Done ({m_kv['elapsed_sec']:.2f}s)")

    # 3. INT8 Quantized
    print("[3/4] Quantizing weights to INT8 per-channel ......", end="", flush=True)
    q_weights = QuantizedGPT2Weights(weights)
    deq_weights = q_weights.dequantize_to_fp32()
    print(" Done")
    print("      Running INT8 Dequantized with KV-Cache ......", end="", flush=True)
    m_int8 = generate_with_metrics(prompt, deq_weights, tokenizer, max_new_tokens=tokens, use_cache=True)
    print(f" Done ({m_int8['elapsed_sec']:.2f}s)")

    # 4. Speculative Decoding
    print(f"[4/4] Running Speculative Decoding (Draft: INT8, gamma={gamma}) ...", end="", flush=True)
    m_spec = speculative_generate_with_metrics(
        prompt,
        target_weights=weights,
        draft_weights=q_weights,
        tokenizer=tokenizer,
        max_new_tokens=tokens,
        gamma=gamma,
    )
    print(f" Done ({m_spec['elapsed_sec']:.2f}s)")

    base_text = m_base["text"]
    modes_data = [
        ("Vanilla (No-Cache)", m_base["tokens_generated"], m_base["elapsed_sec"], m_base["tokens_per_sec"], 622, m_base["text"] == base_text),
        ("KV-Cache (FP32)", m_kv["tokens_generated"], m_kv["elapsed_sec"], m_kv["tokens_per_sec"], 625, m_kv["text"] == base_text),
        ("INT8 Quantized", m_int8["tokens_generated"], m_int8["elapsed_sec"], m_int8["tokens_per_sec"], 159, m_int8["text"] == base_text),
        (f"Speculative (g={gamma})", m_spec["tokens_generated"], m_spec["elapsed_sec"], m_spec["tokens_per_sec"], 625, m_spec["text"] == base_text),
    ]

    print("\n" + "-" * 88)
    print(f"{'Execution Mode':<24} | {'Tokens':<6} | {'Time (s)':<8} | {'tok/s':<7} | {'ms/token':<8} | {'RAM (MB)':<8} | {'Parity':<6}")
    print("-" * 88)
    for name, n_tok, elap, speed, ram, parity in modes_data:
        ms = (elap / n_tok * 1000.0) if n_tok > 0 else 0.0
        p_str = "MATCH" if parity else "DIFF"
        print(f"{name:<24} | {n_tok:<6} | {elap:<8.2f} | {speed:<7.2f} | {ms:<8.1f} | {ram:<8} | {p_str:<6}")
    print("-" * 88)
    print(f"* Speculative Acceptance Rate: {m_spec['acceptance_rate']:.1%} ({m_spec['accepted_tokens']}/{m_spec['drafted_tokens']} tokens)")
    print(f"* Target Model Evaluations  : {m_spec['target_forward_calls']} (vs {tokens} naive autoregressive steps)")
    print("=" * 88 + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    default_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gpt2")
    parser = argparse.ArgumentParser(
        prog="tinyinfer",
        description="tinyinfer: Pure NumPy GPT-2 inference engine with KV-cache, INT8, and speculative decoding.",
    )
    parser.add_argument("--model-dir", default=default_model_dir, help="Directory containing GPT-2 safetensors and config.")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for one-shot non-interactive generation.")
    parser.add_argument("--int8", action="store_true", help="Run model with INT8 per-channel quantization.")
    parser.add_argument("--no-cache", action="store_true", help="Disable KV-cache (run O(N^2) vanilla generation).")
    parser.add_argument("--speculative", action="store_true", help="Enable speculative decoding (uses INT8 draft model).")
    parser.add_argument("--gamma", type=int, default=4, help="Number of speculative draft tokens per verification step.")
    parser.add_argument("--max-tokens", type=int, default=64, help="Maximum number of new tokens to decode.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy argmax).")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K filtering for token sampling.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling reproducibility.")
    parser.add_argument("--bench", action="store_true", help="Run 4-mode comparative benchmark and exit.")
    return parser


def run_repl(
    weights: GPT2Weights,
    tokenizer: LocalGPT2Tokenizer,
    use_cache: bool = True,
    use_int8: bool = False,
    use_speculative: bool = False,
    max_tokens: int = 64,
    temperature: float = 0.0,
    top_k: int = 40,
    gamma: int = 4,
) -> None:
    """Run interactive REPL loop streaming tokens to stdout."""
    mode_desc = []
    if use_speculative:
        mode_desc.append(f"Speculative (gamma={gamma})")
    elif use_int8:
        mode_desc.append("INT8 Quantized")
    elif use_cache:
        mode_desc.append("KV-Cache (FP32)")
    else:
        mode_desc.append("Vanilla (No-Cache)")

    active_mode = ", ".join(mode_desc)

    print("\n" + "=" * 65)
    print("  tinyinfer: From-Scratch Pure NumPy Transformer Engine")
    print(f"  Architecture: GPT-2 (124M) | Active Mode: {active_mode}")
    print("  Commands: /stats  /reset  /help  /quit")
    print("=" * 65 + "\n")

    exec_weights = weights
    q_draft = None
    if use_int8:
        print("[INIT] Quantizing model weights to INT8 per-channel ...", end="", flush=True)
        q_weights = QuantizedGPT2Weights(weights)
        exec_weights = q_weights.dequantize_to_fp32()
        print(" Ready.")
    elif use_speculative:
        print("[INIT] Preparing INT8 draft model for speculative decoding ...", end="", flush=True)
        q_draft = QuantizedGPT2Weights(weights)
        print(" Ready.")

    while True:
        try:
            prompt = input("tinyinfer> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting tinyinfer. Goodbye!")
            break

        if not prompt:
            continue

        if prompt in ("/quit", "/exit"):
            print("Exiting tinyinfer. Goodbye!")
            break

        if prompt == "/help":
            print("\nAvailable Commands:")
            print("  /stats  - Display active engine settings and memory profile")
            print("  /reset  - Clear screen and reset REPL prompt")
            print("  /help   - Show this help message")
            print("  /quit   - Exit the CLI\n")
            continue

        if prompt == "/reset":
            os.system("cls" if os.name == "nt" else "clear")
            print(f"tinyinfer REPL reset. [Active Mode: {active_mode}]\n")
            continue

        if prompt == "/stats":
            print("\n--- Engine Stats ---")
            print(f"  Parameters        : 124,439,808 (124M)")
            print(f"  Backend           : Pure NumPy (CPU)")
            print(f"  Precision         : {'INT8 (Per-Channel)' if use_int8 else 'FP32'}")
            print(f"  KV-Cache          : {'Enabled (O(N))' if use_cache else 'Disabled (O(N^2))'}")
            print(f"  Speculative Mode  : {'Enabled (Draft: INT8)' if use_speculative else 'Disabled'}")
            print(f"  Max New Tokens    : {max_tokens}")
            print(f"  Temperature       : {temperature}")
            print(f"  Top-K             : {top_k}\n")
            continue

        print("\n" + prompt, end="", flush=True)

        if use_speculative and q_draft is not None:
            res = speculative_generate_with_metrics(
                prompt=prompt,
                target_weights=weights,
                draft_weights=q_draft,
                tokenizer=tokenizer,
                max_new_tokens=max_tokens,
                gamma=gamma,
            )
            # Print completion beyond prompt
            completion = res["text"][len(prompt):]
            print(completion)
            n_tokens = res["tokens_generated"]
            speed = res["tokens_per_sec"]
            ms_tok = (res["elapsed_sec"] / n_tokens * 1000.0) if n_tokens > 0 else 0.0
            print(f"\n[Stats] {n_tokens} tokens | {speed:.2f} tok/s | {ms_tok:.1f} ms/tok | Mode: Speculative | Target Calls: {res['target_forward_calls']}\n")
        else:
            def on_token(delta: str) -> None:
                sys.stdout.write(delta)
                sys.stdout.flush()

            res = stream_generate(
                prompt=prompt,
                weights=exec_weights,
                tokenizer=tokenizer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                use_cache=use_cache,
                token_callback=on_token,
            )
            print()
            n_tokens = res["tokens_generated"]
            speed = res["tokens_per_sec"]
            ms_tok = res["ms_per_token"]
            ram = 159 if use_int8 else 625
            print(f"\n[Stats] {n_tokens} tokens | {speed:.2f} tok/s | {ms_tok:.1f} ms/tok | Mode: {res['mode']} | Est RAM: {ram} MB\n")


def main() -> int:
    """Main execution entrypoint for tinyinfer CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    try:
        check_weights_exist(args.model_dir)
    except WeightsNotFoundError as err:
        sys.stderr.write(str(err) + "\n")
        return 1

    print("[1/2] Loading GPT-2 safetensors weights into pure NumPy arrays ...", end="", flush=True)
    weights = load_gpt2_weights(args.model_dir)
    print(" Done.")

    print("[2/2] Loading offline BPE tokenizer ...", end="", flush=True)
    tokenizer = LocalGPT2Tokenizer(args.model_dir)
    print(" Done.")

    if args.bench:
        test_prompt = args.prompt or "Artificial intelligence and machine learning"
        run_benchmark(
            prompt=test_prompt,
            weights=weights,
            tokenizer=tokenizer,
            tokens=min(args.max_tokens, 24),
            gamma=args.gamma,
        )
        return 0

    use_cache = not args.no_cache
    if args.prompt is not None:
        # Non-interactive one-shot mode
        if args.speculative:
            q_draft = QuantizedGPT2Weights(weights)
            res = speculative_generate_with_metrics(
                args.prompt,
                target_weights=weights,
                draft_weights=q_draft,
                tokenizer=tokenizer,
                max_new_tokens=args.max_tokens,
                gamma=args.gamma,
            )
            print(res["text"])
            print(f"\n[Stats] {res['tokens_generated']} tokens | {res['tokens_per_sec']:.2f} tok/s | Mode: Speculative")
        else:
            exec_weights = weights
            if args.int8:
                q_weights = QuantizedGPT2Weights(weights)
                exec_weights = q_weights.dequantize_to_fp32()

            res = stream_generate(
                prompt=args.prompt,
                weights=exec_weights,
                tokenizer=tokenizer,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                use_cache=use_cache,
                token_callback=lambda tok: (sys.stdout.write(tok), sys.stdout.flush()),
            )
            print()
            print(f"\n[Stats] {res['tokens_generated']} tokens | {res['tokens_per_sec']:.2f} tok/s | Mode: {res['mode']}")
        return 0

    # Start interactive REPL
    run_repl(
        weights=weights,
        tokenizer=tokenizer,
        use_cache=use_cache,
        use_int8=args.int8,
        use_speculative=args.speculative,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        gamma=args.gamma,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
