"""Unit tests for the tinyinfer interactive streaming CLI."""

from __future__ import annotations

import os
import pytest

from tinyinfer.cli import (
    WeightsNotFoundError,
    build_parser,
    check_weights_exist,
    stream_generate,
)
from tinyinfer.generate import generate
from tinyinfer.tokenizer import LocalGPT2Tokenizer
from tinyinfer.weights import load_gpt2_weights

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "gpt2"))


@pytest.fixture(scope="module")
def weights_and_tokenizer():
    weights = load_gpt2_weights(MODEL_DIR)
    tokenizer = LocalGPT2Tokenizer(MODEL_DIR)
    return weights, tokenizer


def test_cli_argument_parsing_defaults():
    """Verify CLI parser establishes correct default configuration."""
    parser = build_parser()
    args = parser.parse_args([])

    assert args.prompt is None
    assert args.int8 is False
    assert args.no_cache is False
    assert args.speculative is False
    assert args.gamma == 4
    assert args.max_tokens == 64
    assert args.temperature == 0.0
    assert args.top_k == 40
    assert args.seed is None
    assert args.bench is False


def test_cli_argument_parsing_custom_flags():
    """Verify custom CLI flags parse accurately."""
    parser = build_parser()
    args = parser.parse_args([
        "--prompt", "Hello world",
        "--int8",
        "--no-cache",
        "--speculative",
        "--gamma", "6",
        "--max-tokens", "128",
        "--temperature", "0.7",
        "--top-k", "50",
        "--seed", "42",
        "--bench",
    ])

    assert args.prompt == "Hello world"
    assert args.int8 is True
    assert args.no_cache is True
    assert args.speculative is True
    assert args.gamma == 6
    assert args.max_tokens == 128
    assert args.temperature == 0.7
    assert args.top_k == 50
    assert args.seed == 42
    assert args.bench is True


def test_cli_missing_weights_informative_remediation():
    """Verify missing weights directory raises WeightsNotFoundError pointing to download_weights.py."""
    fake_dir = os.path.join(os.path.dirname(__file__), "nonexistent_model_dir_xyz")
    with pytest.raises(WeightsNotFoundError) as exc_info:
        check_weights_exist(fake_dir)

    err_str = str(exc_info.value)
    assert "[ERROR] GPT-2 model weights not found" in err_str
    assert "download_weights.py" in err_str


def test_cli_stream_generate_parity_with_direct_generate(weights_and_tokenizer):
    """Verify stream_generate() produces identical greedy output to direct generate()."""
    weights, tokenizer = weights_and_tokenizer
    prompt = "The theory of computation demonstrates that"
    max_tokens = 10

    # Direct generate
    direct_out = generate(
        prompt=prompt,
        weights=weights,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=0.0,
        use_cache=True,
    )

    # Streaming generate with callback
    collected_tokens: list[str] = []
    stream_res = stream_generate(
        prompt=prompt,
        weights=weights,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=0.0,
        use_cache=True,
        token_callback=lambda tok: collected_tokens.append(tok),
    )

    assert stream_res["text"] == direct_out
    assert stream_res["tokens_generated"] == max_tokens
    assert len(collected_tokens) == max_tokens
    assert prompt + "".join(collected_tokens) == direct_out
    assert stream_res["tokens_per_sec"] > 0.0
    assert stream_res["ms_per_token"] > 0.0


def test_cli_stream_generate_no_cache_parity(weights_and_tokenizer):
    """Verify non-cached streaming matches direct non-cached generation."""
    weights, tokenizer = weights_and_tokenizer
    prompt = "Alan Turing was a"
    max_tokens = 8

    direct_out = generate(
        prompt=prompt,
        weights=weights,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=0.0,
        use_cache=False,
    )

    stream_res = stream_generate(
        prompt=prompt,
        weights=weights,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=0.0,
        use_cache=False,
    )

    assert stream_res["text"] == direct_out
    assert stream_res["mode"] == "Vanilla (No Cache)"
