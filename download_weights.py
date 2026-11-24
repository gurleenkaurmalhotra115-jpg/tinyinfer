"""Download GPT-2 safetensors and tokenizer assets into tinyinfer/models/gpt2/."""

from __future__ import annotations

import argparse
import os
import urllib.request

BASE_URL = "https://huggingface.co/openai-community/gpt2/resolve/main"
FILES = [
    "config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "vocab.json",
]


def download_gpt2(target_dir: str) -> None:
    """Fetch official GPT-2 files from HuggingFace into target_dir if not present."""
    os.makedirs(target_dir, exist_ok=True)
    for filename in FILES:
        dest = os.path.join(target_dir, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[SKIP] {filename} already exists ({os.path.getsize(dest):,} bytes)")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"[GET ] Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[OK  ] Saved {dest} ({os.path.getsize(dest):,} bytes)")


def main() -> None:
    default_dir = os.path.join(os.path.dirname(__file__), "models", "gpt2")
    parser = argparse.ArgumentParser(description="Download GPT-2 weights for tinyinfer.")
    parser.add_argument("--out-dir", default=default_dir, help="Destination directory for GPT-2 files.")
    args = parser.parse_args()
    download_gpt2(args.out_dir)


if __name__ == "__main__":
    main()
