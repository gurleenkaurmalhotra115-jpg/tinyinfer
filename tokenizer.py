"""Local tokenizer wrapper for tinyinfer."""

from __future__ import annotations
import os
from tokenizers import Tokenizer


class LocalGPT2Tokenizer:
    """Offline tokenizer wrapper loading directly from local tokenizer.json."""

    def __init__(self, model_dir: str):
        tok_json_path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tok_json_path):
            raise FileNotFoundError(f"tokenizer.json not found at {tok_json_path}")
        self._tokenizer = Tokenizer.from_file(tok_json_path)

    def encode(self, text: str) -> list[int]:
        """Convert string to list of token integer IDs."""
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: list[int] | list[list[int]]) -> str:
        """Convert token IDs back to human-readable string."""
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self._tokenizer.decode(token_ids)
