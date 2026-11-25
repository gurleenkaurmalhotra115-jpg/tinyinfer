"""Executable entrypoint for `python -m tinyinfer`."""

from __future__ import annotations

import sys
from tinyinfer.cli import main

if __name__ == "__main__":
    sys.exit(main())
