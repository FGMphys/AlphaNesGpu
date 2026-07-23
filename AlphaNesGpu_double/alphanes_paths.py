"""Resolve the AlphaNesGpu_double code root (for loading .so custom ops).

Override with env ``ALPHANES_DOUBLE_ROOT`` if the tree is relocated.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent


def code_root() -> Path:
    return Path(os.environ.get("ALPHANES_DOUBLE_ROOT", _DEFAULT)).resolve()
