"""Resolve the AlphaNesGpu_float code root (for loading .so custom ops).

Override with env ``ALPHANES_FLOAT_ROOT`` if the tree is relocated.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent


def code_root() -> Path:
    return Path(os.environ.get("ALPHANES_FLOAT_ROOT", _DEFAULT)).resolve()
