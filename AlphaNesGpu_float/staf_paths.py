"""Resolve the STAF code-tree root (for loading .so custom ops).

Prefer ``STAF_FLOAT_ROOT``; ``ALPHANES_FLOAT_ROOT`` is accepted as a
deprecated alias until A2 packaging lands.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent


def code_root() -> Path:
    root = os.environ.get("STAF_FLOAT_ROOT") or os.environ.get(
        "ALPHANES_FLOAT_ROOT", _DEFAULT
    )
    return Path(root).resolve()
