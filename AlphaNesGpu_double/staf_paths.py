"""Resolve the STAF code-tree root (for loading .so custom ops).

Prefer ``STAF_DOUBLE_ROOT``; ``ALPHANES_DOUBLE_ROOT`` is accepted as a
deprecated alias until A2 packaging lands.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent


def code_root() -> Path:
    root = os.environ.get("STAF_DOUBLE_ROOT") or os.environ.get(
        "ALPHANES_DOUBLE_ROOT", _DEFAULT
    )
    return Path(root).resolve()
