"""Resolve STAF custom-op roots (compiled ``.so`` trees).

Python lives under ``STAF/``; float32 and float64 ops stay in
``STAF/ops_float/src`` and ``STAF/ops_double/src`` until the CUDA
``real`` unify lands. Call ``set_ops_root`` / ``set_precision`` before
loading layers that ``tf.load_op_library``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

_STAF_HOME = Path(__file__).resolve().parent
_ops_root: Optional[Path] = None


def staf_home() -> Path:
    return _STAF_HOME


def _normalize_ops_name(raw: Union[str, Path]) -> str:
    s = str(raw).strip().lower()
    if s in ("float", "float32", "fp32", "f32", "32", "ops_float"):
        return "ops_float"
    if s in ("double", "float64", "fp64", "f64", "64", "ops_double"):
        return "ops_double"
    raise ValueError(f"STAF: unknown ops precision {raw!r}")


def set_ops_root(precision: Optional[Union[str, Path]] = None) -> Path:
    """Select ``ops_float`` or ``ops_double`` under STAF home."""
    global _ops_root
    env = os.environ.get("STAF_OPS_ROOT") or os.environ.get("ALPHANES_OPS_ROOT")
    if env:
        _ops_root = Path(env).resolve()
        return _ops_root
    if precision is None:
        # Fall back to keras floatx if already set; else float.
        try:
            import tensorflow as tf

            fx = tf.keras.backend.floatx()
            precision = "double" if fx == "float64" else "float"
        except Exception:
            precision = "float"
    name = _normalize_ops_name(precision)
    _ops_root = (_STAF_HOME / name).resolve()
    os.environ["STAF_OPS_ROOT"] = str(_ops_root)
    return _ops_root


def code_root() -> Path:
    """Directory that contains ``src/.../reforce.so`` for the active precision."""
    global _ops_root
    if _ops_root is not None:
        return _ops_root
    env = os.environ.get("STAF_OPS_ROOT") or os.environ.get("ALPHANES_OPS_ROOT")
    if env:
        _ops_root = Path(env).resolve()
        return _ops_root
    # Legacy aliases
    for key in (
        "STAF_FLOAT_ROOT",
        "STAF_DOUBLE_ROOT",
        "ALPHANES_FLOAT_ROOT",
        "ALPHANES_DOUBLE_ROOT",
    ):
        if os.environ.get(key):
            _ops_root = Path(os.environ[key]).resolve()
            return _ops_root
    return set_ops_root("float")
