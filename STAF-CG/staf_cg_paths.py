"""Resolve STAF-CG custom-op roots (compiled ``.so`` trees).

CUDA sources live in ``STAF-CG/src/`` (``fingerprint`` / ``force`` / ``grad_*``,
no ``mixture/`` folder) and are built into ``STAF-CG/ops_float/src`` or
``STAF-CG/ops_double/src`` by ``install_path.sh``.
Call ``set_ops_root`` before loading layers that ``tf.load_op_library``.

Sprint 1: kernels are still hardcoded double; only ``ops_double`` is the
supported runtime. Float/``staf_real`` arrive in Sprint 2.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

_STAF_CG_HOME = Path(__file__).resolve().parent
_ops_root: Optional[Path] = None


def staf_cg_home() -> Path:
    return _STAF_CG_HOME


def _normalize_ops_name(raw: Union[str, Path]) -> str:
    s = str(raw).strip().lower()
    if s in ("float", "float32", "fp32", "f32", "32", "ops_float"):
        return "ops_float"
    if s in ("double", "float64", "fp64", "f64", "64", "ops_double"):
        return "ops_double"
    raise ValueError(f"STAF-CG: unknown ops precision {raw!r}")


def set_ops_root(precision: Optional[Union[str, Path]] = None) -> Path:
    """Select ``ops_float`` or ``ops_double`` under STAF-CG home."""
    global _ops_root
    env = os.environ.get("STAF_CG_OPS_ROOT")
    if env:
        _ops_root = Path(env).resolve()
        return _ops_root
    if precision is None:
        try:
            import tensorflow as tf

            fx = tf.keras.backend.floatx()
            precision = "double" if fx == "float64" else "float"
        except Exception:
            precision = "double"
    name = _normalize_ops_name(precision)
    _ops_root = (_STAF_CG_HOME / name).resolve()
    os.environ["STAF_CG_OPS_ROOT"] = str(_ops_root)
    return _ops_root


def code_root() -> Path:
    """Directory that contains ``src/.../reforce.so`` for the active precision."""
    global _ops_root
    if _ops_root is not None:
        return _ops_root
    env = os.environ.get("STAF_CG_OPS_ROOT")
    if env:
        _ops_root = Path(env).resolve()
        return _ops_root
    return set_ops_root("double")
