"""Precision helpers for the shared STAF Python path.

Use YAML ``precision: float|double`` (aliases ``float32`` / ``float64``)
plus ``keras.backend.set_floatx``. Hardcoded ``float32`` / ``float64``
strings in training code should go through these helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf

PrecisionName = str  # "float32" | "float64"


def normalize_precision(raw: Any) -> PrecisionName:
    if raw is None:
        raise ValueError("precision is None")
    s = str(raw).strip().lower()
    if s in ("float", "float32", "fp32", "f32", "32"):
        return "float32"
    if s in ("double", "float64", "fp64", "f64", "64"):
        return "float64"
    raise ValueError(
        f"STAF: unknown precision {raw!r}; use float|double (or float32|float64)"
    )


def infer_precision_from_tree(code_root: Union[str, Path]) -> PrecisionName:
    name = Path(code_root).resolve().name.lower()
    if "double" in name:
        return "float64"
    return "float32"


def set_precision(
    raw: Optional[Any] = None,
    *,
    code_root: Optional[Union[str, Path]] = None,
    default: Optional[PrecisionName] = None,
) -> PrecisionName:
    """Set keras floatx and return the resolved TF dtype name."""
    if raw is not None and str(raw).strip() != "":
        dtype = normalize_precision(raw)
    elif default is not None:
        dtype = normalize_precision(default)
    elif code_root is not None:
        dtype = infer_precision_from_tree(code_root)
    else:
        dtype = "float32"
    tf.keras.backend.set_floatx(dtype)
    print(f"STAF: precision set to {dtype}")
    return dtype


def tf_dtype() -> str:
    return tf.keras.backend.floatx()


def np_dtype():
    return np.float64 if tf_dtype() == "float64" else np.float32


def zero(value: float = 0.0):
    return tf.constant(value, dtype=tf_dtype())
