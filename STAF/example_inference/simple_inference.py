"""Minimal STAF inference smoke example.

Usage (from this directory or any cwd):

  python simple_inference.py /path/to/model_dir [pos_file] [box_file]

Defaults: ``pos_0`` / ``box_0`` in the current working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

_STAF_HOME = Path(__file__).resolve().parents[1]
if str(_STAF_HOME) not in sys.path:
    sys.path.insert(0, str(_STAF_HOME))

from staf_models.staf_model_inference_full import staf_full_inference  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

if len(sys.argv) < 2:
    raise SystemExit(__doc__)

model_dir = sys.argv[1]
pos_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("pos_0")
box_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("box_0")

Model = staf_full_inference(model_dir)

Pos = np.loadtxt(pos_path, dtype="float32").reshape((1, -1, 3))
Box = np.loadtxt(box_path, dtype="float32").reshape((1, 6))

output = Model.full_test(Pos, Box)
print("energy[0]", float(np.asarray(output[0]).reshape(-1)[0]))
print("force shape", np.asarray(output[1]).shape)
w = np.asarray(output[2]).reshape(-1)
print("virial_diag", w[0], w[1], w[2])
