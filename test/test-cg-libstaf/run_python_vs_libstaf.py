#!/usr/bin/env python3
"""Sprint 5: 1-frame E/F Python STAF-CG vs libstaf_cg.

Uses the float32 ONNX export under test/test-cg-inference/model_onnx_double/.
Python may run float32 (preferred) or double; tolerances start at 1e-3 and
loosen to 1e-2 if precisions differ.

Writes test/test-cg-libstaf/summary.txt with pass true/false.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/francegm/AlphaNesGpu")
STAF_CG = REPO / "STAF-CG"
STAF = REPO / "STAF"
USCGSITE = Path("/home/francegm/ORIGAMI/INFER_INTRA_TRY2+USCGSITE")
MODEL_ONNX = REPO / "test/test-cg-inference/model_onnx_double"
OUT_DIR = REPO / "test/test-cg-libstaf"
SMOKE = REPO / "libstaf_cg/build/staf_force_smoke"

# 1-epoch origami YAML (freeze_ep10). Not MODEL1896 MD (16/8).
CUTOFF_EP1 = """\
50 24
50 276
25 0
20 0
20 0
10 0
"""


def _setup_path() -> None:
    cg, staf = str(STAF_CG), str(STAF)
    if cg in sys.path:
        sys.path.remove(cg)
    sys.path.insert(0, cg)
    if staf not in sys.path:
        sys.path.insert(1, staf)


def _load_frame0():
    frames = REPO / "test/test-cg-inference/frames"
    if (frames / "pos.npy").is_file() and (frames / "box.npy").is_file():
        pos = np.load(frames / "pos.npy")
        box = np.load(frames / "box.npy")
        src = str(frames)
    else:
        pos = np.load(USCGSITE / "dataset/training/pos.npy", mmap_mode="r")
        box = np.load(USCGSITE / "dataset/training/box.npy", mmap_mode="r")
        src = str(USCGSITE / "dataset/training")
    pos0 = np.array(pos[0], dtype=np.float64).reshape(-1)
    box0 = np.array(box[0], dtype=np.float64).reshape(-1)
    if box0.size == 9:
        box0 = np.array(
            [box0[0], box0[1], box0[2], box0[4], box0[5], box0[8]], dtype=np.float64
        )
    elif box0.size != 6:
        raise SystemExit(f"unexpected box shape {box0.shape} from {src}")
    n = pos0.size // 3
    return pos0, box0, n, src


def _write_frame_bin(path: Path, pos: np.ndarray, box: np.ndarray, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<i", int(n)))
        fh.write(np.asarray(box, dtype=np.float64).tobytes())
        fh.write(np.asarray(pos, dtype=np.float64).tobytes())


def _python_ef(model: Path, pos: np.ndarray, box: np.ndarray, precision: str):
    _setup_path()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from staf.dtype import set_precision, np_dtype
    from staf_cg_paths import set_ops_root

    set_precision(precision)
    set_ops_root(precision)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    from staf_cg_models.alpha_nes_model_inference import alpha_nes_full_inference

    dt = np_dtype()
    n = pos.size // 3
    pos_b = np.asarray(pos, dtype=dt).reshape(1, n * 3)
    box_b = np.asarray(box, dtype=dt).reshape(1, -1)
    model_obj = alpha_nes_full_inference(str(model))
    e, f = model_obj.full_test(pos_b, box_b)
    return float(np.asarray(e).reshape(-1)[0]), np.asarray(f).reshape(-1)


def _has_savedmodel(model: Path) -> bool:
    return (model / "model_type0" / "saved_model.pb").is_file() or (
        model / "model_type0"
    ).is_dir()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_ONNX.is_dir():
        print(f"missing ONNX model {MODEL_ONNX}", file=sys.stderr)
        return 2
    cutoff = MODEL_ONNX / "cutoff_info"
    if not cutoff.is_file():
        cutoff.write_text(CUTOFF_EP1)
        print("wrote", cutoff)
    nnf = MODEL_ONNX / "number_of_nn.dat"
    if not nnf.is_file():
        nnf.write_text("1\n")

    pos, box, n, src = _load_frame0()
    print(f"frame0 from {src}: n={n} box={box}")
    frame_bin = OUT_DIR / "frame0.bin"
    _write_frame_bin(frame_bin, pos, box, n)

    py_prec = "float" if _has_savedmodel(MODEL_ONNX) else "double"
    # Prefer float32 Python if SavedModel is float; else double vs float32 ONNX.
    try:
        e_py, f_py = _python_ef(MODEL_ONNX, pos, box, py_prec)
    except Exception as exc:
        print("float Python infer failed, retrying double:", exc)
        py_prec = "double"
        e_py, f_py = _python_ef(MODEL_ONNX, pos, box, py_prec)

    lib_txt = OUT_DIR / "libstaf_ef.txt"
    if not SMOKE.is_file():
        print(f"missing {SMOKE} — build libstaf_cg first", file=sys.stderr)
        OUT_DIR.joinpath("summary.txt").write_text(
            "pass: false\nblocker: libstaf_cg smoke binary missing\n"
        )
        return 3
    env = os.environ.copy()
    rc = subprocess.run(
        [str(SMOKE), str(MODEL_ONNX), str(frame_bin), str(lib_txt)],
        check=False,
        env=env,
    )
    if rc.returncode != 0 or not lib_txt.is_file():
        print("staf_force_smoke failed", rc.returncode, file=sys.stderr)
        OUT_DIR.joinpath("summary.txt").write_text(
            f"pass: false\nblocker: staf_force_smoke rc={rc.returncode}\n"
        )
        return 4
    lines = lib_txt.read_text().strip().splitlines()
    e_c = float(lines[0])
    f_c = np.array(
        [list(map(float, ln.split())) for ln in lines[1:]], dtype=np.float64
    ).reshape(-1)

    dE = abs(e_py - e_c)
    dF = float(np.max(np.abs(f_py.astype(np.float64) - f_c)))
    tol = 1e-3 if py_prec in ("float", "float32") else 1e-2
    ok = bool(dE < tol and dF < tol)
    summary = {
        "pass": ok,
        "python_precision": py_prec,
        "libstaf_precision": "float32",
        "n_beads": n,
        "E_python": e_py,
        "E_libstaf": e_c,
        "max_abs_dE": dE,
        "max_abs_dF": dF,
        "tol": tol,
        "model": str(MODEL_ONNX),
        "frame_source": src,
        "note": "1-epoch keras checkpoint ONNX (not MODEL1896 SavedModel)",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT_DIR / "summary.txt").write_text(
        f"pass: {str(ok).lower()}\n"
        f"E_python: {e_py:.10g}\n"
        f"E_libstaf: {e_c:.10g}\n"
        f"max|dE|: {dE:.6g}\n"
        f"max|dF|: {dF:.6g}\n"
        f"tol: {tol}\n"
        f"python_precision: {py_prec}\n"
        f"libstaf_precision: float32\n"
        f"n_beads: {n}\n"
        f"model: {MODEL_ONNX}\n"
        f"note: 1-epoch keras ckpt (MODEL1896 is SavedModel-only)\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
