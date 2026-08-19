#!/usr/bin/env python3
"""DEV freeze inference: MODEL1896 (and MODEL1352) on USCGSITE frames.

Does not modify DEV/AlphaNesGpu_double_CG_dv_RC/. Writes JSON next to this script.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEV_CG = REPO / "AlphaNesGpu_double_CG_dv_RC"
HERE = Path(__file__).resolve().parent
USCGSITE = Path("/home/francegm/ORIGAMI/INFER_INTRA_TRY2+USCGSITE")
MODEL1896 = Path(
    "/home/francegm/ORIGAMI/ORIGAMI_DYNAMICS/origami_uscgsite/models/MODEL1896"
)
MODEL1352 = USCGSITE / "only_intra" / "MODEL1352"

# Intra Rc/Rs/buffers: origami standard (MODEL1352 cutoff_info + MD radial/ang blocks).
# Inter cutoffs from DIMERORI16/RUNT400BOX280/nohup.out (MODEL1896 MD).
CUTOFF_1896 = """\
50 24
50 276
25 0
16 0
16 0
8 0
"""


def _prepare_model1896(dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(MODEL1896, dst)
    (dst / "number_of_nn.dat").write_text("2\n")
    (dst / "cutoff_info").write_text(CUTOFF_1896)
    return dst


def _infer(dev_python: str, model_dir: Path, pos: np.ndarray, box: np.ndarray, n_frames: int):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if str(DEV_CG) not in sys.path:
        sys.path.insert(0, str(DEV_CG))
    import tensorflow as tf

    tf.keras.backend.set_floatx("float64")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(exc)
    from alphanes_models.mixture.alpha_nes_model_inference import (
        alpha_nes_full_inference,
    )

    model = alpha_nes_full_inference(str(model_dir))
    e, f = model.full_test(pos[:n_frames], box[:n_frames])
    return np.asarray(e), np.asarray(f)


def main() -> int:
    n_frames = 3
    pos = np.load(USCGSITE / "dataset" / "training" / "pos.npy", mmap_mode="r")
    box = np.load(USCGSITE / "dataset" / "training" / "box.npy", mmap_mode="r")
    pos0 = np.array(pos[:n_frames], dtype=np.float64)
    box0 = np.array(box[:n_frames], dtype=np.float64)

    staging = HERE / "model1896_infer"
    _prepare_model1896(staging)

    results = {}
    print("FREEZE: MODEL1896 inference, frames 0..", n_frames - 1)
    e, f = _infer(sys.executable, staging, pos0, box0, n_frames)
    results["MODEL1896"] = {
        "model": str(MODEL1896),
        "cutoff_info": CUTOFF_1896.strip().splitlines(),
        "n_frames": n_frames,
        "energy": e.reshape(-1).tolist(),
        "force_rms": [float(np.sqrt(np.mean(fi ** 2))) for fi in f],
        "force_frame0_head": f[0].reshape(-1)[:6].tolist(),
        "force_shape": list(f.shape),
    }
    print("  energy", e.reshape(-1))
    print("  force rms", results["MODEL1896"]["force_rms"])

    if MODEL1352.is_dir() and (MODEL1352 / "cutoff_info").is_file():
        print("FREEZE: MODEL1352 inference")
        e2, f2 = _infer(sys.executable, MODEL1352, pos0, box0, n_frames)
        results["MODEL1352"] = {
            "model": str(MODEL1352),
            "n_frames": n_frames,
            "energy": e2.reshape(-1).tolist(),
            "force_rms": [float(np.sqrt(np.mean(fi ** 2))) for fi in f2],
            "force_frame0_head": f2[0].reshape(-1)[:6].tolist(),
        }
        print("  energy", e2.reshape(-1))
        print("  force rms", results["MODEL1352"]["force_rms"])

    out = HERE / "freeze_inference.json"
    out.write_text(json.dumps(results, indent=2))
    print("FREEZE: wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
