#!/usr/bin/env python3
"""STAF-CG energy/force inference CLI.

  python staf_cg_infer.py --model MODEL --pos pos.npy --box box.npy
  python staf_cg_infer.py --model MODEL --frames-dir DIR --frame 0 --json-out out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_STAF_CG_HOME = Path(__file__).resolve().parent
if str(_STAF_CG_HOME) not in sys.path:
    sys.path.insert(0, str(_STAF_CG_HOME))


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    return np.loadtxt(path, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="STAF-CG energy/force inference")
    p.add_argument("--model", required=True, help="Exported model directory")
    p.add_argument("--pos", type=Path, help="Positions file (.npy or text)")
    p.add_argument("--box", type=Path, help="Box file (.npy or text)")
    p.add_argument(
        "--frames-dir",
        type=Path,
        help="Directory with pos.npy / box.npy",
    )
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--n-frames", type=int, default=1, help="How many frames from --frame")
    p.add_argument("--precision", choices=["float", "double"], default="double")
    p.add_argument("--json-out", type=Path, help="Optional JSON metrics path")
    args = p.parse_args(argv)

    _staf = _STAF_CG_HOME.parent / "STAF"
    if _staf.is_dir() and str(_staf) not in sys.path:
        sys.path.insert(1, str(_staf))

    import tensorflow as tf
    from staf.dtype import set_precision, np_dtype
    from staf_cg_paths import set_ops_root

    set_precision(args.precision)
    set_ops_root(args.precision)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(exc)

    from staf_cg_models.alpha_nes_model_inference import alpha_nes_full_inference

    if args.frames_dir is not None:
        pos = np.load(args.frames_dir / "pos.npy")
        box = np.load(args.frames_dir / "box.npy")
    else:
        if args.pos is None or args.box is None:
            p.error("provide --pos and --box, or --frames-dir")
        pos = _load_array(args.pos)
        box = _load_array(args.box)

    pos = np.asarray(pos, dtype=np_dtype())
    box = np.asarray(box, dtype=np_dtype())
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    elif pos.ndim == 3:
        pos = pos.reshape(pos.shape[0], -1)
    if box.ndim == 1:
        box = box.reshape(1, -1)
    i0 = args.frame
    i1 = min(i0 + args.n_frames, pos.shape[0])
    pos = pos[i0:i1]
    box = box[i0:i1]

    model = alpha_nes_full_inference(args.model)
    energy, force = model.full_test(pos, box)
    energy_np = np.asarray(energy)
    force_np = np.asarray(force)
    print("STAF-CG: energy", energy_np)
    print("STAF-CG: force shape", force_np.shape)
    print("STAF-CG: force[0] rms", float(np.sqrt(np.mean(force_np[0] ** 2))))
    if args.json_out is not None:
        payload = {
            "energy": energy_np.reshape(-1).tolist(),
            "force": force_np.reshape(force_np.shape[0], -1).tolist(),
            "frame0": i0,
            "n_frames": int(pos.shape[0]),
            "model": str(args.model),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print("STAF-CG: wrote", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
