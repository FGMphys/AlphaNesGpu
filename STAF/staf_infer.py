#!/usr/bin/env python3
"""STAF inference CLI.

Examples:
  python staf_infer.py --model model_double --precision double --pos pos.npy --box box.npy
  python staf_infer.py --model model_float --precision float --frames-dir ../test/.../frames
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_STAF_HOME = Path(__file__).resolve().parent
if str(_STAF_HOME) not in sys.path:
    sys.path.insert(0, str(_STAF_HOME))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="STAF energy/force inference")
    p.add_argument("--model", required=True, help="Exported model directory")
    p.add_argument("--precision", choices=["float", "double"], required=True)
    p.add_argument("--pos", type=Path, help="Positions file (.npy or text)")
    p.add_argument("--box", type=Path, help="Box file (.npy or text)")
    p.add_argument(
        "--frames-dir",
        type=Path,
        help="Directory with pos.npy / box.npy (uses frame index)",
    )
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--json-out", type=Path, help="Optional JSON metrics path")
    args = p.parse_args(argv)

    import tensorflow as tf
    from staf.dtype import set_precision, np_dtype

    set_precision(args.precision)
    from staf_models.staf_model_inference_full import staf_full_inference

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(exc)

    dtype = np_dtype()
    if args.frames_dir is not None:
        pos = np.load(args.frames_dir / "pos.npy").astype(dtype)[args.frame]
        box = np.load(args.frames_dir / "box.npy").astype(dtype)[args.frame]
    else:
        if args.pos is None or args.box is None:
            p.error("provide --pos/--box or --frames-dir")
        if args.pos.suffix == ".npy":
            pos = np.load(args.pos).astype(dtype)
        else:
            pos = np.loadtxt(args.pos, dtype=dtype)
        if args.box.suffix == ".npy":
            box = np.load(args.box).astype(dtype)
        else:
            box = np.loadtxt(args.box, dtype=dtype)

    pos = np.asarray(pos, dtype=dtype).reshape(1, -1)
    box = np.asarray(box, dtype=dtype).reshape(1, -1)

    model = staf_full_inference(str(Path(args.model).resolve()))
    energy, force = model.full_test(pos, box)[:2]
    e = float(np.asarray(energy).reshape(-1)[0])
    f = np.asarray(force)
    print(f"STAF: energy={e:.10f}")
    print(f"STAF: force shape={f.shape} |F|_rms={float(np.sqrt(np.mean(f * f))):.6e}")

    if args.json_out is not None:
        payload = {
            "precision": args.precision,
            "model": str(Path(args.model).resolve()),
            "energy": e,
            "force_rms": float(np.sqrt(np.mean(f * f))),
            "n_atoms": int(f.size // 3),
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"STAF: wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
