#!/usr/bin/env python3
"""Run STAF-CG inference on prepared frames (float or double, one at a time).

Saves energy and force to inference_{precision}/inference_bundle.npz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["float", "double"], required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO / "STAF-CG"))
    sys.path.insert(1, str(REPO / "STAF"))
    from staf.dtype import set_precision, np_dtype
    from staf_cg_paths import set_ops_root
    from staf_cg_harness import energy_force

    set_precision(args.precision)
    set_ops_root(args.precision)

    dtype = np_dtype()
    model_dir = Path(args.model) if args.model else ROOT / f"model_{args.precision}"
    out_dir = ROOT / f"inference_{args.precision}"

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(exc)

    from staf_cg_models.alpha_nes_model_inference import alpha_nes_full_inference

    frames = ROOT / "frames"
    pos_all = np.load(frames / "pos.npy").astype(dtype)
    box_all = np.load(frames / "box.npy").astype(dtype)
    idx = np.load(frames / "frame_indices.npy")
    if pos_all.ndim == 3:
        pos_all = pos_all.reshape(pos_all.shape[0], -1)

    print(f"Loading model from {model_dir}")
    model = alpha_nes_full_inference(str(model_dir))
    out_dir.mkdir(exist_ok=True)

    energies = []
    forces = []
    for i in range(pos_all.shape[0]):
        e, f = energy_force(model, pos_all[i : i + 1], box_all[i : i + 1])
        energies.append(np.asarray(e, dtype=np.float64).reshape(-1)[0])
        forces.append(np.asarray(f, dtype=np.float64).reshape(-1))
        print(f"frame {int(idx[i])}: E={energies[-1]:.12f}  Frms={np.sqrt(np.mean(forces[-1]**2)):.6e}")

    np.savez_compressed(
        out_dir / "inference_bundle.npz",
        energy=np.asarray(energies),
        force=np.stack(forces, axis=0),
        frame_indices=idx,
        precision=args.precision,
        model=str(model_dir),
    )
    print("wrote", out_dir / "inference_bundle.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
