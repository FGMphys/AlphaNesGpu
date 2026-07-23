#!/usr/bin/env python3
"""Run STAF full inference on prepared frames (float or double, one at a time).

Saves a single inference_bundle.npz with all tensors needed for float/double
compatibility checks (no human-readable side dumps).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def _as_numpy(obj):
    if isinstance(obj, (list, tuple)):
        return [_as_numpy(x) for x in obj]
    if hasattr(obj, "numpy"):
        return np.asarray(obj.numpy())
    return np.asarray(obj)


def _object_array(seq):
    arr = np.empty(len(seq), dtype=object)
    for i, v in enumerate(seq):
        arr[i] = v
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["float", "double"], required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.precision == "float":
        code_root = REPO / "AlphaNesGpu_float"
        dtype = np.float32
        model_dir = Path(args.model) if args.model else ROOT / "model_float"
        out_dir = ROOT / "inference_float"
    else:
        code_root = REPO / "AlphaNesGpu_double"
        dtype = np.float64
        model_dir = Path(args.model) if args.model else ROOT / "model_double"
        out_dir = ROOT / "inference_double"

    sys.path.insert(0, str(code_root))
    from alphanes_models.mixture.alpha_nes_model_inference_full import (
        alpha_nes_full_inference,
    )

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(
                len(gpus),
                "Physical GPUs,",
                len(tf.config.list_logical_devices("GPU")),
                "Logical GPUs",
            )
        except RuntimeError as e:
            print(e)

    frames = ROOT / "frames"
    pos_all = np.load(frames / "pos.npy").astype(dtype)
    box_all = np.load(frames / "box.npy").astype(dtype)
    idx = np.load(frames / "frame_indices.npy")
    n_frames = pos_all.shape[0]

    print(f"Loading model from {model_dir}")
    model = alpha_nes_full_inference(str(model_dir))
    out_dir.mkdir(exist_ok=True)

    store = {
        "energy": [],
        "force": [],
        "force_radial": [],
        "force_angular": [],
        "fingerprint": [],
        "grad_listed": [],
        "x2b": [],
        "x3b": [],
        "x3bsupp": [],
        "int2b": [],
        "int3b": [],
        "intder2b": [],
        "intder3b": [],
        "intder3bsupp": [],
    }
    info_str = ""

    for i in range(n_frames):
        pos = pos_all[i : i + 1]
        box = box_all[i : i + 1]
        print(f"[{args.precision}] frame {i + 1}/{n_frames} (dataset index {int(idx[i])})")
        if i == 0:
            _ = model.full_test(pos, box)
        out = model.full_test(pos, box)
        (
            totenergy,
            force_list,
            grad_listed,
            fingerprint,
            x2b,
            x3b,
            x3bsupp,
            int2b,
            int3b,
            intder2b,
            intder3b,
            intder3bsupp,
            info_str,
        ) = out

        store["energy"].append(_as_numpy(totenergy))
        store["force"].append(
            sum(_as_numpy(force_list[k][0]) for k in range(len(force_list)))
        )
        store["force_radial"].append(
            sum(_as_numpy(force_list[k][1]) for k in range(len(force_list)))
        )
        store["force_angular"].append(
            sum(_as_numpy(force_list[k][2]) for k in range(len(force_list)))
        )
        store["fingerprint"].append(_as_numpy(fingerprint))
        store["grad_listed"].append(_as_numpy(grad_listed))
        store["x2b"].append(_as_numpy(x2b))
        store["x3b"].append(_as_numpy(x3b))
        store["x3bsupp"].append(_as_numpy(x3bsupp))
        store["int2b"].append(_as_numpy(int2b))
        store["int3b"].append(_as_numpy(int3b))
        store["intder2b"].append(_as_numpy(intder2b))
        store["intder3b"].append(_as_numpy(intder3b))
        store["intder3bsupp"].append(_as_numpy(intder3bsupp))

    dense = {"energy", "force", "force_radial", "force_angular"}
    payload = {
        "precision": args.precision,
        "frame_indices": idx,
        "info": np.asarray(info_str),
    }
    for key, vals in store.items():
        payload[key] = np.asarray(vals) if key in dense else _object_array(vals)

    out_path = out_dir / "inference_bundle.npz"
    np.savez_compressed(out_path, **payload)
    print(f"Wrote {out_path} keys={sorted(payload)}")


if __name__ == "__main__":
    main()
