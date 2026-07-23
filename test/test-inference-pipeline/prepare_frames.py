#!/usr/bin/env python3
"""Extract N frames from the training dataset for inference comparison."""
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
# Prefer double dataset (float64); float run uses the same geometry.
DATA = ROOT.parent / "test-training-pipeline" / "run_double" / "dataset"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=10, help="number of frames")
    p.add_argument("-seed", type=int, default=0)
    p.add_argument("--split", default="test", choices=["training", "test"])
    args = p.parse_args()

    out = ROOT / "frames"
    out.mkdir(exist_ok=True)

    pos = np.load(DATA / args.split / "pos.npy")
    box = np.load(DATA / args.split / "box.npy")
    energy = np.load(DATA / args.split / "energy.npy")
    force = np.load(DATA / args.split / "force.npy")

    rng = np.random.default_rng(args.seed)
    n = min(args.n, pos.shape[0])
    idx = np.sort(rng.choice(pos.shape[0], size=n, replace=False))

    np.save(out / "frame_indices.npy", idx)
    np.save(out / "pos.npy", pos[idx].astype(np.float64))
    np.save(out / "box.npy", box[idx].astype(np.float64))
    np.save(out / "energy_ref.npy", energy[idx].astype(np.float64))
    np.save(out / "force_ref.npy", force[idx].astype(np.float64))
    # also plain text for the first frame (simple_inference style)
    np.savetxt(out / "pos_0", pos[idx[0]].reshape(-1))
    np.savetxt(out / "box_0", box[idx[0]].reshape(-1))

    meta = out / "frames_info.txt"
    meta.write_text(
        f"split={args.split}\n"
        f"n_frames={n}\n"
        f"seed={args.seed}\n"
        f"indices={idx.tolist()}\n"
        f"source={DATA / args.split}\n"
        f"natoms={pos.shape[1] // 3}\n"
    )
    print(f"Wrote {n} frames to {out} (indices {idx.tolist()})")


if __name__ == "__main__":
    main()
