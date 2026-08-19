#!/usr/bin/env python3
"""Copy the first N USCGSITE training frames into test/test-cg-inference/frames/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "STAF-CG"))
from staf_cg_harness import USCGSITE  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-frames", type=int, default=3)
    args = p.parse_args()
    src = USCGSITE / "dataset" / "training"
    dest = ROOT / "frames"
    dest.mkdir(parents=True, exist_ok=True)
    n = args.n_frames
    pos = np.load(src / "pos.npy", mmap_mode="r")[:n]
    box = np.load(src / "box.npy", mmap_mode="r")[:n]
    energy = np.load(src / "energy.npy", mmap_mode="r")[:n]
    force = np.load(src / "force.npy", mmap_mode="r")[:n]
    np.save(dest / "pos.npy", np.asarray(pos))
    np.save(dest / "box.npy", np.asarray(box))
    np.save(dest / "energy.npy", np.asarray(energy))
    np.save(dest / "force.npy", np.asarray(force))
    np.save(dest / "frame_indices.npy", np.arange(n, dtype=np.int32))
    print(f"STAF-CG: wrote {n} frames to {dest} pos{pos.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
