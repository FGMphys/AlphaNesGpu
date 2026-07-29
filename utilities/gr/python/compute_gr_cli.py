#!/usr/bin/env python3
"""CLI: fast g(r) OO/OH/HH for STAF lammpstrj vs MB-pol DeepMD set."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # AlphaNesGpu
sys.path.insert(0, str(Path(__file__).resolve().parent))
from staf_gr import compute_gr_frames, read_lammpstrj, read_mbpol_set  # noqa: E402

MBPOL = Path("/home/francegm/MBPOL_PROJECT/MBPOL_dataset/TrainingSet/training")
PAIR_MAP = {"OO": (0, 0), "OH": (0, 1), "HH": (1, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--T", type=int, default=223)
    ap.add_argument("--rmax", type=float, default=10.0)
    ap.add_argument("--dr", type=float, default=0.05)
    ap.add_argument("--pairs", default="OO,OH,HH")
    ap.add_argument("--max-frames-staf", type=int, default=None)
    ap.add_argument("--max-frames-mbpol", type=int, default=200)
    ap.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Keep STAF frames with TIMESTEP >= this (e.g. 10001 for NPT-only after NVT 10k)",
    )
    ap.add_argument("--max-step", type=int, default=None, help="Keep TIMESTEP <= this")
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    args = ap.parse_args()
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)

    staf_frames = list(
        read_lammpstrj(
            args.traj,
            args.max_frames_staf,
            min_step=args.min_step,
            max_step=args.max_step,
        )
    )
    if not staf_frames:
        raise SystemExit("no STAF frames after step filter")
    mb_frames = list(
        read_mbpol_set(MBPOL / f"00_dlpoly_{args.T}K" / "set.000", args.max_frames_mbpol)
    )
    step_note = ""
    if args.min_step is not None or args.max_step is not None:
        step_note = f" steps=[{args.min_step},{args.max_step}]"
    print(
        f"STAF frames={len(staf_frames)}{step_note} n={len(staf_frames[0][0])} | "
        f"MB-pol frames={len(mb_frames)} n={len(mb_frames[0][0])}"
    )

    results = {}
    for key in pairs:
        ta, tb = PAIR_MAP[key]
        t0 = time.time()
        r_s, g_s = compute_gr_frames(staf_frames, ta, tb, args.dr, args.rmax)
        r_m, g_m = compute_gr_frames(mb_frames, ta, tb, args.dr, args.rmax)
        print(f"  {key}: {time.time()-t0:.2f}s")
        np.savetxt(
            args.outdir / f"gr_{key}.dat",
            np.column_stack([r_s, g_s, g_m]),
            header="r_Ang  g_STAF  g_MBpol",
        )
        results[key] = (r_s, g_s, g_m)

    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.6), squeeze=False)
    titles = {"OO": r"$g_{OO}$", "OH": r"$g_{OH}$", "HH": r"$g_{HH}$"}
    for ax, key in zip(axes[0], pairs):
        r, g_s, g_m = results[key]
        ax.plot(r, g_m, label="MB-pol", color="#1f77b4", lw=1.8)
        ax.plot(r, g_s, label="STAF", color="#d62728", lw=1.5)
        ax.set_xlabel(r"$r$ (Å)")
        ax.set_ylabel(titles[key])
        ax.set_xlim(0, args.rmax)
        ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
        ax.legend(frameon=False, fontsize=9)
    title = f"T={args.T} K  STAF({len(staf_frames)} fr"
    if args.min_step is not None or args.max_step is not None:
        title += f", step≥{args.min_step}" if args.max_step is None else f", {args.min_step}–{args.max_step}"
    title += f") vs MB-pol({len(mb_frames)} fr)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    png = args.outdir / "gr_compare.png"
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")

    # OO peak summary
    if "OO" in results:
        r, g_s, g_m = results["OO"]
        m = (r > 2) & (r < 4)
        is_, im = int(np.argmax(g_s[m])), int(np.argmax(g_m[m]))
        (args.outdir / "summary.txt").write_text(
            f"T={args.T}\nstaf_frames={len(staf_frames)}\nmbpol_frames={len(mb_frames)}\n"
            f"OO_peak_STAF: r={r[m][is_]:.3f} g={g_s[m][is_]:.3f}\n"
            f"OO_peak_MBpol: r={r[m][im]:.3f} g={g_m[m][im]:.3f}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
