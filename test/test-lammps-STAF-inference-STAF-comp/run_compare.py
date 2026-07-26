#!/usr/bin/env python3
"""Compare STAF Python inference vs LAMMPS pair_staf on the same frames.

Uses the same trained weights:
  - Python: TF SavedModel export of model_log0 (float)
  - LAMMPS: ORT grad ONNX export of the same model_log0 (model_onnx_grad_float)

Writes under this directory:
  results/summary.json, results/frames.csv
  results/frame_XX/{pos, energy, forces, ...}
  plots/*.png

Usage (repo root, GPU env sourced for LAMMPS):
  source scripts/staf_gpu_env.sh
  python test/test-lammps-STAF-inference-STAF-comp/run_compare.py --n-frames 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STAF = ROOT / "STAF"
FRAMES = ROOT / "test/test-inference-pipeline/frames"
LMP_MODEL = ROOT / "test/test-lammps-smoke/model_onnx_grad_float"
TF_MODEL = HERE / "model_tf_float_log0"
LMP_BIN = Path(
    os.environ.get("LMP", "/home/francegm/programmi/lammps-23Jun2022/src/lmp_staf")
)


def _energy_and_force_py(model, pos, box):
    out = model.full_test(pos, box)
    energy = float(np.asarray(out[0]).reshape(-1)[0])
    force_list = out[1]
    parts = []
    for k in range(len(force_list)):
        fk = force_list[k][0]
        if hasattr(fk, "numpy"):
            fk = fk.numpy()
        parts.append(np.asarray(fk, dtype=np.float64))
    force = sum(parts).reshape(-1)
    return energy, force


def write_lammps_data(path: Path, pos_flat: np.ndarray, box6: np.ndarray, n_type0: int) -> None:
    """Write LAMMPS atomic data; atoms ordered type1 then type2 (0-based types in pos)."""
    xyz = np.asarray(pos_flat, dtype=np.float64).reshape(-1, 3).copy()
    n = xyz.shape[0]
    lx, xy, xz, ly, yz, lz = [float(x) for x in box6.reshape(-1)[:6]]
    # Wrap into [0, L) so data matches LAMMPS boxlo=0 (PBC-equivalent).
    xyz[:, 0] = np.mod(xyz[:, 0], lx)
    xyz[:, 1] = np.mod(xyz[:, 1], ly)
    xyz[:, 2] = np.mod(xyz[:, 2], lz)
    xlo, xhi = 0.0, lx
    ylo, yhi = 0.0, ly
    zlo, zhi = 0.0, lz
    lines = [
        "STAF compare frame\n",
        "\n",
        f"{n} atoms\n",
        "2 atom types\n",
        "\n",
        f"{xlo:.16g} {xhi:.16g} xlo xhi\n",
        f"{ylo:.16g} {yhi:.16g} ylo yhi\n",
        f"{zlo:.16g} {zhi:.16g} zlo zhi\n",
        "\n",
        "Masses\n",
        "\n",
        "1 15.999\n",
        "2 1.008\n",
        "\n",
        "Atoms # atomic\n",
        "\n",
    ]
    # type.dat: n_type0 of species 0 → LAMMPS type 1, rest type 2
    for i in range(n):
        t = 1 if i < n_type0 else 2
        x, y, z = xyz[i]
        lines.append(f"{i+1} {t} {x:.16g} {y:.16g} {z:.16g}\n")
    path.write_text("".join(lines))


def run_lammps(work: Path, data_file: Path, model_dir: Path) -> tuple[float, np.ndarray]:
    """run 0; return (pe, forces[3N] ordered by atom id)."""
    infile = work / "in.compare"
    dump = work / "forces.dump"
    log = work / "log.lammps"
    infile.write_text(
        f"""units           real
atom_style      atomic
boundary        p p p
atom_modify     sort 0 0.0
read_data       {data_file.name}
pair_style      staf 4.5 4.5 float
pair_coeff      * * {model_dir}
neighbor        0.5 bin
neigh_modify    delay 0 every 1 check yes
comm_modify     cutoff 5.0
thermo_style    custom step pe
thermo          1
dump            1 all custom 1 {dump.name} id fx fy fz
dump_modify     1 sort id
run             0
"""
    )
    env = os.environ.copy()
    # Ensure GPU discovery if user sourced staf_gpu_env.sh
    cmd = [str(LMP_BIN), "-in", str(infile.name), "-log", str(log.name)]
    proc = subprocess.run(
        cmd,
        cwd=str(work),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    (work / "lammps.stdout").write_text(proc.stdout + "\n---stderr---\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed rc={proc.returncode}; see {work/'lammps.stdout'}"
        )

    # Parse PE from log (thermo line after "Step ... PotEng" or pe column)
    pe = None
    log_txt = log.read_text() if log.exists() else proc.stdout
    # Prefer last thermo numeric line with two columns: step pe
    for line in log_txt.splitlines():
        m = re.match(r"^\s*(\d+)\s+([-+0-9.eE]+)\s*$", line)
        if m and int(m.group(1)) == 0:
            pe = float(m.group(2))
    if pe is None:
        # fallback: "PotEng" header style from custom thermo in older logs
        raise RuntimeError(f"could not parse PE from {log}")

    # Parse dump
    text = dump.read_text().splitlines()
    # find ITEM: ATOMS
    i = 0
    while i < len(text) and not text[i].startswith("ITEM: ATOMS"):
        i += 1
    if i >= len(text):
        raise RuntimeError(f"bad dump {dump}")
    hdr = text[i].split()[2:]  # id fx fy fz
    i += 1
    rows = []
    while i < len(text) and not text[i].startswith("ITEM:"):
        parts = text[i].split()
        if len(parts) >= 4:
            rows.append([float(x) for x in parts])
        i += 1
    arr = np.asarray(rows, dtype=np.float64)
    # sort by id
    order = np.argsort(arr[:, 0].astype(int))
    arr = arr[order]
    forces = arr[:, 1:4].reshape(-1)
    return pe, forces


def make_plots(results: list[dict], plot_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    frames = [r["frame"] for r in results]
    de = [r["delta_E"] for r in results]
    e_py = [r["E_python"] for r in results]
    e_lp = [r["E_lammps"] for r in results]
    max_df = [r["max_abs_dF"] for r in results]
    rms_df = [r["rms_dF"] for r in results]
    rel_f = [r["rel_rms_dF"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frames, e_py, "o-", label="Python STAF")
    ax.plot(frames, e_lp, "s--", label="LAMMPS STAF")
    ax.set_xlabel("frame")
    ax.set_ylabel("energy")
    ax.legend()
    ax.set_title("Energy: Python vs LAMMPS")
    fig.tight_layout()
    fig.savefig(plot_dir / "energy_vs_frame.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(frames, np.maximum(np.abs(de), 1e-16), "o-")
    ax.set_xlabel("frame")
    ax.set_ylabel("|ΔE|")
    ax.set_title("Absolute energy difference")
    fig.tight_layout()
    fig.savefig(plot_dir / "delta_energy.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(frames, np.maximum(max_df, 1e-16), "o-", label="max|ΔF|")
    ax.semilogy(frames, np.maximum(rms_df, 1e-16), "s--", label="rms ΔF")
    ax.set_xlabel("frame")
    ax.set_ylabel("force error")
    ax.legend()
    ax.set_title("Force difference Python − LAMMPS")
    fig.tight_layout()
    fig.savefig(plot_dir / "delta_force.png", dpi=140)
    plt.close(fig)

    # Scatter F components for first frame
    r0 = results[0]
    fpy = np.load(HERE / "results" / f"frame_{r0['frame']:02d}" / "force_python.npy")
    flp = np.load(HERE / "results" / f"frame_{r0['frame']:02d}" / "force_lammps.npy")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(flp, fpy, ".", ms=2, alpha=0.5)
    lim = float(max(np.max(np.abs(flp)), np.max(np.abs(fpy)), 1e-8))
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8)
    ax.set_xlabel("F LAMMPS")
    ax.set_ylabel("F Python")
    ax.set_title(f"Force parity frame {r0['frame']}")
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(plot_dir / "force_scatter_frame0.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(frames, np.maximum(rel_f, 1e-16), "o-")
    ax.set_xlabel("frame")
    ax.set_ylabel("rms(ΔF) / rms(F_py)")
    ax.set_title("Relative force RMS error")
    fig.tight_layout()
    fig.savefig(plot_dir / "rel_force_rms.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--frames-dir", type=Path, default=FRAMES)
    ap.add_argument("--tf-model", type=Path, default=TF_MODEL)
    ap.add_argument("--lammps-model", type=Path, default=LMP_MODEL)
    ap.add_argument("--lmp", type=Path, default=LMP_BIN)
    args = ap.parse_args()

    if not args.tf_model.is_dir():
        print(f"missing TF model {args.tf_model}; export with save_model_in_float.py", file=sys.stderr)
        return 2
    if not args.lammps_model.is_dir():
        print(f"missing LAMMPS model {args.lammps_model}", file=sys.stderr)
        return 2
    if not args.lmp.is_file():
        print(f"missing lmp binary {args.lmp}", file=sys.stderr)
        return 2

    pos_all = np.load(args.frames_dir / "pos.npy")
    box_all = np.load(args.frames_dir / "box.npy")
    n_type0 = int(np.loadtxt(args.frames_dir / "type.dat").reshape(-1)[0])
    n_frames = min(args.n_frames, pos_all.shape[0])

    # Python STAF (GPU ok)
    sys.path.insert(0, str(STAF))
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    from staf.dtype import set_precision

    set_precision("float")
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    from staf_models.staf_model_inference_full import staf_full_inference

    print(f"Loading Python model {args.tf_model}")
    py_model = staf_full_inference(str(args.tf_model.resolve()))

    results_dir = HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    work_root = HERE / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for fi in range(n_frames):
        print(f"\n=== frame {fi} ===")
        pos = pos_all[fi].astype(np.float32).reshape(1, -1)
        box = box_all[fi].astype(np.float32).reshape(1, -1)

        e_py, f_py = _energy_and_force_py(py_model, pos, box)
        print(f"  Python E={e_py:.8f}  |F|_rms={np.sqrt(np.mean(f_py*f_py)):.6g}")

        fdir = results_dir / f"frame_{fi:02d}"
        fdir.mkdir(parents=True, exist_ok=True)
        np.save(fdir / "pos.npy", pos.reshape(-1))
        np.save(fdir / "box.npy", box.reshape(-1))
        np.save(fdir / "force_python.npy", f_py)
        (fdir / "energy_python.txt").write_text(f"{e_py:.16g}\n")

        wdir = work_root / f"frame_{fi:02d}"
        wdir.mkdir(parents=True, exist_ok=True)
        data = wdir / "data.frame"
        write_lammps_data(data, pos.reshape(-1), box.reshape(-1), n_type0)
        e_lp, f_lp = run_lammps(wdir, data, args.lammps_model.resolve())
        print(f"  LAMMPS E={e_lp:.8f}  |F|_rms={np.sqrt(np.mean(f_lp*f_lp)):.6g}")

        np.save(fdir / "force_lammps.npy", f_lp)
        (fdir / "energy_lammps.txt").write_text(f"{e_lp:.16g}\n")

        dF = f_py - f_lp
        rms_f = float(np.sqrt(np.mean(f_py * f_py)))
        row = {
            "frame": fi,
            "E_python": e_py,
            "E_lammps": e_lp,
            "delta_E": e_py - e_lp,
            "max_abs_dF": float(np.max(np.abs(dF))),
            "rms_dF": float(np.sqrt(np.mean(dF * dF))),
            "rel_rms_dF": float(np.sqrt(np.mean(dF * dF)) / (rms_f + 1e-30)),
            "rms_F_python": rms_f,
        }
        rows.append(row)
        print(
            f"  ΔE={row['delta_E']:.6g}  max|ΔF|={row['max_abs_dF']:.6g}  "
            f"rel_rms_dF={row['rel_rms_dF']:.6g}"
        )

    # CSV + JSON
    with (results_dir / "frames.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "n_frames": n_frames,
        "tf_model": str(args.tf_model),
        "lammps_model": str(args.lammps_model),
        "frames_dir": str(args.frames_dir),
        "max_abs_delta_E": max(abs(r["delta_E"]) for r in rows),
        "max_of_max_abs_dF": max(r["max_abs_dF"] for r in rows),
        "max_rel_rms_dF": max(r["rel_rms_dF"] for r in rows),
        "frames": rows,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_dir = HERE / "plots"
    make_plots(rows, plot_dir)
    print(f"\nWrote {results_dir/'summary.json'} and plots in {plot_dir}")
    print(
        f"max|ΔE|={summary['max_abs_delta_E']:.6g}  "
        f"max max|ΔF|={summary['max_of_max_abs_dF']:.6g}  "
        f"max rel_rms_dF={summary['max_rel_rms_dF']:.6g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
