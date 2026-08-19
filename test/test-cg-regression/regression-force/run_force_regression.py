#!/usr/bin/env python3
"""Force finite-difference regression vs analytical STAF-CG inference forces.

    F_num = -(E_f - E_i) / delta

24-bead origami frames: default is all atoms. GPU float and double must not run together.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
INFER = ROOT.parents[1] / "test-cg-inference"

# corr at δ=0.01 should be essentially 1 if analytic forces match energy.
CORR_PASS = 0.99


def _energy_and_force(model, pos, box):
    e, f = model.full_test(pos, box)
    energy = np.asarray(e, dtype=np.float64).reshape(-1)[0]
    force = np.asarray(f, dtype=np.float64).reshape(-1)
    return energy, force


def main() -> int:
    p = argparse.ArgumentParser(description="STAF-CG analytic vs FD forces (one frame).")
    p.add_argument("--precision", choices=["float", "double"], required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--n-atoms", type=int, default=0, help="0 = all beads")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deltas", type=float, nargs="+", default=[0.1, 0.01, 0.001])
    p.add_argument("--model", type=Path, default=None)
    args = p.parse_args()

    sys.path.insert(0, str(REPO / "STAF-CG"))
    sys.path.insert(1, str(REPO / "STAF"))
    import tensorflow as tf
    from staf.dtype import set_precision, np_dtype
    from staf_cg_paths import set_ops_root

    set_precision(args.precision)
    set_ops_root(args.precision)
    dtype = np_dtype()

    if args.model is not None:
        model_dir = args.model
    elif args.precision == "double":
        model_dir = INFER / "model1896_infer"
    else:
        model_dir = INFER / "model_float"

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(exc)

    from staf_cg_models.alpha_nes_model_inference import alpha_nes_full_inference

    frames = INFER / "frames"
    pos_all = np.load(frames / "pos.npy")
    box_all = np.load(frames / "box.npy")
    if pos_all.ndim == 3:
        pos_all = pos_all.reshape(pos_all.shape[0], -1)
    pos0 = np.asarray(pos_all[args.frame : args.frame + 1], dtype=dtype)
    box0 = np.asarray(box_all[args.frame : args.frame + 1], dtype=dtype)
    n_atoms_total = pos0.shape[1] // 3
    n_probe = n_atoms_total if args.n_atoms <= 0 else min(args.n_atoms, n_atoms_total)
    rng = np.random.default_rng(args.seed)
    if n_probe < n_atoms_total:
        atoms = np.sort(rng.choice(n_atoms_total, size=n_probe, replace=False))
    else:
        atoms = np.arange(n_atoms_total)
    comps = []
    for a in atoms:
        for xyz in range(3):
            comps.append((int(a), xyz, int(a) * 3 + xyz))
    comps = np.asarray(comps, dtype=np.int32)

    print(f"Loading model from {model_dir}")
    model = alpha_nes_full_inference(str(model_dir))
    e_i, f_ref = _energy_and_force(model, pos0, box0)
    f_ref = f_ref[comps[:, 2]]

    out_dir = ROOT / f"results_{args.precision}"
    out_dir.mkdir(exist_ok=True)
    summary = {
        "precision": args.precision,
        "frame": args.frame,
        "n_atoms": n_probe,
        "model_dir": str(model_dir),
        "E_i": float(e_i),
        "deltas": [],
    }
    print(f"E_i={e_i:.12e}  n_probe={n_probe}  n_comp={len(comps)}")

    pass_ok = True
    for delta in args.deltas:
        e_f_list = []
        f_num = np.zeros(len(comps), dtype=np.float64)
        e_i_pred = np.zeros(len(comps), dtype=np.float64)
        for k, (atom, xyz, flat) in enumerate(comps):
            pos_f = pos0.copy()
            pos_f[0, flat] = pos_f[0, flat] + dtype(delta)
            e_f, _ = _energy_and_force(model, pos_f, box0)
            e_f_list.append(float(e_f))
            f_num[k] = -(e_f - e_i) / float(delta)
            e_i_pred[k] = e_f + f_ref[k] * float(delta)
        f_ana = f_ref.astype(np.float64)
        corr = float(np.corrcoef(f_ana, f_num)[0, 1]) if f_ana.size > 1 else 1.0
        slope = float(np.linalg.lstsq(f_ana.reshape(-1, 1), f_num, rcond=None)[0][0])
        mae = float(np.mean(np.abs(f_ana - f_num)))
        rmse = float(np.sqrt(np.mean((f_ana - f_num) ** 2)))
        e_consist_mae = float(np.mean(np.abs(e_i - e_i_pred)))
        entry = {
            "delta": float(delta),
            "correlation": corr,
            "slope": slope,
            "mae": mae,
            "rmse": rmse,
            "E_i_consistency_mae": e_consist_mae,
        }
        summary["deltas"].append(entry)
        print(
            f"δ={delta:g}: corr={corr:.6f} slope={slope:.6f} "
            f"MAE={mae:.3e} RMSE={rmse:.3e} "
            f"|E_i-(E_f+Fδ)| MAE={e_consist_mae:.3e}"
        )
        np.savez_compressed(
            out_dir / f"force_fd_delta_{delta:g}.npz",
            precision=args.precision,
            frame=args.frame,
            n_atoms=n_probe,
            model_dir=str(model_dir),
            delta=delta,
            atoms=comps[:, 0],
            xyz=comps[:, 1],
            flat_index=comps[:, 2],
            F_ana=f_ana,
            F_num=f_num,
            E_i=e_i,
            E_f=np.asarray(e_f_list),
            correlation=corr,
            slope=slope,
            mae=mae,
            rmse=rmse,
        )
        if abs(delta - 0.01) < 1e-12 or abs(delta - 0.001) < 1e-12:
            if not (np.isfinite(corr) and corr >= CORR_PASS):
                pass_ok = False

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# STAF-CG force FD",
        f"precision={args.precision}",
        f"frame={args.frame}",
        f"n_atoms={n_probe}",
        f"model_dir={model_dir}",
        f"E_i={e_i:.12e}",
        "method=forward FD  F_num=-(E_f-E_i)/delta",
        "",
    ]
    for e in summary["deltas"]:
        lines.append(
            f"delta={e['delta']:g}  corr={e['correlation']:.8f}  "
            f"slope={e['slope']:.8f}  mae={e['mae']:.6e}  "
            f"rmse={e['rmse']:.6e}  "
            f"E_i_consist_mae={e['E_i_consistency_mae']:.6e}"
        )
    lines.append("")
    lines.append("PASS" if pass_ok else "FAIL")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("PASS" if pass_ok else "FAIL")
    return 0 if pass_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
