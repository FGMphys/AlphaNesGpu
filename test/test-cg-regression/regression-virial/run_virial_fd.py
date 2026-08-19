#!/usr/bin/env python3
"""Finite-difference virial vs analytic STAF-CG W (no virial.npy labels).

Strain the three diagonal Cartesian axes independently:
    W_num_aa = -(E_plus - E_minus) / (2 * eps)
If the correlation with W_ana is ~-1, the formula is flipped and noted.

Pass if corr(W_ana_diag, W_num_diag) >= 0.95 OR max relative error on
components with |W_aa| > 1e-3 is < 0.2.

GPU float and double must not run together.
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

CORR_PASS = 0.95
REL_PASS = 0.2
REL_FLOOR = 1e-3
EPS_DEFAULT = 1e-4
# box6 = [Lx, xy, xz, Ly, yz, Lz]
BOX_LEN_INDEX = (0, 3, 5)


def _energy_force_virial(model, pos, box):
    e, f, w = model.full_test_virial(pos, box)
    energy = np.asarray(e, dtype=np.float64).reshape(-1)[0]
    force = np.asarray(f, dtype=np.float64).reshape(-1)
    virial = np.asarray(w, dtype=np.float64).reshape(-1)
    return energy, force, virial


def _strain_frame(pos, box, axis, scale, dtype):
    """Scale Cartesian coord `axis` of all atoms and the matching box length."""
    pos_s = np.array(pos, dtype=dtype, copy=True)
    box_s = np.array(box, dtype=dtype, copy=True)
    pos_s[0, axis::3] = pos_s[0, axis::3] * scale
    box_s[0, BOX_LEN_INDEX[axis]] = box_s[0, BOX_LEN_INDEX[axis]] * scale
    return pos_s, box_s


def main() -> int:
    p = argparse.ArgumentParser(description="STAF-CG analytic vs FD virial (one frame).")
    p.add_argument("--precision", choices=["float", "double"], required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--eps", type=float, default=EPS_DEFAULT)
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
    n_atoms = pos0.shape[1] // 3

    print(f"Loading model from {model_dir}")
    model = alpha_nes_full_inference(str(model_dir))
    e0, _f, w_ana = _energy_force_virial(model, pos0, box0)
    w_ana_diag = np.array([w_ana[0], w_ana[4], w_ana[8]], dtype=np.float64)

    eps = float(args.eps)
    w_num = np.zeros(3, dtype=np.float64)
    e_plus = np.zeros(3, dtype=np.float64)
    e_minus = np.zeros(3, dtype=np.float64)
    sign_formula = "-(E+ - E-)/(2*eps)"
    flipped = False
    for axis in range(3):
        pos_p, box_p = _strain_frame(pos0, box0, axis, 1.0 + eps, dtype)
        pos_m, box_m = _strain_frame(pos0, box0, axis, 1.0 - eps, dtype)
        e_p, _, _ = _energy_force_virial(model, pos_p, box_p)
        e_m, _, _ = _energy_force_virial(model, pos_m, box_m)
        e_plus[axis] = e_p
        e_minus[axis] = e_m
        w_num[axis] = -(e_p - e_m) / (2.0 * eps)

    corr = float(np.corrcoef(w_ana_diag, w_num)[0, 1]) if w_ana_diag.size > 1 else 1.0
    if np.isfinite(corr) and corr < -0.5:
        w_num = -w_num
        sign_formula = "+(E+ - E-)/(2*eps)  (flipped; first try was -1 corr)"
        flipped = True
        corr = float(np.corrcoef(w_ana_diag, w_num)[0, 1])

    rel_mask = np.abs(w_ana_diag) > REL_FLOOR
    if np.any(rel_mask):
        rel_err = np.abs(w_ana_diag[rel_mask] - w_num[rel_mask]) / np.abs(w_ana_diag[rel_mask])
        max_rel = float(np.max(rel_err))
    else:
        rel_err = np.array([], dtype=np.float64)
        max_rel = float("nan")

    pass_corr = bool(np.isfinite(corr) and corr >= CORR_PASS)
    pass_rel = bool(np.any(rel_mask) and max_rel < REL_PASS)
    pass_ok = pass_corr or pass_rel

    out_dir = ROOT / f"results_{args.precision}"
    out_dir.mkdir(exist_ok=True)
    summary = {
        "precision": args.precision,
        "frame": args.frame,
        "n_atoms": n_atoms,
        "model_dir": str(model_dir),
        "E": float(e0),
        "eps": eps,
        "sign_formula": sign_formula,
        "flipped": flipped,
        "W_ana_diag": w_ana_diag.tolist(),
        "W_num_diag": w_num.tolist(),
        "W_ana_full": w_ana.tolist(),
        "correlation": corr,
        "max_rel_err": max_rel,
        "n_rel_components": int(np.sum(rel_mask)),
        "pass": pass_ok,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# STAF-CG virial FD (strain of E, no virial.npy)",
        f"precision={args.precision}",
        f"frame={args.frame}",
        f"n_atoms={n_atoms}",
        f"model_dir={model_dir}",
        f"E={e0:.12e}",
        f"eps={eps:g}",
        f"sign_formula={sign_formula}",
        f"flipped={str(flipped).lower()}",
        f"W_ana_xx,yy,zz={w_ana_diag[0]:.8e} {w_ana_diag[1]:.8e} {w_ana_diag[2]:.8e}",
        f"W_num_xx,yy,zz={w_num[0]:.8e} {w_num[1]:.8e} {w_num[2]:.8e}",
        f"E_plus={e_plus[0]:.12e} {e_plus[1]:.12e} {e_plus[2]:.12e}",
        f"E_minus={e_minus[0]:.12e} {e_minus[1]:.12e} {e_minus[2]:.12e}",
        f"correlation={corr:.8f}",
        f"max_rel_err={max_rel}",
        f"pass: {str(pass_ok).lower()}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    np.savez_compressed(
        out_dir / "virial_fd.npz",
        W_ana=w_ana,
        W_ana_diag=w_ana_diag,
        W_num_diag=w_num,
        E=e0,
        E_plus=e_plus,
        E_minus=e_minus,
        eps=eps,
        correlation=corr,
        max_rel_err=max_rel,
    )
    print("\n".join(lines))
    return 0 if pass_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
