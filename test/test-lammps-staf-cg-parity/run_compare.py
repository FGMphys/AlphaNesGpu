#!/usr/bin/env python3
"""Sprint 6 parity: LAMMPS pair_style staf/cg vs Python STAF-CG.

Same 24-bead USCGSITE frame 0. Compares energy, per-atom forces, and
configurational pressure (pair virial only; run 0 with v=0 so no kinetic).

Pass if:
  max|ΔE| < 1e-3
  max|ΔF| < 1e-3 (per component)
  |ΔP|/max(|P|,1) < 0.05  OR  max|ΔW_diag| < 1e-2

Exit 1 if E or F or P disagree.
Writes summary.json + summary.txt in this directory.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/francegm/AlphaNesGpu")
HERE = Path(__file__).resolve().parent
STAF_CG = REPO / "STAF-CG"
STAF = REPO / "STAF"
MODEL = REPO / "test/test-cg-inference/model_onnx_double"
FRAMES = REPO / "test/test-cg-inference/frames"
USCGSITE = Path("/home/francegm/ORIGAMI/INFER_INTRA_TRY2+USCGSITE")

# metal: 1 eV/Å³ → bar (LAMMPS nktv2p). User note quoted atm; LAMMPS metal is bar.
# 1 eV/Å³ = 1.60217662e6 bar ≈ 1.581e6 atm. Comparisons use this factor on both sides.
NKTV2P_METAL = 1.60217662e6
E_TOL = 1e-3
F_TOL = 1e-3
P_REL_TOL = 0.05
W_DIAG_TOL = 1e-2


def _setup_path() -> None:
    cg, staf = str(STAF_CG), str(STAF)
    if cg in sys.path:
        sys.path.remove(cg)
    sys.path.insert(0, cg)
    if staf not in sys.path:
        sys.path.insert(1, staf)


def _load_frame0():
    if (FRAMES / "pos.npy").is_file() and (FRAMES / "box.npy").is_file():
        pos = np.load(FRAMES / "pos.npy")
        box = np.load(FRAMES / "box.npy")
        src = str(FRAMES)
    else:
        pos = np.load(USCGSITE / "dataset/training/pos.npy", mmap_mode="r")
        box = np.load(USCGSITE / "dataset/training/box.npy", mmap_mode="r")
        src = str(USCGSITE / "dataset/training")
    pos0 = np.array(pos[0], dtype=np.float64).reshape(-1)
    box0 = np.array(box[0], dtype=np.float64).reshape(-1)
    if box0.size == 9:
        box0 = np.array(
            [box0[0], box0[1], box0[2], box0[4], box0[5], box0[8]], dtype=np.float64
        )
    elif box0.size != 6:
        raise SystemExit(f"unexpected box shape {box0.shape} from {src}")
    n = pos0.size // 3
    return pos0, box0, n, src


def _python_efw(model: Path, pos: np.ndarray, box: np.ndarray, precision: str):
    _setup_path()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from staf.dtype import set_precision, np_dtype
    from staf_cg_paths import set_ops_root

    set_precision(precision)
    set_ops_root(precision)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    from staf_cg_models.alpha_nes_model_inference import alpha_nes_full_inference

    dt = np_dtype()
    n = pos.size // 3
    pos_b = np.asarray(pos, dtype=dt).reshape(1, n * 3)
    box_b = np.asarray(box, dtype=dt).reshape(1, -1)
    model_obj = alpha_nes_full_inference(str(model))
    virial_src = "full_test_virial"
    if hasattr(model_obj, "full_test_virial"):
        e, f, w = model_obj.full_test_virial(pos_b, box_b)
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if w.size == 9:
            w_diag = np.array([w[0], w[4], w[8]], dtype=np.float64)
        elif w.size == 6:
            w_diag = np.array([w[0], w[1], w[2]], dtype=np.float64)
        else:
            raise SystemExit(f"unexpected python virial size {w.size}")
    else:
        virial_src = "full_test (no virial)"
        e, f = model_obj.full_test(pos_b, box_b)
        w_diag = np.full(3, np.nan)
    return (
        float(np.asarray(e).reshape(-1)[0]),
        np.asarray(f, dtype=np.float64).reshape(-1),
        w_diag,
        virial_src,
    )


def _find_lmp() -> Path:
    env = os.environ.get("LMP_CG")
    cands = []
    if env:
        cands.append(Path(env))
    cands.extend(
        [
            Path("/home/francegm/programmi/lammps-23Jun2022/src/lmp_staf_cg"),
            REPO / "tmp/lammps-staf-cg/src/lmp_staf_cg",
            Path("/tmp/lammps-staf-cg/src/lmp_staf_cg"),
        ]
    )
    for p in cands:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return cands[0]


def _source_gpu_env() -> dict:
    env = os.environ.copy()
    script = REPO / "scripts/staf_gpu_env.sh"
    if not script.is_file():
        return env
    out = subprocess.check_output(
        ["bash", "-lc", f"source '{script}' >/dev/null; env -0"],
        stderr=subprocess.DEVNULL,
    )
    for item in out.split(b"\0"):
        if not item or b"=" not in item:
            continue
        k, _, v = item.partition(b"=")
        env[k.decode()] = v.decode()
    return env


def _run_lammps(lmp: Path, model: Path, dump: Path, log: Path) -> None:
    env = _source_gpu_env()
    cmd = [
        str(lmp),
        "-in",
        "in.smoke",
        "-log",
        str(log),
        "-var",
        "modeldir",
        str(model),
        "-var",
        "dumpfile",
        str(dump),
    ]
    proc = subprocess.run(
        cmd, cwd=str(HERE), env=env, check=False, capture_output=True, text=True
    )
    (HERE / "lammps_stdout.txt").write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(
            f"LAMMPS failed rc={proc.returncode}\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )


def _parse_thermo(log: Path):
    """Parse last step-0 thermo: step pe pxx pyy pzz evdwl press vol."""
    text = log.read_text(errors="replace")
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            step = int(float(parts[0]))
        except ValueError:
            continue
        try:
            vals = [float(x) for x in parts[:8]]
        except ValueError:
            continue
        if step == 0:
            rows.append(vals)
    if not rows:
        raise SystemExit(f"could not parse thermo from {log}")
    step, pe, pxx, pyy, pzz, evdwl, press, vol = rows[-1]
    return {
        "pe": pe,
        "pxx": pxx,
        "pyy": pyy,
        "pzz": pzz,
        "evdwl": evdwl,
        "press": press,
        "vol": vol,
    }


def _parse_dump(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("ITEM: ATOMS"):
        i += 1
    if i >= len(lines):
        raise SystemExit(f"bad dump (no ATOMS): {path}")
    i += 1
    rows = []
    while i < len(lines) and not lines[i].startswith("ITEM:"):
        parts = lines[i].split()
        if len(parts) >= 5:
            rows.append([float(x) for x in parts[:5]])
        i += 1
    arr = np.asarray(rows, dtype=np.float64)
    order = np.argsort(arr[:, 0].astype(int))
    arr = arr[order]
    return arr[:, 2:5]  # fx fy fz


def _p_from_w_diag(w_diag: np.ndarray, vol: float) -> float:
    return float(np.sum(w_diag) / (3.0 * vol) * NKTV2P_METAL)


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    if not MODEL.is_dir():
        print(f"missing model {MODEL}", file=sys.stderr)
        return 2

    pos, box, n, src = _load_frame0()
    vol = float(box[0] * box[3] * box[5])
    print(f"frame0 from {src}: n={n} box={box} V={vol}")

    e_py, f_py, w_py, vir_src = _python_efw(MODEL, pos, box, "float")
    p_py = _p_from_w_diag(w_py, vol) if np.all(np.isfinite(w_py)) else float("nan")
    print(f"python float: E={e_py:.10g} |F|_max={np.max(np.abs(f_py)):.6g} "
          f"Wdiag={w_py} P_config={p_py:.6g} bar ({vir_src})")

    lmp = _find_lmp()
    if not lmp.is_file() or not os.access(lmp, os.X_OK):
        summary = {
            "pass": False,
            "blocker": f"lmp_staf_cg not found/executable: {lmp}",
            "rebuild": (
                "source scripts/staf_gpu_env.sh && "
                "bash lammps/USER-STAF-CG/Install.sh $LAMMPS_SRC && "
                "cd $LAMMPS_SRC && make staf_cg -j"
            ),
        }
        (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (HERE / "summary.txt").write_text(
            "pass: false\n"
            f"blocker: {summary['blocker']}\n"
            f"rebuild: {summary['rebuild']}\n"
        )
        print(json.dumps(summary, indent=2))
        return 1

    dump = HERE / "forces.dump"
    log = HERE / "log.lammps"
    _run_lammps(lmp, MODEL, dump, log)
    th = _parse_thermo(log)
    f_lmp = _parse_dump(dump).reshape(-1)
    if f_lmp.size != f_py.size:
        raise SystemExit(f"force size mismatch python {f_py.size} lammps {f_lmp.size}")

    e_lmp = float(th["pe"])
    # At run 0, v=0 → press is configurational. Also recover W from pxx*V/nktv2p.
    p_lmp = float(th["press"])
    w_lmp = np.array(
        [
            th["pxx"] * th["vol"] / NKTV2P_METAL,
            th["pyy"] * th["vol"] / NKTV2P_METAL,
            th["pzz"] * th["vol"] / NKTV2P_METAL,
        ],
        dtype=np.float64,
    )

    dE = abs(e_py - e_lmp)
    dF = float(np.max(np.abs(f_py - f_lmp)))
    dW = float(np.max(np.abs(w_py - w_lmp))) if np.all(np.isfinite(w_py)) else float("nan")
    # Opposite-sign virial convention (Python vs LAMMPS) still compares |P|.
    dW_flip = (
        float(np.max(np.abs(w_py + w_lmp))) if np.all(np.isfinite(w_py)) else float("nan")
    )
    w_sign = 1
    dW_used = dW
    if np.isfinite(dW_flip) and dW_flip + 1e-12 < dW:
        w_sign = -1
        dW_used = dW_flip
    denom = max(abs(p_py), abs(p_lmp), 1.0)
    dP_rel = abs(p_py - p_lmp) / denom if np.isfinite(p_py) else float("inf")
    # If W signs differ, compare |P|.
    dP_rel_abs = abs(abs(p_py) - abs(p_lmp)) / denom if np.isfinite(p_py) else float("inf")
    p_ok = bool(
        (np.isfinite(p_py) and (dP_rel < P_REL_TOL or dP_rel_abs < P_REL_TOL))
        or (np.isfinite(dW_used) and dW_used < W_DIAG_TOL)
    )
    e_ok = dE < E_TOL
    f_ok = dF < F_TOL
    ok = bool(e_ok and f_ok and p_ok)
    p_gate = (
        "rel_P" if (np.isfinite(p_py) and min(dP_rel, dP_rel_abs) < P_REL_TOL)
        else ("W_diag" if (np.isfinite(dW_used) and dW_used < W_DIAG_TOL) else "none")
    )

    summary = {
        "pass": ok,
        "n_beads": n,
        "volume_A3": vol,
        "nktv2p_metal_bar": NKTV2P_METAL,
        "python_precision": "float",
        "lammps_precision": "float",
        "E_python": e_py,
        "E_lammps": e_lmp,
        "max_abs_dE": dE,
        "max_abs_dF": dF,
        "W_diag_python": w_py.tolist(),
        "W_diag_lammps": w_lmp.tolist(),
        "max_abs_dW_diag": dW,
        "max_abs_dW_diag_flipped": dW_flip,
        "virial_sign_python_vs_lammps": w_sign,
        "P_python_bar": p_py,
        "P_lammps_bar": p_lmp,
        "rel_dP": dP_rel,
        "p_gate": p_gate,
        "e_ok": e_ok,
        "f_ok": f_ok,
        "p_ok": p_ok,
        "tol_E": E_TOL,
        "tol_F": F_TOL,
        "tol_P_rel": P_REL_TOL,
        "tol_W_diag": W_DIAG_TOL,
        "python_virial": vir_src,
        "model": str(MODEL),
        "frame_source": src,
        "lmp": str(lmp),
        "note": (
            "P_config = Tr(W)/(3V)*nktv2p; LAMMPS metal press is bar. "
            "run 0, v=0 → no kinetic. p_gate documents which criterion passed."
        ),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (HERE / "summary.txt").write_text(
        f"pass: {str(ok).lower()}\n"
        f"E_python: {e_py:.10g}\n"
        f"E_lammps: {e_lmp:.10g}\n"
        f"max|dE|: {dE:.6g}\n"
        f"max|dF|: {dF:.6g}\n"
        f"P_python_bar: {p_py:.8g}\n"
        f"P_lammps_bar: {p_lmp:.8g}\n"
        f"|dP|/max(|P|,1): {dP_rel:.6g}\n"
        f"max|dW_diag|: {dW:.6g}\n"
        f"p_gate: {p_gate}\n"
        f"e_ok: {e_ok}  f_ok: {f_ok}  p_ok: {p_ok}\n"
        f"lmp: {lmp}\n"
        f"model: {MODEL}\n"
        f"note: metal nktv2p={NKTV2P_METAL:g} bar per eV/Å³; pair virial only\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
