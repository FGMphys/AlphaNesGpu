#!/usr/bin/env python3
"""MODEL1896: Python STAF-CG (SavedModel) vs libstaf_cg vs LAMMPS staf/cg.

Uses the staged freeze copy (no ORIGAMI NFS) and the 24-bead frame in
test/test-lammps-staf-cg-parity/data.origami24.

Exports float32 ONNX from the SavedModel if missing, then compares E/F
(and P vs LAMMPS). Writes test/test-cg-libstaf/model1896_summary.json.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/francegm/AlphaNesGpu")
STAF_CG = REPO / "STAF-CG"
STAF = REPO / "STAF"
PY_MODEL = REPO / "DEV/staf_cg_freeze/model1896_infer"
ONNX_DIR = REPO / "test/test-cg-inference/model1896_onnx"
LAMMPS_DIR = REPO / "test/test-lammps-staf-cg-parity"
DATA = LAMMPS_DIR / "data.origami24"
SMOKE = REPO / "libstaf_cg/build/staf_force_smoke"
OUT_DIR = REPO / "test/test-cg-libstaf"
NKTV2P_METAL = 1.60217662e6
E_TOL = 1e-3
F_TOL = 1e-3
P_REL_TOL = 0.05


def _setup_path() -> None:
    cg, staf = str(STAF_CG), str(STAF)
    if cg in sys.path:
        sys.path.remove(cg)
    sys.path.insert(0, cg)
    if staf not in sys.path:
        sys.path.insert(1, staf)


def _parse_lammps_data(path: Path):
    text = path.read_text()
    box = None
    pos = []
    xlo = xhi = ylo = yhi = zlo = zhi = None
    in_atoms = False
    for line in text.splitlines():
        ls = line.split()
        if len(ls) >= 4 and ls[-2] == "xlo":
            xlo, xhi = float(ls[0]), float(ls[1])
        elif len(ls) >= 4 and ls[-2] == "ylo":
            ylo, yhi = float(ls[0]), float(ls[1])
        elif len(ls) >= 4 and ls[-2] == "zlo":
            zlo, zhi = float(ls[0]), float(ls[1])
        elif line.startswith("Atoms"):
            in_atoms = True
            continue
        elif in_atoms:
            if not ls or ls[0].startswith("#"):
                continue
            if ls[0] in ("Velocities", "Bonds", "Masses"):
                break
            if len(ls) >= 5:
                pos.append((int(ls[0]), float(ls[2]), float(ls[3]), float(ls[4])))
    pos.sort(key=lambda r: r[0])
    xyz = np.array([[x, y, z] for _i, x, y, z in pos], dtype=np.float64).reshape(-1)
    box = np.array([xhi - xlo, 0.0, 0.0, yhi - ylo, 0.0, zhi - zlo], dtype=np.float64)
    return xyz, box, len(pos)


def _python_efw(model: Path, pos: np.ndarray, box: np.ndarray, precision: str):
    _setup_path()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from staf.dtype import set_precision, np_dtype
    from staf_cg_paths import set_ops_root
    from staf_cg_models.alpha_nes_model_inference import alpha_nes_full_inference

    set_precision(precision)
    set_ops_root(precision)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    dt = np_dtype()
    n = pos.size // 3
    pos_b = np.asarray(pos, dtype=dt).reshape(1, n * 3)
    box_b = np.asarray(box, dtype=dt).reshape(1, -1)
    model_obj = alpha_nes_full_inference(str(model))
    if hasattr(model_obj, "full_test_virial"):
        e, f, w = model_obj.full_test_virial(pos_b, box_b)
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        w_diag = np.array([w[0], w[4], w[8]], dtype=np.float64) if w.size == 9 else w[:3]
    else:
        e, f = model_obj.full_test(pos_b, box_b)
        w_diag = np.full(3, np.nan)
    return (
        float(np.asarray(e).reshape(-1)[0]),
        np.asarray(f, dtype=np.float64).reshape(-1),
        w_diag,
    )


def _write_frame_bin(path: Path, pos: np.ndarray, box: np.ndarray, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<i", int(n)))
        fh.write(np.asarray(box, dtype=np.float64).tobytes())
        fh.write(np.asarray(pos, dtype=np.float64).tobytes())


def _ensure_onnx(py: str) -> None:
    marker = ONNX_DIR / "model_type0.onnx"
    if marker.is_file() and (ONNX_DIR / "model_type1.onnx").is_file():
        print("ONNX already present:", ONNX_DIR)
        return
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    cmd = [
        py,
        str(STAF_CG / "save_models" / "export_mlp_grad_onnx.py"),
        "-imodel",
        str(PY_MODEL),
        "-modelname",
        str(ONNX_DIR),
        "--precision",
        "float32",
    ]
    print("exporting MODEL1896 →", ONNX_DIR)
    subprocess.check_call(cmd, env=env)
    for name in (
        "cutoff_info",
        "number_of_nn.dat",
        "color_type_map.dat",
        "map_color_interaction.dat",
        "map_intra.dat",
    ):
        src = PY_MODEL / name
        if src.is_file() and not (ONNX_DIR / name).is_file():
            import shutil

            shutil.copy(src, ONNX_DIR / name)


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


def _parse_thermo(log: Path):
    rows = []
    for line in log.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            step = int(float(parts[0]))
            vals = [float(x) for x in parts[:8]]
        except ValueError:
            continue
        if step == 0:
            rows.append(vals)
    if not rows:
        raise SystemExit(f"could not parse thermo from {log}")
    step, pe, pxx, pyy, pzz, evdwl, press, vol = rows[-1]
    return {"pe": pe, "pxx": pxx, "pyy": pyy, "pzz": pzz, "press": press, "vol": vol}


def _parse_dump(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("ITEM: ATOMS"):
        i += 1
    i += 1
    rows = []
    while i < len(lines) and not lines[i].startswith("ITEM:"):
        parts = lines[i].split()
        if len(parts) >= 5:
            rows.append([float(x) for x in parts[:5]])
        i += 1
    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0].astype(int))]
    return arr[:, 2:5].reshape(-1)


def main() -> int:
    py = str(REPO / ".venv/bin/python")
    if not PY_MODEL.is_dir():
        print(f"missing staged MODEL1896 {PY_MODEL}", file=sys.stderr)
        return 2
    pos, box, n = _parse_lammps_data(DATA)
    vol = float(box[0] * box[3] * box[5])
    print(f"frame data.origami24 n={n} box={box} V={vol}")

    _ensure_onnx(py)

    print("Python STAF-CG double on SavedModel…")
    e_py, f_py, w_py = _python_efw(PY_MODEL, pos, box, "double")
    p_py = float(np.sum(w_py) / (3.0 * vol) * NKTV2P_METAL) if np.all(np.isfinite(w_py)) else float("nan")
    print(f"  E={e_py:.10g} max|F|={np.max(np.abs(f_py)):.6g} P={p_py:.6g} bar")

    frame_bin = OUT_DIR / "model1896_frame0.bin"
    _write_frame_bin(frame_bin, pos, box, n)
    lib_txt = OUT_DIR / "model1896_libstaf_ef.txt"
    env = _source_gpu_env()
    print("libstaf_cg…")
    rc = subprocess.run(
        [str(SMOKE), str(ONNX_DIR), str(frame_bin), str(lib_txt)],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    (OUT_DIR / "model1896_libstaf_stdout.txt").write_text(rc.stdout + "\n" + rc.stderr)
    if rc.returncode != 0 or not lib_txt.is_file():
        print("staf_force_smoke failed", rc.returncode, rc.stderr[-2000:], file=sys.stderr)
        return 4
    lines = lib_txt.read_text().strip().splitlines()
    e_c = float(lines[0])
    f_c = np.array(
        [list(map(float, ln.split())) for ln in lines[1:]], dtype=np.float64
    ).reshape(-1)
    dE_c = abs(e_py - e_c)
    dF_c = float(np.max(np.abs(f_py - f_c)))
    print(f"  E={e_c:.10g} max|ΔE|={dE_c:.6g} max|ΔF|={dF_c:.6g}")

    lmp = Path("/home/francegm/programmi/lammps-23Jun2022/src/lmp_staf_cg")
    dump = OUT_DIR / "model1896_forces.dump"
    log = OUT_DIR / "model1896_log.lammps"
    print("LAMMPS staf/cg…")
    proc = subprocess.run(
        [
            str(lmp),
            "-in",
            "in.smoke",
            "-log",
            str(log),
            "-var",
            "modeldir",
            str(ONNX_DIR),
            "-var",
            "dumpfile",
            str(dump),
        ],
        cwd=str(LAMMPS_DIR),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    (OUT_DIR / "model1896_lammps_stdout.txt").write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        print("LAMMPS failed", proc.returncode, proc.stderr[-2000:], file=sys.stderr)
        return 5
    th = _parse_thermo(log)
    f_lmp = _parse_dump(dump)
    e_lmp = float(th["pe"])
    p_lmp = float(th["press"])
    dE_l = abs(e_py - e_lmp)
    dF_l = float(np.max(np.abs(f_py - f_lmp)))
    denom = max(abs(p_py), abs(p_lmp), 1.0)
    dP_rel = abs(abs(p_py) - abs(p_lmp)) / denom if np.isfinite(p_py) else float("inf")
    print(f"  E={e_lmp:.10g} max|ΔE|={dE_l:.6g} max|ΔF|={dF_l:.6g} |ΔP|_rel={dP_rel:.6g}")

    ok_c = bool(dE_c < E_TOL and dF_c < F_TOL)
    ok_l = bool(dE_l < E_TOL and dF_l < F_TOL and dP_rel < P_REL_TOL)
    summary = {
        "pass": bool(ok_c and ok_l),
        "pass_libstaf": ok_c,
        "pass_lammps": ok_l,
        "n_beads": n,
        "python_model": str(PY_MODEL),
        "onnx_model": str(ONNX_DIR),
        "python_precision": "double",
        "md_precision": "float32",
        "E_python": e_py,
        "E_libstaf": e_c,
        "E_lammps": e_lmp,
        "max_abs_dE_libstaf": dE_c,
        "max_abs_dF_libstaf": dF_c,
        "max_abs_dE_lammps": dE_l,
        "max_abs_dF_lammps": dF_l,
        "P_python_bar": p_py,
        "P_lammps_bar": p_lmp,
        "rel_dP": dP_rel,
        "tol_E": E_TOL,
        "tol_F": F_TOL,
        "tol_P_rel": P_REL_TOL,
        "note": (
            "Python = STAF-CG SavedModel float64 MODEL1896; "
            "libstaf_cg/LAMMPS = float32 ONNX rebuilt from the same SavedModel."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "model1896_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT_DIR / "model1896_summary.txt").write_text(
        f"pass: {str(summary['pass']).lower()}\n"
        f"E_python: {e_py:.10g}\n"
        f"E_libstaf: {e_c:.10g}\n"
        f"E_lammps: {e_lmp:.10g}\n"
        f"max|dE|_libstaf: {dE_c:.6g}\n"
        f"max|dF|_libstaf: {dF_c:.6g}\n"
        f"max|dE|_lammps: {dE_l:.6g}\n"
        f"max|dF|_lammps: {dF_l:.6g}\n"
        f"|dP|/max(|P|,1): {dP_rel:.6g}\n"
        f"python: double SavedModel MODEL1896\n"
        f"md: float32 ONNX {ONNX_DIR}\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
