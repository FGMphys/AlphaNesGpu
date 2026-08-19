#!/usr/bin/env python3
"""Compare lcurve_notmean across distribute modes (none / horovod).

Runs 1-epoch training per mode (same Seed / YAML base), then checks that
Loss_E / Loss_F / Loss_Bound match within a tolerance.

Usage (from repo root, .venv active, one GPU — sequential float then double):

  python test/test-training-pipeline/compare_distribute_lcurve.py --precision float
  python test/test-training-pipeline/compare_distribute_lcurve.py --precision double
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
STAF_TRAIN = REPO / "STAF" / "staf_train.py"
MODES = ("none", "horovod")


def _run_dir(precision: str) -> Path:
    return Path(__file__).resolve().parent / f"run_{precision}"


def _write_yaml(base: Path, out: Path, distribute: str, n_epochs: int) -> None:
    with open(base) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["distribute"] = distribute
    cfg["number_of_epochs"] = n_epochs
    # Dense host logging so lcurve_notmean has every step.
    cfg["log_batch_freq"] = 1
    cfg["displ_freq"] = max(int(cfg.get("displ_freq", 10)), 10)
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def _clean_artifacts(cwd: Path) -> None:
    for name in (
        "lcurve.out",
        "lcurve_notmean",
        "time_story.dat",
        "lr_step.dat",
        "shuffle_dataset_vec",
        "type_map.dat",
    ):
        p = cwd / name
        if p.exists():
            p.unlink()
    for p in cwd.glob("model_log*"):
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)


def _load_lcurve_notmean(path: Path) -> np.ndarray:
    """Columns: step Loss_E Loss_F Loss_Bound (grace headers skipped)."""
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("@"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                rows.append([float(x) for x in parts[:4]])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"no numeric rows in {path}")
    return np.asarray(rows, dtype=np.float64)


def run_mode(precision: str, mode: str, n_epochs: int, out_root: Path) -> Path:
    run_dir = _run_dir(precision)
    base_yaml = run_dir / "input_4test.yaml"
    if not base_yaml.is_file():
        raise FileNotFoundError(base_yaml)
    yaml_path = run_dir / f"input_parity_{mode}.yaml"
    _write_yaml(base_yaml, yaml_path, mode, n_epochs)
    _clean_artifacts(run_dir)

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    cmd = [sys.executable, str(STAF_TRAIN), yaml_path.name]
    if mode == "horovod":
        cmd = ["mpirun", "-np", "1"] + cmd

    log_path = out_root / f"train_{precision}_{mode}.log"
    print(f"STAF parity: running precision={precision} distribute={mode} …")
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(run_dir),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"train failed mode={mode} rc={proc.returncode}; see {log_path}"
        )

    src = run_dir / "lcurve_notmean"
    if not src.is_file():
        raise FileNotFoundError(f"missing {src} after {mode}")
    dest = out_root / f"lcurve_notmean_{precision}_{mode}.dat"
    shutil.copy2(src, dest)
    # Sanity: mode string in log
    text = log_path.read_text(errors="replace")
    needle = {
        "none": "distribute=none",
        "horovod": "distribute=horovod",
    }[mode]
    if needle not in text:
        raise RuntimeError(f"log {log_path} missing {needle!r} — wrong branch?")
    return dest


def compare(arrays: dict[str, np.ndarray], rtol: float, atol: float) -> dict:
    ref_name = "none"
    if ref_name not in arrays:
        raise KeyError("need none as reference")
    ref = arrays[ref_name]
    report = {"n_rows_ref": int(ref.shape[0]), "pairs": {}}
    ok_all = True
    for name, arr in arrays.items():
        if name == ref_name:
            continue
        n = min(ref.shape[0], arr.shape[0])
        if ref.shape[0] != arr.shape[0]:
            print(
                f"WARN: row count {ref_name}={ref.shape[0]} vs {name}={arr.shape[0]}; "
                f"comparing first {n}"
            )
        a, b = ref[:n], arr[:n]
        # column 0 = step (must match)
        step_ok = np.array_equal(a[:, 0].astype(np.int64), b[:, 0].astype(np.int64))
        diffs = {}
        pair_ok = step_ok
        for j, col in enumerate(("Loss_E", "Loss_F", "Loss_Bound"), start=1):
            max_abs = float(np.max(np.abs(a[:, j] - b[:, j])))
            # relative where |ref| is not tiny
            denom = np.maximum(np.abs(a[:, j]), atol)
            max_rel = float(np.max(np.abs(a[:, j] - b[:, j]) / denom))
            close = np.allclose(a[:, j], b[:, j], rtol=rtol, atol=atol)
            diffs[col] = {
                "max_abs": max_abs,
                "max_rel": max_rel,
                "allclose": close,
            }
            pair_ok = pair_ok and close
        report["pairs"][f"{ref_name}_vs_{name}"] = {
            "n": n,
            "steps_match": bool(step_ok),
            "ok": bool(pair_ok),
            "cols": diffs,
        }
        ok_all = ok_all and pair_ok
    report["ok"] = ok_all
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision", choices=("float", "double"), required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="relative tol (default: 1e-4 float / 1e-10 double)",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=None,
        help="absolute tol (default: 1e-5 float / 1e-12 double)",
    )
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="only compare existing lcurve_notmean_* files",
    )
    args = p.parse_args()

    out_root = Path(__file__).resolve().parent / "parity_distribute" / args.precision
    out_root.mkdir(parents=True, exist_ok=True)

    # Float GPU atomics/order → ~1e-6 Loss_F noise; double is essentially bit-identical.
    if args.precision == "float":
        rtol = 1e-4 if args.rtol is None else args.rtol
        atol = 1e-5 if args.atol is None else args.atol
    else:
        rtol = 1e-10 if args.rtol is None else args.rtol
        atol = 1e-12 if args.atol is None else args.atol

    paths = {}
    if not args.skip_run:
        for mode in MODES:
            paths[mode] = run_mode(args.precision, mode, args.epochs, out_root)
    else:
        for mode in MODES:
            paths[mode] = out_root / f"lcurve_notmean_{args.precision}_{mode}.dat"

    arrays = {m: _load_lcurve_notmean(paths[m]) for m in MODES}
    report = compare(arrays, rtol=rtol, atol=atol)

    summary_path = out_root / "summary.txt"
    lines = [
        f"precision={args.precision} epochs={args.epochs}",
        f"rtol={rtol} atol={atol}",
        f"n_rows_ref={report['n_rows_ref']}",
        f"OVERALL_OK={report['ok']}",
        "",
    ]
    for pair, info in report["pairs"].items():
        lines.append(
            f"{pair}: ok={info['ok']} steps_match={info['steps_match']} n={info['n']}"
        )
        for col, d in info["cols"].items():
            lines.append(
                f"  {col}: max_abs={d['max_abs']:.6e} max_rel={d['max_rel']:.6e} "
                f"allclose={d['allclose']}"
            )
        lines.append("")
    text = "\n".join(lines)
    summary_path.write_text(text)
    print(text)
    print(f"Wrote {summary_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
