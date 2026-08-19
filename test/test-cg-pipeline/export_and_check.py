#!/usr/bin/env python3
"""Export 1-epoch checkpoint to float64 + float32 inference dirs; check RMSE vs freeze."""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "STAF-CG"))
from staf_cg_harness import finalize_train_export, write_cutoff_info  # noqa: E402

# DEV freeze RMSE_f (Seed 60, 80/20, batch 8).
FREEZE_RMSE_F = 38.352597573817725
RMSE_F_TOL = 0.01


def _find_ckpt(work: Path) -> Path:
    cands = [
        c
        for c in sorted(work.glob("staf_cg_freeze_ep1*"))
        if c.is_dir() and (c / "net_model_type0").exists()
    ]
    if not cands:
        raise SystemExit(f"no 1-epoch checkpoint under {work}")
    return cands[-1]


def _parse_rmse_f(lcurve: Path) -> float:
    lines = [ln.strip() for ln in lcurve.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise SystemExit(f"empty {lcurve}")
    last = lines[-1]
    m = re.search(r"RMSE_f=([0-9.eE+-]+)", last)
    if m:
        return float(m.group(1))
    parts = last.split()
    return float(parts[3])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, default=ROOT / "work")
    args = p.parse_args()
    work = args.work
    ckpt = _find_ckpt(work)
    write_cutoff_info(ckpt / "cutoff_info")
    infer = ROOT.parent / "test-cg-inference"
    ddir = infer / "model_double"
    fdir = infer / "model_float"
    if ddir.exists():
        shutil.rmtree(ddir)
    if fdir.exists():
        shutil.rmtree(fdir)
    print("export double from", ckpt)
    finalize_train_export(ckpt, ddir, float32=False)
    print("export float from", ckpt)
    finalize_train_export(ckpt, fdir, float32=True)
    rmse_f = _parse_rmse_f(work / "lcurve.out")
    delta = abs(rmse_f - FREEZE_RMSE_F)
    ok = delta <= RMSE_F_TOL
    report = (
        f"RMSE_f={rmse_f:.12f}  freeze={FREEZE_RMSE_F:.12f}  |Δ|={delta:.3e}  "
        f"tol={RMSE_F_TOL:g}  {'PASS' if ok else 'FAIL'}\n"
    )
    (work / "rmse_vs_freeze.txt").write_text(report)
    print(report, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
