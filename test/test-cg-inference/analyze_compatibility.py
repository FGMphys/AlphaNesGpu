#!/usr/bin/env python3
"""Compare STAF-CG float vs double inference energy and forces."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# Same energy/force thresholds as STAF test-inference-pipeline.
ENERGY_MAX_ABS = 1e-3
FORCE_MAX_ABS = 1e-3


def _stats(a, b, name):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-30
    finite = np.isfinite(diff)
    return {
        "name": name,
        "shape": list(a.shape),
        "max_abs": float(diff[finite].max()) if finite.any() else 0.0,
        "mean_abs": float(diff[finite].mean()) if finite.any() else 0.0,
        "rms": float(np.sqrt((diff[finite] ** 2).mean())) if finite.any() else 0.0,
        "median_rel": float(np.median((diff / denom)[finite])) if finite.any() else 0.0,
    }


def main() -> int:
    d = np.load(ROOT / "inference_double" / "inference_bundle.npz", allow_pickle=True)
    f = np.load(ROOT / "inference_float" / "inference_bundle.npz", allow_pickle=True)
    report = {
        "frames": np.asarray(d["frame_indices"]).tolist(),
        "energy": _stats(d["energy"], f["energy"], "energy"),
        "force": _stats(d["force"], f["force"], "force"),
    }
    e = report["energy"]["max_abs"]
    frc = report["force"]["max_abs"]
    verdicts = [
        ("energy", e < ENERGY_MAX_ABS, f"max|Δ|={e:.3e} (tol {ENERGY_MAX_ABS:g})"),
        ("force", frc < FORCE_MAX_ABS, f"max|Δ|={frc:.3e} (tol {FORCE_MAX_ABS:g})"),
    ]
    ok = all(v[1] for v in verdicts)
    lines = [
        "STAF-CG float vs double inference",
        f"frames={report['frames']}",
        "",
    ]
    for name, passed, detail in verdicts:
        tag = "PASS" if passed else "FAIL"
        lines.append(f"{tag}  {name}: {detail}")
    lines.append("")
    lines.append("Compatible" if ok else "NOT Compatible")
    text = "\n".join(lines) + "\n"
    (ROOT / "comparison_summary.txt").write_text(text)
    (ROOT / "comparison_summary.json").write_text(json.dumps(report, indent=2))
    print(text, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
