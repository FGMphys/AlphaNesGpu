#!/usr/bin/env python3
"""Stage MODEL1896 with cutoff_info + number_of_nn.dat for CG inference tests."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "STAF-CG"))
from staf_cg_harness import stage_model1896  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "model1896_infer",
        help="Destination inference directory",
    )
    args = p.parse_args()
    dest = stage_model1896(args.dest)
    print("STAF-CG: staged MODEL1896 at", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
