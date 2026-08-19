#!/usr/bin/env python3
"""Batch inference over a dataset folder with pos.npy / box.npy."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from staf_cg_infer import main

if __name__ == "__main__":
    raise SystemExit(main())
