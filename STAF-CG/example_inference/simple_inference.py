#!/usr/bin/env python3
"""Point example_inference at staf_cg_infer.py (Sprint 1)."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from staf_cg_infer import main

if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--model",
        sys.argv[1] if len(sys.argv) > 1 else "model29",
        "--pos",
        str(Path(__file__).parent / "pos_0"),
        "--box",
        str(Path(__file__).parent / "box_0"),
    ]
    raise SystemExit(main())
