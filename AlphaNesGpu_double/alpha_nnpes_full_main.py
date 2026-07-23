"""Deprecated — use STAF/staf_train.py."""
import runpy
import sys
from pathlib import Path
sys.stderr.write("STAF: alpha_nnpes_full_main.py is deprecated; use STAF/staf_train.py\n")
staf = Path(__file__).resolve().parents[1] / "STAF" / "staf_train.py"
sys.argv[0] = str(staf)
runpy.run_path(str(staf), run_name="__main__")
