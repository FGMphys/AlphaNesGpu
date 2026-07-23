"""Deprecated entrypoint — use ``staf_train.py``."""
import runpy
import sys
from pathlib import Path

sys.stderr.write("STAF: alpha_nnpes_full_main.py is deprecated; use staf_train.py\n")
runpy.run_path(str(Path(__file__).resolve().parent / "staf_train.py"), run_name="__main__")
