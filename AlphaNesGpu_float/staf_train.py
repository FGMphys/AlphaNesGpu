"""Deprecated entrypoint — use ``../STAF/staf_train.py`` with ``precision: float``."""
import runpy
import sys
from pathlib import Path

sys.stderr.write(
    "STAF: AlphaNesGpu_float/staf_train.py is deprecated; use STAF/staf_train.py\n"
)
staf = Path(__file__).resolve().parents[1] / "STAF" / "staf_train.py"
sys.argv[0] = str(staf)
runpy.run_path(str(staf), run_name="__main__")
