"""Shared helpers for STAF-CG tests (path order, MODEL1896 staging, E/F)."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STAF_CG = REPO / "STAF-CG"
STAF = REPO / "STAF"
MODEL1896 = Path(
    "/home/francegm/ORIGAMI/ORIGAMI_DYNAMICS/origami_uscgsite/models/MODEL1896"
)
USCGSITE = Path("/home/francegm/ORIGAMI/INFER_INTRA_TRY2+USCGSITE")

# Intra origami defaults + MODEL1896 MD inter cutoffs (RUNT400BOX280/nohup.out).
CUTOFF_1896 = """\
50 24
50 276
25 0
16 0
16 0
8 0
"""


def setup_sys_path() -> None:
    """STAF-CG first so source_routine/ is CG, then STAF for staf.dtype."""
    cg = str(STAF_CG)
    staf = str(STAF)
    if cg in sys.path:
        sys.path.remove(cg)
    sys.path.insert(0, cg)
    if staf not in sys.path:
        sys.path.insert(1, staf)


def write_cutoff_info(
    dest: Path,
    *,
    rc=50.0,
    rad_buff=24,
    rc_ang=50.0,
    ang_buff=276,
    rs=25.0,
    rc_inter=20.0,
    ra_inter=20.0,
    rs_inter=10.0,
) -> None:
    dest.write_text(
        f"{rc} {rad_buff}\n{rc_ang} {ang_buff}\n{rs} 0\n"
        f"{rc_inter} 0\n{ra_inter} 0\n{rs_inter} 0\n"
    )


def stage_model1896(dst: Path) -> Path:
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(MODEL1896, dst)
    (dst / "number_of_nn.dat").write_text("2\n")
    (dst / "cutoff_info").write_text(CUTOFF_1896)
    return dst


def finalize_train_export(src: Path, dest: Path, *, float32: bool = False) -> Path:
    """Export a training checkpoint to an inference directory (SavedModel + maps)."""
    dest = Path(dest)
    src = Path(src)
    if dest.exists():
        shutil.rmtree(dest)
    script = "save_model_in_float.py" if float32 else "save_model.py"
    import subprocess

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    subprocess.check_call(
        [
            sys.executable,
            str(STAF_CG / "save_models" / script),
            "-imodel",
            str(src),
            "-modelname",
            str(dest),
        ],
        env=env,
    )
    n_nn = 0
    for k in range(100):
        if (dest / f"model_type{k}").exists() or (src / f"net_model_type{k}").exists():
            n_nn += 1
        else:
            break
    (dest / "number_of_nn.dat").write_text(f"{max(n_nn, 1)}\n")
    if not (dest / "cutoff_info").exists():
        write_cutoff_info(dest / "cutoff_info")
    for name in (
        "color_type_map.dat",
        "map_color_interaction.dat",
        "map_intra.dat",
        "model_error",
    ):
        s = src / name
        if s.exists() and not (dest / name).exists():
            shutil.copy(s, dest / name)
    return dest


def energy_force(model, pos, box):
    e, f = model.full_test(pos, box)
    energy = np.asarray(e).reshape(-1)
    force = np.asarray(f).reshape(pos.shape[0], -1)
    return energy, force
