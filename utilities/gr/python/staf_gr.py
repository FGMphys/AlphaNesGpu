"""Python API for utilities/gr (C cell-list g(r))."""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Iterable

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB_CANDIDATES = [
    _HERE.parent / "libstaf_gr.so",
    _HERE.parent / "build" / "libstaf_gr.so",
]


def _load_lib() -> ctypes.CDLL:
    for p in _LIB_CANDIDATES:
        if p.is_file():
            return ctypes.CDLL(str(p))
    raise FileNotFoundError(
        "libstaf_gr.so not found; run `make -C utilities/gr` in AlphaNesGpu"
    )


class _StafGr(ctypes.Structure):
    pass


def _bind(lib: ctypes.CDLL):
    lib.staf_gr_create.argtypes = [ctypes.c_double, ctypes.c_double]
    lib.staf_gr_create.restype = ctypes.POINTER(_StafGr)
    lib.staf_gr_free.argtypes = [ctypes.POINTER(_StafGr)]
    lib.staf_gr_free.restype = None
    lib.staf_gr_reset.argtypes = [ctypes.POINTER(_StafGr)]
    lib.staf_gr_reset.restype = None
    lib.staf_gr_accumulate.argtypes = [
        ctypes.POINTER(_StafGr),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.staf_gr_accumulate.restype = ctypes.c_int
    lib.staf_gr_normalize.argtypes = [
        ctypes.POINTER(_StafGr),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.staf_gr_normalize.restype = ctypes.c_int
    return lib


_LIB = _bind(_load_lib())


def compute_gr_frames(
    frames: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ta: int,
    tb: int,
    dr: float = 0.05,
    rmax: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    frames: iterable of (pos[n,3], types[n], box[3]) with types 0-based.
    Returns (r, g) for pair (ta, tb).
    """
    gr = _LIB.staf_gr_create(ctypes.c_double(dr), ctypes.c_double(rmax))
    if not gr:
        raise RuntimeError("staf_gr_create failed")
    try:
        sum_na = 0.0
        sum_rho_b = 0.0
        nframes = 0
        for pos, types, box in frames:
            pos = np.ascontiguousarray(pos, dtype=np.float64)
            types = np.ascontiguousarray(types, dtype=np.int32)
            box = np.ascontiguousarray(box, dtype=np.float64)
            if pos.ndim != 2 or pos.shape[1] != 3:
                raise ValueError("pos must be (n,3)")
            n = pos.shape[0]
            V = float(np.prod(box))
            n_a = int(np.sum(types == ta))
            n_b = int(np.sum(types == tb))
            if n_a == 0 or n_b == 0:
                continue
            rc = _LIB.staf_gr_accumulate(
                gr,
                pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                types.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ctypes.c_int(n),
                box.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(ta),
                ctypes.c_int(tb),
            )
            if rc != 0:
                raise RuntimeError(f"staf_gr_accumulate rc={rc}")
            sum_na += n_a
            sum_rho_b += n_b / V
            nframes += 1
        if nframes == 0:
            raise RuntimeError("no frames accumulated")
        nbin = int(np.floor(rmax / dr))
        r = np.empty(nbin, dtype=np.float64)
        g = np.empty(nbin, dtype=np.float64)
        same = 1 if ta == tb else 0
        rc = _LIB.staf_gr_normalize(
            gr,
            ctypes.c_double(sum_na / nframes),
            ctypes.c_double(sum_rho_b / nframes),
            ctypes.c_int(same),
            r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"staf_gr_normalize rc={rc}")
        return r, g
    finally:
        _LIB.staf_gr_free(gr)


def read_lammpstrj(
    path: Path | str,
    max_frames: int | None = None,
    min_step: int | None = None,
    max_step: int | None = None,
):
    """Yield (pos, types0, box) with LAMMPS types 1,2 → 0,1.

    Optional min_step/max_step filter on ITEM: TIMESTEP (inclusive).
    """
    path = Path(path)
    with path.open() as f:
        n_out = 0
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(f.readline())
            assert f.readline().startswith("ITEM: NUMBER")
            natoms = int(f.readline())
            assert f.readline().startswith("ITEM: BOX")
            box = []
            for _ in range(3):
                lo, hi = map(float, f.readline().split()[:2])
                box.append(hi - lo)
            hdr = f.readline()
            cols = hdr.split()[2:]
            ti, xi, yi, zi = (
                cols.index("type"),
                cols.index("x"),
                cols.index("y"),
                cols.index("z"),
            )
            data = np.loadtxt(f, max_rows=natoms)
            if min_step is not None and step < min_step:
                continue
            if max_step is not None and step > max_step:
                continue
            types = data[:, ti].astype(np.int32) - 1
            pos = data[:, [xi, yi, zi]].astype(np.float64)
            yield pos, types, np.asarray(box, dtype=np.float64)
            n_out += 1
            if max_frames is not None and n_out >= max_frames:
                break


def read_mbpol_set(set_dir: Path | str, max_frames: int | None = None):
    """DeepMD set.000 + parent type.raw (0=O,1=H)."""
    set_dir = Path(set_dir)
    types = np.loadtxt(set_dir.parent / "type.raw", dtype=np.int32)
    coord = np.load(set_dir / "coord.npy")
    box9 = np.load(set_dir / "box.npy")
    nframes, natoms = coord.shape[0], len(types)
    coord = coord.reshape(nframes, natoms, 3)
    if max_frames is not None:
        nframes = min(nframes, max_frames)
    for i in range(nframes):
        box = np.array([box9[i, 0], box9[i, 4], box9[i, 8]], dtype=np.float64)
        yield coord[i].astype(np.float64), types, box
