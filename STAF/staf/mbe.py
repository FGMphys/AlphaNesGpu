"""A6: isolated-cluster multi-body energy decomposition.

For each n-body term, every clique of n atoms with all pairwise MIC distances
≤ rcut is placed **alone in vacuum** (no other atoms) and the STAF energy of
that n-particle system is evaluated. The n-body contribution is the **sum** of
those isolated-cluster energies.

This is *not* the inclusion-exclusion MBE (triplet energies still contain the
pair interactions). The sums are therefore not expected to add up to E_full.

TODO(FGM): closed-form 2-body from AF parameters / microscopic interpretation
(latex). Do not dump α parameters here until that formula is in docs/.

Only energies (no forces / virial).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

EnergyFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def mic_delta(dxyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Minimum-image displacement. Orthorhombic box: [lx, 0, 0, ly, 0, lz]."""
    box = np.asarray(box, dtype=np.float64).reshape(-1)
    if box.size < 6:
        raise ValueError("box must have 6 components (lx,xy,xz,ly,yz,lz)")
    if abs(float(box[1])) + abs(float(box[2])) + abs(float(box[4])) > 1e-8:
        raise NotImplementedError("A6 MBE currently supports orthorhombic boxes")
    lx, ly, lz = float(box[0]), float(box[3]), float(box[5])
    out = np.asarray(dxyz, dtype=np.float64).copy()
    out[..., 0] -= lx * np.round(out[..., 0] / lx)
    out[..., 1] -= ly * np.round(out[..., 1] / ly)
    out[..., 2] -= lz * np.round(out[..., 2] / lz)
    return out


def mic_dist2(ri: np.ndarray, rj: np.ndarray, box: np.ndarray) -> float:
    d = mic_delta(np.asarray(rj) - np.asarray(ri), box)
    return float(np.dot(d, d))


def vacuum_box(rcut: float, span: float) -> np.ndarray:
    L = max(20.0, 2.0 * float(rcut) + 4.0, 2.0 * float(span) + 4.0)
    return np.array([L, 0.0, 0.0, L, 0.0, L], dtype=np.float64)


def pack_cluster_vacuum(
    pos: np.ndarray, box: np.ndarray, idxs: Sequence[int], rcut: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Place cluster near the centre of a vacuum orthorhombic box (MIC unwrap)."""
    pos = np.asarray(pos)
    idxs = list(idxs)
    rel = np.zeros((len(idxs), 3), dtype=pos.dtype)
    p0 = pos[idxs[0]]
    span = 0.0
    for a, i in enumerate(idxs[1:], start=1):
        d = mic_delta(pos[i] - p0, box)
        rel[a] = d.astype(pos.dtype, copy=False)
        span = max(span, float(np.linalg.norm(d)))
    vbox = vacuum_box(rcut, span)
    L = float(vbox[0])
    rel = rel + (L * 0.5)
    return rel, vbox.astype(pos.dtype, copy=False)


def neighbor_pairs(
    pos: np.ndarray, box: np.ndarray, rcut: float
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    pos = np.asarray(pos).reshape(-1, 3)
    n = pos.shape[0]
    r2 = float(rcut) * float(rcut)
    neigh: List[List[int]] = [[] for _ in range(n)]
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        ri = pos[i]
        for j in range(i + 1, n):
            if mic_dist2(ri, pos[j], box) <= r2:
                neigh[i].append(j)
                neigh[j].append(i)
                pairs.append((i, j))
    return neigh, pairs


def enumerate_cliques(
    pos: np.ndarray, box: np.ndarray, rcut: float, max_body: int
) -> Dict[int, List[Tuple[int, ...]]]:
    """Cliques in the rcut neighbor graph (all pairs within rcut)."""
    if max_body < 2 or max_body > 5:
        raise ValueError("A6: max_body must be 2..5 (got %s)" % (max_body,))
    neigh, pairs = neighbor_pairs(pos, box, rcut)
    sets = [set(x) for x in neigh]
    out: Dict[int, List[Tuple[int, ...]]] = {2: [(i, j) for i, j in pairs]}
    if max_body >= 3:
        tri: List[Tuple[int, ...]] = []
        for i, j in pairs:
            for k in sets[i] & sets[j]:
                if k > j:
                    tri.append((i, j, k))
        out[3] = tri
    if max_body >= 4:
        quad: List[Tuple[int, ...]] = []
        for i, j, k in out[3]:
            common = sets[i] & sets[j] & sets[k]
            for ell in common:
                if ell > k:
                    quad.append((i, j, k, ell))
        out[4] = quad
    if max_body >= 5:
        five: List[Tuple[int, ...]] = []
        for i, j, k, ell in out[4]:
            common = sets[i] & sets[j] & sets[k] & sets[ell]
            for m in common:
                if m > ell:
                    five.append((i, j, k, ell, m))
        out[5] = five
    return out


def sum_isolated_cluster_energies(
    pos: np.ndarray,
    box: np.ndarray,
    types: np.ndarray,
    rcut: float,
    max_body: int,
    energy_fn: EnergyFn,
    max_clusters: Optional[int] = None,
    progress_every: int = 200,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[int, dict]:
    """Sum STAF energies of isolated n-atom vacuum clusters (cliques, n=2..max_body)."""

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    pos = np.asarray(pos).reshape(-1, 3)
    types = np.asarray(types, dtype=np.int32).reshape(-1)
    if pos.shape[0] != types.shape[0]:
        raise ValueError("pos/types length mismatch")
    cliques = enumerate_cliques(pos, box, rcut, max_body)
    results: Dict[int, dict] = {}
    for n in range(2, max_body + 1):
        clusters = cliques[n]
        if max_clusters is not None:
            clusters = clusters[: int(max_clusters)]
        total = 0.0
        ncl = len(clusters)
        _log("A6: n-body=%d  clusters=%d  (cliques with all pairs ≤ rcut=%.4f)" % (n, ncl, rcut))
        for ic, idxs in enumerate(clusters):
            xyz, vbox = pack_cluster_vacuum(pos, box, idxs, rcut)
            tloc = types[list(idxs)]
            total += float(energy_fn(xyz, tloc, vbox))
            if progress_every and (ic + 1) % int(progress_every) == 0:
                _log("A6:   n=%d  %d/%d" % (n, ic + 1, ncl))
        results[n] = {"n_clusters": ncl, "sum_E": total}
        _log("A6: n-body=%d  sum_E=%.10f" % (n, total))
    return results
