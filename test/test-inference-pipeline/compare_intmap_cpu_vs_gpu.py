#!/usr/bin/env python3
"""Compare interaction maps from two inference_bundle.npz (CPU-NL vs GPU-NL)."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def _load_intmaps(bundle_path: Path):
    z = np.load(bundle_path, allow_pickle=True)
    int2b = z["int2b"]  # object array of per-frame lists/arrays
    int3b = z["int3b"]
    return int2b, int3b, z


def _as_arr(x):
    if isinstance(x, (list, tuple)):
        # multi-type split: concat along atom axis if needed
        parts = [np.asarray(p) for p in x]
        if len(parts) == 1:
            return parts[0]
        # shapes (1, N_type, ...) -> concat on axis 1
        return np.concatenate(parts, axis=1)
    return np.asarray(x)


def compare_one(cpu_path: Path, gpu_path: Path, label: str):
    print(f"\n===== {label} =====")
    print(f"CPU: {cpu_path}")
    print(f"GPU: {gpu_path}")
    c2, c3, zc = _load_intmaps(cpu_path)
    g2, g3, zg = _load_intmaps(gpu_path)
    n = len(c2)
    assert n == len(g2) == len(c3) == len(g3)

    mism_howmany = 0
    mism_neigh = 0
    mism_order = 0
    mism_int3b = 0
    total_atoms = 0
    total_bonds = 0
    identical_frames_2b = 0
    identical_frames_3b = 0

    for i in range(n):
        a2 = _as_arr(c2[i])
        b2 = _as_arr(g2[i])
        a3 = _as_arr(c3[i])
        b3 = _as_arr(g3[i])

        if a2.shape != b2.shape:
            print(f"  frame {i}: int2b shape mismatch {a2.shape} vs {b2.shape}")
            mism_neigh += 1
            continue
        if a3.shape != b3.shape:
            print(f"  frame {i}: int3b shape mismatch {a3.shape} vs {b3.shape}")
            mism_int3b += 1
            continue

        # intmap2b layout: [howmany, neigh0, neigh1, ...] per atom (last dim = radbuff+1)
        how_a = a2[..., 0].astype(np.int64)
        how_b = b2[..., 0].astype(np.int64)
        total_atoms += how_a.size
        if not np.array_equal(how_a, how_b):
            mism_howmany += int(np.sum(how_a != how_b))
            # show a few
            bad = np.argwhere(how_a != how_b)
            for idx in bad[:5]:
                idx = tuple(idx)
                print(f"  frame {i} howmany mismatch at {idx}: CPU={how_a[idx]} GPU={how_b[idx]}")

        # neighbor lists: compare as sets (membership) and ordered sequences
        # a2 shape typically (1, N, radbuff+1) or (N, radbuff+1)
        flat_a = a2.reshape(-1, a2.shape[-1])
        flat_b = b2.reshape(-1, b2.shape[-1])
        how_af = flat_a[:, 0].astype(np.int64)
        how_bf = flat_b[:, 0].astype(np.int64)
        frame_2b_ok = True
        for p in range(flat_a.shape[0]):
            ha, hb = int(how_af[p]), int(how_bf[p])
            total_bonds += ha
            na = flat_a[p, 1 : 1 + ha].astype(np.int64)
            nb = flat_b[p, 1 : 1 + hb].astype(np.int64)
            if ha != hb or set(na.tolist()) != set(nb.tolist()):
                mism_neigh += 1
                frame_2b_ok = False
                if mism_neigh <= 8:
                    print(
                        f"  frame {i} atom {p}: neigh SET mismatch "
                        f"CPU({ha})={na[:min(ha,8)]}... GPU({hb})={nb[:min(hb,8)]}..."
                    )
            elif not np.array_equal(na, nb):
                mism_order += 1
                frame_2b_ok = False
                if mism_order <= 8:
                    print(
                        f"  frame {i} atom {p}: same set, DIFFERENT ORDER "
                        f"CPU={na[:min(ha,8)]} GPU={nb[:min(hb,8)]}"
                    )
        if frame_2b_ok and np.array_equal(how_a, how_b):
            identical_frames_2b += 1

        if np.array_equal(a3, b3):
            identical_frames_3b += 1
        else:
            mism_int3b += 1
            # int3b is pairs; compare packed valid entries if possible
            diff = np.sum(a3 != b3)
            if mism_int3b <= 5:
                print(f"  frame {i}: int3b not identical ({diff} differing entries)")

    print(
        f"frames={n} atoms_checked≈{total_atoms} "
        f"identical_int2b_frames={identical_frames_2b}/{n} "
        f"identical_int3b_frames={identical_frames_3b}/{n}"
    )
    print(
        f"mismatches: howmany_atoms={mism_howmany} "
        f"neigh_set={mism_neigh} neigh_order_only={mism_order} int3b_frames={mism_int3b}"
    )
    ok = (
        mism_howmany == 0
        and mism_neigh == 0
        and mism_order == 0
        and mism_int3b == 0
    )
    print("VERDICT:", "IDENTICAL" if ok else "DIFFERENT")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu-float", type=Path, default=ROOT / "intmap_cpu_float" / "inference_bundle.npz")
    p.add_argument("--gpu-float", type=Path, default=ROOT / "intmap_gpu_float" / "inference_bundle.npz")
    p.add_argument("--cpu-double", type=Path, default=ROOT / "intmap_cpu_double" / "inference_bundle.npz")
    p.add_argument("--gpu-double", type=Path, default=ROOT / "intmap_gpu_double" / "inference_bundle.npz")
    args = p.parse_args()

    ok_f = compare_one(args.cpu_float, args.gpu_float, "FLOAT intmap CPU vs GPU")
    ok_d = compare_one(args.cpu_double, args.gpu_double, "DOUBLE intmap CPU vs GPU")
    print("\n===== OVERALL =====")
    print("FLOAT:", "IDENTICAL" if ok_f else "DIFFERENT")
    print("DOUBLE:", "IDENTICAL" if ok_d else "DIFFERENT")
    sys.exit(0 if (ok_f and ok_d) else 1)


if __name__ == "__main__":
    main()
