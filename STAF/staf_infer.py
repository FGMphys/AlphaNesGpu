#!/usr/bin/env python3
"""STAF inference CLI.

Examples:
  python staf_infer.py --model model_double --precision double --pos pos.npy --box box.npy
  python staf_infer.py --model model_float --precision float --frames-dir ../test/.../frames
  python staf_infer.py --model MODEL --precision float --pos pos.npy --box box.npy --decompose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_STAF_HOME = Path(__file__).resolve().parent
if str(_STAF_HOME) not in sys.path:
    sys.path.insert(0, str(_STAF_HOME))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="STAF energy/force inference")
    p.add_argument("--model", required=True, help="Exported model directory")
    p.add_argument("--precision", choices=["float", "double"], required=True)
    p.add_argument("--pos", type=Path, help="Positions file (.npy or text)")
    p.add_argument("--box", type=Path, help="Box file (.npy or text)")
    p.add_argument(
        "--frames-dir",
        type=Path,
        help="Directory with pos.npy / box.npy (uses frame index)",
    )
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--json-out", type=Path, help="Optional JSON metrics path")
    p.add_argument(
        "--decompose",
        action="store_true",
        help="A6: sum isolated-cluster energies (pairs, triplets, …) — energy only",
    )
    p.add_argument(
        "--max-body",
        type=int,
        default=3,
        help="A6: highest n-body clique (2..5, default 3)",
    )
    p.add_argument(
        "--rcut",
        type=float,
        default=None,
        help="A6: clique cutoff (default max(Rc, Rc_ang) from the model)",
    )
    p.add_argument(
        "--max-clusters",
        type=int,
        default=None,
        help="A6: cap clusters per order (smoke / debug)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="A6: log every N isolated-cluster evaluations",
    )
    args = p.parse_args(argv)

    import tensorflow as tf
    from staf.dtype import set_precision, np_dtype

    set_precision(args.precision)
    from staf_models.staf_model_inference_full import staf_full_inference

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(exc)

    dtype = np_dtype()
    if args.frames_dir is not None:
        pos = np.load(args.frames_dir / "pos.npy").astype(dtype)[args.frame]
        box = np.load(args.frames_dir / "box.npy").astype(dtype)[args.frame]
    else:
        if args.pos is None or args.box is None:
            p.error("provide --pos/--box or --frames-dir")
        if args.pos.suffix == ".npy":
            pos = np.load(args.pos).astype(dtype)
        else:
            pos = np.loadtxt(args.pos, dtype=dtype)
        if args.box.suffix == ".npy":
            box = np.load(args.box).astype(dtype)
        else:
            box = np.loadtxt(args.box, dtype=dtype)

    pos_b = np.asarray(pos, dtype=dtype).reshape(1, -1)
    box_b = np.asarray(box, dtype=dtype).reshape(1, -1)

    model = staf_full_inference(str(Path(args.model).resolve()))
    energy, force, virial = model.full_test(pos_b, box_b)
    e = float(np.asarray(energy).reshape(-1)[0])
    f = np.asarray(force)
    w = np.asarray(virial).reshape(-1)
    wxx, wyy, wzz = (
        (float(w[0]), float(w[4]), float(w[8]))
        if w.size >= 9
        else (float(w[0]), float(w[1]), float(w[2]))
    )
    print(f"STAF: energy={e:.10f}")
    print(f"STAF: force shape={f.shape} |F|_rms={float(np.sqrt(np.mean(f * f))):.6e}")
    print(f"STAF: virial_diag (eV) Wxx={wxx:.6f} Wyy={wyy:.6f} Wzz={wzz:.6f}")

    payload_extra: dict = {"decompose": False}

    if args.decompose:
        from staf.mbe import sum_isolated_cluster_energies

        if args.max_body < 2 or args.max_body > 5:
            p.error("--max-body must be 2..5")
        rcut = float(args.rcut) if args.rcut is not None else max(model.rc, model.rc_ang)
        pos_xyz = np.asarray(pos, dtype=dtype).reshape(-1, 3)
        types = np.asarray(model.type_map.numpy(), dtype=np.int32).reshape(-1)
        if pos_xyz.shape[0] != types.shape[0]:
            raise SystemExit(
                f"A6: N_pos={pos_xyz.shape[0]} != N_type_map={types.shape[0]}"
            )
        print("STAF A6: isolated-cluster energies (only those n particles in vacuum)")
        print("STAF A6: TODO(FGM) closed-form 2-body from AF parameters (latex)")
        print(
            "STAF A6: sums are raw cluster energies (not MBE inclusion-exclusion); "
            "they need not add to E_full"
        )
        parts = sum_isolated_cluster_energies(
            pos_xyz,
            np.asarray(box, dtype=dtype).reshape(-1),
            types,
            rcut=rcut,
            max_body=int(args.max_body),
            energy_fn=model.energy_of_n_atoms,
            max_clusters=args.max_clusters,
            progress_every=args.progress_every,
            log=print,
        )
        payload_extra = {
            "decompose": True,
            "rcut": rcut,
            "max_body": int(args.max_body),
            "E_full": e,
            "orders": {
                str(n): {"n_clusters": v["n_clusters"], "sum_E": v["sum_E"]}
                for n, v in parts.items()
            },
            "note_2body_closed_form": "TODO(FGM) latex for analytic 2-body / AF parameters",
        }
        for n, v in parts.items():
            print(
                f"STAF A6: n={n}  n_clusters={v['n_clusters']}  "
                f"sum_E={v['sum_E']:.10f}"
            )

    if args.json_out is not None:
        payload = {
            "precision": args.precision,
            "model": str(Path(args.model).resolve()),
            "energy": e,
            "force_rms": float(np.sqrt(np.mean(f * f))),
            "n_atoms": int(f.size // 3),
            "virial": [float(x) for x in np.asarray(w).reshape(-1)],
            "virial_diag": [wxx, wyy, wzz],
            **payload_extra,
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"STAF: wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
