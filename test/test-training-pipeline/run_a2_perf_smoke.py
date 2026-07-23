#!/usr/bin/env python3
"""A2 perf smoke: timing of train step vs freeze baseline (ms/frame).

Runs a short steady-state window on model_log1 without a full epoch sweep.
Does not write into the training run directories; uses the same ops as training.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "test" / "test-training-pipeline"

# V100 freeze baseline (ACCEPTANCE / performance_baseline.txt)
BASELINE_MS = {"float": 91.5, "double": 149.7}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--precision", choices=["float", "double"], required=True)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-timed", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    run_dir = TRAIN / f"run_{args.precision}"
    code_root = REPO / (
        "AlphaNesGpu_float" if args.precision == "float" else "AlphaNesGpu_double"
    )
    sys.path = [str(code_root)] + [x for x in sys.path if x != str(code_root)]

    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    dtype = "float32" if args.precision == "float" else "float64"
    np_dtype = np.float32 if args.precision == "float" else np.float64
    sys.path = [str(REPO)] + [x for x in sys.path if x != str(REPO)]
    from staf.dtype import set_precision

    set_precision(args.precision)
    # keep local aliases for arrays
    assert tf.keras.backend.floatx() == dtype

    from gradient_utility.mixture import register_force_3bAFs_grad  # noqa: F401
    from gradient_utility.mixture import register_force_2bAFs_grad  # noqa: F401
    from gradient_utility.mixture import register_3bAFs_grad  # noqa: F401
    from gradient_utility.mixture import register_2bAFs_grad  # noqa: F401
    from staf_models.mixture.staf_model import staf_full
    from source_routine.mixture.physics_layer_mod import physics_layer, lognorm_layer
    from source_routine.mixture.force_layer_mod import force_layer
    from source_routine.descriptor_builder import descriptor_layer
    from init_params.init_AFs_param import init_AFs_param

    with open(run_dir / "input_4test.yaml") as fh:
        full_param = yaml.load(fh, Loader=yaml.FullLoader)

    tipos_raw = np.loadtxt(run_dir / "dataset" / "type.dat", dtype=int).reshape(-1)
    tipos = [int(x) for x in tipos_raw]
    type_map = []
    for t, n in enumerate(tipos):
        type_map.extend([t] * n)
    type_map = np.asarray(type_map, dtype=np.int32)
    nt = len(tipos)
    N = len(type_map)

    seed_par = int(full_param.get("Seed", 60))
    np.random.seed(seed_par)
    tf.random.set_seed(seed_par + 1)

    e = np.load(run_dir / "dataset" / "test" / "energy.npy").astype(np_dtype)
    f = np.load(run_dir / "dataset" / "test" / "force.npy").astype(np_dtype)
    pos = np.load(run_dir / "dataset" / "test" / "pos.npy").astype(np_dtype)
    box = np.load(run_dir / "dataset" / "test" / "box.npy").astype(np_dtype)

    bs = int(args.batch_size)
    n_need = (args.n_warmup + args.n_timed) * bs
    assert pos.shape[0] >= n_need, f"need {n_need} frames, have {pos.shape[0]}"

    restart = str((run_dir / "model_log1").resolve())
    opt_net = tf.keras.optimizers.Adam(1e-3)
    opt_phys = tf.keras.optimizers.Adam(1e-3)
    rng_state = np.random.get_state()
    init_a2, init_a3, init_mu, init_te, rng_state = init_AFs_param(
        restart, full_param, nt, rng_state
    )
    np.random.set_state(rng_state)

    rc = float(full_param["Rc"])
    rb = int(full_param["Radial_Buffer"])
    rca = float(full_param["Rc_Angular"])
    ab = int(full_param["Max_Angular_Neigh"])
    Rs = float(full_param["Rs"])
    Desc = descriptor_layer(rc, rb, rca, ab, N, box[0], Rs, bs)
    Phys = [physics_layer(init_a2[k], init_a3[k], init_te[k]) for k in range(nt)]
    Log = [lognorm_layer(init_mu[k]) for k in range(nt)]
    Force = force_layer(rb, ab)

    def mse(ypred, y):
        return tf.reduce_mean(tf.square(ypred - y))

    model = staf_full(
        Phys,
        Force,
        int(full_param["number_of_decoding_layers"]),
        [int(k) for k in full_param["number_of_decoding_nodes"].split()],
        full_param["activation_function"],
        1,
        mse,
        mse,
        opt_net,
        opt_phys,
        float(full_param.get("alpha_bound", 5.0)),
        Log,
        tipos,
        type_map,
        restart,
        seed_par,
    )

    pe = tf.constant(1.0, dtype=dtype)
    pf = tf.constant(1.0, dtype=dtype)
    pb = tf.constant(0.0, dtype=dtype)

    times = []
    cursor = 0
    for step in range(args.n_warmup + args.n_timed):
        sl = slice(cursor, cursor + bs)
        cursor += bs
        t0 = time.perf_counter()
        d = Desc(tf.constant(pos[sl]), tf.constant(box[sl]))
        pack = (
            d[0],
            d[1],
            d[2],
            d[3],
            d[5],
            d[4],
            d[6],
            d[7],
            d[8],
            tf.constant(e[sl]),
            tf.constant(f[sl]),
            pe,
            pf,
            pb,
        )
        out = model.full_train_e_f(*pack)
        _ = float(out[0].numpy())
        dt = time.perf_counter() - t0
        if step >= args.n_warmup:
            times.append(dt)

    times = np.asarray(times, dtype=np.float64)
    ms_batch = 1e3 * times
    ms_frame = ms_batch / bs
    base = BASELINE_MS[args.precision]
    mean = float(ms_frame.mean())
    std = float(ms_frame.std())
    ratio = mean / base
    print(
        f"precision={args.precision}  n_timed={len(times)}  batch={bs}\n"
        f"ms/frame: {mean:.2f} ± {std:.2f}   baseline={base:.1f}   ratio={ratio:.3f}\n"
        f"ms/batch: {float(ms_batch.mean()):.1f} ± {float(ms_batch.std()):.1f}"
    )
    # Soft gate: allow 15% regression on this micro-bench (compile/noise).
    if ratio > 1.15:
        print("PERF WARN: slower than freeze baseline by >15%")
        return 2
    print("PERF OK vs freeze baseline (≤15% slower)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
