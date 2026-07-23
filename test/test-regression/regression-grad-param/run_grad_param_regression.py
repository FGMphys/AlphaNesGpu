#!/usr/bin/env python3
"""Parameter-gradient FD regression vs analytic grads (training path).

Confronta, separatamente per Loss_E e Loss_F (MSE):

    g_num = (L(w+dw) - L(w)) / dw
    g_ana = ∂L/∂w   (via tf.gradients, stessa chain di full_train_e_f)

Famiglie sondate (--n-per-family, default 100 ciascuna):
  - dense_kernel, dense_bias
  - alpha2b (tutti i tipi; se <100 disponibili si usano tutti)
  - alpha3b β / γ / δ (slot attivi, tutti i tipi)

Usa il checkpoint di training ``model_log1`` e il path ``full_train_e_f``
*senza* ``apply_gradients``.

Esempio:

    python run_grad_param_regression.py --precision double --n-per-family 100
    python run_grad_param_regression.py --precision float --n-per-family 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
TRAIN = ROOT.parents[1] / "test-training-pipeline"

ANG_COMP_NAMES = {0: "beta", 1: "gamma", 2: "delta"}
FAMILIES = (
    "dense_kernel",
    "dense_bias",
    "alpha2b",
    "alpha3b_beta",
    "alpha3b_gamma",
    "alpha3b_delta",
)


def _make_typemap(tipos):
    list_tmap = []
    num = 0
    for el in tipos:
        for _ in range(int(el)):
            list_tmap.append(num)
        num += 1
    return list_tmap


def _read_cutoff(full_param):
    rs = float(full_param["Rs"])
    rc = float(full_param["Rc"])
    rad_buff = int(full_param["Radial_Buffer"])
    rc_ang = float(full_param["Rc_Angular"])
    maxneigh = int(full_param["Max_Angular_Neigh"])
    ang_buff = int(maxneigh * (maxneigh - 1) / 2)
    return rc, rad_buff, rc_ang, ang_buff, rs


def _unravel(shape, flat):
    return tuple(int(x) for x in np.unravel_index(int(flat), shape))


def _sample_slots(slots, n, rng):
    """Sample up to n slots without replacement. slots: list of dicts."""
    if not slots:
        return []
    n_take = min(int(n), len(slots))
    idx = rng.choice(len(slots), size=n_take, replace=False)
    return [slots[int(i)] for i in idx]


def _build_forward_fn(model, tf, mse):
    @tf.function
    def forward_ef(
        x1,
        x2,
        x3bsupp,
        int2b,
        intder2b,
        int3b,
        intder3b,
        intder3bsupp,
        numtriplet,
        etrue,
        ftrue,
    ):
        nt = model.ntipos
        x2b = tf.split(x1, model.tipos, axis=1)
        x3b = tf.split(x2, model.tipos, axis=1)
        x3bsupp_s = tf.split(x3bsupp, model.tipos, axis=1)
        int2b_s = tf.split(int2b, model.tipos, axis=1)
        int3b_s = tf.split(int3b, model.tipos, axis=1)
        intder2b_s = tf.split(intder2b, model.tipos, axis=1)
        intder3b_s = tf.split(intder3b, model.tipos, axis=1)
        intder3bsupp_s = tf.split(intder3bsupp, model.tipos, axis=1)
        numtriplet_s = tf.split(numtriplet, model.tipos, axis=1)

        fingerprint = [
            model.physics_layer[k](
                x2b[k],
                x3bsupp_s[k],
                int2b_s[k],
                x3b[k],
                int3b_s[k],
                numtriplet_s[k],
                model.type_map,
            )
            for k in range(nt)
        ]
        log_norm = [
            model.lognorm_layer[k](finger) for k, finger in enumerate(fingerprint)
        ]
        energy = [model.nets[k](cp) for k, cp in enumerate(log_norm)]
        grad_ene = [tf.gradients(energy[k], fingerprint[k]) for k in range(nt)]
        totene = tf.concat(energy, axis=1)
        totenergy = tf.reduce_mean(totene, axis=(-1, -2)) * 0.5

        grad_listed = [
            tf.split(
                grad_ene[k][0],
                [
                    model.physics_layer[k].nalpha_r,
                    model.physics_layer[k].nalpha_a,
                ],
                axis=2,
            )
            for k in range(nt)
        ]
        force_list = [
            model.force_layer(
                grad_listed[k][0],
                x2b[k],
                intder2b_s[k],
                int2b_s[k],
                model.physics_layer[k].alpha2b,
                grad_listed[k][1],
                x3b[k],
                x3bsupp_s[k],
                intder3b_s[k],
                intder3bsupp_s[k],
                int3b_s[k],
                numtriplet_s[k],
                model.physics_layer[k].alpha3b,
                model.physics_layer[k].type_emb_2b,
                model.physics_layer[k].type_emb_3b,
                model.type_map,
                model.tipos,
                k,
            )
            for k in range(nt)
        ]
        force = tf.math.add_n(force_list)
        return mse(totenergy, etrue), mse(force, ftrue)

    return forward_ef


def _build_grad_fn(model, tf, mse, param_vars):
    @tf.function
    def losses_and_grads(
        x1,
        x2,
        x3bsupp,
        int2b,
        intder2b,
        int3b,
        intder3b,
        intder3bsupp,
        numtriplet,
        etrue,
        ftrue,
    ):
        nt = model.ntipos
        x2b = tf.split(x1, model.tipos, axis=1)
        x3b = tf.split(x2, model.tipos, axis=1)
        x3bsupp_s = tf.split(x3bsupp, model.tipos, axis=1)
        int2b_s = tf.split(int2b, model.tipos, axis=1)
        int3b_s = tf.split(int3b, model.tipos, axis=1)
        intder2b_s = tf.split(intder2b, model.tipos, axis=1)
        intder3b_s = tf.split(intder3b, model.tipos, axis=1)
        intder3bsupp_s = tf.split(intder3bsupp, model.tipos, axis=1)
        numtriplet_s = tf.split(numtriplet, model.tipos, axis=1)

        fingerprint = [
            model.physics_layer[k](
                x2b[k],
                x3bsupp_s[k],
                int2b_s[k],
                x3b[k],
                int3b_s[k],
                numtriplet_s[k],
                model.type_map,
            )
            for k in range(nt)
        ]
        log_norm = [
            model.lognorm_layer[k](finger) for k, finger in enumerate(fingerprint)
        ]
        energy = [model.nets[k](cp) for k, cp in enumerate(log_norm)]
        grad_ene = [tf.gradients(energy[k], fingerprint[k]) for k in range(nt)]
        totene = tf.concat(energy, axis=1)
        totenergy = tf.reduce_mean(totene, axis=(-1, -2)) * 0.5

        grad_listed = [
            tf.split(
                grad_ene[k][0],
                [
                    model.physics_layer[k].nalpha_r,
                    model.physics_layer[k].nalpha_a,
                ],
                axis=2,
            )
            for k in range(nt)
        ]
        force_list = [
            model.force_layer(
                grad_listed[k][0],
                x2b[k],
                intder2b_s[k],
                int2b_s[k],
                model.physics_layer[k].alpha2b,
                grad_listed[k][1],
                x3b[k],
                x3bsupp_s[k],
                intder3b_s[k],
                intder3bsupp_s[k],
                int3b_s[k],
                numtriplet_s[k],
                model.physics_layer[k].alpha3b,
                model.physics_layer[k].type_emb_2b,
                model.physics_layer[k].type_emb_3b,
                model.type_map,
                model.tipos,
                k,
            )
            for k in range(nt)
        ]
        force = tf.math.add_n(force_list)
        loss_E = mse(totenergy, etrue)
        loss_F = mse(force, ftrue)
        grads_E = [tf.gradients(loss_E, v)[0] for v in param_vars]
        grads_F = [tf.gradients(loss_F, v)[0] for v in param_vars]
        return loss_E, loss_F, grads_E, grads_F

    return losses_and_grads


def _candidate_dense(model, kind):
    """kind in {'kernel','bias'} → lista (type, var, flat_idx, multi_index)."""
    out = []
    for t, net in enumerate(model.nets):
        for v in net.trainable_variables:
            if kind == "kernel" and "kernel" not in v.name:
                continue
            if kind == "bias" and "bias" not in v.name:
                continue
            n = int(np.prod(v.shape))
            for flat in range(n):
                out.append(
                    {
                        "variable": v,
                        "type_atom": t,
                        "flat": flat,
                        "index": _unravel(tuple(v.shape), flat),
                    }
                )
    return out


def _candidate_alpha2b(model):
    out = []
    for t in range(model.ntipos):
        v = model.physics_layer[t].alpha2b
        n = int(np.prod(v.shape))
        for flat in range(n):
            out.append(
                {
                    "variable": v,
                    "type_atom": t,
                    "flat": flat,
                    "index": _unravel(tuple(v.shape), flat),
                }
            )
    return out


def _build_probes_dense_and_radial(model, rng, n_per_family):
    probes = []
    counts = {}

    for family, kind in (
        ("dense_kernel", "kernel"),
        ("dense_bias", "bias"),
    ):
        slots = _candidate_dense(model, kind)
        picked = _sample_slots(slots, n_per_family, rng)
        counts[family] = {
            "requested": n_per_family,
            "available": len(slots),
            "used": len(picked),
        }
        for i, s in enumerate(picked):
            probes.append(
                {
                    "name": f"{family}_{i:03d}",
                    "family": family,
                    "variable": s["variable"],
                    "index": s["index"],
                    "type_atom": s["type_atom"],
                }
            )

    slots2 = _candidate_alpha2b(model)
    picked2 = _sample_slots(slots2, n_per_family, rng)
    counts["alpha2b"] = {
        "requested": n_per_family,
        "available": len(slots2),
        "used": len(picked2),
    }
    for i, s in enumerate(picked2):
        probes.append(
            {
                "name": f"alpha2b_{i:03d}",
                "family": "alpha2b",
                "variable": s["variable"],
                "index": s["index"],
                "type_atom": s["type_atom"],
            }
        )

    # Placeholder angular: filled after analytic grads.
    for comp in (0, 1, 2):
        family = f"alpha3b_{ANG_COMP_NAMES[comp]}"
        counts[family] = {
            "requested": n_per_family,
            "available": None,
            "used": 0,
        }
        # one stub probe per type-variable so unique_vars includes all alpha3b
        # (actual probes added in _fill_active_alpha3b_family)
        _ = family

    return probes, counts


def _fill_active_alpha3b_family(
    model, grads_E, grads_F, var_id_to_slot, rng, n_per_family, eps=1e-12
):
    """Per β/γ/δ prende slot energy-attivi (|g_E|>eps), top-|g_E|+|g_F|."""
    probes = []
    counts = {}
    for comp in (0, 1, 2):
        family = f"alpha3b_{ANG_COMP_NAMES[comp]}"
        slots = []
        for t in range(model.ntipos):
            v = model.physics_layer[t].alpha3b
            slot = var_id_to_slot[id(v)]
            gE = np.asarray(grads_E[slot].numpy(), dtype=np.float64)
            gF = np.asarray(grads_F[slot].numpy(), dtype=np.float64)
            nt_couple, width = gE.shape
            nalpha_a = width // 3
            for couple in range(nt_couple):
                for iaf in range(nalpha_a):
                    col = 3 * iaf + comp
                    gE_abs = float(np.abs(gE[couple, col]))
                    gF_abs = float(np.abs(gF[couple, col]))
                    slots.append(
                        {
                            "variable": v,
                            "type_atom": t,
                            "index": (couple, col),
                            "couple": int(couple),
                            "iaf": int(iaf),
                            "ang_component": comp,
                            "ang_component_name": ANG_COMP_NAMES[comp],
                            "activity": gE_abs + gF_abs,
                            "activity_E": gE_abs,
                        }
                    )
        # Require energy-active AF (guards against ghost g_F on unused couples).
        e_active = [s for s in slots if s["activity_E"] > eps]
        e_active.sort(key=lambda s: s["activity"], reverse=True)
        n_take = min(int(n_per_family), len(e_active) if e_active else len(slots))
        pool_src = e_active if e_active else slots
        pool_src = sorted(pool_src, key=lambda s: s["activity"], reverse=True)
        pool = pool_src[: max(n_take, min(len(pool_src), 2 * n_take))]
        if len(pool) >= n_take and n_take > 0:
            picked = _sample_slots(pool, n_take, rng)
        else:
            picked = pool_src[:n_take]
        counts[family] = {
            "requested": n_per_family,
            "available": len(slots),
            "n_active_eps": int(sum(s["activity"] > eps for s in slots)),
            "n_energy_active": len(e_active),
            "used": len(picked),
        }
        for i, s in enumerate(picked):
            probes.append(
                {
                    "name": f"{family}_{i:03d}",
                    "family": family,
                    **{k: v for k, v in s.items() if k != "activity_E"},
                }
            )
    return probes, counts


def _corr_metrics(g_ana, g_num, outlier_rel=50.0):
    """Corr/MAE ignorando NaN/Inf e outlier FD (|err| >> |g_ana|)."""
    g_ana = np.asarray(g_ana, dtype=np.float64)
    g_num = np.asarray(g_num, dtype=np.float64)
    ok = np.isfinite(g_ana) & np.isfinite(g_num)
    scale = np.maximum(np.abs(g_ana), 1e-8)
    ok &= np.abs(g_num - g_ana) <= outlier_rel * scale + 1.0
    n_total = int(g_ana.size)
    n_used = int(ok.sum())
    if n_used < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), n_used, n_total
    ga = g_ana[ok]
    gn = g_num[ok]
    corr = float(np.corrcoef(ga, gn)[0, 1])
    denom = float(np.dot(ga, ga)) + 1e-30
    slope = float(np.dot(ga, gn) / denom)
    mae = float(np.mean(np.abs(gn - ga)))
    rmse = float(np.sqrt(np.mean((gn - ga) ** 2)))
    return corr, slope, mae, rmse, n_used, n_total


def _build_model(precision: str, n_frames: int, tf):
    run_dir = TRAIN / f"run_{precision}"
    yaml_path = run_dir / "input_4test.yaml"
    model_dir = run_dir / "model_log1"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint: {model_dir}")

    code_root = REPO / "STAF"
    sys.path = [str(code_root)] + [p for p in sys.path if p != str(code_root)]

    dtype_str = "float32" if precision == "float" else "float64"
    np_dtype = np.float32 if precision == "float" else np.float64
    from staf.dtype import set_precision
    set_precision(precision)
    assert tf.keras.backend.floatx() == dtype_str

    from gradient_utility import register_force_3bAFs_grad  # noqa: F401
    from gradient_utility import register_force_2bAFs_grad  # noqa: F401
    from gradient_utility import register_3bAFs_grad  # noqa: F401
    from gradient_utility import register_2bAFs_grad  # noqa: F401

    from staf_models.staf_model import staf_full
    from source_routine.physics_layer_mod import physics_layer, lognorm_layer
    from source_routine.force_layer_mod import force_layer
    from source_routine.descriptor_builder import descriptor_layer
    from init_params.init_AFs_param import init_AFs_param

    with open(yaml_path) as fh:
        full_param = yaml.load(fh, Loader=yaml.FullLoader)

    dataset_dir = run_dir / "dataset"
    tipos_raw = np.loadtxt(dataset_dir / "type.dat", dtype=int).reshape(-1)
    tipos = [int(x) for x in tipos_raw]
    type_map = _make_typemap(tipos)
    nt = len(tipos)
    N = len(type_map)

    seed_par = int(full_param.get("Seed", 60))
    np.random.seed(seed_par)
    tf.random.set_seed(seed_par + 1)

    e_all = np.load(dataset_dir / "test" / "energy.npy").astype(np_dtype)
    f_all = np.load(dataset_dir / "test" / "force.npy").astype(np_dtype)
    pos_all = np.load(dataset_dir / "test" / "pos.npy").astype(np_dtype)
    box_all = np.load(dataset_dir / "test" / "box.npy").astype(np_dtype)

    n_frames = min(n_frames, pos_all.shape[0])
    frame_idx = np.arange(n_frames, dtype=int)

    actfun = full_param["activation_function"]
    nhl = int(full_param["number_of_decoding_layers"])
    nD = [int(k) for k in full_param["number_of_decoding_nodes"].split()]
    alpha_bound = float(full_param.get("alpha_bound", 5.0))

    restart = str(model_dir.resolve())
    opt_net = tf.keras.optimizers.Adam(1e-3)
    opt_phys = tf.keras.optimizers.Adam(1e-3)

    rng_state = np.random.get_state()
    init_alpha2b, init_alpha3b, init_mu, initial_type_emb, new_rng = init_AFs_param(
        restart, full_param, nt, rng_state
    )
    np.random.set_state(new_rng)

    rc, rad_buff, rc_ang, ang_buff, Rs = _read_cutoff(full_param)
    Descriptor_Layer = descriptor_layer(
        rc, rad_buff, rc_ang, ang_buff, N, box_all[0], Rs, max(n_frames, 1)
    )
    Physics_Layers = [
        physics_layer(init_alpha2b[k], init_alpha3b[k], initial_type_emb[k])
        for k in range(nt)
    ]
    Lognorm_Layers = [lognorm_layer(init_mu[k]) for k in range(nt)]
    Force_Layer = force_layer(rad_buff, ang_buff)

    def mse(ypred, y):
        return tf.reduce_mean(tf.square(ypred - y))

    model = staf_full(
        Physics_Layers,
        Force_Layer,
        nhl,
        nD,
        actfun,
        1,
        mse,
        mse,
        opt_net,
        opt_phys,
        alpha_bound,
        Lognorm_Layers,
        tipos,
        type_map,
        restart,
        seed_par,
    )

    pos = tf.constant(pos_all[frame_idx])
    box = tf.constant(box_all[frame_idx])
    etrue = tf.constant(e_all[frame_idx])
    ftrue = tf.constant(f_all[frame_idx])
    desc = Descriptor_Layer(pos, box)
    desc_pack = (
        desc[0],
        desc[1],
        desc[2],
        desc[3],
        desc[5],
        desc[4],
        desc[6],
        desc[7],
        desc[8],
    )
    meta = {
        "run_dir": str(run_dir),
        "yaml": str(yaml_path),
        "model_dir": restart,
        "code_root": str(code_root),
        "n_frames": n_frames,
        "frame_indices": frame_idx.tolist(),
        "dtype": dtype_str,
        "N_atoms": N,
        "ntipos": nt,
    }
    return model, desc_pack, etrue, ftrue, mse, meta


def main():
    p = argparse.ArgumentParser(
        description=(
            "Regressione FD dei gradienti di Loss_E / Loss_F rispetto a "
            "pesi densità e parametri AF (path training, MSE)."
        )
    )
    p.add_argument("--precision", choices=["float", "double"], required=True)
    p.add_argument("--n-frames", type=int, default=16)
    p.add_argument(
        "--n-per-family",
        type=int,
        default=100,
        help="quanti punti per famiglia (dense_kernel, dense_bias, alpha2b, β/γ/δ)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[1e-2, 1e-3, 1e-4],
    )
    args = p.parse_args()

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    model, desc, etrue, ftrue, mse, meta = _build_model(
        args.precision, args.n_frames, tf
    )
    rng = np.random.default_rng(args.seed)

    probes, family_counts = _build_probes_dense_and_radial(
        model, rng, args.n_per_family
    )

    # Include tutte le Variable alpha3b per i gradienti analitici.
    unique_vars = []
    var_id_to_slot = {}

    def _register_var(v):
        vid = id(v)
        if vid not in var_id_to_slot:
            var_id_to_slot[vid] = len(unique_vars)
            unique_vars.append(v)
        return var_id_to_slot[vid]

    for pr in probes:
        pr["var_slot"] = _register_var(pr["variable"])
    for t in range(model.ntipos):
        _register_var(model.physics_layer[t].alpha3b)

    forward_ef = _build_forward_fn(model, tf, mse)
    losses_and_grads = _build_grad_fn(model, tf, mse, unique_vars)

    print(
        f"flags: precision={args.precision} n_frames={meta['n_frames']} "
        f"n_per_family={args.n_per_family} seed={args.seed} deltas={args.deltas}"
    )
    print(f"model_dir={meta['model_dir']}")

    _ = forward_ef(*desc, etrue, ftrue)
    loss_E0, loss_F0, grads_E, grads_F = losses_and_grads(*desc, etrue, ftrue)
    loss_E0_f = float(loss_E0.numpy())
    loss_F0_f = float(loss_F0.numpy())

    ang_probes, ang_counts = _fill_active_alpha3b_family(
        model,
        grads_E,
        grads_F,
        var_id_to_slot,
        rng,
        args.n_per_family,
        eps=1e-12,
    )
    for pr in ang_probes:
        pr["var_slot"] = var_id_to_slot[id(pr["variable"])]
    probes.extend(ang_probes)
    family_counts.update(ang_counts)

    print("family counts:")
    for fam in FAMILIES:
        c = family_counts[fam]
        print(
            f"  {fam}: used={c['used']} available={c['available']} "
            f"requested={c['requested']}"
        )
    print(f"total probes={len(probes)}")
    print(f"L_E(w)={loss_E0_f:.8e}  L_F(w)={loss_F0_f:.8e}")

    # Analytic scalars (numpy cache per Variable).
    grad_cache_E = {}
    grad_cache_F = {}
    for slot, v in enumerate(unique_vars):
        if grads_E[slot] is None or grads_F[slot] is None:
            raise RuntimeError(f"Gradiente None per Variable {v.name}")
        grad_cache_E[slot] = np.asarray(grads_E[slot].numpy(), dtype=np.float64)
        grad_cache_F[slot] = np.asarray(grads_F[slot].numpy(), dtype=np.float64)

    ana = {}
    for pr in probes:
        slot = pr["var_slot"]
        idx = pr["index"]
        ana[pr["name"]] = {
            "loss_E": float(grad_cache_E[slot][idx]),
            "loss_F": float(grad_cache_F[slot][idx]),
        }

    out_dir = ROOT / f"results_{args.precision}"
    out_dir.mkdir(parents=True, exist_ok=True)

    flags = {
        "precision": args.precision,
        "n_frames": meta["n_frames"],
        "n_per_family": args.n_per_family,
        "frame_indices": meta["frame_indices"],
        "seed": args.seed,
        "deltas": [float(d) for d in args.deltas],
        "model_dir": meta["model_dir"],
        "yaml": meta["yaml"],
        "code_root": meta["code_root"],
        "loss": "MSE (energy and force separately)",
        "method": "forward FD  g_num=(L(w+dw)-L(w))/dw",
        "analytic": "tf.gradients mirroring full_train_e_f (no apply)",
        "family_counts": family_counts,
    }

    results = {
        "flags": flags,
        "L_E": loss_E0_f,
        "L_F": loss_F0_f,
        "family_counts": family_counts,
        "n_probes": len(probes),
        "by_delta": [],
    }

    for delta in args.deltas:
        print(f"\n=== δ={delta:g}  ({len(probes)} probes) ===")
        gE_ana = {f: [] for f in FAMILIES}
        gE_num = {f: [] for f in FAMILIES}
        gF_ana = {f: [] for f in FAMILIES}
        gF_num = {f: [] for f in FAMILIES}

        for k, pr in enumerate(probes):
            var = pr["variable"]
            idx = pr["index"]
            w0 = var.numpy().copy()
            ge_a = ana[pr["name"]]["loss_E"]
            gf_a = ana[pr["name"]]["loss_F"]

            ge_n = float("nan")
            gf_n = float("nan")
            for _attempt in range(2):
                w1 = w0.copy()
                w1[idx] = w0[idx] + delta
                var.assign(w1)
                loss_E1, loss_F1 = forward_ef(*desc, etrue, ftrue)
                var.assign(w0)
                e1 = float(loss_E1.numpy())
                f1 = float(loss_F1.numpy())
                if not (np.isfinite(e1) and np.isfinite(f1)):
                    var.assign(w0)
                    _ = forward_ef(*desc, etrue, ftrue)
                    continue
                ge_n = (e1 - loss_E0_f) / delta
                gf_n = (f1 - loss_F0_f) / delta
                # Glitch tipico: L_F(w+dw)≈0 → |g_num|≈L_F/dw
                ghost = abs(loss_F0_f) / max(abs(delta), 1e-30)
                bad_F = (not np.isfinite(gf_n)) or (
                    abs(gf_n) > 20.0 * max(abs(gf_a), 1e-6) + 1.0
                    and abs(abs(gf_n) - ghost) < 0.25 * ghost
                )
                if not bad_F:
                    break
                var.assign(w0)
                _ = forward_ef(*desc, etrue, ftrue)
                ge_n = float("nan")
                gf_n = float("nan")

            fam = pr["family"]
            gE_ana[fam].append(ge_a)
            gE_num[fam].append(ge_n)
            gF_ana[fam].append(gf_a)
            gF_num[fam].append(gf_n)

            if (k + 1) % 50 == 0 or k == 0:
                print(f"  {k+1}/{len(probes)}  last={pr['name']}")

        fam_stats = []
        for fam in FAMILIES:
            cE, sE, maeE, rmseE, nE, nEtot = _corr_metrics(gE_ana[fam], gE_num[fam])
            cF, sF, maeF, rmseF, nF, nFtot = _corr_metrics(gF_ana[fam], gF_num[fam])
            entry = {
                "family": fam,
                "n": len(gE_ana[fam]),
                "loss_E": {
                    "correlation": cE,
                    "slope": sE,
                    "mae": maeE,
                    "rmse": rmseE,
                    "n_used": nE,
                    "n_total": nEtot,
                    "g_ana": gE_ana[fam],
                    "g_num": gE_num[fam],
                },
                "loss_F": {
                    "correlation": cF,
                    "slope": sF,
                    "mae": maeF,
                    "rmse": rmseF,
                    "n_used": nF,
                    "n_total": nFtot,
                    "g_ana": gF_ana[fam],
                    "g_num": gF_num[fam],
                },
            }
            fam_stats.append(entry)
            print(
                f"  {fam}: n={entry['n']}  "
                f"E corr={cE:.6f} mae={maeE:.3e} (used {nE}/{nEtot})  "
                f"F corr={cF:.6f} mae={maeF:.3e} (used {nF}/{nFtot})"
            )

        results["by_delta"].append({"delta": float(delta), "families": fam_stats})

        np.savez_compressed(
            out_dir / f"grad_fd_delta_{delta:g}.npz",
            delta=delta,
            **{
                f"{fam}_E_ana": np.asarray(gE_ana[fam])
                for fam in FAMILIES
            },
            **{
                f"{fam}_E_num": np.asarray(gE_num[fam])
                for fam in FAMILIES
            },
            **{
                f"{fam}_F_ana": np.asarray(gF_ana[fam])
                for fam in FAMILIES
            },
            **{
                f"{fam}_F_num": np.asarray(gF_num[fam])
                for fam in FAMILIES
            },
        )

    # summary senza vettori lunghi
    summary_light = {
        "flags": flags,
        "L_E": loss_E0_f,
        "L_F": loss_F0_f,
        "family_counts": family_counts,
        "n_probes": len(probes),
        "by_delta": [
            {
                "delta": de["delta"],
                "families": [
                    {
                        "family": f["family"],
                        "n": f["n"],
                        "loss_E": {
                            k: f["loss_E"][k]
                            for k in (
                                "correlation",
                                "slope",
                                "mae",
                                "rmse",
                                "n_used",
                                "n_total",
                            )
                        },
                        "loss_F": {
                            k: f["loss_F"][k]
                            for k in (
                                "correlation",
                                "slope",
                                "mae",
                                "rmse",
                                "n_used",
                                "n_total",
                            )
                        },
                    }
                    for f in de["families"]
                ],
            }
            for de in results["by_delta"]
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_light, indent=2) + "\n")

    lines = [
        "# Input flags (CLI)",
        f"precision={args.precision}",
        f"n_frames={meta['n_frames']}",
        f"n_per_family={args.n_per_family}",
        f"seed={args.seed}",
        f"deltas={' '.join(str(float(d)) for d in args.deltas)}",
        f"model_dir={meta['model_dir']}",
        f"yaml={meta['yaml']}",
        f"code_root={meta['code_root']}",
        "loss=MSE (Loss_E and Loss_F separately)",
        "method=forward FD  g_num=(L(w+dw)-L(w))/dw",
        "",
        "# Family counts",
    ]
    for fam in FAMILIES:
        c = family_counts[fam]
        lines.append(
            f"{fam}: used={c['used']} available={c['available']} "
            f"requested={c['requested']}"
        )
    lines += [
        "",
        "# Reference losses",
        f"L_E={loss_E0_f:.12e}",
        f"L_F={loss_F0_f:.12e}",
        "",
        "# Results per delta / family",
    ]
    for de in summary_light["by_delta"]:
        lines.append(f"delta={de['delta']:g}")
        for f in de["families"]:
            lines.append(
                f"  {f['family']} n={f['n']}  "
                f"E: corr={f['loss_E']['correlation']:.8f} "
                f"slope={f['loss_E']['slope']:.8f} "
                f"mae={f['loss_E']['mae']:.6e} "
                f"rmse={f['loss_E']['rmse']:.6e} "
                f"used={f['loss_E']['n_used']}/{f['loss_E']['n_total']}  "
                f"F: corr={f['loss_F']['correlation']:.8f} "
                f"slope={f['loss_F']['slope']:.8f} "
                f"mae={f['loss_F']['mae']:.6e} "
                f"rmse={f['loss_F']['rmse']:.6e} "
                f"used={f['loss_F']['n_used']}/{f['loss_F']['n_total']}"
            )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")

    # Plots: corr vs delta + scatter at middle delta.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        deltas = [float(d) for d in args.deltas]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, channel, title in (
            (axes[0], "loss_E", r"Loss$_E$"),
            (axes[1], "loss_F", r"Loss$_F$"),
        ):
            for fam in FAMILIES:
                corrs = []
                for de in summary_light["by_delta"]:
                    rec = next(x for x in de["families"] if x["family"] == fam)
                    corrs.append(rec[channel]["correlation"])
                ax.semilogx(deltas, corrs, "o-", label=fam)
            ax.set_xlabel(r"$dw$")
            ax.set_ylabel("corr($g_{ana}$, $g_{num}$)")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7, loc="best")
        fig.suptitle(
            f"STAF param-grad corr ({args.precision})  "
            f"n_per_family≈{args.n_per_family}  n_frames={meta['n_frames']}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "grad_param_corr_vs_delta.png", dpi=150)

        # Scatter at preferred delta (1e-3 if present, else middle).
        pref = 1e-3 if 1e-3 in deltas else deltas[len(deltas) // 2]
        de = next(x for x in results["by_delta"] if abs(x["delta"] - pref) < 1e-15)
        nfam = len(FAMILIES)
        fig2, axes2 = plt.subplots(2, nfam, figsize=(3.1 * nfam, 6.2), squeeze=False)
        for j, fam in enumerate(FAMILIES):
            rec = next(x for x in de["families"] if x["family"] == fam)
            for i, channel, ylab in (
                (0, "loss_E", r"$g_E$"),
                (1, "loss_F", r"$g_F$"),
            ):
                ax = axes2[i, j]
                ga = np.asarray(rec[channel]["g_ana"])
                gn = np.asarray(rec[channel]["g_num"])
                ax.scatter(ga, gn, s=10, alpha=0.7, edgecolors="none")
                lim = max(np.max(np.abs(ga)), np.max(np.abs(gn)), 1e-12)
                ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(
                    f"{fam}\ncorr={rec[channel]['correlation']:.4f}", fontsize=8
                )
                if i == 1:
                    ax.set_xlabel(r"$g_{\mathrm{ana}}$")
                if j == 0:
                    ax.set_ylabel(ylab + r" $g_{\mathrm{num}}$")
                ax.grid(True, alpha=0.3)
        fig2.suptitle(
            f"STAF param-grad scatter ({args.precision})  dw={pref:g}  "
            f"n_frames={meta['n_frames']}",
            fontsize=11,
        )
        fig2.tight_layout()
        fig2.savefig(out_dir / "grad_param_scatter.png", dpi=140)
        print(f"Wrote {out_dir / 'grad_param_corr_vs_delta.png'}")
        print(f"Wrote {out_dir / 'grad_param_scatter.png'}")
    except ImportError:
        print("matplotlib non disponibile: skip plot")

    print(f"Wrote {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
