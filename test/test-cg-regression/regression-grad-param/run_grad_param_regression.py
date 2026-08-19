#!/usr/bin/env python3
"""Parameter-gradient FD vs analytic grads on the STAF-CG training path.

Compares MSE Loss_F (same convention as STAF regression-grad-param):

    g_num = (L(w+dw) - L(w)) / dw
    g_ana = ∂L_F/∂w   via tf.gradients (no apply_gradients, no alpha bound)

Families: dense_kernel, dense_bias, alpha2b, alpha3b.
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
PIPE = ROOT.parents[1] / "test-cg-pipeline"

FAMILIES = ("dense_kernel", "dense_bias", "alpha2b", "alpha3b")
CORR_PASS = 0.95
DW_PASS = 1e-3


def _unravel(shape, flat):
    return tuple(int(x) for x in np.unravel_index(int(flat), shape))


def _sample_top(g_all, n_take, rng, *, extra_mask=None, gmax=50.0, gmin=1e-12):
    g_all = np.asarray(g_all, dtype=np.float64)
    mask = np.isfinite(g_all) & (np.abs(g_all) > gmin) & (np.abs(g_all) < gmax)
    if extra_mask is not None:
        mask = mask & extra_mask
    pool = np.flatnonzero(mask)
    if pool.size == 0:
        return np.array([], dtype=np.int64)
    mag = np.abs(g_all[pool])
    order = np.argsort(-mag)
    n_take = min(int(n_take), pool.size)
    # mix top-magnitude with a little randomness among the top 3n
    top = order[: max(n_take * 3, n_take)]
    pick = rng.choice(top, size=n_take, replace=False) if top.size > n_take else top
    return pool[np.sort(pick)]


def _corr_metrics(g_ana, g_num, outlier_rel=50.0):
    g_ana = np.asarray(g_ana, dtype=np.float64)
    g_num = np.asarray(g_num, dtype=np.float64)
    ok = np.isfinite(g_ana) & np.isfinite(g_num)
    scale = np.maximum(np.abs(g_ana), 1e-8)
    ok &= np.abs(g_num - g_ana) <= outlier_rel * scale + 1.0
    n_used = int(ok.sum())
    if n_used < 2:
        return float("nan"), float("nan"), n_used
    ga = g_ana[ok]
    gn = g_num[ok]
    corr = float(np.corrcoef(ga, gn)[0, 1])
    mae = float(np.mean(np.abs(gn - ga)))
    return corr, mae, n_used


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--precision", choices=["float", "double"], required=True)
    p.add_argument("--n-per-family", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dws", type=float, nargs="+", default=[1e-3, 1e-4])
    p.add_argument("--yaml", type=Path, default=PIPE / "work" / "input_epoch1.yaml")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--n-frames", type=int, default=8)
    args = p.parse_args()

    sys.path.insert(0, str(REPO / "STAF-CG"))
    sys.path.insert(1, str(REPO / "STAF"))

    import tensorflow as tf
    from staf.dtype import set_precision, tf_dtype, np_dtype
    from staf_cg_paths import set_ops_root

    set_precision(args.precision)
    set_ops_root(args.precision)
    dt = np_dtype()

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(exc)

    from source_routine.descriptor_builder import descriptor_layer
    from source_routine.force_layer_mod import force_layer
    from source_routine.physics_layer_mod import physics_layer, lognorm_layer
    from staf_cg_models.alpha_nes_model import alpha_nes_full
    from staf_cg_harness import USCGSITE
    from init_params.init_AFs_param import init_AFs_param
    from optimizer_learning_rate_utility import build_learning_rate, build_optimizer
    from gradient_utility import register_force_3bAFs_grad  # noqa: F401
    from gradient_utility import register_force_2bAFs_grad  # noqa: F401
    from gradient_utility import register_3bAFs_grad  # noqa: F401
    from gradient_utility import register_2bAFs_grad  # noqa: F401

    def read_cutoff_info(full_param):
        rs = float(full_param["Rs"])
        rc = float(full_param["Rc"])
        rc_inter = float(full_param["Rc_Inter"])
        rs_inter = float(full_param["Rs_Inter"])
        ra_inter = float(full_param["Rc_Angular_Inter"])
        rad_buff = int(full_param["Radial_Buffer"])
        rc_ang = float(full_param["Rc_Angular"])
        maxneigh = int(full_param["Max_Angular_Neigh"])
        ang_buff = int(maxneigh * (maxneigh - 1) / 2)
        return rc, rad_buff, rc_ang, ang_buff, rs, rs_inter, rc_inter, ra_inter

    def make_loss(full_param):
        def mse(ypred, y):
            return tf.reduce_mean(tf.square((ypred - y)))

        if str(full_param.get("loss_method", "huber")) == "huber":
            model_loss = tf.keras.losses.Huber(
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
            )
        else:
            model_loss = mse
        pe = tf.constant(float(full_param.get("loss_energy_prefactor", 1.0)), dtype=tf_dtype())
        pf = tf.constant(float(full_param.get("loss_force_prefactor", 1.0)), dtype=tf_dtype())
        pb = tf.constant(1.0, dtype=tf_dtype())
        return model_loss, mse, pe, pf, pb

    full_param = yaml.safe_load(args.yaml.read_text())
    ckpt = args.checkpoint
    if ckpt is None:
        work = PIPE / "work"
        cands = sorted(work.glob("staf_cg_freeze_ep1*"))
        cands = [c for c in cands if c.is_dir() and (c / "net_model_type0").exists()]
        if not cands:
            raise SystemExit(f"no checkpoint with net_model_type0 under {work}")
        ckpt = cands[-1]
    ckpt = Path(ckpt)
    print("checkpoint", ckpt)

    full_param["restart"] = str(ckpt)
    full_param["color_interaction_file"] = str(ckpt / "map_color_interaction.dat")
    full_param["map_intra_file"] = str(ckpt / "map_intra.dat")

    src = USCGSITE / "dataset" / "training"
    n = args.n_frames
    pos = np.asarray(np.load(src / "pos.npy", mmap_mode="r")[:n], dtype=dt)
    box = np.asarray(np.load(src / "box.npy", mmap_mode="r")[:n], dtype=dt)
    energy = np.asarray(np.load(src / "energy.npy", mmap_mode="r")[:n], dtype=dt)
    force = np.asarray(np.load(src / "force.npy", mmap_mode="r")[:n], dtype=dt)
    if pos.ndim == 3:
        pos = pos.reshape(pos.shape[0], -1)
    if force.ndim == 3:
        force = force.reshape(force.shape[0], -1)

    color_type_map = np.loadtxt(ckpt / "color_type_map.dat", dtype=np.int32).reshape(-1)
    map_intra = np.loadtxt(ckpt / "map_intra.dat", dtype=np.int32).reshape((-1, 1))
    n_part = int(color_type_map.shape[0])

    rc, rad_buff, rc_ang, ang_buff, Rs, rs_inter, rc_inter, ra_inter = read_cutoff_info(
        full_param
    )
    Descriptor = descriptor_layer(
        rc, rad_buff, rc_ang, ang_buff, n_part, box[0], Rs, n, rs_inter, rc_inter, ra_inter
    )
    descr = Descriptor(tf.constant(pos), tf.constant(box), tf.constant(map_intra))

    number_of_interaction = 3
    rng_state = np.random.RandomState(int(full_param.get("Seed", 60))).get_state()
    init_alpha2b, init_alpha3b, init_mu, initial_type_emb, _ = init_AFs_param(
        str(ckpt), full_param, number_of_interaction, rng_state
    )
    Physics = [
        physics_layer(init_alpha2b[k], init_alpha3b[k], initial_type_emb[k])
        for k in range(len(init_alpha2b))
    ]
    Lognorm = [lognorm_layer(init_mu[k]) for k in range(len(init_mu))]
    Force = force_layer(rad_buff, ang_buff)
    model_loss, val_loss, pe, pf, pb = make_loss(full_param)
    lr_net = build_learning_rate(full_param["lr_dense_net"].split(), 1, 1, 1, "net", 0)
    opt_net = build_optimizer(full_param["optimizer_net"].split(), lr_net, 0)
    model = alpha_nes_full(
        Physics,
        Force,
        1,
        model_loss,
        val_loss,
        opt_net,
        float(full_param.get("alpha_bound", 7.0)),
        Lognorm,
        color_type_map,
        str(ckpt),
        int(full_param.get("Seed", 60)),
        full_param,
    )

    tensors = {
        "x1": descr[0],
        "x2": descr[1],
        "x3bsupp": descr[2],
        "int2b": descr[3],
        "int3b": descr[4],
        "intder2b": descr[5],
        "intder3b": descr[6],
        "intder3bsupp": descr[7],
        "numtriplet": descr[8],
        "etrue": tf.constant(energy),
        "ftrue": tf.constant(force),
        "pe": pe,
        "pf": pf,
        "pb": pb,
    }

    probe_names = ("dense_kernel", "dense_bias", "alpha2b", "alpha3b")

    def _call():
        return model.full_mse_grads_e_f(
            tensors["x1"],
            tensors["x2"],
            tensors["x3bsupp"],
            tensors["int2b"],
            tensors["intder2b"],
            tensors["int3b"],
            tensors["intder3b"],
            tensors["intder3bsupp"],
            tensors["numtriplet"],
            tensors["etrue"],
            tensors["ftrue"],
        )

    loss_e, loss_f, grad_E, grad_F = _call()
    family_vars = {
        "dense_kernel": model.nets[0].trainable_variables[0],
        "dense_bias": model.nets[0].trainable_variables[1],
        "alpha2b": model.physics_layer[0].alpha2b,
        "alpha3b": model.physics_layer[0].alpha3b,
    }
    ana_E = {}
    ana_F = {}
    for name, gE, gF in zip(probe_names, grad_E, grad_F):
        ana_E[name] = np.asarray(gE.numpy(), dtype=np.float64).reshape(-1)
        ana_F[name] = np.asarray(gF.numpy(), dtype=np.float64).reshape(-1)
        print(
            f"analytic {name}  E_rms={np.sqrt(np.mean(ana_E[name]**2)):.4e}  "
            f"F_rms={np.sqrt(np.mean(ana_F[name]**2)):.4e}"
        )

    def losses():
        tot, lf, le = model.full_test_e_f(
            tensors["x1"],
            tensors["x2"],
            tensors["x3bsupp"],
            tensors["int2b"],
            tensors["intder2b"],
            tensors["int3b"],
            tensors["intder3b"],
            tensors["intder3bsupp"],
            tensors["numtriplet"],
            tensors["etrue"],
            tensors["ftrue"],
        )
        return float(np.asarray(le)), float(np.asarray(lf))

    rng = np.random.default_rng(args.seed)
    out_dir = ROOT / f"results_{args.precision}"
    out_dir.mkdir(exist_ok=True)
    summary = {"checkpoint": str(ckpt), "precision": args.precision, "families": {}}
    pass_ok = True

    lE0 = float(np.asarray(loss_e))
    lF0 = float(np.asarray(loss_f))
    a2_shape = np.asarray(family_vars["alpha2b"].numpy()).shape
    sticky_mask = np.zeros(int(np.prod(a2_shape)), dtype=bool)
    if len(a2_shape) == 2:
        sticky_mask.reshape(a2_shape)[-1, :] = True
    a3_shape = np.asarray(family_vars["alpha3b"].numpy()).shape
    ang_mask = np.zeros(int(np.prod(a3_shape)), dtype=bool)
    if len(a3_shape) == 2 and a3_shape[0] >= 3:
        # map_ang_afs {0:[0,0,5,0,0,0]} → interaction block index 2
        ang_mask.reshape(a3_shape)[2, :] = True
    extra = {"alpha2b": sticky_mask, "alpha3b": ang_mask}

    for name, var in family_vars.items():
        arr0 = np.asarray(var.numpy())
        # Prefer the loss that actually depends on this family (E for dense, F for AF).
        g_pick = ana_E[name] if name.startswith("dense") else ana_F[name]
        slots = _sample_top(g_pick, args.n_per_family, rng, extra_mask=extra.get(name))
        if slots.size < 2:
            print(f"{name}: skip (not enough active finite slots)")
            summary["families"][name] = {"n": int(slots.size), "dws": [], "skipped": True}
            continue
        family_report = {"n": int(slots.size), "dws": []}
        for dw in args.dws:
            gE_num = np.zeros(slots.size, dtype=np.float64)
            gF_num = np.zeros(slots.size, dtype=np.float64)
            for i, sl in enumerate(slots):
                idx = _unravel(arr0.shape, sl)
                bumped = arr0.copy()
                bumped[idx] = bumped[idx] + dw
                var.assign(bumped)
                le1, lf1 = losses()
                gE_num[i] = (le1 - lE0) / dw
                gF_num[i] = (lf1 - lF0) / dw
                var.assign(arr0)
            cE, maeE, nE = _corr_metrics(ana_E[name][slots], gE_num)
            cF, maeF, nF = _corr_metrics(ana_F[name][slots], gF_num)
            family_report["dws"].append(
                {
                    "dw": float(dw),
                    "corr_E": cE,
                    "corr_F": cF,
                    "mae_E": maeE,
                    "mae_F": maeF,
                    "n_used_E": nE,
                    "n_used_F": nF,
                }
            )
            print(
                f"{name} dw={dw:g}: E corr={cE:.6f} mae={maeE:.3e}  "
                f"F corr={cF:.6f} mae={maeF:.3e}"
            )
            if abs(dw - DW_PASS) < 1e-18:
                best = np.nanmax([cE, cF])
                if not (np.isfinite(best) and best >= CORR_PASS):
                    pass_ok = False
        summary["families"][name] = family_report

    summary["pass"] = pass_ok
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [f"checkpoint={ckpt}", f"precision={args.precision}", ""]
    for name, fam in summary["families"].items():
        for d in fam["dws"]:
            lines.append(
                f"{name} dw={d['dw']:g} n={fam['n']}  "
                f"E corr={d['corr_E']:.8f} mae={d['mae_E']:.6e}  "
                f"F corr={d['corr_F']:.8f} mae={d['mae_F']:.6e}"
            )
    lines.append("PASS" if pass_ok else "FAIL")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("PASS" if pass_ok else "FAIL")
    return 0 if pass_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
