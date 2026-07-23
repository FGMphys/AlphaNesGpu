#!/usr/bin/env python3
"""Force finite-difference regression vs analytical forces from STAF inference.

Confronta le forze analitiche dell'inferenza STAF con una differenza finita
in avanti sull'energia, su un singolo frame.

Formula numerica (forward FD):

    F_num = -(E_f - E_i) / delta

dove E_i è l'energia della geometria di riferimento e E_f quella dopo lo
spostamento di una sola coordinata Cartesiana di ``delta`` Å.

Controllo di consistenza (stessa FD):

    E_i ≈ E_f + F_ana * delta

Scan tipico di delta: 0.1, 0.01, 0.001 Å.

Esempio:

    python run_force_regression.py --precision double
    python run_force_regression.py --precision float --frame 0 --n-atoms 40 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Directory di questo script, repo root, e cartella dell'inference pipeline
# (modelli esportati + frames condivisi).
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
INFER = ROOT.parents[1] / "test-inference-pipeline"


def _energy_and_force(model, pos, box):
    """Esegue un'inferenza e restituisce (energia scalare, forze flat 3N).

    ``full_test`` ritorna una lista; out[0] = energia, out[1] = lista per tipo
    atomico di (forcetot, force_radial, force_angular). Sommiamo forcetot su
    tutti i tipi per ottenere le forze totali.
    """
    out = model.full_test(pos, box)
    energy = np.asarray(out[0].numpy(), dtype=np.float64).reshape(-1)[0]
    force_list = out[1]
    # force_list[k] = (forcetot, force_radial, force_angular)
    ftot = sum(
        np.asarray(force_list[k][0].numpy(), dtype=np.float64)
        for k in range(len(force_list))
    )
    force = ftot.reshape(-1)  # (3N,)
    return energy, force


def main():
    p = argparse.ArgumentParser(
        description=(
            "Regressione forze analitiche vs FD sull'energia (un frame). "
            "I flag usati vengono scritti in results_*/summary.txt e summary.json."
        )
    )
    p.add_argument(
        "--precision",
        choices=["float", "double"],
        required=True,
        help="float → AlphaNesGpu_float + model_float; double → double + model_double",
    )
    p.add_argument(
        "--frame",
        type=int,
        default=0,
        help="indice nel file frames/pos.npy (0 = primo frame preparato)",
    )
    p.add_argument(
        "--n-atoms",
        type=int,
        default=40,
        help="quanti atomi campionare (tutte e 3 le componenti xyz)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed RNG per la scelta degli atomi da sondare",
    )
    p.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[0.1, 0.01, 0.001],
        help="ampiezze di spostamento in Å (scan FD)",
    )
    args = p.parse_args()

    # Scelta codice + modello esportato in base alla precisione.
    if args.precision == "float":
        code_root = REPO / "AlphaNesGpu_float"
        dtype = np.float32
        model_dir = INFER / "model_float"
    else:
        code_root = REPO / "AlphaNesGpu_double"
        dtype = np.float64
        model_dir = INFER / "model_double"

    sys.path.insert(0, str(code_root))
    sys.path.insert(0, str(REPO))
    import tensorflow as tf
    from staf.dtype import set_precision

    set_precision(args.precision)
    from staf_models.mixture.staf_model_inference_full import (
        staf_full_inference,
    )

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    # Frame preparati dall'inference pipeline (stessi per float e double).
    pos_all = np.load(INFER / "frames" / "pos.npy")
    box_all = np.load(INFER / "frames" / "box.npy")
    # Indici originali nel dataset test (es. frame=0 → dataset_index=4).
    frame_ids = np.load(INFER / "frames" / "frame_indices.npy")

    pos0 = pos_all[args.frame].astype(dtype).reshape(1, -1)
    box0 = box_all[args.frame].astype(dtype).reshape(1, -1)
    n_atoms_total = pos0.shape[1] // 3
    dataset_index = int(frame_ids[args.frame])

    # Campiona n_probe atomi (senza ripetizione); per ciascuno sposta x, y, z.
    rng = np.random.default_rng(args.seed)
    n_probe = min(args.n_atoms, n_atoms_total)
    atoms = np.sort(rng.choice(n_atoms_total, size=n_probe, replace=False))
    comps = []
    for a in atoms:
        for c in range(3):
            comps.append((int(a), c, int(3 * a + c)))
    comps = np.array(comps, dtype=int)  # colonne: atom, xyz, flat_idx in (3N,)

    print(f"Loading {model_dir}")
    model = staf_full_inference(str(model_dir))
    # Warm-up GPU / graph, poi misura di riferimento (E_i, F_ana).
    _ = _energy_and_force(model, pos0, box0)
    e_i, f_ana = _energy_and_force(model, pos0, box0)

    print(
        f"flags: precision={args.precision} frame={args.frame} "
        f"dataset_index={dataset_index} n_atoms={n_probe} seed={args.seed} "
        f"deltas={args.deltas}"
    )
    print(f"model_dir={model_dir}")
    print(f"E_i={e_i:.10f}")
    print(f"probing {n_probe} atoms × 3 = {len(comps)} force components")

    out_dir = ROOT / f"results_{args.precision}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Blocco "flags" ripetuto in summary.txt / summary.json per tracciabilità.
    flags = {
        "precision": args.precision,
        "frame": args.frame,
        "dataset_index": dataset_index,
        "n_atoms": n_probe,
        "n_atoms_requested": args.n_atoms,
        "n_atoms_total": n_atoms_total,
        "seed": args.seed,
        "deltas": [float(d) for d in args.deltas],
        "model_dir": str(model_dir),
        "frames_dir": str(INFER / "frames"),
        "method": "forward FD  F_num=-(E_f-E_i)/delta",
        "consistency": "E_i ≈ E_f + F_ana*delta",
    }

    summary = {
        "flags": flags,
        # Campi top-level (comodi da leggere) — stessi valori di flags.
        "precision": args.precision,
        "frame": args.frame,
        "dataset_index": dataset_index,
        "seed": args.seed,
        "n_atoms": n_probe,
        "n_atoms_probed": n_probe,
        "deltas_requested": [float(d) for d in args.deltas],
        "model_dir": str(model_dir),
        "E_i": float(e_i),
        "atoms": atoms.tolist(),
        "deltas": [],
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        have_plot = True
    except ImportError:
        have_plot = False

    if have_plot:
        fig, axes = plt.subplots(
            1, len(args.deltas), figsize=(4.2 * len(args.deltas), 4.0), squeeze=False
        )

    for idel, delta in enumerate(args.deltas):
        f_num = np.zeros(len(comps), dtype=np.float64)
        f_ref = np.zeros(len(comps), dtype=np.float64)
        e_f_list = np.zeros(len(comps), dtype=np.float64)
        e_i_pred = np.zeros(len(comps), dtype=np.float64)

        for k, (atom, xyz, flat) in enumerate(comps):
            # Sposta una sola coordinata Cartesiana di +delta.
            pos_f = pos0.copy()
            pos_f[0, flat] = pos_f[0, flat] + dtype(delta)
            e_f, _ = _energy_and_force(model, pos_f, box0)
            # Forward FD: F = -dE/dx
            f_num[k] = -(e_f - e_i) / delta
            f_ref[k] = f_ana[flat]
            e_f_list[k] = e_f
            # Consistenza: E_i ≈ E_f + F_ana * delta
            e_i_pred[k] = e_f + f_ref[k] * delta

            if (k + 1) % 20 == 0 or k == 0:
                print(
                    f"  δ={delta:g}  {k+1}/{len(comps)}  "
                    f"atom={atom} xyz={xyz}  "
                    f"F_ana={f_ref[k]:.6e} F_num={f_num[k]:.6e}"
                )

        corr = (
            float(np.corrcoef(f_ref, f_num)[0, 1]) if len(f_ref) > 1 else float("nan")
        )
        # Slope least-squares: F_num ≈ slope * F_ana (ideale → 1).
        denom = float(np.dot(f_ref, f_ref)) + 1e-30
        slope = float(np.dot(f_ref, f_num) / denom)
        mae = float(np.mean(np.abs(f_num - f_ref)))
        rmse = float(np.sqrt(np.mean((f_num - f_ref) ** 2)))
        e_consist_mae = float(np.mean(np.abs(e_i_pred - e_i)))

        entry = {
            "delta": float(delta),
            "correlation": corr,
            "slope": slope,
            "mae": mae,
            "rmse": rmse,
            "E_i_consistency_mae": e_consist_mae,
        }
        summary["deltas"].append(entry)
        print(
            f"δ={delta:g}: corr={corr:.6f} slope={slope:.6f} "
            f"MAE={mae:.3e} RMSE={rmse:.3e} "
            f"|E_i-(E_f+Fδ)| MAE={e_consist_mae:.3e}"
        )

        np.savez_compressed(
            out_dir / f"force_fd_delta_{delta:g}.npz",
            # Metadati run (stessi flag CLI).
            precision=args.precision,
            frame=args.frame,
            dataset_index=dataset_index,
            seed=args.seed,
            n_atoms=n_probe,
            model_dir=str(model_dir),
            delta=delta,
            atoms=comps[:, 0],
            xyz=comps[:, 1],
            flat_index=comps[:, 2],
            F_ana=f_ref,
            F_num=f_num,
            E_i=e_i,
            E_f=e_f_list,
            E_i_pred=e_i_pred,
            correlation=corr,
            slope=slope,
            mae=mae,
            rmse=rmse,
        )

        if have_plot:
            ax = axes[0, idel]
            lim = max(np.max(np.abs(f_ref)), np.max(np.abs(f_num)), 1e-8)
            ax.scatter(f_ref, f_num, s=12, alpha=0.75, edgecolors="none")
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y=x")
            ax.set_xlabel(r"$F_{\mathrm{ana}}$")
            ax.set_ylabel(r"$F_{\mathrm{num}}=-(E_f-E_i)/\delta$")
            ax.set_title(f"δ={delta:g}  corr={corr:.4f}")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # summary.txt: prima tutti i flag CLI, poi le metriche per delta.
    lines = [
        "# Input flags (CLI)",
        f"precision={args.precision}",
        f"frame={args.frame}",
        f"dataset_index={dataset_index}",
        f"n_atoms={n_probe}",
        f"n_atoms_requested={args.n_atoms}",
        f"n_atoms_total={n_atoms_total}",
        f"seed={args.seed}",
        f"deltas={' '.join(str(float(d)) for d in args.deltas)}",
        f"model_dir={model_dir}",
        f"frames_dir={INFER / 'frames'}",
        "",
        "# Reference",
        f"E_i={e_i:.12e}",
        "method=forward FD  F_num=-(E_f-E_i)/delta",
        "consistency: E_i ≈ E_f + F_ana*delta",
        "",
        "# Results per delta",
    ]
    for e in summary["deltas"]:
        lines.append(
            f"delta={e['delta']:g}  corr={e['correlation']:.8f}  "
            f"slope={e['slope']:.8f}  mae={e['mae']:.6e}  "
            f"rmse={e['rmse']:.6e}  "
            f"E_i_consist_mae={e['E_i_consistency_mae']:.6e}"
        )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")

    if have_plot:
        fig.suptitle(
            (
                f"STAF force regression ({args.precision})  "
                f"frame={args.frame} (dataset_idx={dataset_index})  "
                f"n_atoms={n_probe} seed={args.seed}"
            ),
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "force_regression.png", dpi=150)
        print(f"Wrote {out_dir / 'force_regression.png'}")

    print(f"Wrote {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
