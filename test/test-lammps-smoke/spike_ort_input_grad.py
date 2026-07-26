#!/usr/bin/env python3
"""Spike: ORT training artifacts → ∂sum(atomic)/∂af without hand-written Dense backprop.

Verdict (2026-07-24): GREEN for Python/CPU path when:
  1. Forward ONNX ends in a *scalar* ReduceSum (vector ReduceSum breaks Sum_Grad).
  2. generate_artifacts(requires_grad=["af"], frozen_params=weights, loss=None).
  3. Training graph is post-processed to expose intermediate `af_grad` as an output
     (default `af_grad.accumulation.out` is a bool from InPlaceAccumulatorV2).
  4. Inference feeds: af, frozen weights (from checkpoint), zeros buffer, lazy_reset=True.

Grad ops used: com.microsoft::TanhGrad (+ standard MatMul/Gemm grads).
InPlaceAccumulatorV2 can be stripped for a cleaner MD graph; TanhGrad still needs an
ORT build that ships Microsoft contrib / training gradient ops.

Usage (repo venv with onnxruntime-training + torch CPU + tf/tf2onnx):
  CUDA_VISIBLE_DEVICES=-1 python test/test-lammps-smoke/spike_ort_input_grad.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "test/test-lammps-smoke/model_onnx_float"
OUT_DIR = ROOT / "test/test-lammps-smoke/ort_grad_spike"
CKPT = ROOT / "test/test-training-pipeline/run_float/model_log0"

REPORT: dict = {"status": "incomplete", "steps": [], "errors": [], "parity": {}}


def log(msg: str) -> None:
    print(msg, flush=True)
    REPORT["steps"].append(msg)


def _bake_constant_inputs(model) -> None:
    """Turn tf2onnx constant graph-inputs into pure initializers (ORT training-safe)."""
    from onnx import numpy_helper

    init_names = {i.name for i in model.graph.initializer}
    mu = np.loadtxt(MODEL_DIR / "type0_alpha_mu.dat", dtype=np.float32)
    known = {
        "add/y:0": np.array(1e-3, dtype=np.float32),
        "sub/y:0": mu,
        "mul/y:0": np.array(0.5, dtype=np.float32),
    }
    keep = []
    for vi in list(model.graph.input):
        if vi.name == "af":
            keep.append(vi)
            continue
        if vi.name in init_names:
            continue
        if vi.name in known:
            model.graph.initializer.append(
                numpy_helper.from_array(known[vi.name], name=vi.name)
            )
            log(f"baked {vi.name} → initializer")
            continue
        log(f"WARNING: dropping unexpected graph input {vi.name}")
    del model.graph.input[:]
    model.graph.input.extend(keep)


def step_export_forward() -> Path:
    """af [B,N,n_af] → scalar sum(atomic) with lognorm + Dense in-graph."""
    import onnx
    import tensorflow as tf
    import tf2onnx

    tf.keras.backend.set_floatx("float32")
    mu = np.loadtxt(MODEL_DIR / "type0_alpha_mu.dat", dtype=np.float32)
    n_af = int(mu.shape[0])
    net = tf.keras.models.load_model(str(CKPT / "net_model_type0"))
    rebuilt = tf.keras.Sequential()
    rebuilt.add(tf.keras.Input(shape=(n_af,), dtype=tf.float32))
    for layer in net.layers:
        rebuilt.add(layer)

    mu_c = tf.constant(mu, dtype=tf.float32)
    eps = tf.constant(np.float32(1e-3), dtype=tf.float32)

    @tf.function(
        input_signature=[tf.TensorSpec([None, None, n_af], tf.float32, name="af")]
    )
    def forward(af):
        logdes = tf.math.log(af + eps) - mu_c
        flat = tf.reshape(logdes, [-1, n_af])
        out = rebuilt(flat)
        # Scalar is required: vector ReduceSum breaks ORT Sum_Grad (Unsqueeze axis).
        return {"sum_atomic": tf.reduce_sum(out)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT_DIR / "type0_af_to_sum_scalar.onnx"
    model_proto, _ = tf2onnx.convert.from_function(
        forward,
        input_signature=[tf.TensorSpec([None, None, n_af], tf.float32, name="af")],
        opset=17,
        output_path=str(onnx_path),
    )
    _bake_constant_inputs(model_proto)
    keep_outs = [o for o in model_proto.graph.output if o.name == "sum_atomic"]
    del model_proto.graph.output[:]
    model_proto.graph.output.extend(keep_outs)
    onnx.save(model_proto, str(onnx_path))
    inits = {i.name for i in model_proto.graph.initializer}
    inns = [i.name for i in model_proto.graph.input if i.name not in inits]
    log(f"exported {onnx_path} inputs={inns} outs={[o.name for o in model_proto.graph.output]}")
    return onnx_path


def step_generate_and_expose_grad(onnx_path: Path) -> tuple[Path, Path]:
    """Build training artifacts and rewrite outputs to expose af_grad."""
    import onnx
    from onnx import helper, TensorProto
    from onnxruntime.training import artifacts

    model = onnx.load(str(onnx_path))
    inits = [i.name for i in model.graph.initializer]
    art_dir = OUT_DIR / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    artifacts.generate_artifacts(
        model,
        requires_grad=["af"],
        frozen_params=inits,
        loss=None,
        optimizer=None,
        artifact_directory=str(art_dir),
        prefix="spike_",
    )
    train_raw = art_dir / "spike_training_model.onnx"
    log(f"generate_artifacts OK → {train_raw}")

    tm = onnx.load(str(train_raw))
    # Ensure af_grad value info exists
    has_af_grad = any(vi.name == "af_grad" for vi in tm.graph.value_info)
    if not has_af_grad:
        af_in = next(i for i in tm.graph.input if i.name == "af")
        dims = [
            d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in af_in.type.tensor_type.shape.dim
        ]
        tm.graph.value_info.append(
            helper.make_tensor_value_info("af_grad", TensorProto.FLOAT, dims)
        )

    af_grad_vi = next(vi for vi in tm.graph.value_info if vi.name == "af_grad")
    new_outs = []
    for o in tm.graph.output:
        if o.name == "af_grad.accumulation.out":
            continue
        new_outs.append(o)
    if not any(o.name == "af_grad" for o in new_outs):
        new_outs.append(af_grad_vi)
    del tm.graph.output[:]
    tm.graph.output.extend(new_outs)

    train_path = art_dir / "spike_training_model_afgrad.onnx"
    onnx.save(tm, str(train_path))
    ckpt = art_dir / "spike_checkpoint"
    log(f"exposed af_grad → {train_path}; checkpoint={ckpt}")
    REPORT["training_io"] = {
        "inputs": [i.name for i in tm.graph.input],
        "outputs": [o.name for o in tm.graph.output],
        "ops": sorted({n.op_type for n in tm.graph.node}),
    }
    return train_path, ckpt


def _rebuild_keras_mlp():
    import tensorflow as tf

    tf.keras.backend.set_floatx("float32")
    mu = np.loadtxt(MODEL_DIR / "type0_alpha_mu.dat", dtype=np.float32)
    n_af = int(mu.shape[0])
    net = tf.keras.models.load_model(str(CKPT / "net_model_type0"))
    rebuilt = tf.keras.Sequential()
    rebuilt.add(tf.keras.Input(shape=(n_af,), dtype=tf.float32))
    for layer in net.layers:
        rebuilt.add(layer)
    return rebuilt, mu, n_af


def step_parity(train_path: Path, ckpt_path: Path) -> bool:
    import onnxruntime as ort
    import tensorflow as tf
    from onnxruntime.training.api import CheckpointState

    rebuilt, mu, n_af = _rebuild_keras_mlp()
    st = CheckpointState.load_checkpoint(str(ckpt_path))
    params = {n: np.asarray(p.data) for n, p in st.parameters}
    sess = ort.InferenceSession(str(train_path), providers=["CPUExecutionProvider"])

    results = []
    for seed, n_atoms in [(0, 16), (1, 8), (2, 32), (3, 100)]:
        af = np.random.default_rng(seed).uniform(0.01, 2.0, size=(1, n_atoms, n_af)).astype(
            np.float32
        )
        feeds: dict = {"af": af}
        for inp in sess.get_inputs():
            name = inp.name
            if name in params:
                arr = np.asarray(params[name])
                if "int64" in inp.type:
                    arr = arr.astype(np.int64)
                elif "int32" in inp.type:
                    arr = arr.astype(np.int32)
                elif "float" in inp.type:
                    arr = arr.astype(np.float32)
                feeds[name] = arr
            elif "accumulation.buffer" in name:
                feeds[name] = np.zeros_like(af)
            elif name == "lazy_reset_grad":
                feeds[name] = np.array([True])

        outs = sess.run(None, feeds)
        by = {m.name: np.asarray(v) for m, v in zip(sess.get_outputs(), outs)}

        af_tf = tf.constant(af)
        with tf.GradientTape() as tape:
            tape.watch(af_tf)
            s = tf.reduce_sum(
                rebuilt(tf.reshape(tf.math.log(af_tf + 1e-3) - mu, [-1, n_af]))
            )
        g_tf = tape.gradient(s, af_tf).numpy()
        g_ort = by["af_grad"]
        max_abs = float(np.max(np.abs(g_tf - g_ort)))
        rel = max_abs / (float(np.max(np.abs(g_tf))) + 1e-12)
        e_abs = abs(float(s.numpy()) - float(by["sum_atomic"]))
        row = {
            "n_atoms": n_atoms,
            "max_abs_grad": max_abs,
            "rel_grad": rel,
            "abs_sum": e_abs,
            "sum_tf": float(s.numpy()),
            "sum_ort": float(by["sum_atomic"]),
        }
        results.append(row)
        log(
            f"parity n_atoms={n_atoms}: max|Δgrad|={max_abs:.3g} rel={rel:.3g} |Δsum|={e_abs:.3g}"
        )

    REPORT["parity"] = {"cases": results}
    ok = all(r["max_abs_grad"] < 1e-4 and r["abs_sum"] < 1e-4 for r in results)
    return ok


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        onnx_path = step_export_forward()
        train_path, ckpt = step_generate_and_expose_grad(onnx_path)
        ok = step_parity(train_path, ckpt)
        REPORT["status"] = "green" if ok else "red"
        REPORT["verdict"] = (
            "ORT training artifacts give TF-parity ∂sum(atomic)/∂af; "
            "can replace C++ analytical Dense backprop (needs ORT with TanhGrad)."
            if ok
            else "parity failed"
        )
        return 0 if ok else 3
    except Exception as e:
        REPORT["status"] = "red"
        REPORT["errors"].append(str(e))
        REPORT["verdict"] = "spike crashed"
        traceback.print_exc()
        return 1
    finally:
        out = OUT_DIR / "SPIKE_REPORT.json"
        out.write_text(json.dumps(REPORT, indent=2))
        log(f"wrote {out}")
        print("\n=== VERDICT:", REPORT.get("status"), "-", REPORT.get("verdict"), "===\n")


if __name__ == "__main__":
    sys.exit(main())
