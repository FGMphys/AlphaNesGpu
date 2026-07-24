#!/usr/bin/env python3
"""Export STAF Dense MLP for libstaf (ORT + analytical grads).

Produces per type k:
  model_type{k}.onnx   — Sequential only: logdes -> atomic_e
                         input  logdes: [batch, n_atoms, n_AF]
                         output atomic_e: [batch, n_atoms]
  mlp_type{k}.npz    — mu, W*, b*, activations (for lognorm + ∂E/∂AF)

libstaf applies: af -> log(af+1e-3)-mu -> ORT/native Dense -> E=0.5*sum(atomic_e)
and analytical backprop for ∂E/∂AF (ONNX has no TF gradient ops).

Usage:
  python export_mlp_onnx.py -imodel model_log0 -modelname out --precision float32
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf


def _detect_ntypes(input_model: str) -> int:
    nt = 0
    for guess in range(100):
        if os.path.exists(os.path.join(input_model, f"net_model_type{guess}")):
            nt += 1
        else:
            break
    return nt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-imodel", required=True)
    parser.add_argument("-modelname", required=True)
    parser.add_argument(
        "--precision",
        default="float32",
        choices=("float32", "float", "float64", "double"),
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if args.precision in ("float", "float32"):
        dtype = tf.float32
        np_dtype = np.float32
        tf.keras.backend.set_floatx("float32")
    else:
        dtype = tf.float64
        np_dtype = np.float64
        tf.keras.backend.set_floatx("float64")

    try:
        import tf2onnx
    except ImportError:
        print("STAF: pip install tf2onnx", file=sys.stderr)
        return 1

    input_model = args.imodel
    out_dir = args.modelname
    nt = _detect_ntypes(input_model)
    if nt == 0:
        print(f"STAF: no net_model_type* under {input_model}", file=sys.stderr)
        return 1
    print(f"STAF: exporting {nt} type network(s) → {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    err_src = os.path.join(input_model, "model_error")
    if os.path.isfile(err_src):
        shutil.copy(err_src, out_dir)

    for k in range(nt):
        mu = np.loadtxt(
            os.path.join(input_model, f"type{k}_alpha_mu.dat"), dtype=np_dtype
        )
        if mu.ndim == 0:
            mu = np.array([mu], dtype=np_dtype)
        n_af = int(mu.shape[0])

        net = tf.keras.models.load_model(
            os.path.join(input_model, f"net_model_type{k}")
        )
        rebuilt = tf.keras.Sequential(name=f"staf_mlp_type{k}")
        rebuilt.add(tf.keras.Input(shape=(n_af,), dtype=dtype, name="logdes_flat"))
        acts = []
        weights = {"mu": mu}
        for li, layer in enumerate(net.layers):
            rebuilt.add(layer)
            w = layer.get_weights()
            if len(w) >= 2:
                weights[f"W{li}"] = np.asarray(w[0], dtype=np_dtype)
                weights[f"b{li}"] = np.asarray(w[1], dtype=np_dtype)
                act = layer.activation.__name__ if hasattr(layer, "activation") else "linear"
                acts.append(act)
        weights["n_layers"] = np.array([len(acts)], dtype=np.int32)
        weights["activations"] = np.array(acts)
        np.savez(os.path.join(out_dir, f"mlp_type{k}.npz"), **weights)

        # Simple binary for libstaf (little-endian)
        bin_path = os.path.join(out_dir, f"mlp_type{k}.bin")
        with open(bin_path, "wb") as bf:
            bf.write(b"STAFMLP1")
            prec_code = 1 if np_dtype == np.float64 else 0
            bf.write(np.array([prec_code, n_af, len(acts)], dtype=np.int32).tobytes())
            bf.write(np.asarray(mu, dtype=np_dtype).tobytes())
            for li, act in enumerate(acts):
                W = weights[f"W{li}"]
                b = weights[f"b{li}"]
                act_code = {"linear": 0, "tanh": 1, "relu": 2}.get(act, -1)
                if act_code < 0:
                    print(f"STAF: unsupported activation {act}", file=sys.stderr)
                    return 1
                bf.write(
                    np.array(
                        [act_code, W.shape[0], W.shape[1]], dtype=np.int32
                    ).tobytes()
                )
                bf.write(np.asarray(W, dtype=np_dtype).tobytes())
                bf.write(np.asarray(b, dtype=np_dtype).tobytes())
        print(f"    wrote {bin_path}")

        # Wrapper: [B,N,F] -> Sequential on last dim -> [B,N]
        @tf.function(
            input_signature=[
                tf.TensorSpec([None, None, n_af], dtype, name="logdes")
            ]
        )
        def forward(logdes):
            flat = tf.reshape(logdes, [-1, n_af])
            out = rebuilt(flat)
            n_atoms = tf.shape(logdes)[1]
            atomic_e = tf.reshape(out, [-1, n_atoms])
            return {"atomic_e": atomic_e}

        onnx_path = os.path.join(out_dir, f"model_type{k}.onnx")
        model_proto, _ = tf2onnx.convert.from_function(
            forward,
            input_signature=[
                tf.TensorSpec([None, None, n_af], dtype, name="logdes")
            ],
            opset=args.opset,
            output_path=onnx_path,
        )
        print(
            f"  type{k}: n_AF={n_af} {acts} → {onnx_path} + mlp_type{k}.npz "
            f"(outs={[o.name for o in model_proto.graph.output]})"
        )

        for name in (
            f"type{k}_alpha_2body.dat",
            f"type{k}_alpha_3body.dat",
            f"type{k}_alpha_mu.dat",
            f"type{k}_type_emb_2b.dat",
            f"type{k}_type_emb_3b.dat",
            f"type{k}_type_emb_2b_sq.dat",
            f"type{k}_type_emb_3b_sq.dat",
        ):
            src = os.path.join(input_model, name)
            if os.path.isfile(src):
                shutil.copy(src, out_dir)

    with open(os.path.join(out_dir, "EXPORT_ONNX.txt"), "w", encoding="utf-8") as fh:
        fh.write("STAF MLP export\n")
        fh.write(f"source: {os.path.abspath(input_model)}\n")
        fh.write(f"precision: {args.precision}\n")
        fh.write(f"ntypes: {nt}\n")
        fh.write("onnx input: logdes [batch,n_atoms,n_AF]  (after lognorm)\n")
        fh.write("onnx output: atomic_e [batch,n_atoms]\n")
        fh.write("libstaf: lognorm + ORT/native Dense + analytical dE/daf\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
