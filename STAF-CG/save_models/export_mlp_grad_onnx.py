#!/usr/bin/env python3
"""Export STAF-CG MLP ONNX with autodiff ∂sum(atomic)/∂af for inference ORT.

Builds a training grad graph via onnxruntime-training, then rewrites it so it
runs on *inference* ORT (no com.microsoft::TanhGrad / InPlaceAccumulatorV2):

  TanhGrad(dY, Y)  →  dY * (1 - Y*Y)   (standard Mul/Sub)
  drop accumulator + buffer / lazy_reset_grad
  bake weights from ORT checkpoint as initializers

Per type k under -modelname:
  model_type{k}.onnx   input  af:      [batch, n_atoms, n_AF]
                       output energy:  [1] or []  (= 0.5 * sum atomic)
                       output dE_daf:  [batch, n_atoms, n_AF]  (= ∂sum/∂af)
  mlp_type{k}.bin      native fallback (unchanged binary format)
  + type{k}_alpha_*.dat copies

Requires: onnxruntime-training, torch (CPU ok), tf, tf2onnx, onnx, numpy.

Usage:
  CUDA_VISIBLE_DEVICES=-1 python export_mlp_grad_onnx.py \\
      -imodel model_log0 -modelname out_dir --precision float32
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _detect_ntypes(input_model: str) -> int:
    nt = 0
    for guess in range(100):
        if os.path.exists(os.path.join(input_model, f"net_model_type{guess}")):
            nt += 1
        else:
            break
    if nt:
        return nt
    for guess in range(100):
        sm = os.path.join(input_model, f"model_type{guess}")
        if os.path.isdir(sm) and (
            os.path.isfile(os.path.join(sm, "saved_model.pb"))
            or os.path.isdir(os.path.join(sm, "variables"))
        ):
            nt += 1
        else:
            break
    return nt


def _dense_index(name: str) -> int:
    import re

    m = re.search(r"dense(?:_(\d+))?", name)
    if not m:
        return 10**9
    return int(m.group(1) or 0)


def _n_af_from_alpha(input_model: str, k: int):
    rad = os.path.join(input_model, f"type{k}_alpha_2body.dat")
    ang = os.path.join(input_model, f"type{k}_alpha_3body.dat")
    if not (os.path.isfile(rad) and os.path.isfile(ang)):
        return None
    a2 = np.loadtxt(rad, dtype=np.float64).reshape(3, -1)
    a3 = np.loadtxt(ang, dtype=np.float64).reshape(6, -1)
    return int(a2.shape[1] + a3.shape[1])


def _collect_dense_vars(sm):
    """TF SavedModel TestModel keeps Dense weights on .newmodel, not .variables."""
    for src in (getattr(sm, "newmodel", None), sm):
        if src is None:
            continue
        vs = getattr(src, "variables", None)
        if vs:
            return list(vs)
        vs = getattr(src, "trainable_variables", None)
        if vs:
            return list(vs)
        keras_api = getattr(src, "keras_api", None)
        if keras_api is not None:
            vs = getattr(keras_api, "variables", None)
            if vs:
                return list(vs)
    return []


def _keras_layers_from_newmodel(sm):
    net = getattr(sm, "newmodel", None)
    if net is None:
        return []
    for src in (net, getattr(net, "keras_api", None)):
        if src is None or not hasattr(src, "layers"):
            continue
        try:
            layers = [
                el
                for el in src.layers
                if type(el).__name__ != "InputLayer"
            ]
        except Exception:
            continue
        if layers:
            return layers
    return []


def _keras_from_savedmodel(sm_dir: str, dtype, np_dtype):
    """Rebuild a Dense Sequential + mu from an inference SavedModel (MODEL1896)."""
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input

    sm = tf.saved_model.load(sm_dir)
    mu = None
    am = getattr(sm, "alphamu", None)
    if am is not None:
        mu = np.asarray(am.numpy() if hasattr(am, "numpy") else am, dtype=np_dtype).reshape(-1)

    net = getattr(sm, "newmodel", None)
    layers = _keras_layers_from_newmodel(sm)

    dtype_name = "float32" if np_dtype == np.float32 else "float64"
    rebuilt = tf.keras.Sequential(name=os.path.basename(sm_dir) + "_mlp")
    acts: list[str] = []

    if layers:
        try:
            first = layers[0]
            n_af = int(first.kernel.shape[0]) if hasattr(first, "kernel") else int(mu.shape[0])
            rebuilt.add(Input(shape=(n_af,), dtype=dtype, name="logdes_flat"))
            for el in layers:
                cfg = el.get_config()
                if "dtype" in cfg:
                    cfg["dtype"] = dtype_name
                new_el = el.__class__.from_config(cfg)
                rebuilt.add(new_el)
                w = el.get_weights()
                if w:
                    new_el.set_weights([np.asarray(x, dtype=np_dtype) for x in w])
                if hasattr(el, "activation"):
                    acts.append(
                        el.activation.__name__ if hasattr(el.activation, "__name__") else "linear"
                    )
            if mu is None:
                mu = np.zeros((n_af,), dtype=np_dtype)
            _validate_savedmodel_mlp(sm, rebuilt, mu, dtype)
            return rebuilt, mu, acts
        except Exception as exc:
            print(f"    newmodel clone failed ({exc}); falling back to variables")
            rebuilt = tf.keras.Sequential(name=os.path.basename(sm_dir) + "_mlp")
            acts = []

    kernels = []
    rank1 = []
    for v in _collect_dense_vars(sm):
        arr = np.asarray(v.numpy())
        name = v.name
        if arr.ndim == 2:
            kernels.append((name, np.asarray(arr, dtype=np_dtype)))
        elif arr.ndim == 1:
            rank1.append((name, np.asarray(arr, dtype=np_dtype)))
        elif arr.ndim == 0:
            continue

    if not kernels:
        raise RuntimeError(f"no Dense kernels in SavedModel {sm_dir}")
    kernels.sort(key=lambda kv: (_dense_index(kv[0]), kv[0]))

    used = set()
    pairs = []
    for kn, W in kernels:
        prefix = kn.rsplit("/", 1)[0] if "/" in kn else kn.split(":")[0]
        b = None
        for i, (bn, bv) in enumerate(rank1):
            if i in used:
                continue
            if bv.shape[0] != W.shape[1]:
                continue
            if prefix in bn or _dense_index(bn) == _dense_index(kn):
                b = bv
                used.add(i)
                break
        if b is None:
            for i, (bn, bv) in enumerate(rank1):
                if i not in used and bv.shape[0] == W.shape[1]:
                    b = bv
                    used.add(i)
                    break
        if b is None:
            raise RuntimeError(f"no bias for kernel {kn} shape {W.shape} in {sm_dir}")
        pairs.append((W, b))

    n_af = int(pairs[0][0].shape[0])
    if mu is None:
        for i, (bn, bv) in enumerate(rank1):
            if i in used:
                continue
            if bv.shape[0] == n_af:
                mu = bv
                used.add(i)
                break
    if mu is None:
        raise RuntimeError(f"could not find alphamu in {sm_dir} (n_AF={n_af})")

    rebuilt.add(Input(shape=(n_af,), dtype=dtype, name="logdes_flat"))
    for i, (W, b) in enumerate(pairs):
        last = i == len(pairs) - 1
        act = "linear" if last else "tanh"
        layer = Dense(int(W.shape[1]), activation=act, dtype=dtype_name)
        rebuilt.add(layer)
        layer.build((None, int(W.shape[0])))
        layer.set_weights([W, b])
        acts.append(act)

    _validate_savedmodel_mlp(sm, rebuilt, mu, dtype)
    return rebuilt, np.asarray(mu, dtype=np_dtype).reshape(-1), acts


def _validate_savedmodel_mlp(sm, keras_net, mu, dtype) -> None:
    import tensorflow as tf

    n_af = int(np.asarray(mu).reshape(-1).shape[0])
    rng = np.random.default_rng(0)
    n_atoms = 6
    af = rng.random((1, n_atoms, n_af)) * 0.2 + 0.05
    af_tf = tf.constant(af, dtype=tf.float64)
    e_sm, _g = sm.testmodel(af_tf)
    e_sm = float(np.asarray(e_sm).reshape(-1)[0])
    mu64 = np.asarray(mu, dtype=np.float64).reshape(-1)
    logdes = np.log(af + 1e-3) - mu64
    atomic = keras_net(tf.constant(logdes.reshape(-1, n_af), dtype=dtype), training=False)
    e_k = 0.5 * float(np.sum(np.asarray(atomic, dtype=np.float64)))
    err = abs(e_sm - e_k)
    scale = max(abs(e_sm), 1.0)
    if err > 1e-6 and err / scale > 1e-7:
        raise RuntimeError(
            f"SavedModel MLP rebuild mismatch |ΔE|={err:.3g} "
            f"(SM={e_sm:.10g} keras={e_k:.10g})"
        )
    print(f"    SavedModel↔keras MLP |ΔE|={err:.3g} (n_AF={n_af})")


def _load_keras_mlp(input_model: str, k: int, dtype, np_dtype):
    """Keras Sequential + mu from net_model_type{k} or model_type{k} SavedModel."""
    import tensorflow as tf

    keras_dir = os.path.join(input_model, f"net_model_type{k}")
    mu_path = os.path.join(input_model, f"type{k}_alpha_mu.dat")
    mu = None
    if os.path.isfile(mu_path):
        mu = np.loadtxt(mu_path, dtype=np_dtype)
        if mu.ndim == 0:
            mu = np.array([mu], dtype=np_dtype)

    if os.path.exists(keras_dir):
        net = tf.keras.models.load_model(keras_dir)
        if mu is None:
            raise RuntimeError(f"missing {mu_path} for keras net type{k}")
        return net, mu

    sm_dir = os.path.join(input_model, f"model_type{k}")
    if not os.path.isdir(sm_dir):
        raise RuntimeError(f"no net_model_type{k} or model_type{k} under {input_model}")
    # MODEL1896 SavedModel is float64; rebuild in f64 then let export cast.
    net, mu_sm, _acts = _keras_from_savedmodel(sm_dir, tf.float64, np.float64)
    if mu is None:
        mu = np.asarray(mu_sm, dtype=np_dtype)
    n_af_alpha = _n_af_from_alpha(input_model, k)
    n_mu = int(np.asarray(mu).reshape(-1).shape[0])
    if n_af_alpha is not None and n_mu != n_af_alpha:
        print(
            f"  WARNING type{k}: mu n_AF={n_mu} vs alpha files n_AF={n_af_alpha} "
            "(using SavedModel mu)"
        )
    return net, np.asarray(mu, dtype=np_dtype).reshape(-1)


def _bake_constant_inputs(model: onnx.ModelProto, mu: np.ndarray) -> None:
    init_names = {i.name for i in model.graph.initializer}
    known = {
        "add/y:0": np.asarray(1e-3, dtype=mu.dtype),
        "sub/y:0": np.asarray(mu, dtype=mu.dtype),
        "mul/y:0": np.asarray(0.5, dtype=mu.dtype),
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
            continue
    del model.graph.input[:]
    model.graph.input.extend(keep)


def _write_bin(out_dir: str, k: int, mu: np.ndarray, acts: list, weights: dict, np_dtype) -> None:
    bin_path = os.path.join(out_dir, f"mlp_type{k}.bin")
    with open(bin_path, "wb") as bf:
        bf.write(b"STAFMLP1")
        prec_code = 1 if np_dtype == np.float64 else 0
        bf.write(np.array([prec_code, int(mu.shape[0]), len(acts)], dtype=np.int32).tobytes())
        bf.write(np.asarray(mu, dtype=np_dtype).tobytes())
        for li, act in enumerate(acts):
            W = weights[f"W{li}"]
            b = weights[f"b{li}"]
            act_code = {"linear": 0, "tanh": 1, "relu": 2}.get(act, -1)
            if act_code < 0:
                raise RuntimeError(f"unsupported activation {act}")
            bf.write(np.array([act_code, W.shape[0], W.shape[1]], dtype=np.int32).tobytes())
            bf.write(np.asarray(W, dtype=np_dtype).tobytes())
            bf.write(np.asarray(b, dtype=np_dtype).tobytes())
    print(f"    wrote {bin_path}")


def _replace_tanhgrad(model: onnx.ModelProto, np_dtype) -> int:
    """Replace com.microsoft::TanhGrad(dY, Y) with dY*(1-Y*Y). Returns count."""
    new_nodes = []
    n_replaced = 0
    # unique ones initializer
    one_name = "staf_const_one"
    if not any(i.name == one_name for i in model.graph.initializer):
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(1.0, dtype=np_dtype), name=one_name)
        )
    for node in model.graph.node:
        if node.op_type == "TanhGrad" and (node.domain or "") in ("", "com.microsoft"):
            if len(node.input) < 2 or len(node.output) < 1:
                raise RuntimeError(f"unexpected TanhGrad: {node.name}")
            dy, y = node.input[0], node.input[1]
            dx = node.output[0]
            yy = f"{node.name}__yy"
            one_m_yy = f"{node.name}__one_m_yy"
            new_nodes.append(helper.make_node("Mul", [y, y], [yy], name=f"{node.name}__MulY"))
            new_nodes.append(
                helper.make_node("Sub", [one_name, yy], [one_m_yy], name=f"{node.name}__Sub")
            )
            new_nodes.append(
                helper.make_node("Mul", [dy, one_m_yy], [dx], name=f"{node.name}__MulDy")
            )
            n_replaced += 1
        else:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return n_replaced


def _prune_accumulator(model: onnx.ModelProto) -> None:
    drop_ops = {"InPlaceAccumulatorV2", "InPlaceAccumulator"}
    drop_inputs = {"af_grad.accumulation.buffer", "lazy_reset_grad"}
    drop_outputs = {"af_grad.accumulation.out"}
    keep_nodes = [n for n in model.graph.node if n.op_type not in drop_ops]
    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)

    keep_in = [i for i in model.graph.input if i.name not in drop_inputs]
    del model.graph.input[:]
    model.graph.input.extend(keep_in)

    keep_out = [o for o in model.graph.output if o.name not in drop_outputs]
    del model.graph.output[:]
    model.graph.output.extend(keep_out)


def _bake_params_from_checkpoint(model: onnx.ModelProto, ckpt_path: str) -> None:
    from onnxruntime.training.api import CheckpointState

    st = CheckpointState.load_checkpoint(ckpt_path)
    params = {n: np.asarray(p.data) for n, p in st.parameters}
    init_names = {i.name for i in model.graph.initializer}
    for name, arr in params.items():
        if name in init_names:
            # replace
            for i, init in enumerate(model.graph.initializer):
                if init.name == name:
                    model.graph.initializer[i].CopyFrom(numpy_helper.from_array(arr, name=name))
                    break
        else:
            model.graph.initializer.append(numpy_helper.from_array(arr, name=name))
            init_names.add(name)

    # Drop baked tensors from graph.input (keep only af)
    keep = [i for i in model.graph.input if i.name == "af"]
    del model.graph.input[:]
    model.graph.input.extend(keep)


def _expose_outputs(model: onnx.ModelProto, np_dtype) -> None:
    """Outputs: energy = 0.5*sum_atomic, dE_daf = af_grad."""
    # Ensure af_grad is a graph output
    names = {o.name for o in model.graph.output}
    if "af_grad" not in names:
        af_in = next(i for i in model.graph.input if i.name == "af")
        dims = [
            d.dim_value if d.HasField("dim_value") else (d.dim_param or None)
            for d in af_in.type.tensor_type.shape.dim
        ]
        model.graph.output.append(
            helper.make_tensor_value_info("af_grad", TensorProto.FLOAT if np_dtype == np.float32 else TensorProto.DOUBLE, dims)
        )

    half = "staf_half"
    if not any(i.name == half for i in model.graph.initializer):
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(0.5, dtype=np_dtype), name=half)
        )
    if not any(n.output and n.output[0] == "energy" for n in model.graph.node):
        model.graph.node.append(
            helper.make_node("Mul", ["sum_atomic", half], ["energy"], name="staf_energy_half")
        )

    elem = TensorProto.FLOAT if np_dtype == np.float32 else TensorProto.DOUBLE
    new_outs = [
        helper.make_tensor_value_info("energy", elem, []),
        helper.make_tensor_value_info(
            "dE_daf",
            elem,
            [
                d.dim_value if d.HasField("dim_value") else (d.dim_param or None)
                for d in next(i for i in model.graph.input if i.name == "af").type.tensor_type.shape.dim
            ],
        ),
    ]
    # Rename af_grad -> dE_daf via Identity if needed
    if not any(n.output and n.output[0] == "dE_daf" for n in model.graph.node):
        model.graph.node.append(
            helper.make_node("Identity", ["af_grad"], ["dE_daf"], name="staf_rename_dE_daf")
        )
    del model.graph.output[:]
    model.graph.output.extend(new_outs)


def _export_type(
    input_model: str,
    out_dir: str,
    k: int,
    dtype,
    np_dtype,
    opset: int,
) -> None:
    import tensorflow as tf
    import tf2onnx
    from onnxruntime.training import artifacts

    net, mu = _load_keras_mlp(input_model, k, dtype, np_dtype)
    mu = np.asarray(mu, dtype=np_dtype).reshape(-1)
    n_af = int(mu.shape[0])
    np.savetxt(os.path.join(out_dir, f"type{k}_alpha_mu.dat"), mu)

    rebuilt = tf.keras.Sequential(name=f"staf_mlp_type{k}")
    rebuilt.add(tf.keras.Input(shape=(n_af,), dtype=dtype, name="logdes_flat"))
    acts = []
    weights = {"mu": mu}
    dtype_name = "float32" if np_dtype == np.float32 else "float64"
    li = 0
    for layer in net.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        cfg = layer.get_config()
        if "dtype" in cfg:
            cfg["dtype"] = dtype_name
        new_el = layer.__class__.from_config(cfg)
        rebuilt.add(new_el)
        w = layer.get_weights()
        if w:
            new_el.set_weights([np.asarray(x, dtype=np_dtype) for x in w])
        if len(w) >= 2:
            weights[f"W{li}"] = np.asarray(w[0], dtype=np_dtype)
            weights[f"b{li}"] = np.asarray(w[1], dtype=np_dtype)
            act = layer.activation.__name__ if hasattr(layer, "activation") else "linear"
            acts.append(act)
            li += 1
    weights["n_layers"] = np.array([len(acts)], dtype=np.int32)
    weights["activations"] = np.array(acts)
    np.savez(os.path.join(out_dir, f"mlp_type{k}.npz"), **weights)
    _write_bin(out_dir, k, mu, acts, weights, np_dtype)

    mu_c = tf.constant(mu, dtype=dtype)
    eps = tf.constant(np_dtype(1e-3), dtype=dtype)

    @tf.function(input_signature=[tf.TensorSpec([None, None, n_af], dtype, name="af")])
    def forward(af):
        logdes = tf.math.log(af + eps) - mu_c
        flat = tf.reshape(logdes, [-1, n_af])
        out = rebuilt(flat)
        return {"sum_atomic": tf.reduce_sum(out)}

    with tempfile.TemporaryDirectory(prefix="staf_grad_") as tmp:
        fwd_path = os.path.join(tmp, "fwd.onnx")
        model_proto, _ = tf2onnx.convert.from_function(
            forward,
            input_signature=[tf.TensorSpec([None, None, n_af], dtype, name="af")],
            opset=opset,
            output_path=fwd_path,
        )
        _bake_constant_inputs(model_proto, mu)
        keep_outs = [o for o in model_proto.graph.output if o.name == "sum_atomic"]
        del model_proto.graph.output[:]
        model_proto.graph.output.extend(keep_outs)
        onnx.save(model_proto, fwd_path)

        art = Path(tmp) / "art"
        art.mkdir()
        inits = [i.name for i in model_proto.graph.initializer]
        artifacts.generate_artifacts(
            model_proto,
            requires_grad=["af"],
            frozen_params=inits,
            loss=None,
            optimizer=None,
            artifact_directory=str(art),
            prefix="t_",
        )
        train = onnx.load(str(art / "t_training_model.onnx"))
        # expose af_grad before prune
        if not any(o.name == "af_grad" for o in train.graph.output):
            # value_info may exist
            af_vi = None
            for vi in train.graph.value_info:
                if vi.name == "af_grad":
                    af_vi = vi
                    break
            if af_vi is None:
                af_in = next(i for i in train.graph.input if i.name == "af")
                dims = [
                    d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in af_in.type.tensor_type.shape.dim
                ]
                af_vi = helper.make_tensor_value_info(
                    "af_grad",
                    TensorProto.FLOAT if np_dtype == np.float32 else TensorProto.DOUBLE,
                    dims,
                )
            train.graph.output.append(af_vi)

        _prune_accumulator(train)
        n_tg = _replace_tanhgrad(train, np_dtype)
        _bake_params_from_checkpoint(train, str(art / "t_checkpoint"))
        _expose_outputs(train, np_dtype)

        # Drop unused microsoft domain if empty — keep opset imports that remain
        # Strip com.microsoft opset if no remaining microsoft nodes
        has_ms = any((n.domain or "") == "com.microsoft" for n in train.graph.node)
        if not has_ms:
            keep_imp = [
                oi
                for oi in train.opset_import
                if oi.domain not in ("com.microsoft",)
            ]
            del train.opset_import[:]
            train.opset_import.extend(keep_imp)

        onnx_path = os.path.join(out_dir, f"model_type{k}.onnx")
        onnx.save(train, onnx_path)
        try:
            onnx.checker.check_model(train)
        except Exception as e:
            print(f"  WARNING onnx.checker: {e}")
        print(
            f"  type{k}: n_AF={n_af} acts={acts} TanhGrad→std={n_tg} → {onnx_path} "
            f"outs={[o.name for o in train.graph.output]}"
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

    import tensorflow as tf

    if args.precision in ("float", "float32"):
        dtype = tf.float32
        np_dtype = np.float32
        tf.keras.backend.set_floatx("float32")
    else:
        dtype = tf.float64
        np_dtype = np.float64
        tf.keras.backend.set_floatx("float64")
        print("STAF: float64 grad export is experimental", file=sys.stderr)

    try:
        import tf2onnx  # noqa: F401
        from onnxruntime.training import artifacts  # noqa: F401
    except ImportError as e:
        print(f"STAF: need tf2onnx + onnxruntime-training (+ torch): {e}", file=sys.stderr)
        return 1

    input_model = args.imodel
    out_dir = args.modelname
    nt = _detect_ntypes(input_model)
    if nt == 0:
        print(
            f"STAF: no net_model_type* or model_type* SavedModel under {input_model}",
            file=sys.stderr,
        )
        return 1
    print(f"STAF-CG: grad-export {nt} type(s) → {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    for name in (
        "color_type_map.dat",
        "map_color_interaction.dat",
        "map_intra.dat",
        "cutoff_info",
        "number_of_nn.dat",
        "model_error",
    ):
        src = os.path.join(input_model, name)
        if os.path.isfile(src):
            shutil.copy(src, out_dir)
            print(f"    copied {name}")

    for k in range(nt):
        _export_type(input_model, out_dir, k, dtype, np_dtype, args.opset)

    nn_path = os.path.join(out_dir, "number_of_nn.dat")
    if not os.path.isfile(nn_path):
        with open(nn_path, "w", encoding="utf-8") as fh:
            fh.write(f"{nt}\n")
        print(f"    wrote {nn_path}")

    with open(os.path.join(out_dir, "EXPORT_ONNX.txt"), "w", encoding="utf-8") as fh:
        fh.write("STAF-CG MLP grad export (inference-ORT friendly, float32 for libstaf_cg)\n")
        fh.write(f"source: {os.path.abspath(input_model)}\n")
        fh.write(f"precision: {args.precision}\n")
        fh.write(f"ntypes: {nt}\n")
        fh.write("onnx input:  af [batch,n_atoms,n_AF]\n")
        fh.write("onnx output: energy [] = 0.5*sum(atomic)\n")
        fh.write("onnx output: dE_daf [batch,n_atoms,n_AF] = d(sum atomic)/daf\n")
        fh.write("TanhGrad rewritten to Mul/Sub; no training-only ops\n")
        fh.write("CG maps: color_type_map.dat, map_color_interaction.dat, map_intra.dat\n")
        fh.write(
            "Accepts net_model_type* (keras train export) or model_type* "
            "SavedModel (e.g. MODEL1896); writes type{k}_alpha_mu.dat for libstaf_cg.\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
