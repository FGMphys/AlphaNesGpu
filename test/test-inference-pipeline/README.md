# STAF inference float/double compatibility test

Goal: export the same trained checkpoint (`source_model_log1`, from
`test-training-pipeline/run_double/model_log1`) as float32 and float64 inference
models, then compare **all** NPZ tensors (descriptors, interaction maps,
fingerprints, network gradients, forces, energy) on the same frames.

## Layout

- `source_model_log1/` — training checkpoint (+ `type.dat`, `cutoff_info`)
- `model_double/` / `model_float/` — exported inference models
- `frames/` — shared input frames
- `inference_double/` / `inference_float/` — `inference_bundle.npz`
- `analyze_compatibility.py` — full float↔double report
- `comparison_summary.txt` / `compatibility_report.json`

## Workflow

```bash
source ../../.venv/bin/activate
cd test/test-inference-pipeline

python ../../STAF/save_models/save_model.py \
  -imodel source_model_log1 -modelname model_double
python ../../STAF/save_models/save_model_in_float.py \
  -imodel source_model_log1 -modelname model_float

python prepare_frames.py
python run_inference.py --precision double   # then float (not in parallel)
python run_inference.py --precision float
python analyze_compatibility.py
```

## Result (10 frames)

**Compatible.** `int2b`/`int3b` bit-identical; energy/force/fingerprint/grads ~1e-5–1e-7.
Large raw `x2b`/`intder*` diffs are **padded** neighbor slots only; active slots ~1e-6.
