# STAF Python inference vs LAMMPS `pair_staf` (same weights)

Compares energy and forces on shared water frames from
`test/test-inference-pipeline/frames/`.

| Side | Model |
|------|--------|
| Python | `model_tf_float_log0` — TF SavedModel from `run_float/model_log0` |
| LAMMPS | `../test-lammps-smoke/model_onnx_grad_float` — ORT ONNX from the same checkpoint |

## Run

```bash
source scripts/staf_gpu_env.sh
python test/test-lammps-STAF-inference-STAF-comp/run_compare.py --n-frames 5
```

Outputs: `results/summary.json`, `results/frames.csv`, `results/frame_XX/`, `plots/*.png`.

`libstaf` converts LAMMPS cartesian coordinates to fractional (JMD convention)
and packs atoms by species before the CUDA AF path — required for E/F parity
with Python STAF.

## Re-export TF model (if missing)

```bash
cd STAF
CUDA_VISIBLE_DEVICES=-1 python save_models/save_model_in_float.py \
  -imodel ../test/test-training-pipeline/run_float/model_log0 \
  -modelname ../test/test-lammps-STAF-inference-STAF-comp/model_tf_float_log0
cp ../test/test-training-pipeline/run_float/dataset/type.dat \
   ../test/test-lammps-STAF-inference-STAF-comp/model_tf_float_log0/
cp ../test/test-inference-pipeline/model_float/cutoff_info \
   ../test/test-lammps-STAF-inference-STAF-comp/model_tf_float_log0/
```
