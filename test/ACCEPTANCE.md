# Acceptance gates (pre-refactor freeze + A2)

Any modification of **STAF** must stay comparable to these baselines
**before** it is considered acceptable.

Frozen reference: git tag **`pre-refactor`** (see `git describe --tags`).
Post-A2 official tree: **`STAF/`** (`precision: float|double` in YAML;
CUDA from `STAF/src/{descriptor_builder,fingerprint,force,grad_*}/` →
`ops_{float,double}/`). Package helpers: `STAF/staf/dtype.py`.
Infer CLI: `STAF/staf_infer.py`. Prep notes for multi-GPU: `test/A3_PREP.md`.
Leonardo data bundle + checklist: `test/LEONARDO.md` (tar via `test/pack_leonardo_bundle.sh`).

Longer roadmap: `/home/francegm/PIANO_ALPHANES_DEBUG_LAMMPS_CG.md` (Linea A→B).
Derivator policy for training: keep **`tf.gradients` inside `@tf.function`**
unless a controlled A/B on this gate shows a clear win (do not switch to
`GradientTape` by default).

## Required gates (float **and** double, sequentially on GPU)

From repo root, with `.venv` active:

```bash
# 0) Build ops (once per machine / after CUDA edits)
cd STAF && bash install_path.sh all && cd ..

# 1) Inference float↔double compatibility
cd test/test-inference-pipeline
python analyze_compatibility.py
# expect: Compatible (see comparison_summary.txt)

# 2) Analytical vs FD forces (inference SavedModel)
cd ../test-regression/regression-force
python run_force_regression.py --precision double
python run_force_regression.py --precision float
# expect: corr → 1 for δ≈0.01–0.001 (see results_*/summary.txt)

# 3) Parameter grads vs FD (training path, model_log1, MSE Loss_E / Loss_F)
cd ../regression-grad-param
python run_grad_param_regression.py --precision double --n-per-family 100
python run_grad_param_regression.py --precision float  --n-per-family 100
# expect: per-family corr ≈ 1 at dw=1e-3
#   alpha2b may be n=80 (= all available)
#   alpha3b β/γ/δ sample energy-active slots only (|g_E|>0); n may be < requested

# 4) Performance (training wall-time, same hardware class)
# Compare new time_story.dat steady-state ms/frame to:
#   test/test-training-pipeline/comparison/performance_baseline.txt
# V100 reference: float32 ≈ 91.5 ms/frame, float64 ≈ 150 ms/frame (~1.64×)
# Quick check: 1-epoch via STAF/staf_train.py (≤15% slower than baseline)
```

Do **not** run float and double GPU jobs in parallel on a single GPU.

## Optional / deferred

| Gate | Status |
|------|--------|
| Parity vs `neuralmdGPU` / `jmd_nn` export | Deferred to Linea **B** (LAMMPS / libstaf) |
| CPU OpenMP parity | Deferred to **A4** |
| Multi-GPU | **Closed:** A3 CUDA per-device ctx + **A5** `distribute: horovod` (Leonardo 1×4 / 2×4). `mirrored` removed. |
| DEV/ CG trees | Linea **C**: official tree is **`STAF-CG/`** (see CG gates below). `DEV/` remains archive. |
| RDF / reweighting | Linea **E** — prototipo `DEV/AlphaNesGpu_double_RDF`; **waiting on FGM latex** |
| Multi-body inference | **A6** `--decompose`: isolated n-mer vacuum energies (n=2..5). Closed-form 2-body: **TODO(FGM) latex** |

## STAF-CG gates (Sprint 3)

Any modification of **STAF-CG** must stay comparable to the freeze in
[`DEV/staf_cg_freeze/FREEZE_NUMBERS.md`](../DEV/staf_cg_freeze/FREEZE_NUMBERS.md)
before it is considered acceptable.

From repo root, with `.venv` active. **Do not** run float and double GPU
jobs in parallel.

```bash
# 0) Build ops (once per machine / after CUDA edits)
cd STAF-CG && bash install_path.sh all && cd ..

# Full sequence (double then float):
bash test/test-cg-pipeline/run_sprint3.sh
```

Or piece-wise:

```bash
# 1) 1-epoch subsample vs freeze RMSE_f ≈ 38.3526 (Seed 60, 80/20, batch 8)
bash test/test-cg-pipeline/run_one_epoch.sh
python test/test-cg-pipeline/export_and_check.py

# 2) Inference float↔double compatibility (1-epoch export)
cd test/test-cg-inference
python run_inference.py --precision double --model model_double
python run_inference.py --precision float --model model_float
python analyze_compatibility.py
# expect: Compatible  (energy and force max|Δ| < 1e-3)

# 3) Analytical vs FD forces
cd ../test-cg-regression/regression-force
python run_force_regression.py --precision double --model ../../test-cg-inference/model1896_infer
python run_force_regression.py --precision float --model ../../test-cg-inference/model_float
# expect: corr ≥ 0.99 for δ=0.01 and 0.001

# 4) Parameter grads vs FD (training path, 1-epoch checkpoint)
cd ../regression-grad-param
python run_grad_param_regression.py --precision double --n-per-family 20
# expect: Loss_E corr ≈ 1 at dw=1e-3 on dense and AF families
#   (Loss_F is reported too; alpha3b F is noisier unless active angular slots are used)
```

Checklist: [`DEV/STAF_CG_SPRINTS.md`](../DEV/STAF_CG_SPRINTS.md).


## A1 residual

See `test/A1_RESIDUAL.md` (historical). Production CUDA now uses `STAF/include/staf_real.h`.
