# Test training pipeline

Smoke test and float32/float64 compatibility check for STAF training on the MB-pol water dataset.

## Layout

- `run_float/` — float32 training (`AlphaNesGpu_float`), dataset float32, `input_4test.yaml`
- `run_double/` — float64 training (`AlphaNesGpu_double`), dataset float64, `input_4test.yaml`
- `comparison/` — overlays, plots, performance baseline from `time_story`

## What we did

1. Built both precisions into the repo `.venv` and compiled CUDA ops (`install_path.sh`).
2. Prepared twin datasets from `dataset_MBPOL_278_223_248_full` (source float32; double converted to float64; folder `train` → `training`).
3. Ran short trainings with the same YAML (Seed 60, Rc=4.5 Å, batch 4) and compared `lcurve_notmean` / `lcurve.out`.
4. Confirmed early-training compatibility (Loss_F corr ≈ 0.98; Loss_E smoother corr ≈ 0.96).
5. Added xmgrace headers to learning-curve writers and documented V100 timings as a performance baseline (~38.2 neighbors within Rc).

## How to rerun

```bash
source ../../.venv/bin/activate
cd run_float   # or run_double
python ../../AlphaNesGpu_float/staf_train.py input_4test.yaml
# double: ../../AlphaNesGpu_double/staf_train.py
```

Dataset `.npy` files are gitignored; regenerate with the conversion used in the initial setup if needed.
