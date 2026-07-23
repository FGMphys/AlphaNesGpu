# A2 progress notes

## Slice 1 — naming (2026-07-23)

Official trees (`AlphaNesGpu_{float,double}`):

| Old | New |
| --- | --- |
| `alphanes_models/` | `staf_models/` |
| `alpha_nes_model.py` | `staf_model.py` (`staf_full`) |
| `alpha_nes_model_inference*.py` | `staf_model_inference*.py` (`staf_full_inference`) |
| `alpha_nnpes_full_main.py` | `staf_train.py` |

Compat shims kept (deprecated): `alphanes_models/*`, `alpha_nnpes_full_main.py`,
class aliases `alpha_nes_full` / `alpha_nes_full_inference`.

### Slice 1 gates (2026-07-23)

| Gate | Result |
| --- | --- |
| Force FD float/double | OK (corr ≈ 0.999 at δ=0.001) |
| Perf smoke (`run_a2_perf_smoke.py`) | OK float/double (micro-bench only; not ACCEPTANCE gate 4) |
| `time_story` 1-epoch (V100) | float **87.0** ms/frame (base 91.5, ratio 0.95); double **160.2** ms/frame (base 149.7, ratio 1.07 ≤1.15) |

## Slice 2 — precision-agnostic Python (2026-07-23)

- Shared package: `staf/dtype.py` (`precision` YAML → `set_floatx`)
- Official trees share identical `staf_train.py`, `staf_models/*`, `init_AFs_param.py`, `optimizer_learning_rate_utility.py`
- Test YAML: `precision: float|double` in `run_{float,double}/input_4test.yaml`
- Tree name still used as fallback if YAML omits `precision`

### Slice 2 gates (2026-07-23)

| Gate | Result |
| --- | --- |
| Force FD float/double | OK (corr ≈ 0.9998 / 0.9999 at δ=0.001) |
| `time_story` 1-epoch (V100) | float **89.8** ms/frame (ratio 0.98); double **152.5** ms/frame (ratio 1.02) |

Still pending A2:

- Unify float/double into one installable package layout (move modules under `staf/`)
- CUDA `real` typedef / single `src/`
- Rename directories `AlphaNesGpu_*` → `STAF_*` (or thin wrappers)
- Remove shims after callers migrate
- DEV/ trees (left untouched this slice)
