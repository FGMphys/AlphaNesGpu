# A2 progress notes

## Slice 1 — naming (2026-07-23)

| Old | New |
| --- | --- |
| `alphanes_models/` | `staf_models/` |
| `alpha_nes_model*.py` | `staf_model*.py` |
| `alpha_nnpes_full_main.py` | `staf_train.py` |

## Slice 2 — precision-agnostic Python (2026-07-23)

Shared `staf/dtype.py`; YAML `precision: float|double`.

## Slice 3 — unified `STAF/` tree (2026-07-23)

- Canonical tree: **`STAF/`** (no separate float/double Python trees)
- Flattened out of `mixture/`: `staf_models/`, `source_routine/`, `gradient_utility/`
- Ops remain split until CUDA `real` unify: `STAF/ops_float/src`, `STAF/ops_double/src`
- `AlphaNesGpu_{float,double}/` → thin deprecated redirects
- `alphanes_models` removed from official trees

### Slice 3 gates (2026-07-23)

| Gate | Result |
| --- | --- |
| Force FD float/double | OK (corr ≈ 0.9998 / 0.9999 at δ=0.001) |
| `time_story` 1-epoch (V100) | float **91.1** ms/frame (×0.996); double **153.1** ms/frame (×1.02) |

Still pending A2:

- CUDA `real` typedef / single `src/` (collapse `ops_float` + `ops_double`)
- Remove deprecated `AlphaNesGpu_*` redirects when callers migrated
- DEV/ trees
