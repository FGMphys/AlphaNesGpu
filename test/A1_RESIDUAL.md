# A1 residual notes (pre A2)

Audit date: 2026-07-23. Tree: `AlphaNesGpu_double` (official).

## D1 — float math on double buffers (CUDA) — **fixed on production paths**

Replaced `expf` / `cosf` / `sinf` / `0.f` / `1.f` / … with double variants in:

- `AlphaNesGpu_double/src/mixture/**` (fingerprint, force, grad_*)
- `AlphaNesGpu_double/src/descriptor_builder/` (wired builder)

Left untouched: `descriptor_builder_develop/` (experimental).

Recompile with `cd AlphaNesGpu_double && bash install_path.sh`, then re-run
ACCEPTANCE gates 2–3 (refresh perf baseline if timings move).

## D2 — energy-only `loss_force` dtype — **fixed**

`alpha_nes_model.py` (double): `loss_force` constant now `float64`.

## D7 — hardcoded `root_path` — **fixed**

Loaders use `alphanes_paths.code_root()` with optional
`ALPHANES_DOUBLE_ROOT` / `ALPHANES_FLOAT_ROOT`.
`install_path.sh` no longer `sed`s Python sources.

## Deferred

- **jmd_nn / neuralmdGPU parity** → Linea B (`PIANO_…` B1–B3).
- Optimizer split `opt_net` / `opt_phys`, `type_emb` grads (D4/D5) → A2 Python.
- Single `src/` with `real` typedef → A2 CUDA.
