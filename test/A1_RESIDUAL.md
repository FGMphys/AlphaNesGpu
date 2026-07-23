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

Loaders use `staf_paths.code_root()` with preferred
`STAF_DOUBLE_ROOT` / `STAF_FLOAT_ROOT` (deprecated aliases
`ALPHANES_*_ROOT` still work). `install_path.sh` no longer `sed`s Python
sources. Module `alphanes_paths` remains a thin re-export until A2 renames
packages (`alphanes_models` → STAF packaging).

## D8 — divergent `__syncthreads` in angular grads — **mitigated**

`grad_force/ang` (and `grad_finger/ang`) had `__syncthreads()` + block
reduction **inside** `if (t < prod)`. That is illegal CUDA (divergent barrier)
and showed up after D1 as intermittent **ghost** `∂Loss_F/∂α3b` on
energy-inactive AF slots (FD = 0, analytic ≠ 0), poisoning the grad-param
gate when sampling ranked by `|g_E|+|g_F|`.

Mitigation:

- Move barrier + reduction outside the `t` filter; zero shared mem then sync;
  same change on float tree for parity.
- Grad-param sampler for α3b now requires `|g_E| > eps` (energy-active slots).

A2 CUDA should replace the thread-0 shared reduction with a proper block reduce.

## Deferred

- **jmd_nn / neuralmdGPU parity** → Linea B (`PIANO_…` B1–B3).
- Optimizer split `opt_net` / `opt_phys`, `type_emb` grads (D4/D5) → A2 Python.
- Single `src/` with `real` typedef → A2 CUDA.
