# A3 prep — CUDA global / static state audit (2026-07-23)

Production sources live under `STAF/src/`. Experimental trees are under
`STAF/experimental/` and are **not** built by `install_path.sh`.

## Device-global state that blocks naive MirroredStrategy

| Location | Kind | Risk for multi-GPU |
|----------|------|--------------------|
| `src/descriptor_builder/reforce.cc` | many `static` host/device pointers + scalars (`Radbuff`, cell lists, …) | **High** — one process / one logical device assumed |
| `src/*/force|grad_*/{rad,ang}/reforce.cu.cc` | `static int BLOCK_DIM` | Medium — init-once per process |
| Widespread `cudaDeviceSynchronize()` | host sync after launches | Perf / multi-stream; not correctness alone |
| Quarantined `experimental/descriptor_builder_develop/` | same pattern, `double`-hardcoded | Out of build; ignore for A3 |

## Hygiene already done pre-A3

- Flattened `src/mixture/` → `src/{fingerprint,force,grad_finger,grad_force}/`
- Removed unwired `op_2bAFs_serial.cc`
- Quarantined `descriptor_builder_develop`
- `example_inference/simple_inference.py` no longer hardcodes `/home/francegm/...`
- `install_path.sh` prefers `STAF_NVCC` / `nvcc` on `PATH` over fixed home paths

## Pre-A3 hygiene gates (2026-07-23, V100, post flatten)

| Gate | float | double |
| --- | --- | --- |
| Force FD δ=0.001 | corr=0.99983 | corr=0.99994 |
| Grad-param dw=1e-3 | families ≈1 | families ≈1 |
| Inference float↔double | Compatible | Compatible |
| `time_story` 1-epoch | 93.0 ms (×1.02) | 152.4 ms (×1.02) |

## A3 follow-ups (do not “fix” blindly here)

1. Replace file-scope `static` buffers with per-op / per-device context (or thread-local keyed by `tf.device`).
2. Ensure `Init*` kernels are idempotent per replica.
3. Gate `cudaDeviceSynchronize` behind debug builds once streams are correct.
