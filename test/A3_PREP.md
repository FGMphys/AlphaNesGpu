# A3 prep — CUDA global / static state audit (2026-07-23)

Production sources live under `STAF/src/`. Experimental trees are under
`STAF/experimental/` and are **not** built by `install_path.sh`.

## Device-global state that blocks naive MirroredStrategy

| Location | Kind | Risk for multi-GPU |
|----------|------|--------------------|
| `src/descriptor_builder/reforce.cc` | many `static` host/device pointers + scalars (`Radbuff`, cell lists, …) | **High** — one process / one logical device assumed |
| `src/*/force|grad_*/{rad,ang}/reforce.cu.cc` | `static int BLOCK_DIM` | Medium — init-once per process |
| ~~Widespread `cudaDeviceSynchronize()`~~ | removed in production after Eigen-stream wiring | Was perf / multi-stream hazard; see section below |
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
3. ~~Gate `cudaDeviceSynchronize` behind debug builds once streams are correct.~~ Done for production: Eigen stream + no device sync.

## A+B single-GPU sync trim (2026-07-23)

- **A:** `staf_train.py` logs batch losses / lr only every `log_batch_freq` (default=`displ_freq`); flush aligned to `displ_freq`.
- **B:** removed `cudaDeviceSynchronize` after zero-fill kernels; **kept** one sync at end of each compute launcher (required until launchers use TF's Eigen GPU stream). Full sync removal without stream wiring caused `CUDA_ERROR_ILLEGAL_ADDRESS`.

Gates after A+B: force/compat/grad OK; float `time_story` ≈ 88 ms/frame (×0.96 vs 91.5).

## TF Eigen GPU stream wiring (2026-07-23)

- Production launchers / `set_tensor_to_zero_*` take `cudaStream_t stream`.
- Each GPU `OpKernel::Compute` passes `context->eigen_device<Eigen::GpuDevice>().stream()`.
- Removed remaining production `cudaDeviceSynchronize` (sync after zero-fill and end-of-launcher).
- `init_block_dim` stays host-only (no stream) — it only sets `BLOCK_DIM`.
- `staf_real.h` always includes `<cuda_runtime.h>` so stream types resolve on host.

Gates after stream wiring (V100, sequential float→double):

| Gate | float | double |
| --- | --- | --- |
| Force FD δ=0.001 | corr=0.99985 | corr=0.99994 |
| Grad-param dw=1e-3 | families ≈1 | families ≈1 |
| Inference float↔double | Compatible | Compatible |
| `time_story` 1-epoch | 87.2 ms (×0.95 vs 91.5) | 154.0 ms (×1.03 vs 149.7) |

Next single-GPU win after this: wire/fix GPU neighbor list (`celle_gpu` / descriptor cell list) — still D2H→CPU→H2D per frame today.
