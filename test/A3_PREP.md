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

## C — GPU neighbor list (2026-07-23)

- Descriptor `ComputeDescriptorsLight` uses `celle_gpu.cu.cc` (cell build + IME + per-particle insertion sort).
- No thrust-in-kernel; positions stay on GPU; only box (6×nf) touches the host for cell sizing.
- `compila.sh` links `celle_gpu` instead of CPU `cell_list` / `interaction_map` (sources kept).
- Streams: all NL launches on TF Eigen stream.

Gates after C (V100):

| Gate | float | double |
| --- | --- | --- |
| Force FD δ=0.001 | corr=0.99980 | corr=0.99994 |
| Grad-param dw=1e-3 | families ≈1 | families ≈1 |
| Inference float↔double | Compatible | Compatible |
| `time_story` 1-epoch | **80.1 ms** (×0.88 vs 91.5) | **145.5 ms** (×0.97 vs 149.7) |

## D — prefetch + reduce_retracing (2026-07-23)

- Host mmap buffer copies overlap GPU via `ThreadPoolExecutor(1)`.
- `e`/`f` uploaded once per buffer; angular-buffer `.numpy()` check once per epoch.
- `full_train_*` / `full_test_*` use `@tf.function(reduce_retracing=True)`.

## E — fuse angular grad launches (2026-07-23)

- `grad_finger/ang` and `grad_force/ang`: host nested `(alpha,sum)` launch loops → single 2D grid (`blockIdx.y` selects pair). Same block reduction as before (per-thread atomics regressed float badly).

Gates after D+E (V100): float **76.4 ms/frame** (×0.84); double **142.8 ms/frame** (×0.95); grad-param α3b families ≈1.

## Intmap CPU-NL vs GPU-NL (2026-07-23)

A/B on `test/test-inference-pipeline` frames (10 frames, float and double):
`int2b` / `int3b` (howmany, neighbor set, order) are **bit-identical** between pre-C CPU neighbor list and current GPU NL.
Harness: `test/test-inference-pipeline/compare_intmap_cpu_vs_gpu.py`.

## A3 slice — per-device CUDA ctx + `distribute` (2026-07-24)

**CUDA (same-process multi-GPU prep)**

- `descriptor_builder/reforce.cc`: file-scope static buffers → `StafDescriptorConfig` + per-device `StafDescriptorCtx` (mutex + `unordered_map` keyed by `cudaGetDevice()`).
- `force|grad_force/{rad,ang}/reforce.cu.cc`: `BLOCK_DIM` is per-device (`init_block_dim` / `current_block_dim`).

**Training YAML**

```yaml
distribute: none          # none | mirrored | horovod
# devices: [0, 1]         # mirrored only; horovod uses hvd.local_rank()
```

- `none`: current single-device path.
- `mirrored`: `tf.distribute.MirroredStrategy`; model/opts under `strategy.scope()`.
  - **1 GPU:** train step is the normal path (no `strategy.run`) so Huber
    `SUM_OVER_BATCH_SIZE` stays valid; smoke YAML:
    `test/test-training-pipeline/run_float/input_mirrored_smoke.yaml`.
  - **≥2 GPU:** `strategy.run` + Huber `Reduction.SUM` (mean across replicas
    in the wrapper). Full scaling / global-batch LR equivalence still TBD on
    Leonardo.
- `horovod`: see **A5** below.

**Not done in A3:** true multi-GPU scaling validation (needs ≥2 GPUs); MultiWorker.

### Gates after A3 slice (V100, 2026-07-24)

| Gate | float | double |
| --- | --- | --- |
| Force FD δ=0.001 | corr=0.99981 | corr=0.99994 |
| Grad-param dw=1e-3 | families ≈1 | families ≈1 |
| Inference float↔double | Compatible | Compatible |
| `time_story` 1-epoch (none) | **~77.7 ms/frame** (×0.85 vs 91.5) | **~142.1 ms/frame** (×0.95 vs 149.7) |
| Mirrored 1-GPU smoke | OK (~78.5 ms/frame, `model_log0`) | OK (~145.8 ms/frame, `model_log0`) |

## A5 — Horovod MPI+GPU (2026-07-24)

Same repo / YAML switch. Requires `horovod` + MPI launcher.

```bash
# smoke (1 rank / 1 GPU)
cd test/test-training-pipeline/run_float
mpirun -np 1 python ../../../STAF/staf_train.py input_horovod_smoke.yaml
# Leonardo example (4 GPU/node):
# mpirun -np 4 python staf_train.py input.yaml
```

Wiring in `STAF/staf_train.py`:
- GPU init deferred until after YAML; `hvd.init()` + pin `gpus[local_rank]`
- `hvd.DistributedOptimizer` on net/phys opts; initial LR × `hvd.size()`
- train buffer shard `idx_str_tr[rank::size]`; test/save/logs on rank 0 only
- `hvd.broadcast_variables` after first train step (or restart warm-up)

Smoke YAMLs: `run_{float,double}/input_horovod_smoke.yaml`.

Local gates (V100, `mpirun -np 1`, Horovod 0.28.1): float OK + `model_log0`; double OK + `model_log0`. Multi-rank scaling still Leonardo.

### Distribute loss parity (`lcurve_notmean`)

Same Seed / 1 epoch / every-step `log_batch_freq=1`, compare `none` vs `mirrored` vs `horovod` (`mpirun -np 1`):

```bash
python test/test-training-pipeline/compare_distribute_lcurve.py --precision float
python test/test-training-pipeline/compare_distribute_lcurve.py --precision double
```

Results (V100, 2026-07-24), 125 steps:

| precision | none vs mirrored | none vs horovod | notes |
| --- | --- | --- | --- |
| **double** | max‖ΔF‖ ≈ 1.6e-13 | max‖ΔF‖ ≈ 3.3e-14 | bit-identical within tol |
| **float** | max‖ΔF‖ ≈ 2.0e-6 | max‖ΔF‖ ≈ 3.9e-6 | GPU order noise; all ‖ΔF‖ under 1e-5 |

Artifacts: `test/test-training-pipeline/parity_distribute/{float,double}/`.
