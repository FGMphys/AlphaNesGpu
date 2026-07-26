# Linea B — Architecture (default path)

**Status:** default road frozen 2026-07-24  
**Goal:** `pair_style staf` in LAMMPS with Allegro-style domain decomposition.

## Default inference chain

```text
LAMMPS positions + ghosts (DD)
        │
        ▼
  CUDA STAF: neighbor / descriptors / AF     (libstaf, custom kernels)
        │  AF per atom-type
        ▼
  ONNX Runtime (CUDA EP): MLP               (lognorm + Dense only)
        │  energy, ∂E/∂AF
        ▼
  CUDA STAF: force kernels (2b/3b chain rule)
        │
        ▼
  LAMMPS forces / virial / reverse_comm
```

Training stays **TensorFlow**. Custom CUDA ops are **never** inside the ONNX graph.

## Repo layout

```text
AlphaNesGpu/
  STAF/                      # train + export
    save_models/
      export_mlp_onnx.py     # Keras → model_type{k}.onnx
      save_model.py          # legacy SavedModel (parity / jmd_nn)
  libstaf/                   # C API runtime
    include/staf.h           # public LAMMPS-facing API
    include/staf_mlp.h       # pluggable MLP backend
    src/mlp/                 # ORT default implementation
  lammps/USER-STAF/          # pair_staf + Install.sh
  test/B_ARCHITECTURE.md     # this file
```

## ONNX contract (per atomic type `k`)

Default export: `export_mlp_grad_onnx.py` (preferred). Legacy forward-only:
`export_mlp_onnx.py` (libstaf then uses analytical grads from `.bin`).

| Item | Spec |
|------|------|
| File | `{model_dir}/model_type{k}.onnx` |
| Input name | `af` |
| Input shape | `[batch, n_atoms, n_AF_k]` (dynamic batch & n_atoms) |
| Output `energy` | scalar — `0.5 * sum(atomic)` for that type |
| Output `dE_daf` | `[batch, n_atoms, n_AF_k]` — `∂sum(atomic)/∂af` (jmd force convention) |
| Preprocess in-graph | `log(af + 1e-3) - μ` then Dense Sequential |
| Grad ops | rewritten to standard ONNX (`TanhGrad` → `Mul`/`Sub`); no training-only ops |
| Precision | separate float / double exports (FP64 grad export experimental) |

Ancillary ASCII (unchanged vs `jmd_nn`):

- `type{k}_alpha_2body.dat`, `type{k}_alpha_3body.dat`
- embeddings if multi-type
- cutoff / type map as documented by export

## `libstaf` responsibilities

1. Load AF parameters + ONNX sessions (one ORT session per type, per MPI rank).
2. Build AF on device from local+ghost coordinates (or from LAMMPS neigh list).
3. Call `staf_mlp_eval` → ORT → `(E, dE/dAF)`.
4. Apply force CUDA kernels; fill `f[nall*3]` and virial for the rank.
5. No global `MPI_Allgather` of the full system inside `staf_compute`.

MLP backend is swappable (`ort` default; future `tf_c` / `native`) without changing `pair_staf`.

## LAMMPS usage (target)

```lammps
units           real
atom_style      atomic
pair_style      staf 6.0 6.0
pair_coeff      * * /path/to/model_dir
comm_modify     cutoff 6.0
```

```bash
#SBATCH --nodes=1 --ntasks-per-node=4 --gres=gpu:4
srun lmp_mpi_staf -in in.staf_water
```

## Acceptance (B)

1. Export ONNX: parity `(E, dE/dAF)` vs TF SavedModel on fixed AF tensors.
2. Single-rank MD: energy/forces vs `jmd_nn` / Python inference.
3. DD: 1 vs 2 vs 4 ranks — total E and forces on shared atoms within tol.
4. Packaging: link `libonnxruntime` + CUDA; no `libtensorflow` required for default path.

## Non-goals (MVP)

- Putting custom STAF ops into ONNX.
- TensorRT as required (optional ORT EP later).
- Rewriting training in PyTorch.
- Kokkos rewrite (DD/ghost contract first).


## B1 smoke status (2026-07-24)

- Export: `STAF/save_models/export_mlp_grad_onnx.py` → `model_type*.onnx` with `energy` + `dE_daf`
- Runtime MLP default: **ORT CUDA** (no analytical `.bin` fallback on `STAF_MLP_ORT`)
- AF/force: `libstaf/vendor/jmd/nn_nn_mlp.cu` + neuralmdGPU CUDA objects
- LAMMPS: `lmp_staf` + `test/test-lammps-smoke/run_staf_smoke_gpu.sh` (5 NVE, 1 rank, ORT CUDA EP) OK

## Spike: ORT autodiff for ∂E/∂af (2026-07-24) — GREEN

Script: `test/test-lammps-smoke/spike_ort_input_grad.py`  
Report: `test/test-lammps-smoke/ort_grad_spike/SPIKE_REPORT.json`

**Result:** `onnxruntime.training.artifacts.generate_artifacts` builds a usable
`∂sum(atomic)/∂af` graph with TF parity (~1e-6) for dynamic `n_atoms`.

| Requirement | Detail |
|-------------|--------|
| Forward output | **Scalar** `ReduceSum` over all atomic energies (vector ReduceSum breaks ORT `Sum_Grad`) |
| `requires_grad` | Graph input `af`; all Dense weights + μ/ε frozen |
| Post-process | Expose intermediate `af_grad`; drop bool `af_grad.accumulation.out` |
| Grad ops | rewritten to standard `Mul`/`Sub` for inference ORT (no `TanhGrad` at runtime) |
| Thermo energy | Still `0.5 * sum(atomic)`; `dE_daf` = ∂sum/∂af (jmd force convention) |

## Grad export → inference ORT (landed)

1. `STAF/save_models/export_mlp_grad_onnx.py` → MD ONNX (`af` → `energy`, `dE_daf`), standard ops only.
2. This box: ORT **1.16.3 CUDA11** at `third_party/onnxruntime-cuda11` + CUDA 11.8 + cuDNN 8.
3. `libstaf` `STAF_MLP_ORT`: requires `energy`+`dE_daf`; fails hard if missing (native analytical only for `STAF_MLP_NATIVE`).

```bash
source scripts/staf_gpu_env.sh
CUDA_VISIBLE_DEVICES=-1 python STAF/save_models/export_mlp_grad_onnx.py \
  -imodel path/to/model_log0 -modelname out_dir --precision float32

cmake -S libstaf -B libstaf/build -DSTAF_WITH_ORT=ON -DSTAF_WITH_JMD=ON \
  -DORT_ROOT="$ORT_ROOT" -DCUDA_HOME="$CUDA_HOME" -DCUDNN_LIB="$CUDNN_LIB" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX"
cmake --build libstaf/build -j
./libstaf/build/staf_mlp_smoke out_dir
./test/test-lammps-smoke/run_staf_smoke_gpu.sh
```

Smoke model: `test/test-lammps-smoke/model_onnx_grad_float/`.

## GPU stack & portable discovery

| Role | This machine (driver 470) | New machine |
|------|---------------------------|-------------|
| CUDA | 11.8 `/home/francegm/programmi/cuda` | whatever `scripts/staf_gpu_env.sh` finds (or `CUDA_HOME=`) |
| cuDNN | 8.x `programmi/cudaNN/cuda-11.8/lib` | `CUDNN_LIB=` or next to CUDA |
| ORT | `third_party/onnxruntime-cuda11` (1.16.3) | CUDA12 → `third_party/onnxruntime` (1.19.2) when `libcublas.so.12` resolves |

Helper: [`scripts/staf_gpu_env.sh`](../scripts/staf_gpu_env.sh) — honors `CUDA_HOME` / `CUDNN_LIB` / `ORT_ROOT`, else probes common paths and exports `LD_LIBRARY_PATH` + `CUDACXX`.

Do **not** install CUDA 12 on this host until the NVIDIA driver is upgraded (≥525/570 depending on toolkit).

## B2 MPI DD status (2026-07-24)

- `pair_staf` injects LAMMPS **full + ghost** neighbor list into `libstaf` (indices only; distances from positions). JMD `celle`/`ime` skipped on this path; PBC `rint` off when ghosts carry images.
- Neighbors are filtered to STAF cutoff (not skin) and sorted by distance before inject (JMD ime convention).
- Ghost force contributions merge via pair **`reverse_comm`** (`comm_reverse=3`).
- Multi-rank allowed; `comm_modify cutoff` must be ≥ `max(rcut_r, rcut_a)` (1× cutoff DD).
- GPU pin: `staf_cuda_set_device` uses `local_rank % deviceCount` (shared GPU ok for smoke).
- Parity harness: `test/test-lammps-dd-parity/run_dd_parity.sh` — **PASS** np=1/2/4 on water smoke (`PE≈−99.53378`, max|ΔE|~1e-5, max|ΔF|~1e-5).
