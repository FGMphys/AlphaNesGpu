# libstaf

C/C++/CUDA runtime for STAF MD (LAMMPS `pair_staf` and standalone).

## Default path

```text
CUDA AF  →  ONNX Runtime (MLP)  →  CUDA forces
```

- Public API: [`include/staf.h`](include/staf.h)
- MLP plugin API: [`include/staf_mlp.h`](include/staf_mlp.h)
- Architecture: [`../test/B_ARCHITECTURE.md`](../test/B_ARCHITECTURE.md)

## Status

| Component | Status |
|-----------|--------|
| `staf.h` / `staf_mlp.h` | API frozen (v0); default backend `STAF_MLP_ORT` |
| ORT backend | requires `energy`+`dE_daf` ONNX; CUDA EP required |
| Grad export | `STAF/save_models/export_mlp_grad_onnx.py` |
| CUDA AF/force | `vendor/jmd` + neuralmdGPU objects |
| GPU env | `../scripts/staf_gpu_env.sh` (portable CUDA/ORT/cuDNN discovery) |

## Build sketch

```bash
source scripts/staf_gpu_env.sh
cmake -S libstaf -B libstaf/build \
  -DSTAF_WITH_ORT=ON -DSTAF_WITH_JMD=ON \
  -DORT_ROOT="$ORT_ROOT" -DCUDA_HOME="$CUDA_HOME" -DCUDNN_LIB="$CUDNN_LIB" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX"
cmake --build libstaf/build -j
```

Override `CUDA_HOME` / `ORT_ROOT` / `CUDNN_LIB` on a new machine (or rely on discovery).
Without ORT (`-DSTAF_WITH_ORT=OFF`) only the native `.bin` backend is available.
