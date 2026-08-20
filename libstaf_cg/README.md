# libstaf_cg

C/C++/CUDA runtime for STAF-CG origami MD (future LAMMPS `pair_staf/cg`).

CUDA AF / force come from
`neuralmdGPU/DEV/CG_and_WCA_LJ2_inter/src/` (dual cutoff + `map_intra` +
color maps), compiled from source in this tree. **Not** the full-atom
`libstaf/vendor/jmd` tree and **not** prebuilt neuralmdGPU `.o` files.

## Default path

```text
CUDA AF (intra/inter cutoffs, color maps)  →  ONNX Runtime MLP  →  CUDA forces
```

- Public API: [`include/staf.h`](include/staf.h) (`staf_load` / `staf_compute`)
- MLP: [`include/staf_mlp.h`](include/staf_mlp.h) (same as libstaf)
- ONNX export: [`../STAF-CG/save_models/export_mlp_grad_onnx.py`](../STAF-CG/save_models/export_mlp_grad_onnx.py)
- **float32 ONNX** — FP64 ORT training export is broken; matches full-atom libstaf

## CG model directory

Besides `model_type{k}.onnx` / `mlp_type{k}.bin` and `type{k}_alpha_*.dat`:

| File | Role |
|------|------|
| `color_type_map.dat` | bead → origami color |
| `map_color_interaction.dat` | sticky color pairing |
| `map_intra.dat` | intra vs inter molecule id per bead |
| `cutoff_info` | Rc, buffers, Rs, Rc_inter / Ra_inter / Rs_inter |
| `number_of_nn.dat` | number of MLPs (usually 1) |

`staf_load` writes `.staf_jmd_auto.cfg` with dual cutoffs and those map paths.
Beads are **not** type-sorted by chemical species; colors come from the maps.

WCA/LJ extras from the origami MD tree are not linked. This is STAF-only.

Ghosts: `staf_compute` with a LAMMPS neighbor list supports `nall = nlocal + nghost`
(PBC / DD reverse_comm). Bead colors for ghosts come from LAMMPS tags (`tag-1` index
into `color_type_map.dat`). Owned `nlocal` must equal the model bead count (empty
ranks skip compute). Sprint 5 1-rank `howmany==NULL` path is unchanged.

Sprint 5 used a **1-epoch keras checkpoint** (`net_model_type0`) for ONNX.
`export_mlp_grad_onnx.py` also accepts MODEL1896-style `model_type*` SavedModels
(rebuilds Dense + writes `type{k}_alpha_mu.dat`). Parity vs Python STAF-CG on
the 24-bead frame: `python test/test-cg-libstaf/run_model1896_md_parity.py`.

## Build

```bash
source scripts/staf_gpu_env.sh
cmake -S libstaf_cg -B libstaf_cg/build \
  -DSTAF_WITH_ORT=ON -DSTAF_WITH_JMD=ON \
  -DORT_ROOT="$ORT_ROOT" -DCUDA_HOME="$CUDA_HOME" -DCUDNN_LIB="$CUDNN_LIB" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=sm_70
cmake --build libstaf_cg/build -j
```

## 1-frame parity

```bash
python test/test-cg-libstaf/run_python_vs_libstaf.py
```
