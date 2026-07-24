# vendor/jmd — neuralmdGPU patched for libstaf MLP

This directory contains a patched copy of the GPU-MD kernel from
`neuralmdGPU/full_atom/src/`, with TensorFlow replaced by the libstaf native
Dense MLP (`StafMlp`).

## Files in this directory

| File | Origin | Notes |
|---|---|---|
| `nn_nn_mlp.cu` | `nn_nn.cu` | **Main patch** – TF removed, libstaf MLP wired in |
| `nn_nn.h` | `nn_nn.h` | TF types removed; `staf_jmd_set_mlp()` added |
| `vector.h` | `vector.h` | GSL guard added (`JMD_NO_GSL`) |
| `interaction_map.h` | verbatim | |
| `celle_gpu.h` | verbatim | |
| `global_definitions.h` | verbatim | |
| `nn_io.h` | verbatim | |
| `nn_smart_allocator.h` | verbatim | |
| `nn_smart_allocator_gpu.h` | verbatim | |
| `io.h` | verbatim | |
| `secure_search.h` | verbatim | |
| `src_nn/descriptor_builder/reforce.h` | verbatim | |
| `src_nn/fingerprint/rad/reforce.h` | verbatim | |
| `src_nn/fingerprint/ang/reforce.h` | verbatim | |
| `src_nn/force/rad/reforce.h` | verbatim | |
| `src_nn/force/ang/reforce.h` | verbatim | |

## Compilation of nn_nn_mlp.cu

```bash
nvcc -dc -O2 -std=c++17 \
     -I. \
     -I<AlphaNesGpu>/libstaf/include \
     nn_nn_mlp.cu -o nn_nn_mlp.o
```

`-I.` picks up the local header copies.  
`-I<AlphaNesGpu>/libstaf/include` exposes `staf_mlp.h` and `staf.h`.

## Object files to link from neuralmdGPU

Link **all** `.o` files listed below from `neuralmdGPU/full_atom/src/`
**except `nn_nn.o`** (that TF-linked object is replaced by `nn_nn_mlp.o`):

```
alle_pairs.o
bilista.o
celle_gpu.o
cell_list.o
cluster.o
interaction_map.o
io.o
lennard_jones.o
log.o
main.o
md.o
nn.o
nn_io.o
nn_smart_allocator.o
nn_smart_allocator_gpu.o
order_parameters.o
ordinator.o
random.o
restart.o
saving_scale.o
secure_search.o
sus.o
sw.o
thermostats.o
vector.o
src_nn/descriptor_builder/reforce.o
src_nn/fingerprint/rad/reforce.o
src_nn/fingerprint/ang/reforce.o
src_nn/force/rad/reforce.o
src_nn/force/ang/reforce.o
```

Plus the replacement:

```
<AlphaNesGpu>/libstaf/vendor/jmd/nn_nn_mlp.o
```

## Linker flags

```
-L<AlphaNesGpu>/libstaf/build -lstaf \
-lcuda -lcudart \
-lgsl -lgslcblas \
-lm
```

## Usage from C/C++

```c
#include "libstaf/include/staf_mlp.h"
#include "libstaf/vendor/jmd/nn_nn.h"

// 1. Create the MLP (precision=1 → double; matches neuralmdGPU)
StafMlp *mlp = staf_mlp_create(STAF_MLP_NATIVE, "/path/to/model_dir",
                                /*precision=*/1, /*device=*/0);

// 2. Hand it to the JMD layer before calling initializenn_
staf_jmd_set_mlp(mlp);

// 3. Initialise descriptors and AF buffers from config
FILE *cfg = fopen("input_nn.dat", "r");
initializenn_(cfg, N);
fclose(cfg);

// 4. MD loop
calculateforces(pos, box, ime, &energy, force, &virial, &virialxyz);

// 5. Teardown
nnDestructor();
staf_mlp_destroy(mlp);
```

## Key design notes

* **Double precision only** — all AF, gradient, and energy arrays are `double`.
  `staf_mlp_eval` is called with `precision=1` (`af_f64` / `dE_daf_f64`).
* **AF packing** — `Compute_NNEnergyandGradient_all` packs all types'
  AFs into one contiguous host buffer (type 0 atoms first, then type 1, …),
  calls `staf_mlp_eval` once, then scatters per-type gradients back to device.
  This matches `StafMlpEval`'s packed layout contract.
* **`Compute_NNEnergyandGradient(type, …)`** is kept for API completeness
  but delegates to `_all`; it should only be used when `NumTypes == 1`.
* **`NNmodelRoot`** is still read from the config file and used by
  `Constructor_AFS` to load alpha/embedding files.  The MLP weights
  (`mlp_type{k}.bin` or `.onnx`) are loaded independently by the caller
  through `staf_mlp_create`.
* **`deletetensor` / `nnDestructor`** are no-ops for TF resources;
  `nnDestructor` frees only the host/device gradient buffers.
