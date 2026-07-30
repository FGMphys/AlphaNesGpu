# STAF TODOs

## Virial parity: TF inference vs libstaf/ORT

**Status:** pending  
**Why:** Diagonal virial was added to TF force ops (`ComputeForceRadialVirial` /
`ComputeForceTriplVirial`) and wired through `staf_model_inference_full`.
A smoke run returns finite `(E, F, W)`, but we still need a **numeric parity**
check against the already-validated jmd path.

**Test to run (when GPU is free):**
1. Export a float checkpoint with `save_models/save_model_in_float.py`
   (needs `model_type*`, `cutoff_info`, `type.dat`).
2. Same frame(s): TF `staf_infer.py --precision float` → `virial_diag`.
3. Same model/frame via ONNX + `libstaf/build/staf_virial_batch` (or
   `STAF_sweep_md/scripts/compute_rmse_p.py`).
4. Assert `|W_TF - W_jmd|` small (same order as force float noise; expect
   agreement on `Wxx,Wyy,Wzz` to ~1e-3 relative or better for double).

**Related code:**
- `STAF/src/force/{rad,ang}/reforce*.{cc,cu.cc}` — `*Virial` ops
- `STAF/source_routine/force_layer_mod.py` — `force_virial_layer`
- `STAF/staf_models/staf_model_inference_full.py`
- `libstaf/src/runtime/staf_virial_batch.cpp`
