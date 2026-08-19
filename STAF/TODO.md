# STAF TODOs

## Virial parity: TF inference vs libstaf/ORT

**Status:** done (2026-07-30)  
**Bug fixed:** TF virial kernels treated Cartesian `pos` as fractional (rint MIC).
Now convert cart→frac with the same inverse-box as the descriptor / `staf_api`
(`staf_force_pbc.cuh::staf_min_image_cart_from_cart`).

**Parity (model_log172, test frame 200 / T223):**
`|W_TF - W_ORT| ≈ 9e-4` (rel ~1e-4); `|E_TF - E_ORT| ≈ 5e-5`.
See `STAF_sweep_md/md/diag/long_run_virial_eval/parity_tf_ort_frame200.json`.

**Related code:**
- `STAF/src/force/{rad,ang}/reforce*.{cc,cu.cc}` — `*Virial` ops
- `STAF/source_routine/force_layer_mod.py` — `force_virial_layer`
- `STAF/staf_models/staf_model_inference_full.py`
- `libstaf/src/runtime/staf_virial_batch.cpp`

## Energy + force + virial training

**Status:** implemented (2026-07-31)

- Full W tensor `(batch,9)` row-major, `W_ab -= f_a * r_b` (MIC cart)
- Labels: `virial.npy` total eV, **no `/N`**; hard-fail if missing
- `type_of_training: energy+force+virial`, `loss_virial_prefactor` (default 1, Huber)
- Grad: `ComputeForce*VirialGrad` + RegisterGradient
- Run folder: `STAF_sweep_md/best/set2/long_run_efv_fullW/`
- FD test: `test/test-force-virial-grad/`
