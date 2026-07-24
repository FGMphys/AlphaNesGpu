# LAMMPS STAF B1 smoke

## What works
- Export `model_log0` → ONNX (forward) + `mlp_type{k}.bin` (analytical ∂E/∂AF)
- `libstaf`: native Dense MLP + JMD CUDA AF/force (`nn_nn_mlp.cu`)
- `lmp_staf`: `pair_style staf` single-rank NVE

## Run
```bash
# MLP-only
./libstaf/build/staf_mlp_smoke test/test-lammps-smoke/model_onnx_float

# Force eval
./libstaf/build/staf_force_smoke test/test-lammps-smoke/model_onnx_double \
  test/test-lammps-smoke/data.water_smoke

# LAMMPS 5 steps
cd test/test-lammps-smoke
mpirun -np 1 /path/to/lmp_staf -in in.staf_smoke
```

## Notes
- B1 is **1 MPI rank** only (no DD/ghost STAF yet).
- MLP path: ONNX export is forward-only; runtime uses `.bin` + analytical tanh backprop (ORT optional).
- Energy drift on 5 steps from a jittered frame is expected; parity vs `jmd_nn` is next.
