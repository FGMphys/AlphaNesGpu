# USER-STAF — LAMMPS package for pair_style staf
#
# Links libstaf (CUDA AF/force + ONNX Runtime MLP). See:
#   ../../test/B_ARCHITECTURE.md
#   ../../libstaf/README.md
#
# Units: STAF models are trained in **eV** / **eV/Å**. For absolute pressure
# (NPT), use `units metal` (press in bars). `units real` is fine for E/F
# parity tests that only compare numerical PE/forces, but kinetic+virial
# pressure conversion will be wrong.
#
# Virial: `pair_staf` sets `no_virial_fdotr_compute=1` and tallies the diagonal
# virial from libstaf. Do not rely on `virial_fdotr_compute` (ghost force
# pieces live in `f_ghost` until reverse_comm).
#
# Install (after libstaf is built):
#   cd $LAMMPS_SRC
#   bash /path/to/AlphaNesGpu/lammps/USER-STAF/Install.sh
#   # then rebuild LAMMPS with Makefile that links -lstaf -lonnxruntime -lcudart ...
#
# Pressure smoke: `test/test-lammps-smoke/run_staf_press_check.sh`
