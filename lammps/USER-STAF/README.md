# USER-STAF — LAMMPS package for pair_style staf
#
# Links libstaf (CUDA AF/force + ONNX Runtime MLP). See:
#   ../../test/B_ARCHITECTURE.md
#   ../../libstaf/README.md
#
# Install (after libstaf is built):
#   cd $LAMMPS_SRC
#   bash /path/to/AlphaNesGpu/lammps/USER-STAF/Install.sh
#   # then rebuild LAMMPS with Makefile that links -lstaf -lonnxruntime -lcudart ...
