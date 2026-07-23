#!/bin/sh
# Compile STAF custom ops into ops_float and/or ops_double.
# Usage: ./install_path.sh [float|double|all]
# Default: all

set -e
PRECISION="${1:-all}"
STAF_HOME="$(cd "$(dirname "$0")" && pwd)"
export STAF_INC="$STAF_HOME/include"
REPO_ROOT="$(cd "$STAF_HOME/.." && pwd)"

NVCC_PATH="/home/francegm/programmi/cuda/bin/nvcc"
GPP_PATH="/usr/bin/g++"
CUDA_LIB64_PATH="/home/francegm/programmi/cuda/lib64"
CUDA_INCLUDE_PATH="/home/francegm/programmi/cuda/include"

if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON_PATH="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_PATH="/home/francegm/miniconda3/envs/tensorgpu/bin/python"
fi
COMPCAP=$("$PYTHON_PATH" "$STAF_HOME/get_compcap.py")
export PATH="$(dirname "$PYTHON_PATH"):$PATH"

compile_ops() {
  ops_dir="$1"
  echo "==== Compiling $ops_dir ===="
  cd "$ops_dir"
  cd src/descriptor_builder
  echo Compiling Descriptors
  rm -f *.o *.so
  bash compila.sh "$NVCC_PATH" "$GPP_PATH" "$CUDA_LIB64_PATH" "$CUDA_INCLUDE_PATH" "$PYTHON_PATH" "$COMPCAP"
  cd ../..

  cd src/mixture
  echo Compiling fingerprint/force/grad libraries
  for folder in $(ls -d *); do
    echo "Compiling folder $folder radial"
    cd "$folder/rad"
    rm -f *.o *.so
    bash compila.sh "$NVCC_PATH" "$GPP_PATH" "$CUDA_LIB64_PATH" "$CUDA_INCLUDE_PATH" "$PYTHON_PATH" "$COMPCAP"
    cd ../..
    echo "Compiling folder $folder angular"
    cd "$folder/ang"
    rm -f *.o *.so
    bash compila.sh "$NVCC_PATH" "$GPP_PATH" "$CUDA_LIB64_PATH" "$CUDA_INCLUDE_PATH" "$PYTHON_PATH" "$COMPCAP"
    cd ../..
  done
  cd "$STAF_HOME"
}

case "$PRECISION" in
  float|float32)
    compile_ops "$STAF_HOME/ops_float"
    ;;
  double|float64)
    compile_ops "$STAF_HOME/ops_double"
    ;;
  all)
    compile_ops "$STAF_HOME/ops_float"
    compile_ops "$STAF_HOME/ops_double"
    ;;
  *)
    echo "Usage: $0 [float|double|all]" >&2
    exit 1
    ;;
esac

echo "STAF: install_path done ($PRECISION)"
