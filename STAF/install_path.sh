#!/bin/sh
# Compile STAF custom ops from the single source tree STAF/src
# into precision-specific output trees ops_{float,double}/src.
#
# Usage: ./install_path.sh [float|double|all]
# Default: all

set -e
PRECISION="${1:-all}"
STAF_HOME="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$STAF_HOME/.." && pwd)"
export STAF_INC="$STAF_HOME/include"
STAF_SRC="$STAF_HOME/src"

# Prefer env overrides, then common local CUDA installs, then PATH.
NVCC_PATH="${STAF_NVCC:-${NVCC:-}}"
if [ -z "$NVCC_PATH" ] && [ -x /home/francegm/programmi/cuda/bin/nvcc ]; then
  NVCC_PATH="/home/francegm/programmi/cuda/bin/nvcc"
fi
if [ -z "$NVCC_PATH" ]; then
  NVCC_PATH="$(command -v nvcc || true)"
fi
if [ -z "$NVCC_PATH" ] || [ ! -x "$NVCC_PATH" ]; then
  echo "STAF: nvcc not found; set STAF_NVCC or install CUDA" >&2
  exit 1
fi

CUDA_HOME="$(cd "$(dirname "$NVCC_PATH")/.." && pwd)"
GPP_PATH="${STAF_GPP:-/usr/bin/g++}"
CUDA_LIB64_PATH="${STAF_CUDA_LIB64:-$CUDA_HOME/lib64}"
CUDA_INCLUDE_PATH="${STAF_CUDA_INCLUDE:-$CUDA_HOME/include}"

if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON_PATH="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_PATH="$(command -v python3)"
else
  PYTHON_PATH="/home/francegm/miniconda3/envs/tensorgpu/bin/python"
fi
COMPCAP=$("$PYTHON_PATH" "$STAF_HOME/get_compcap.py")
export PATH="$(dirname "$PYTHON_PATH"):$PATH"

# Op families under STAF/src/ (no intermediate mixture/ folder).
OP_FAMILIES="fingerprint force grad_finger grad_force"

compile_ops() {
  prec="$1"   # float | double
  out_dir="$STAF_HOME/ops_${prec}"
  echo "==== Compiling $prec → $out_dir (from $STAF_SRC) ===="

  if [ "$prec" = "double" ]; then
    export STAF_PREC_FLAGS="-I${STAF_INC} -DSTAF_REAL_DOUBLE"
  else
    export STAF_PREC_FLAGS="-I${STAF_INC}"
  fi

  mkdir -p "$out_dir"
  # Sync sources into the output tree, then compile in-place (writes .so next to sources).
  rsync -a --delete \
    --exclude='*.so' --exclude='*.o' --exclude='nohup.out' \
    "$STAF_SRC/" "$out_dir/src/"
  # rsync --exclude='*.so' leaves stale .so trees; drop known leftovers.
  rm -rf "$out_dir/src/mixture" "$out_dir/src/descriptor_builder_develop"

  cd "$out_dir"
  cd src/descriptor_builder
  echo Compiling Descriptors
  rm -f *.o *.so
  bash compila.sh "$NVCC_PATH" "$GPP_PATH" "$CUDA_LIB64_PATH" "$CUDA_INCLUDE_PATH" "$PYTHON_PATH" "$COMPCAP"
  cd ../..

  echo Compiling fingerprint/force/grad libraries
  for folder in $OP_FAMILIES; do
    echo "Compiling folder $folder radial"
    cd "src/$folder/rad"
    rm -f *.o *.so
    bash compila.sh "$NVCC_PATH" "$GPP_PATH" "$CUDA_LIB64_PATH" "$CUDA_INCLUDE_PATH" "$PYTHON_PATH" "$COMPCAP"
    cd ../../..
    echo "Compiling folder $folder angular"
    cd "src/$folder/ang"
    rm -f *.o *.so
    bash compila.sh "$NVCC_PATH" "$GPP_PATH" "$CUDA_LIB64_PATH" "$CUDA_INCLUDE_PATH" "$PYTHON_PATH" "$COMPCAP"
    cd ../../..
  done
  cd "$STAF_HOME"
}

case "$PRECISION" in
  float|float32)
    compile_ops float
    ;;
  double|float64)
    compile_ops double
    ;;
  all)
    compile_ops float
    compile_ops double
    ;;
  *)
    echo "Usage: $0 [float|double|all]" >&2
    exit 1
    ;;
esac

echo "STAF: install_path done ($PRECISION)"
