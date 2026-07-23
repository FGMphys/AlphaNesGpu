#!/bin/sh

#NVCC_PATH="/usr/local/cuda-11.2/bin/nvcc"
#GPP_PATH="/usr/bin/g++"
#CUDA_LIB64_PATH="/usr/local/cuda-11.2/lib64"
#CUDA_INCLUDE_PATH="/usr/local/cuda-11.2/include"

NVCC_PATH="/home/francegm/programmi/cuda/bin/nvcc" #11.8
GPP_PATH="/usr/bin/g++"
CUDA_LIB64_PATH="/home/francegm/programmi/cuda/lib64"
CUDA_INCLUDE_PATH="/home/francegm/programmi/cuda/include"
# Prefer repo-local .venv if present
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON_PATH="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_PATH="/home/francegm/miniconda3/envs/tensorgpu/bin/python"
fi
COMPCAP=$($PYTHON_PATH get_compcap.py)
export PATH="$(dirname "$PYTHON_PATH"):$PATH"

# Custom-op .so paths are resolved at runtime via alphanes_paths.code_root()
# (optional override: ALPHANES_DOUBLE_ROOT). No sed of Python sources.




cd src
cd descriptor_builder
echo Compiling Descriptors
rm *.o *.so
bash compila.sh $NVCC_PATH $GPP_PATH $CUDA_LIB64_PATH $CUDA_INCLUDE_PATH $PYTHON_PATH $COMPCAP
cd ../..

cd src/mixture
echo Compiling for mixtures
for folder in $(ls -d *)
do
echo Compiling folder $folder radial 
cd $folder'/rad'
rm *.o *.so
bash compila.sh $NVCC_PATH $GPP_PATH $CUDA_LIB64_PATH $CUDA_INCLUDE_PATH $PYTHON_PATH $COMPCAP
cd ../..
echo Compiling folder $folder radial
cd $folder'/ang'
rm *.o *.so
bash compila.sh $NVCC_PATH $GPP_PATH $CUDA_LIB64_PATH $CUDA_INCLUDE_PATH $PYTHON_PATH $COMPCAP
cd ../..
done

cd ..
