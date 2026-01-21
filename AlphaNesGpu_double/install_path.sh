#!/bin/sh

#NVCC_PATH="/usr/local/cuda-11.2/bin/nvcc"
#GPP_PATH="/usr/bin/g++"
#CUDA_LIB64_PATH="/usr/local/cuda-11.2/lib64"
#CUDA_INCLUDE_PATH="/usr/local/cuda-11.2/include"

NVCC_PATH="/home/francegm/programmi/cuda/bin/nvcc" #11.8
GPP_PATH="/usr/bin/g++"
CUDA_LIB64_PATH="/home/francegm/programmi/cuda/lib64"
CUDA_INCLUDE_PATH="/home/francegm/programmi/cuda/include"
PYTHON_PATH="/home/francegm/miniconda3/envs/tensorgpu/bin/python"
COMPCAP=$($PYTHON_PATH get_compcap.py)
actual_path=$(pwd)

sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' source_routine/descriptor_builder.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' source_routine/mixture/physics_layer_mod.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' source_routine/mixture/force_layer_mod.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/mixture/register_3bAFs_grad.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/mixture/register_2bAFs_grad.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/mixture/register_force_2bAFs_grad.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/mixture/register_force_3bAFs_grad.py




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
