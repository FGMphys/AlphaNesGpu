#!/bin/sh

NVCC_PATH="put here full path to nvcc compiler"
GPP_PATH="put here full path to g++ compiler"
CUDA_LIB64_PATH="put here full path to cuda/lib64"
CUDA_INCLUDE_PATH="put here full path to cuda/include"


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
bash compila.sh $NVCC_PATH $GPP_PATH $CUDA_LIB64_PATH $CUDA_INCLUDE_PATH
cd ../..

cd src/mixture
echo Compiling for mixtures
for folder in $(ls -d *)
do
echo Compiling folder $folder radial 
cd $folder'/rad'
bash compila.sh $NVCC_PATH $GPP_PATH $CUDA_LIB64_PATH $CUDA_INCLUDE_PATH
cd ../..
echo Compiling folder $folder radial
cd $folder'/ang'
bash compila.sh $NVCC_PATH $GPP_PATH $CUDA_LIB64_PATH $CUDA_INCLUDE_PATH
cd ../..
done

cd ..
