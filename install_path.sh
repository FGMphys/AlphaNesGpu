#!/bin/sh
actual_path=$(pwd)

sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' source_routine/descriptor_builder.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' source_routine/mixture/physics_layer_mod.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' source_routine/notype/physics_layer_mod.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' source_routine/mixture/force_layer_mod.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' source_routine/notype/force_layer_mod.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/mixture/register_3bAFs_grad.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/mixture/register_2bAFs_grad.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/mixture/register_force_2bAFs_grad.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/mixture/register_force_3bAFs_grad.py

sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/notype/register_3bAFs_grad.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/notype/register_2bAFs_grad.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/notype/register_force_3bAFs_grad.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/notype/register_force_2bAFs_grad.py
sed -i   's@root_path=.*@root_path='"\'$actual_path\'"'@' debug_mode/debug_alpha_force.py


cd src/mixture
echo Compiling for mixtures
for folder in $(ls -d *)
do
echo Compiling folder $folder radial 
cd $folder'/rad'
bash compila.sh ubu
cd ../..
echo Compiling folder $folder radial
cd $folder'/ang'
bash compila.sh ubu
cd ../..
done

cd ..
cd descriptor_builder
echo Compiling Descriptors
bash compila.sh ubu
cd ../../..
