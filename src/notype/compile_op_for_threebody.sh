TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

sysop="$1"
if [[ "$sysop" == "mac" ]]
then
   undef="dynamic_lookup"
fi

if [[ "$sysop" == "ubu" ]]
then
   undef=''
fi

conda_prefix=$CONDA_PREFIX

g++ -std=c++14 -shared op_3bAFs.cc -o bin/op_3bAFs.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm -undefined $undef -I$conda_prefix/include

g++ -std=c++14 -shared alphagrad_3body.cc -o bin/alphagrad_3body.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm -undefined $undef -I$conda_prefix/include

g++ -std=c++14 -shared op_force_3bAFs.cc -o bin/op_force_3bAFs.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm -undefined $undef -I$conda_prefix/include

g++ -std=c++14 -shared op_grad_of_force_3bAFs.cc -o bin/op_grad_of_force_3bAFs.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm -undefined $undef -I$conda_prefix/include

#g++ -std=c++14 -shared  op_force_and_pressure_3bAFs.cc -o bin/op_force_and_pressure_3bAFs.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm -undefined $undef -I$conda_prefix/include
