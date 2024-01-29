TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


conda_prefix=$CONDA_PREFIX

sysop="$1"
if [[ "$sysop" == "mac" ]]
then
   undef="dynamic_lookup"
fi

if [[ "$sysop" == "ubu" ]]
then
   undef=''
fi


g++ -std=c++14 -shared alphagrad_2body.cc -o bin/alphagrad_2body.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm -undefined $undef -I$conda_prefix/include

g++ -std=c++14 -shared op_2bAFs.cc -o bin/op_2bAFs.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm -undefined $undef -I$conda_prefix/include

g++ -std=c++14 -shared op_force_2bAFs.cc -o bin/op_force_2bAFs.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm -undefined $undef -I$conda_prefix/include

g++ -std=c++14 -shared op_grad_of_force_2bAFs.cc -o bin/op_grad_of_force_2bAFs.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm -undefined $undef -I$conda_prefix/include

#g++ -std=c++14 -shared op_force_and_pressure_2bAFs.cc -o bin/op_force_and_pressure_2bAFs.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm -undefined $undef -I$conda_prefix/include
