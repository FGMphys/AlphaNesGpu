#gcc op_force_3bAFs.c -lm -o op_force_3bAFs.o
#g++ -std=c++14 -shared tfinterface.cc -o tfinterface.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm -undefined $undef -I$conda_prefix/include


export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

####Compilo i kernel (non sono chiamati cell_list o interaction_map)
#/usr/local/cuda/bin/nvcc -std=c++14 -c -o reforce.cu.o reforce.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings
#
####Compilo cell_list e interaction_map
#/usr/local/cuda/bin/nvcc -std=c++14 -c -o cell_list.cu.o cell_list.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings
#/usr/local/cuda/bin/nvcc -std=c++14  -c -o interaction_map.cu.o interaction_map.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings
#/usr/local/cuda/bin/nvcc -std=c++14  -c -o utilities.cu.o utilities.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings
###Linko tutto e compilo il main dell'operazione (reforce.cc)
g++ -std=c++14 -shared  reforce.cc utilities.cu.o cell_list.cu.o interaction_map.cu.o reforce.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64 -I /usr/local/cuda/include -o reforce.so


