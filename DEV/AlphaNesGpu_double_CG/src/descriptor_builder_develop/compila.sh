#export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2>/dev/null ))
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null ))

###Compilo i kernel (non sono chiamati cell_list o interaction_map)
$1  -arch=sm_70 -c -o reforce.cu.o reforce.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings

####Compilo cell_list e interaction_map
#$1 -arch=sm_70 -c -o cell_list.cu.o cell_list.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings
#$1  -arch=sm_70 -c -o interaction_map.cu.o interaction_map.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings
#$1   -arch=sm_70 -c -o utilities.cu.o utilities.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings
#Linko tutto e compilo il main dell'operazione (reforce.cc)
$1 -arch=sm_70 -c -o celle_gpu.cu.o celle_gpu.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings
$2 -shared  reforce.cc celle_gpu.cu.o reforce.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L $3 -I $4 -o reforce.so
