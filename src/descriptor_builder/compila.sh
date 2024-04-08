#export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2>/dev/null ))
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null ))

###Compilo i kernel (non sono chiamati cell_list o interaction_map)
/leonardo/prod/opt/compilers/cuda/11.8/none/bin/nvcc -std=c++14 -c -o reforce.cu.o reforce.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings

####Compilo cell_list e interaction_map
/leonardo/prod/opt/compilers/cuda/11.8/none/bin/nvcc -std=c++14 -c -o cell_list.cu.o cell_list.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings
/leonardo/prod/opt/compilers/cuda/11.8/none/bin/nvcc -std=c++14  -c -o interaction_map.cu.o interaction_map.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings
/leonardo/prod/opt/compilers/cuda/11.8/none/bin/nvcc -std=c++14  -c -o utilities.cu.o utilities.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings
#Linko tutto e compilo il main dell'operazione (reforce.cc)
g++ -std=c++14 -shared  reforce.cc utilities.cu.o cell_list.cu.o interaction_map.cu.o reforce.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /leonardo/prod/opt/compilers/cuda/11.8/none/lib64 -I /leonardo/prod/opt/compilers/cuda/11.8/none/include -o reforce.so
