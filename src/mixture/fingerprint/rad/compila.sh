#export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2>/dev/null ))
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null ))

/leonardo/prod/opt/compilers/cuda/11.8/none/bin/nvcc -std=c++14 -c -o reforce.cu.o reforce.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings

g++ -std=c++14 -shared -o reforce.so reforce.cc reforce.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /leonardo/prod/opt/compilers/cuda/11.8/none/lib64 #-I /usr/local/cuda/include
