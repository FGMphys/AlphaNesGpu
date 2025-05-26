NVCC_PATH=/leonardo/prod/opt/compilers/cuda/11.8/none/bin/nvcc
GPP_PATH=/usr/bin/g++
CUDA_INCLUDE_PATH=/leonardo/prod/opt/compilers/cuda/11.8/none/include
CUDA_LIB64_PATH=/leonardo/prod/opt/compilers/cuda/11.8/none/lib64


python=/leonardo/pub/userexternal/fguidare/python_envs/tensorgpu/bin/python
TF_CFLAGS=( $($python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2>/dev/null ))
TF_LFLAGS=( $($python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null ))

$1  -arch=sm_70 -c -o reforce.cu.o reforce.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings

$2 -shared -o reforce.so reforce.cc reforce.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L $3 -I $4
