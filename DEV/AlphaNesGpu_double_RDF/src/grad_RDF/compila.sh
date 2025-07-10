
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2>/dev/null ))
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null ))


g++ -shared -o reforce.so reforce.cc  calculate_gr_reweight_grad.c ${TF_CFLAGS[@]} -fPIC ${TF_LFLAGS[@]} 
