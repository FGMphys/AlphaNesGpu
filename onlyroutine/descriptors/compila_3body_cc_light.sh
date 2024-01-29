#!/bin/bash
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


g++ -std=c++14 -shared ops_descriptors2and3body.c interaction_map.c cell_list.c utilities.c log.c events.c  -o ops_descriptors.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -lm -undefined $undef -I$conda_prefix/include 
