# STAF-CG precision flags (staf_real.h lives in STAF/include)
if [ -z "$STAF_INC" ]; then
  _d="$(cd "$(dirname "$0")" && pwd)"
  while [ "$_d" != / ]; do
    if [ -f "$_d/include/staf_real.h" ]; then STAF_INC="$_d/include"; break; fi
    if [ -f "$_d/STAF/include/staf_real.h" ]; then STAF_INC="$_d/STAF/include"; break; fi
    _d="$(dirname "$_d")"
  done
fi
STAF_PREC_FLAGS="${STAF_PREC_FLAGS:--I${STAF_INC}}"

TF_CFLAGS=( $($5 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2>/dev/null ))
TF_LFLAGS=( $($5 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null ))

### Descriptor kernels (cell_list / interaction_map still linked in the CG tree)
$1 -arch=$6 -c -o reforce.cu.o reforce.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings ${STAF_PREC_FLAGS}
echo Compilo celle e interactionmaps
$1 -arch=$6 -c -o cell_list.cu.o cell_list.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings ${STAF_PREC_FLAGS}
$1 -arch=$6 -c -o interaction_map.cu.o interaction_map.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings ${STAF_PREC_FLAGS}
echo Compilo utilities
$1 -arch=$6 -c -o utilities.cu.o utilities.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings ${STAF_PREC_FLAGS}
echo Compilo la libreria
$2 -shared reforce.cc utilities.cu.o cell_list.cu.o interaction_map.cu.o reforce.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L $3 -I $4 -o reforce.so ${STAF_PREC_FLAGS}
