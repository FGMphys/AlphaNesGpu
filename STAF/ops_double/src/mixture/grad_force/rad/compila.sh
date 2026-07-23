# STAF precision (set STAF_INC by install_path.sh; fallback walk-up)
if [ -z "$STAF_INC" ]; then
  _d="$(cd "$(dirname "$0")" && pwd)"
  while [ "$_d" != / ]; do
    if [ -f "$_d/include/staf_real.h" ]; then STAF_INC="$_d/include"; break; fi
    if [ -f "$_d/../include/staf_real.h" ]; then STAF_INC="$_d/../include"; break; fi
    _d="$(dirname "$_d")"
  done
fi
STAF_PREC_FLAGS="-I${STAF_INC} -DSTAF_REAL_DOUBLE"


TF_CFLAGS=( $($5 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2>/dev/null ))
TF_LFLAGS=( $($5 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null ))

$1  -arch=$6 -c -o reforce.cu.o reforce.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED --disable-warnings ${STAF_PREC_FLAGS}

$2 -shared -o reforce.so reforce.cc reforce.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L $3 -I $4 ${STAF_PREC_FLAGS}
