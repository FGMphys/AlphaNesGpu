#!/usr/bin/env bash
# Portable STAF GPU environment discovery.
#
# Source this file (do not execute):
#   source /path/to/AlphaNesGpu/scripts/staf_gpu_env.sh
#
# Honors overrides if already set: CUDA_HOME, CUDNN_LIB, ORT_ROOT, STAF_ROOT.
# On a new machine, either set those or rely on the probe order below.

# Resolve repo root from this script location unless STAF_ROOT is set.
_STAF_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
: "${STAF_ROOT:=$(cd "${_STAF_ENV_DIR}/.." && pwd)}"
export STAF_ROOT

_staf_has_lib() {
  local dir="$1" pattern="$2"
  [ -d "$dir" ] || return 1
  compgen -G "${dir}/${pattern}" >/dev/null 2>&1
}

_staf_cuda_major_from_home() {
  local home="$1" ver=""
  if [ -f "${home}/version.json" ]; then
    ver=$(sed -n 's/.*"cuda"[[:space:]]*:[[:space:]]*"\([0-9][0-9]*\)\..*/\1/p' "${home}/version.json" | head -1)
  fi
  if [ -z "$ver" ] && [ -x "${home}/bin/nvcc" ]; then
    ver=$("${home}/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)
  fi
  if [ -z "$ver" ] && [ -f "${home}/version.txt" ]; then
    ver=$(sed -n 's/.*CUDA Version \([0-9][0-9]*\)\..*/\1/p' "${home}/version.txt" | head -1)
  fi
  echo "${ver:-}"
}

# --- CUDA_HOME ---------------------------------------------------------------
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
    CUDA_HOME="${CONDA_PREFIX}"
  elif command -v nvcc >/dev/null 2>&1; then
    _nvcc=$(command -v nvcc)
    # .../bin/nvcc -> ...
    CUDA_HOME=$(cd "$(dirname "${_nvcc}")/.." && pwd)
  else
    for cand in \
      /home/francegm/programmi/cuda \
      /usr/local/cuda \
      /usr/local/cuda-12.8 \
      /usr/local/cuda-12.6 \
      /usr/local/cuda-12.4 \
      /usr/local/cuda-12.2 \
      /usr/local/cuda-12.1 \
      /usr/local/cuda-12.0 \
      /usr/local/cuda-11.8 \
      /usr/local/cuda-11.7 \
      /usr/local/cuda-11.4
    do
      if [ -x "${cand}/bin/nvcc" ] || [ -d "${cand}/lib64" ]; then
        CUDA_HOME="${cand}"
        break
      fi
    done
  fi
fi
export CUDA_HOME="${CUDA_HOME:-}"

_STAF_CUDA_MAJOR=""
if [ -n "${CUDA_HOME}" ]; then
  _STAF_CUDA_MAJOR=$(_staf_cuda_major_from_home "${CUDA_HOME}")
fi
export STAF_CUDA_MAJOR="${_STAF_CUDA_MAJOR}"

# --- CUDNN_LIB ---------------------------------------------------------------
if [ -z "${CUDNN_LIB:-}" ]; then
  if [ -n "${CUDA_HOME}" ] && _staf_has_lib "${CUDA_HOME}/lib64" "libcudnn.so*"; then
    CUDNN_LIB="${CUDA_HOME}/lib64"
  elif [ -n "${CUDA_HOME}" ] && _staf_has_lib "${CUDA_HOME}/lib" "libcudnn.so*"; then
    CUDNN_LIB="${CUDA_HOME}/lib"
  elif _staf_has_lib "/home/francegm/programmi/cudaNN/cuda-11.8/lib" "libcudnn.so*"; then
    CUDNN_LIB="/home/francegm/programmi/cudaNN/cuda-11.8/lib"
  else
    for cand in \
      /home/francegm/programmi/cudaNN/cuda-12*/lib \
      /home/francegm/programmi/cudnn*/lib \
      /usr/local/cudnn/lib \
      /usr/lib/x86_64-linux-gnu
    do
      # shellcheck disable=SC2086
      for d in ${cand}; do
        if _staf_has_lib "${d}" "libcudnn.so*"; then
          CUDNN_LIB="${d}"
          break 2
        fi
      done
    done
  fi
fi
export CUDNN_LIB="${CUDNN_LIB:-}"

# --- ORT_ROOT ----------------------------------------------------------------
# Prefer CUDA11 ORT when CUDA major is 11 (or unknown on this box); CUDA12 ORT
# when libcublas.so.12 is resolvable / CUDA major >= 12.
_ort11="${STAF_ROOT}/third_party/onnxruntime-cuda11"
_ort12="${STAF_ROOT}/third_party/onnxruntime"

_staf_ort_ok() {
  local root="$1"
  [ -f "${root}/lib/libonnxruntime.so" ] || [ -f "${root}/lib/libonnxruntime.so.1" ]
}

if [ -z "${ORT_ROOT:-}" ]; then
  _want12=0
  if [ "${STAF_CUDA_MAJOR}" = "12" ] || [ "${STAF_CUDA_MAJOR}" = "13" ]; then
    _want12=1
  elif [ -n "${CUDA_HOME}" ] && _staf_has_lib "${CUDA_HOME}/lib64" "libcublas.so.12*"; then
    _want12=1
  fi
  if [ "${_want12}" = "1" ] && _staf_ort_ok "${_ort12}"; then
    ORT_ROOT="${_ort12}"
  elif _staf_ort_ok "${_ort11}"; then
    ORT_ROOT="${_ort11}"
  elif _staf_ort_ok "${_ort12}"; then
    ORT_ROOT="${_ort12}"
  fi
fi
export ORT_ROOT="${ORT_ROOT:-}"

# --- PATH / LD_LIBRARY_PATH / CUDACXX ----------------------------------------
if [ -n "${CUDA_HOME}" ] && [ -d "${CUDA_HOME}/bin" ]; then
  case ":${PATH}:" in
    *":${CUDA_HOME}/bin:"*) ;;
    *) PATH="${CUDA_HOME}/bin:${PATH}" ;;
  esac
  export PATH
  if [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
  fi
fi

_prepend_ld() {
  local d="$1"
  [ -n "$d" ] && [ -d "$d" ] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${d}:"*) ;;
    *) LD_LIBRARY_PATH="${d}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

_prepend_ld "${ORT_ROOT:+${ORT_ROOT}/lib}"
_prepend_ld "${CUDNN_LIB}"
_prepend_ld "${CUDA_HOME:+${CUDA_HOME}/lib64}"
_prepend_ld "${CUDA_HOME:+${CUDA_HOME}/lib}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# --- One-line summary --------------------------------------------------------
_ort_ver="?"
if [ -f "${ORT_ROOT}/VERSION_NUMBER" ]; then
  _ort_ver=$(tr -d '[:space:]' < "${ORT_ROOT}/VERSION_NUMBER")
fi
echo "staf_gpu_env: CUDA_HOME=${CUDA_HOME:-unset} (major=${STAF_CUDA_MAJOR:-?}) CUDNN_LIB=${CUDNN_LIB:-unset} ORT_ROOT=${ORT_ROOT:-unset} (v${_ort_ver})"

unset -f _staf_has_lib _staf_cuda_major_from_home _staf_ort_ok _prepend_ld 2>/dev/null || true
unset _STAF_ENV_DIR _STAF_CUDA_MAJOR _nvcc _ort11 _ort12 _want12 _ort_ver 2>/dev/null || true
