#!/bin/bash
# Copy USER-STAF sources into a LAMMPS src/ tree.
# Usage: bash Install.sh /path/to/lammps/src

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DST="${1:-}"

if [[ -z "${DST}" || ! -d "${DST}" ]]; then
  echo "Usage: $0 /path/to/lammps/src" >&2
  exit 1
fi

cp -f "${HERE}/pair_staf.h" "${HERE}/pair_staf.cpp" "${DST}/"
cp -f "${HERE}/fix_staf.h" "${HERE}/fix_staf.cpp" "${DST}/"
echo "USER-STAF: installed pair_staf + fix_staf into ${DST}"
echo "USER-STAF: rebuild LAMMPS linking libstaf + onnxruntime (see Makefile.staf.example)"
