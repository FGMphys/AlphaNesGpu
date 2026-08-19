#!/bin/bash
# Copy USER-STAF-CG sources into a LAMMPS src/ tree (alongside pair_staf).
# Usage:
#   bash Install.sh /path/to/lammps/src
#   # or, from inside $LAMMPS/src/USER-STAF-CG (LAMMPS package mode):
#   bash Install.sh 1

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DST="${1:-}"

if [[ -z "${DST}" ]]; then
  echo "Usage: $0 /path/to/lammps/src" >&2
  echo "   or: $0 0|1|2   (uninstall/install/update when this dir is src/USER-STAF-CG)" >&2
  exit 1
fi

# LAMMPS package Install.sh protocol: mode 0/1/2, files live in this directory.
if [[ "${DST}" == "0" || "${DST}" == "1" || "${DST}" == "2" ]]; then
  mode="${DST}"
  action() {
    local f="$1"
    if [[ "${mode}" == "0" ]]; then
      rm -f "../${f}"
    elif ! cmp -s "${HERE}/${f}" "../${f}" 2>/dev/null; then
      cp -f "${HERE}/${f}" "../${f}"
      if [[ "${mode}" == "2" ]]; then
        echo "  updating src/${f}"
      fi
    fi
  }
  action pair_staf_cg.h
  action pair_staf_cg.cpp
  echo "USER-STAF-CG: package mode ${mode} (pair_staf_cg alongside pair_staf)"
  exit 0
fi

if [[ ! -d "${DST}" ]]; then
  echo "USER-STAF-CG: destination is not a directory: ${DST}" >&2
  exit 1
fi

cp -f "${HERE}/pair_staf_cg.h" "${HERE}/pair_staf_cg.cpp" "${DST}/"
if [[ -d "${DST}/MAKE/MINE" && -f "${HERE}/Makefile.staf_cg.example" ]]; then
  cp -f "${HERE}/Makefile.staf_cg.example" "${DST}/MAKE/MINE/Makefile.staf_cg"
  echo "USER-STAF-CG: installed Makefile.staf_cg into ${DST}/MAKE/MINE/"
fi
echo "USER-STAF-CG: installed pair_staf_cg into ${DST} (does not replace pair_staf)"
echo "USER-STAF-CG: rebuild with Makefile.staf_cg linking libstaf_cg (see Makefile.staf_cg.example)"
echo "USER-STAF-CG:   cd ${DST} && source \$STAF_ROOT/scripts/staf_gpu_env.sh && make staf_cg -j"
echo "USER-STAF-CG: produces lmp_staf_cg; do not overwrite lmp_staf"
