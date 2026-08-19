#!/usr/bin/env bash
# 1-rank STAF-CG LAMMPS smoke (24-bead origami dimer, run 0).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../../scripts/staf_gpu_env.sh
source "${ROOT}/scripts/staf_gpu_env.sh"

LMP_CG="${LMP_CG:-/home/francegm/programmi/lammps-23Jun2022/src/lmp_staf_cg}"
# Writable-tree fallback from Sprint 6 rebuild.
if [[ ! -x "${LMP_CG}" ]]; then
  for cand in \
    "${ROOT}/tmp/lammps-staf-cg/src/lmp_staf_cg" \
    /tmp/lammps-staf-cg/src/lmp_staf_cg
  do
    if [[ -x "${cand}" ]]; then
      LMP_CG="${cand}"
      break
    fi
  done
fi

MODEL="${MODEL:-${ROOT}/test/test-cg-inference/model_onnx_double}"
DUMP="${DUMP:-${HERE}/forces.dump}"
LOG="${LOG:-${HERE}/log.lammps}"

if [[ ! -x "${LMP_CG}" ]]; then
  echo "run_smoke: ERROR: lmp_staf_cg not executable: ${LMP_CG}" >&2
  echo "  Set LMP_CG or rebuild (see lammps/USER-STAF-CG/README.md)" >&2
  exit 1
fi
if [[ ! -d "${MODEL}" ]]; then
  echo "run_smoke: ERROR: missing model dir ${MODEL}" >&2
  exit 1
fi

cd "${HERE}"
echo "run_smoke: ${LMP_CG}  model=${MODEL}"
exec "${LMP_CG}" -in in.smoke -log "${LOG}" \
  -var modeldir "${MODEL}" -var dumpfile "${DUMP}" "$@"
