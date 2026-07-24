#!/usr/bin/env bash
# Run STAF LAMMPS smoke with discovered CUDA/ORT/cuDNN (see scripts/staf_gpu_env.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=../../scripts/staf_gpu_env.sh
source "${ROOT}/scripts/staf_gpu_env.sh"

LMP="${LMP:-/home/francegm/programmi/lammps-23Jun2022/src/lmp_staf}"
cd "${ROOT}/test/test-lammps-smoke"
echo "Running: ${LMP} -in in.staf_smoke"
exec "${LMP}" -in in.staf_smoke "$@"
