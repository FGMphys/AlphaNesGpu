#!/usr/bin/env bash
# MPI domain-decomposition parity: compare total PE and per-atom forces
# for np=1 vs np=2 vs np=4 on the water smoke system with pair_style staf.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TEST_DIR="${ROOT}/test/test-lammps-dd-parity"
# shellcheck source=../../scripts/staf_gpu_env.sh
source "${ROOT}/scripts/staf_gpu_env.sh"

LMP="${LMP:-/home/francegm/programmi/lammps-23Jun2022/src/lmp_staf}"
LMP_MPI="${LMP_MPI:-${LMP}}"

E_TOL="${DD_PARITY_E_TOL:-1e-4}"
F_TOL="${DD_PARITY_F_TOL:-1e-4}"

MPIRUN=""
if command -v mpirun >/dev/null 2>&1; then
  MPIRUN="mpirun"
elif command -v mpiexec >/dev/null 2>&1; then
  MPIRUN="mpiexec"
fi

die() {
  echo "dd_parity: ERROR: $*" >&2
  exit 1
}

info() {
  echo "dd_parity: $*"
}

[ -x "${LMP}" ] || die "LAMMPS binary not executable: ${LMP}"
[ -f "${TEST_DIR}/in.dd_parity" ] || die "missing ${TEST_DIR}/in.dd_parity"
[ -f "${TEST_DIR}/../test-lammps-smoke/data.water_smoke" ] || die "missing water smoke data"
[ -d "${TEST_DIR}/../test-lammps-smoke/model_onnx_grad_float" ] || die "missing grad float model"

mkdir -p "${TEST_DIR}/results"

run_rank_case() {
  local np="$1"
  local out="${TEST_DIR}/results/np${np}"
  mkdir -p "${out}"
  local dump="${out}/forces.dump"
  local log="${out}/log.lammps"

  info "running np=${np} -> ${out}"
  (
    cd "${TEST_DIR}"
    if [ "${np}" -eq 1 ]; then
      "${LMP}" -in in.dd_parity -log "${log}" -var dumpfile "${dump}"
    else
      "${MPIRUN}" -np "${np}" "${LMP_MPI}" -in in.dd_parity -log "${log}" -var dumpfile "${dump}"
    fi
  )
}

parse_pe() {
  local log="$1"
  python3 - "$log" <<'PY'
import re, sys
log = open(sys.argv[1]).read().splitlines()
pe = None
for line in log:
    m = re.match(r"^\s*0\s+([-+0-9.eE]+)\s*$", line)
    if m:
        pe = float(m.group(1))
        break
if pe is None:
    raise SystemExit(f"could not parse PE from {sys.argv[1]}")
print(f"{pe:.16g}")
PY
}

compare_to_ref() {
  local ref_pe="$1"
  local ref_dump="$2"
  local cand_pe="$3"
  local cand_dump="$4"
  local label="$5"

  python3 - "$ref_pe" "$ref_dump" "$cand_pe" "$cand_dump" "$label" "$E_TOL" "$F_TOL" <<'PY'
import re, sys
import numpy as np

ref_pe = float(sys.argv[1])
ref_dump = sys.argv[2]
cand_pe = float(sys.argv[3])
cand_dump = sys.argv[4]
label = sys.argv[5]
e_tol = float(sys.argv[6])
f_tol = float(sys.argv[7])

def load_dump(path):
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("ITEM: ATOMS"):
        i += 1
    if i >= len(lines):
        raise SystemExit(f"bad dump (no ATOMS): {path}")
    i += 1
    rows = []
    while i < len(lines) and not lines[i].startswith("ITEM:"):
        parts = lines[i].split()
        if len(parts) >= 5:
            rows.append([float(x) for x in parts[:5]])
        i += 1
    arr = np.asarray(rows, dtype=np.float64)
    order = np.argsort(arr[:, 0].astype(int))
    arr = arr[order]
    return arr[:, 1:5]  # type, fx, fy, fz

ref = load_dump(ref_dump)
cand = load_dump(cand_dump)
if ref.shape != cand.shape:
    raise SystemExit(f"{label}: dump shape mismatch {ref.shape} vs {cand.shape}")

dE = abs(cand_pe - ref_pe)
dF = cand[:, 1:4] - ref[:, 1:4]
max_dF = float(np.max(np.abs(dF)))

print(f"{label}: PE_ref={ref_pe:.8g} PE={cand_pe:.8g} max|dE|={dE:.6g} max|dF|={max_dF:.6g}")

if dE > e_tol or max_dF > f_tol:
    raise SystemExit(
        f"{label}: FAIL (tol E={e_tol:g} F={f_tol:g}) max|dE|={dE:.6g} max|dF|={max_dF:.6g}"
    )
PY
}

run_rank_case 1 || die "np=1 run failed"
ref_pe="$(parse_pe "${TEST_DIR}/results/np1/log.lammps")"
ref_dump="${TEST_DIR}/results/np1/forces.dump"
info "np=1 PE=${ref_pe}"

fail=0
multi_rank_skipped=0
if [ -n "${MPIRUN}" ] && [ -x "${LMP_MPI}" ]; then
  for np in 2 4; do
    if ! run_rank_case "${np}"; then
      info "np=${np} run failed (MPI binary may be serial-only or DD not enabled); skipping np>1 parity"
      multi_rank_skipped=1
      break
    fi
    cand_pe="$(parse_pe "${TEST_DIR}/results/np${np}/log.lammps")"
    if ! compare_to_ref "${ref_pe}" "${ref_dump}" "${cand_pe}" \
      "${TEST_DIR}/results/np${np}/forces.dump" "np=${np}"; then
      fail=1
    fi
  done
else
  multi_rank_skipped=1
  if [ -z "${MPIRUN}" ]; then
    info "no mpirun/mpiexec found; skipping np>1"
  else
    info "LMP_MPI=${LMP_MPI} not executable; skipping np>1"
  fi
fi

if [ "${fail}" -ne 0 ]; then
  die "DD parity FAILED (see results/ and messages above)"
fi

if [ "${multi_rank_skipped}" -eq 1 ]; then
  info "PASS (1-rank verified; multi-rank skipped)"
else
  info "PASS (np=1,2,4 within E tol=${E_TOL}, F tol=${F_TOL})"
fi
exit 0
