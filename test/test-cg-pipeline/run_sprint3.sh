#!/bin/bash
# Sprint 3 gates: double then float, never in parallel.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${STAF_CG_PYTHON:-$REPO/.venv/bin/python}"
INFER="$REPO/test/test-cg-inference"
FORCE="$REPO/test/test-cg-regression/regression-force"
GRAD="$REPO/test/test-cg-regression/regression-grad-param"
export PYTHONPATH="$REPO/STAF-CG:$REPO/STAF${PYTHONPATH:+:$PYTHONPATH}"

echo "===== 0. frames + MODEL1896 stage ====="
"$PY" "$INFER/prepare_frames.py"
"$PY" "$INFER/stage_model1896.py"

echo "===== 1. force FD double (MODEL1896) ====="
"$PY" "$FORCE/run_force_regression.py" --precision double --model "$INFER/model1896_infer"

echo "===== 2. 1-epoch subsample ====="
bash "$HERE/run_one_epoch.sh"

echo "===== 3. export double+float + RMSE vs freeze ====="
"$PY" "$HERE/export_and_check.py"

echo "===== 4. infer double then float ====="
"$PY" "$INFER/run_inference.py" --precision double --model "$INFER/model_double"
"$PY" "$INFER/run_inference.py" --precision float --model "$INFER/model_float"
"$PY" "$INFER/analyze_compatibility.py"

echo "===== 5. force FD float (1-epoch float export) ====="
"$PY" "$FORCE/run_force_regression.py" --precision float --model "$INFER/model_float"

echo "===== 6. grad-param double ====="
"$PY" "$GRAD/run_grad_param_regression.py" --precision double --n-per-family 20

echo "===== Sprint 3 gates finished ====="
