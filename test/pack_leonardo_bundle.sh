#!/usr/bin/env bash
# Build STAF_leonardo_test_bundle.tar.gz at repo root (datasets + YAML + howto).
# Does not include STAF source — clone git on Leonardo, then unpack this tar.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="${ROOT}/STAF_leonardo_test_bundle"
OUT="${ROOT}/STAF_leonardo_test_bundle.tar.gz"

echo "STAF: staging bundle under ${STAGE}"
rm -rf "${STAGE}"
mkdir -p \
  "${STAGE}/training/run_float" \
  "${STAGE}/training/run_double" \
  "${STAGE}/inference_models" \
  "${STAGE}/baselines"

# Howto (same text as in-repo LEONARDO.md)
cp "${ROOT}/test/LEONARDO.md" "${STAGE}/README.md"

# Expected numbers already measured on V100 (also tracked in git)
{
  echo "# Expected baselines (V100 reference — already in git)"
  echo ""
  echo "## Performance"
  cat "${ROOT}/test/test-training-pipeline/comparison/performance_baseline.txt"
  echo ""
  echo "## Distribute loss parity (lcurve_notmean, 1 epoch)"
  echo "### float"
  cat "${ROOT}/test/test-training-pipeline/parity_distribute/float/summary.txt"
  echo ""
  echo "### double"
  cat "${ROOT}/test/test-training-pipeline/parity_distribute/double/summary.txt"
  echo ""
  echo "## Inference compatibility (excerpt)"
  head -n 40 "${ROOT}/test/test-inference-pipeline/comparison_summary.txt" || true
} > "${STAGE}/EXPECTED_BASELINES.md"

# Copy baseline artifacts into tar for offline reading
cp -a "${ROOT}/test/test-training-pipeline/comparison/." "${STAGE}/baselines/training_comparison/"
mkdir -p "${STAGE}/baselines/parity_distribute"
cp -a "${ROOT}/test/test-training-pipeline/parity_distribute/float" \
      "${ROOT}/test/test-training-pipeline/parity_distribute/double" \
      "${STAGE}/baselines/parity_distribute/"
cp -a "${ROOT}/test/test-inference-pipeline/comparison_summary.txt" \
      "${STAGE}/baselines/" 2>/dev/null || true
cp -a "${ROOT}/test/ACCEPTANCE.md" "${STAGE}/baselines/ACCEPTANCE.md"
cp -a "${ROOT}/test/A3_PREP.md" "${STAGE}/baselines/A3_PREP.md"

copy_run() {
  local prec="$1"
  local src="${ROOT}/test/test-training-pipeline/run_${prec}"
  local dst="${STAGE}/training/run_${prec}"
  if [[ ! -d "${src}/dataset" ]]; then
    echo "ERROR: missing ${src}/dataset" >&2
    exit 1
  fi
  cp -a "${src}/dataset" "${dst}/"
  # Training YAMLs (skip generated parity_* if you prefer; include all input_*.yaml)
  shopt -s nullglob
  for y in "${src}"/input_*.yaml; do
    cp -a "$y" "${dst}/"
  done
  shopt -u nullglob
  # Point dataset_folder relative for portability inside extracted tree
  # (YAMLs already use dataset_folder: dataset)
}

copy_run float
copy_run double

# Inference SavedModels (force / compat on Leonardo)
if [[ -d "${ROOT}/test/test-inference-pipeline/model_float" ]]; then
  cp -a "${ROOT}/test/test-inference-pipeline/model_float" "${STAGE}/inference_models/"
fi
if [[ -d "${ROOT}/test/test-inference-pipeline/model_double" ]]; then
  cp -a "${ROOT}/test/test-inference-pipeline/model_double" "${STAGE}/inference_models/"
fi

# Install hint for overlaying onto a git clone
cat > "${STAGE}/INSTALL_OVER_CLONE.txt" <<'EOF'
After `git clone` of AlphaNesGpu on Leonardo:

  tar -xzf STAF_leonardo_test_bundle.tar.gz
  rsync -a STAF_leonardo_test_bundle/training/run_float/  test/test-training-pipeline/run_float/
  rsync -a STAF_leonardo_test_bundle/training/run_double/ test/test-training-pipeline/run_double/
  rsync -a STAF_leonardo_test_bundle/inference_models/model_float/  test/test-inference-pipeline/model_float/
  rsync -a STAF_leonardo_test_bundle/inference_models/model_double/ test/test-inference-pipeline/model_double/

Then follow README.md (same as test/LEONARDO.md in the repo).
EOF

echo "STAF: creating ${OUT}"
tar -C "${ROOT}" -czf "${OUT}" STAF_leonardo_test_bundle
du -sh "${OUT}" "${STAGE}"
echo "STAF: done. Transfer ${OUT} to Leonardo (scp/rsync). Staging dir left at ${STAGE}"
