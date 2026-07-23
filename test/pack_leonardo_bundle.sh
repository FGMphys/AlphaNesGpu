#!/usr/bin/env bash
# Build STAF_leonardo_test_bundle.tar.gz at repo root.
# Includes the FULL MB-pol STAF datasets (float32 + float64, 2400 train / 300 test),
# acceptance YAMLs, production YAML + afsparam, inference SavedModels, baselines.
# STAF source is NOT included — git clone on Leonardo, then unpack this tar.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="${ROOT}/STAF_leonardo_test_bundle"
OUT="${ROOT}/STAF_leonardo_test_bundle.tar.gz"
AFSPARAM_SRC="${AFSPARAM_SRC:-/home/francegm/MBPOL_PROJECT/alphaGPU_223-248-278-stab-full/afsparam_2}"

echo "STAF: staging bundle under ${STAGE}"
rm -rf "${STAGE}"
mkdir -p \
  "${STAGE}/datasets" \
  "${STAGE}/training/run_float" \
  "${STAGE}/training/run_double" \
  "${STAGE}/production" \
  "${STAGE}/inference_models" \
  "${STAGE}/baselines"

# Howto
cp "${ROOT}/test/LEONARDO.md" "${STAGE}/README.md"

# --- Full MB-pol datasets (already float32/float64 in test pipeline; 2400+300 frames) ---
SRC_F="${ROOT}/test/test-training-pipeline/run_float/dataset"
SRC_D="${ROOT}/test/test-training-pipeline/run_double/dataset"
if [[ ! -d "${SRC_F}" || ! -d "${SRC_D}" ]]; then
  echo "ERROR: need test/test-training-pipeline/run_{float,double}/dataset" >&2
  exit 1
fi

echo "STAF: copying full float32 dataset → datasets/mbpol_full_float"
cp -a "${SRC_F}" "${STAGE}/datasets/mbpol_full_float"
echo "STAF: copying full float64 dataset → datasets/mbpol_full_double"
cp -a "${SRC_D}" "${STAGE}/datasets/mbpol_full_double"

# Hardlink trees into acceptance run_* so the tar stores each .npy once
echo "STAF: hardlinking datasets into training/run_* (single copy in tar)"
cp -al "${STAGE}/datasets/mbpol_full_float" "${STAGE}/training/run_float/dataset"
cp -al "${STAGE}/datasets/mbpol_full_double" "${STAGE}/training/run_double/dataset"

# Acceptance / smoke / parity YAMLs
for prec in float double; do
  src="${ROOT}/test/test-training-pipeline/run_${prec}"
  dst="${STAGE}/training/run_${prec}"
  shopt -s nullglob
  for y in "${src}"/input_*.yaml; do
    cp -a "$y" "${dst}/"
  done
  shopt -u nullglob
done

# --- afsparam ---
if [[ -d "${AFSPARAM_SRC}" ]]; then
  echo "STAF: copying afsparam_2 from ${AFSPARAM_SRC}"
  cp -a "${AFSPARAM_SRC}" "${STAGE}/production/afsparam_2"
else
  echo "WARN: afsparam not found at ${AFSPARAM_SRC}; production only_afs restart will need it" >&2
  mkdir -p "${STAGE}/production/afsparam_2"
fi

# Expected baselines
{
  echo "# Expected baselines (V100 reference — also tracked in git)"
  echo ""
  echo "## Performance"
  cat "${ROOT}/test/test-training-pipeline/comparison/performance_baseline.txt"
  echo ""
  echo "## Distribute loss parity (lcurve_notmean, 1 epoch, subsampled YAML)"
  echo "### float"
  cat "${ROOT}/test/test-training-pipeline/parity_distribute/float/summary.txt"
  echo ""
  echo "### double"
  cat "${ROOT}/test/test-training-pipeline/parity_distribute/double/summary.txt"
  echo ""
  echo "## Inference compatibility (excerpt)"
  head -n 40 "${ROOT}/test/test-inference-pipeline/comparison_summary.txt" || true
} > "${STAGE}/EXPECTED_BASELINES.md"

cp -a "${ROOT}/test/test-training-pipeline/comparison/." "${STAGE}/baselines/training_comparison/"
mkdir -p "${STAGE}/baselines/parity_distribute"
cp -a "${ROOT}/test/test-training-pipeline/parity_distribute/float" \
      "${ROOT}/test/test-training-pipeline/parity_distribute/double" \
      "${STAGE}/baselines/parity_distribute/"
cp -a "${ROOT}/test/test-inference-pipeline/comparison_summary.txt" \
      "${STAGE}/baselines/" 2>/dev/null || true
cp -a "${ROOT}/test/ACCEPTANCE.md" "${STAGE}/baselines/ACCEPTANCE.md"
cp -a "${ROOT}/test/A3_PREP.md" "${STAGE}/baselines/A3_PREP.md"

# Inference SavedModels
[[ -d "${ROOT}/test/test-inference-pipeline/model_float" ]] && \
  cp -a "${ROOT}/test/test-inference-pipeline/model_float" "${STAGE}/inference_models/"
[[ -d "${ROOT}/test/test-inference-pipeline/model_double" ]] && \
  cp -a "${ROOT}/test/test-inference-pipeline/model_double" "${STAGE}/inference_models/"

cat > "${STAGE}/INSTALL_OVER_CLONE.txt" <<'EOF'
After `git clone` of AlphaNesGpu on Leonardo:

  tar -xzf STAF_leonardo_test_bundle.tar.gz

# 1) Full datasets (transport once) — already inside the tar:
#    STAF_leonardo_test_bundle/datasets/mbpol_full_{float,double}/

# 2) Overlay acceptance paths used by scripts in the repo:
  rsync -a STAF_leonardo_test_bundle/training/run_float/  test/test-training-pipeline/run_float/
  rsync -a STAF_leonardo_test_bundle/training/run_double/ test/test-training-pipeline/run_double/
  rsync -a STAF_leonardo_test_bundle/inference_models/model_float/  test/test-inference-pipeline/model_float/
  rsync -a STAF_leonardo_test_bundle/inference_models/model_double/ test/test-inference-pipeline/model_double/

# 3) Production training (full set, no subsample), from bundle tree:
  cd STAF_leonardo_test_bundle/production
  # edit distribute / devices if needed, then e.g.:
  mpirun -np 4 python ../../STAF/staf_train.py input_staf_float.yaml

See README.md for the full test checklist.
EOF

export STAGE_DIR="${STAGE}"
export ROOT_DIR="${ROOT}"
STAGE_DIR="${STAGE}" ROOT_DIR="${ROOT}" python3 - <<'PY'
import numpy as np
from pathlib import Path
import os
stage = Path(os.environ["STAGE_DIR"])
out = stage / "datasets" / "DATASET_INFO.txt"
lines = [
    "MB-pol water — FULL STAF arrays (not a subsample on disk).",
    "Family: dataset_MBPOL_278_223_248 / alphaGPU_223-248-278-stab-full",
    "Layout: <dataset>/{type.dat, training/*.npy, test/*.npy}",
    "Acceptance YAMLs may set subsampling: 500 100; production uses subsampling: no.",
    "",
]
for name in ("mbpol_full_float", "mbpol_full_double"):
    base = stage / "datasets" / name
    lines.append(f"== {name} ==")
    for split in ("training", "test"):
        pos = np.load(base / split / "pos.npy", mmap_mode="r")
        ene = np.load(base / split / "energy.npy", mmap_mode="r")
        lines.append(
            f"  {split}: frames={pos.shape[0]}  pos={tuple(pos.shape)} {pos.dtype}  "
            f"energy={tuple(ene.shape)} {ene.dtype}"
        )
    lines.append(f"  type.dat: {(base / 'type.dat').read_text().strip()}")
    lines.append("")
out.write_text("\n".join(lines))
print(out.read_text())
PY

# Rewrite production YAMLs with portable relative paths
STAGE_DIR="${STAGE}" ROOT_DIR="${ROOT}" python3 - <<'PY'
import os
from pathlib import Path
import yaml
stage = Path(os.environ["STAGE_DIR"])
staf_yaml = Path(os.environ["ROOT_DIR"]) / "STAF" / "input_staf.yaml"
with open(staf_yaml) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

def dump(path, precision, dataset_rel, distribute):
    c = dict(cfg)
    c["precision"] = precision
    c["distribute"] = distribute
    c["dataset_folder"] = dataset_rel
    c["afs_param_folder"] = "afsparam_2"
    c["subsampling"] = "no"
    header = (
        f"# STAF Leonardo production input (run with cwd = production/).\n"
        f"# Full MB-pol dataset ({precision}): 2400 train / 300 test.\n"
        f"# Example: mpirun -np 4 python ../../STAF/staf_train.py {path.name}\n\n"
    )
    with open(path, "w") as f:
        f.write(header)
        yaml.dump(c, f, default_flow_style=False, sort_keys=False)

prod = stage / "production"
dump(prod / "input_staf_float.yaml", "float", "../datasets/mbpol_full_float", "horovod")
dump(prod / "input_staf_double.yaml", "double", "../datasets/mbpol_full_double", "horovod")
dump(prod / "input_staf_float_mirrored.yaml", "float", "../datasets/mbpol_full_float", "mirrored")
dump(prod / "input_staf_float_none.yaml", "float", "../datasets/mbpol_full_float", "none")
print("production YAMLs OK")
PY

echo "STAF: creating ${OUT} (hardlinks → one physical copy of each .npy)"
tar -C "${ROOT}" -czf "${OUT}" STAF_leonardo_test_bundle
du -sh "${OUT}" "${STAGE}"
echo "STAF: done → ${OUT}"
