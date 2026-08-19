#!/bin/bash
# 1-epoch subsample on official STAF-CG (Seed 60), same YAML as DEV freeze.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
USCGSITE=/home/francegm/ORIGAMI/INFER_INTRA_TRY2+USCGSITE
MODEL1896=/home/francegm/ORIGAMI/ORIGAMI_DYNAMICS/origami_uscgsite/models/MODEL1896
PY="${STAF_CG_PYTHON:-$REPO/.venv/bin/python}"
WORK="$HERE/work"

rm -rf "$WORK"
mkdir -p "$WORK/dataset_uscgsite"
ln -s "$USCGSITE/dataset/training" "$WORK/dataset_uscgsite/training"
ln -s "$USCGSITE/dataset/test" "$WORK/dataset_uscgsite/test"
cp "$MODEL1896/color_type_map.dat" "$WORK/dataset_uscgsite/color_type_map.dat"
cp "$MODEL1896/map_intra.dat" "$WORK/map_intra.dat"
cp "$MODEL1896/map_color_interaction.dat" "$WORK/map_color_interaction.dat"
cp "$HERE/input_epoch1.yaml" "$WORK/input_epoch1.yaml"

export PYTHONPATH="$REPO/STAF-CG:$REPO/STAF${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORK"
echo "STAF-CG: 1-epoch train CWD=$WORK"
"$PY" "$REPO/STAF-CG/staf_cg_train.py" input_epoch1.yaml
echo "STAF-CG: 1-epoch done"
if [ -f lcurve.out ]; then
  echo "===== lcurve.out ====="
  cat lcurve.out
fi
