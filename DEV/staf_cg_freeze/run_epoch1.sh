#!/bin/bash
# 1-epoch subsample freeze on DEV CG (does not edit the DEV source tree).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DEV_CG="$(cd "$HERE/../AlphaNesGpu_double_CG_dv_RC" && pwd)"
USCGSITE=/home/francegm/ORIGAMI/INFER_INTRA_TRY2+USCGSITE
MODEL1896=/home/francegm/ORIGAMI/ORIGAMI_DYNAMICS/origami_uscgsite/models/MODEL1896
PY="${STAF_CG_PYTHON:-/home/francegm/miniconda3/envs/tensorgpu/bin/python}"

cd "$HERE"
rm -rf dataset_uscgsite
mkdir -p dataset_uscgsite
ln -s "$USCGSITE/dataset/training" dataset_uscgsite/training
ln -s "$USCGSITE/dataset/test" dataset_uscgsite/test
cp "$MODEL1896/color_type_map.dat" dataset_uscgsite/color_type_map.dat
cp "$MODEL1896/map_intra.dat" map_intra.dat
cp "$MODEL1896/map_color_interaction.dat" map_color_interaction.dat

export PYTHONPATH="$DEV_CG${PYTHONPATH:+:$PYTHONPATH}"
echo "FREEZE: 1-epoch DEV train CWD=$HERE PYTHONPATH=$DEV_CG"
"$PY" "$DEV_CG/alpha_nnpes_full_main.py" "$HERE/input_epoch1.yaml"
echo "FREEZE: 1-epoch done"
if [ -f lcurve.out ]; then
  echo "===== lcurve.out ====="
  cat lcurve.out
fi
if [ -f lcurve_notmean ]; then
  echo "===== lcurve_notmean (tail) ====="
  tail -20 lcurve_notmean
fi
