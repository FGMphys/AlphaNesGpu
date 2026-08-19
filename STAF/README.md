# STAF-AI Potential (unified)

Official code tree for **STAF-AI Potential — Self Trained Atomic Fingerprint AI Potential**.

```bash
# compile ops from STAF/src → ops_{float,double}/src
bash install_path.sh float
bash install_path.sh double
bash install_path.sh all

# train (precision from YAML)
python staf_train.py input_staf.yaml
python staf_infer.py --model MODEL --precision double --pos pos.npy --box box.npy
python staf_infer.py --model MODEL --precision float --pos pos.npy --box box.npy --decompose
```

YAML:

```yaml
precision: float   # or double / float32 / float64
```

Layout:

| Path | Role |
|------|------|
| `staf_models/` | Training + inference models |
| `source_routine/` | Descriptor, physics, force layers |
| `gradient_utility/` | Custom-op gradient registrations |
| `src/` | **Single** CUDA/C++ source tree (`fingerprint`/`force`/`grad_*`, `real` via `include/staf_real.h`) |
| `staf/` | Shared helpers (`dtype`, precision, **A6** isolated-cluster MBE) |
| `experimental/` | Unwired CUDA (not built) |
| `ops_float/`, `ops_double/` | Build outputs (gitignored); `.so` loaded at runtime |
| `staf_paths.py` | Resolves active ops root from precision |

Experimental CG / RDF forks remain under `../DEV/` as historical freeze. Official CG port: `../STAF-CG/` ([`DEV/STAF_CG_SPRINTS.md`](../DEV/STAF_CG_SPRINTS.md)). RDF reweighting → **Linea E** (`docs/PIANO_ALPHANES.md`); waiting on latex.

## A6 multi-body inference

Isolated n-atom vacuum clusters (cliques within `rcut`); sum of STAF **energies** only:

```bash
python staf_infer.py --model MODEL --precision float --pos pos.npy --box box.npy --decompose --max-body 3
```

TODO(FGM): closed-form 2-body from AF parameters (latex). See `docs/PIANO_ALPHANES.md` §A6.

## MD export (Linea B, ONNX)

Dense MLP for LAMMPS/`libstaf` (ORT), no custom ops in the graph:

```bash
python save_models/export_mlp_onnx.py -imodel model_log0 -modelname model_onnx
# requires: pip install tf2onnx
```

See `../test/B_ARCHITECTURE.md`. Legacy SavedModel: `save_models/save_model.py`.

