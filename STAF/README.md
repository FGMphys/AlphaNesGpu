# STAF (unified)

Single official tree for the Soft Two-body Angular Fingerprint potential.

```bash
# compile ops from STAF/src → ops_{float,double}/src
bash install_path.sh float
bash install_path.sh double
bash install_path.sh all

# train (precision from YAML)
python staf_train.py input_staf.yaml
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
| `src/` | **Single** CUDA/C++ source tree (`real` via `include/staf_real.h`) |
| `ops_float/`, `ops_double/` | Build outputs (gitignored); `.so` loaded at runtime |
| `staf_paths.py` | Resolves active ops root from precision |

Experimental CG / RDF forks remain under `../DEV/` (out of A2 scope).
