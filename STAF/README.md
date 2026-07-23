# STAF (unified)

Single official tree for the Soft Two-body Angular Fingerprint potential.

```bash
# compile ops (float, double, or both)
bash install_path.sh float    # → ops_float/src/**/reforce.so
bash install_path.sh double
bash install_path.sh all

# train (precision from YAML)
python staf_train.py input_staf.yaml
```

YAML key:

```yaml
precision: float   # or double / float32 / float64
```

Layout:

| Path | Role |
|------|------|
| `staf_models/` | Training + inference models (no `mixture/` nesting) |
| `source_routine/` | Descriptor, physics, force layers |
| `gradient_utility/` | Custom-op gradient registrations |
| `ops_float/`, `ops_double/` | Compiled CUDA `.so` (until `real` unify) |
| `staf_paths.py` | Resolves active ops root from precision |

Deprecated entry points: `../AlphaNesGpu_float/`, `../AlphaNesGpu_double/` (thin redirects).
