# Experimental / unwired CUDA

Not part of `install_path.sh` or the production Python path.

| Path | Notes |
|------|--------|
| `descriptor_builder_develop/` | Alternate descriptor builder with heavy `static` GPU state. Quarantined before A3 so it is not rsync’d into `ops_*`. |

Do not load these `.so` from training or inference.
