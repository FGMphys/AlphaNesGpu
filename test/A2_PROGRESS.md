# A2 progress notes

## Closed (2026-07-23) — official full-atom path

| Slice | Deliverable |
| --- | --- |
| 1 | Naming: `staf_models`, `staf_train`, STAF logs |
| 2 | `staf/dtype.py` + YAML `precision` |
| 3 | Unified Python tree `STAF/` (no `mixture/` nesting) |
| 4 | `include/staf_real.h` (`real`, `staf_exp`, `sizeof(real)`) |
| 5 | **Single CUDA `STAF/src/`**; build → `ops_{float,double}/` (gitignored) |
| — | Removed `AlphaNesGpu_{float,double}/` redirects |

### Final gates (post single-`src/` rebuild, V100)

| Gate | float | double |
| --- | --- | --- |
| Force FD δ=0.001 | corr=0.99983 | corr=0.99994 |
| `time_story` 1-epoch | 89.0 ms/frame (×0.97 vs 91.5) | 156.5 ms/frame (×1.05 vs 149.7) |

Logs use `STAF:` (no `Alpha_nes:`). Both performance ratios ≤ 1.15.

## Out of A2 scope (deferred)

- **`DEV/`** CG / RDF forks (Linea C) — not migrated; keep using their local trees
- Structured CSV/JSON logging, dead-file cleanup polish (A2.2 leftovers)
- A3+ multi-GPU / A4 CPU MPI

A2 exit criterion for full-atom: one install command, float **or** double, regression + performance gates green — **met**.
