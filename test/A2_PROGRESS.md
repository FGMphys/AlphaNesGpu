# A2 progress notes

## Closed (2026-07-23) — official full-atom path

| Slice | Deliverable |
| --- | --- |
| 1 | Naming: `staf_models`, `staf_train`, STAF logs |
| 2 | `staf/dtype.py` + YAML `precision` (package under `STAF/staf/`) |
| 3 | Unified Python tree `STAF/` (no `mixture/` nesting) |
| 4 | `include/staf_real.h` (`real`, `staf_exp`, `sizeof(real)`) |
| 5 | **Single CUDA `STAF/src/`**; build → `ops_{float,double}/` (gitignored) |
| — | Removed `AlphaNesGpu_{float,double}/` redirects |
| 6 | Pre-A3 hygiene: flatten `src/mixture/`, quarantine develop, `staf_infer`, metrics CSV/JSONL |

### Final gates (post single-`src/` rebuild, V100)

| Gate | float | double |
| --- | --- | --- |
| Force FD δ=0.001 | corr=0.99983 | corr=0.99994 |
| `time_story` 1-epoch | 89.0 ms/frame (×0.97 vs 91.5) | 156.5 ms/frame (×1.05 vs 149.7) |

See `test/A3_PREP.md` for CUDA static-state audit before multi-GPU.

## Out of A2 scope (deferred)

- **`DEV/`** CG forks (Linea C, after A6) and RDF (Linea E, waiting on latex) — not migrated
- A4 CPU MPI (A3 CUDA ctx + A5 Horovod are in tree)

A2 exit criterion for full-atom: one install command, float **or** double, regression + performance gates green — **met**.
