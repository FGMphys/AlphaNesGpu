# A1 residual notes (historical)

Audit date: 2026-07-23. Originally against `AlphaNesGpu_double`.

Superseded for the official full-atom path by **A2**: single `STAF/src/`
with `STAF/include/staf_real.h` (`real`, `staf_exp`, `sizeof(real)`).

## Status after A2

| Item | Status |
|------|--------|
| D1 float math on double buffers | Fixed via `staf_real` / `staf_exp` |
| D2 energy-only `loss_force` dtype | Fixed via `staf.dtype` |
| D7 hardcoded `root_path` | Fixed via `staf_paths.code_root()` |
| D8 divergent `__syncthreads` | Mitigated (see freeze commit); proper block reduce still optional |

## Deferred

- **jmd_nn / neuralmdGPU parity** → Linea **B**
- **DEV/ CG** → Linea **C**
