# LAMMPS STAF MPI domain-decomposition parity

Compares total potential energy and per-atom forces from `pair_style staf` on the
same water smoke system (`test/test-lammps-smoke/data.water_smoke`) for **1, 2, and 4
MPI ranks**. Reference is always the 1-rank run.

## Prerequisites

- Built `lmp_staf` (or set `LMP` / `LMP_MPI`)
- Model: `test/test-lammps-smoke/model_onnx_grad_float`
- GPU stack: `source scripts/staf_gpu_env.sh`

## Run

```bash
source /home/francegm/AlphaNesGpu/scripts/staf_gpu_env.sh
./test/test-lammps-dd-parity/run_dd_parity.sh
```

Manual single-rank (default dump path):

```bash
cd test/test-lammps-dd-parity
lmp_staf -in in.dd_parity -var dumpfile forces.dump
```

Optional overrides:

| Variable | Default | Meaning |
|----------|---------|---------|
| `LMP` | `.../lmp_staf` | Serial / 1-rank binary |
| `LMP_MPI` | `$LMP` | MPI-capable binary for `np>1` |
| `DD_PARITY_E_TOL` | `1e-4` | max \|ΔE\| vs 1-rank |
| `DD_PARITY_F_TOL` | `1e-4` | max \|ΔF\| vs 1-rank |

Outputs land in `results/np{1,2,4}/` (`log.lammps`, `forces.dump`).

If `mpirun`/`mpiexec` is missing or the MPI binary fails a 2-rank probe, the script
**skips** `np=2,4` with a message but still verifies the 1-rank case and exits 0.

## What is exercised

- `in.dd_parity`: `read_data`, `pair_style staf 4.5 4.5 float`, `comm_modify cutoff 5.0`,
  `thermo pe`, `dump custom id type fx fy fz`, then `run 0` (single force evaluation).
- Domain decomposition uses the **LAMMPS neighbor list** (full + ghost atoms) and
  **`reverse_comm`** to accumulate ghost force contributions back onto owned atoms.
- `pair_staf` sets `cutghost = max(rcut)` (here 4.5 Å) for ghost communication.

## Pass criteria

`max|ΔE| ≤ 1e-4` and `max|ΔF| ≤ 1e-4` for each multi-rank case vs the 1-rank reference.
