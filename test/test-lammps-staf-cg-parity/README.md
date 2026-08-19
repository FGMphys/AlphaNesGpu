# LAMMPS `pair_style staf/cg` smoke + Python parity + DD

24-bead origami dimer (USCGSITE training frame 0, cubic box 280 Å) vs Python
STAF-CG on the Sprint 5 float32 ONNX export
(`test/test-cg-inference/model_onnx_double/`, 1-epoch keras checkpoint).

## Run

```bash
source /home/francegm/AlphaNesGpu/scripts/staf_gpu_env.sh
export LMP_CG=${LMP_CG:-/home/francegm/programmi/lammps-23Jun2022/src/lmp_staf_cg}

./run_smoke.sh
python3 run_compare.py          # required: E, F, P_config; exit 1 on fail
./run_dd_parity.sh              # np=1 vs 2 vs 4; skip np>1 if CUDA ctx crashes
```

## Pass (Python vs LAMMPS)

| Quantity | Tol |
|----------|-----|
| max\|ΔE\| | < 1e-3 |
| max\|ΔF\| per component | < 1e-3 |
| configurational P | \|ΔP\|/max(\|P\|,1) < 0.05 **or** max\|ΔW_diag\| < 1e-2 |

`p_gate` in `summary.json` records which pressure criterion passed.
Pressure is pair virial only (`run 0`, velocities 0). Units metal: nktv2p =
1.60217662e6 **bar** per eV/Å³ (LAMMPS metal; not atm).

DD vs np=1: max\|ΔE\|, max\|ΔF\|, \|ΔP\| ≤ 1e-4 (same as full-atom
`test/test-lammps-dd-parity`). Ranks may share the single V100.

## Files

| File | Role |
|------|------|
| `data.origami24` | 24 beads, type 1, box 280 |
| `in.smoke` | `staf/cg 50 50 float`, `comm_modify cutoff 50`, `newton on`, `run 0` |
| `run_smoke.sh` | 1-rank driver (`LMP_CG`) |
| `run_compare.py` | Python STAF-CG vs LAMMPS |
| `run_dd_parity.sh` | 1 vs 2 vs 4 MPI ranks |
