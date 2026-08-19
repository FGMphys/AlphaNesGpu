# STAF-CG (origami dual-cutoff)

Official CG tree, sibling of [`STAF/`](../STAF/). Ported from
[`DEV/AlphaNesGpu_double_CG_dv_RC/`](../DEV/AlphaNesGpu_double_CG_dv_RC/)
(freeze; do not edit). Sprint checklist: [`DEV/STAF_CG_SPRINTS.md`](../DEV/STAF_CG_SPRINTS.md).

```bash
cd STAF-CG
bash install_path.sh all          # ops_float + ops_double from src/
python staf_cg_train.py input_staf_cg.yaml   # precision: float|double
python staf_cg_infer.py --model MODEL --precision double --pos pos.npy --box box.npy
```

YAML `precision: float|double`. CUDA uses `real` via [`STAF/include/staf_real.h`](../STAF/include/staf_real.h). Horovod: `distribute: none|horovod` (no LR×N). Virial: `type_of_training: energy+force+virial`.

Regression gates: [`test/ACCEPTANCE.md`](../test/ACCEPTANCE.md) (STAF-CG section). One-shot:

```bash
bash test/test-cg-pipeline/run_sprint3.sh
```

| Path | Role |
|------|------|
| `staf_cg_models/` | Training + inference models |
| `source_routine/` | Descriptor, physics, force layers (flattened; no `mixture/`) |
| `gradient_utility/` | Custom-op gradient registrations |
| `src/` | Single CUDA/C++ source tree (`fingerprint` / `force` / `grad_*`) |
| `ops_float/`, `ops_double/` | Build outputs (gitignored) |
| `staf_cg_paths.py` | Resolves active ops root |

Do **not** load `STAF/ops_*` from this tree. Dual cutoff (`Rc` / `Rc_Inter`), `map_intra`, and sticky colors stay here, not in `STAF/src/`.

C/CUDA MD runtime: [`libstaf_cg/`](../libstaf_cg/) (ONNX MLP + CG dual-cutoff kernels). Export: `save_models/export_mlp_grad_onnx.py` (float32).

LAMMPS (Sprint 6): `pair_style staf/cg` in [`lammps/USER-STAF-CG/`](../lammps/USER-STAF-CG/). Binary `lmp_staf_cg` (do not overwrite `lmp_staf`). Origami 1-epoch intra `Rc=50` needs a 50 Å ghost cutoff:

```lammps
units           metal
atom_style      atomic
pair_style      staf/cg 50 50 float
pair_coeff      * * /path/to/model_onnx_float32
comm_modify     cutoff 50
newton          on
```

Parity gate: `python test/test-lammps-staf-cg-parity/run_compare.py` (energy, forces, configurational pair-virial pressure vs Python STAF-CG).
