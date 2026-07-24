# STAF

**STAF** — Soft Two-body Angular Fingerprint deep neural network potential for water and biomolecular systems.

![STAF neural potential](assets/staf_hero.png)

> DOI: **10.1063/5.0139245**

Official code: **`STAF/`** (one tree; precision from YAML):

```bash
cd STAF
bash install_path.sh all          # builds ops_float + ops_double from src/
python staf_train.py input_staf.yaml   # precision: float|double
```

**Linea B (LAMMPS, in progress):** default MD path is **ONNX + ONNX Runtime** for the Dense MLP; custom CUDA stays in `libstaf`. See [`test/B_ARCHITECTURE.md`](test/B_ARCHITECTURE.md), [`libstaf/`](libstaf/), [`lammps/USER-STAF/`](lammps/USER-STAF/). Export:

```bash
python STAF/save_models/export_mlp_onnx.py -imodel model_log0 -modelname model_onnx
```

Experimental variants remain under `DEV/` (not part of the A2 full-atom unify).

See `STAF/README.md`, `test/ACCEPTANCE.md`, and `test/A2_PROGRESS.md`.

## Performance baseline (reference)

Measured on **Tesla V100-PCIE-16GB** (driver 470.256.02, CUDA 11.8, TensorFlow 2.14) with the MB-pol water test in `test/test-training-pipeline/run_{float,double}` (`dataset_MBPOL_278_223_248_full`).

| Quantity | Value |
| --- | --- |
| Atoms | 768 (256 O + 512 H) |
| Box | orthorhombic, \(L \approx 19.5\)–\(19.7\) Å |
| Cutoffs | \(R_c = R_c^\mathrm{ang} = 4.5\) Å, \(R_s = 2.25\) Å |
| Mean neighbors within \(R_c\) | **38.2 ± 0.8** (40-frame sample, MIC PBC) |
| Neighbor buffers | `Radial_Buffer` = `Max_Angular_Neigh` = 60 |
| Batch | `batch_size = 4`, `buffer_stream_dim_tr = 4`, energy+force |

| Precision | Per frame | Frames / s | Per batch (4 frames) | ≈ train / epoch (120 steps) |
| --- | --- | --- | --- | --- |
| **float32** | **91.5 ms** | 10.9 | 0.366 s | ≈ 44 s |
| **float64** | **149.7 ms** | 6.7 | 0.599 s | ≈ 72 s |

float64 / float32 wall-time ratio on this workload: **≈ 1.64×**.

Raw logs: `test/test-training-pipeline/comparison/performance_baseline.txt`.

## Citation

If you use this code, please cite:

```
Francesco Guidarelli Mattioli, Federico Dogo, Franz Saija, and Marco Fabrizio,
"A Soft Two-body Angular Fingerprint approach for the development of neural network potentials for water and biomolecular systems",
J. Chem. Phys. 158, 104101 (2023). https://doi.org/10.1063/5.0139245
```
