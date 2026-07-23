# STAF

**STAF** — Soft Two-body Angular Fingerprint deep neural network potential for water and biomolecular systems.

![STAF neural potential](assets/staf_hero.png)

> DOI: **10.1063/5.0139245**

Official GPU trees (legacy directory names; Python package is ``staf_models``):

- `AlphaNesGpu_float/` — float32 STAF (`staf_train.py`)
- `AlphaNesGpu_double/` — float64 STAF (`staf_train.py`)

Experimental variants live under `DEV/`. Runtime `.so` roots resolve via
`staf_paths.code_root()` (`STAF_FLOAT_ROOT` / `STAF_DOUBLE_ROOT`).

See `test/A2_PROGRESS.md` for the consolidation checklist.

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

Steady-state timings from `time_story.dat` (`displ_freq = 10` → each sample covers 10 batches / 40 frames; reports > 8 s excluded as compile / validation spikes):

| Precision | Per frame | Frames / s | Per batch (4 frames) | ≈ train / epoch (120 steps) |
| --- | --- | --- | --- | --- |
| **float32** | **91.5 ms** | 10.9 | 0.366 s | ≈ 44 s |
| **float64** | **149.7 ms** | 6.7 | 0.599 s | ≈ 72 s |

float64 / float32 wall-time ratio on this workload: **≈ 1.64×**.

Raw logs and notes: `test/test-training-pipeline/comparison/performance_baseline.txt`, `time_story_baseline_float.txt`, `time_story_baseline_double.txt`.

See also `test/ACCEPTANCE.md` (gates before any refactor), `test/A2_PROGRESS.md`,
`test/test-training-pipeline/README.md`, and `test/test-inference-pipeline/README.md`.

## Citation

If you use this code, please cite:

```
DOI: 10.1063/5.0139245
```

## License

BSD 3-Clause License

## Contact

francesco.guidarellimattioli@gmail.com
