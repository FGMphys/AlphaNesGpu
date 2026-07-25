# STAF-AI Potential

**STAF-AI Potential — Self Trained Atomic Fingerprint AI Potential**

![STAF neural potential](assets/staf_hero.png)

> DOI: **10.1063/5.0139245**

Official code: **`STAF/`** (one tree; precision from YAML):

```bash
cd STAF
bash install_path.sh all          # builds ops_float + ops_double from src/
python staf_train.py input_staf.yaml   # precision: float|double
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

## Multi-GPU (Horovod) — none vs 1×4 vs 2×4

Leonardo A100, float32, `batch_size=8`, full MB-pol. Throughput scales ~linearly with ranks; after removing `LR × hvd.size()`, Horovod loss/RMSE track the single-GPU baseline.

| Run | GPUs | Train s/epoch | Global frames/s | Notes |
| --- | ---: | ---: | ---: | --- |
| none | 1 | **147.7** | **16.3** | 20 ep |
| Horovod 1×4 | 4 | **36.9** (~4.0×) | **68** | 100 ep (LR × 4 still OK) |
| Horovod 2×4 | 8 | **18.5** (~8.0×) | **129** | 200 ep, no LR × N |

![Scaling timing](assets/scaling_none_1x4_2x4_timing.png)

![Scaling loss / RMSE](assets/scaling_none_1x4_2x4_loss_rmse.png)

![Scaling loss / RMSE early epochs](assets/scaling_none_1x4_2x4_loss_rmse_ep20.png)

### LR × N bug (2×4 was stuck)

With `LR × hvd.size()` on **2×4** (×8), training stalled (`Loss_Tot` ≈ 0.23, \(RMSE_F\) flat ≈ 0.72). Without that scale, loss decreases normally.

| Run | Setup | Result |
| --- | --- | --- |
| before | 2×4, batch 8, LR × 8 | Loss_Tot stuck ≈ 0.23; \(RMSE_F\) flat ≈ 0.72 eV/Å |
| after | 2×4, batch 8, no LR × N | Loss_Tot → ≈ 0.0047; \(RMSE_F\) → ≈ 0.097 eV/Å (200 ep) |
| after | 2×4, batch 4, no LR × N | Loss_Tot → ≈ 0.0049; \(RMSE_F\) → ≈ 0.099 eV/Å (200 ep) |

![HVD loss bug vs fix](assets/hvd_loss_unstuck_vs_bug.png)

![HVD batch Loss_Tot bug vs fix](assets/hvd_loss_batch_bug_vs_fix.png)

![HVD validation RMSE after fix](assets/hvd_validation_rmse.png)

Data: `jobs/bench_20260724/{none_b8_20ep,hvd_1x4_b8_100ep,hvd_2x4_b8_200ep}` and `jobs/bench_post-LR-not-scaling-fix/hvd_2x4_{b8,b4}_200ep`.

## Citation

If you use this code, please cite:

```
Guidarelli Mattioli, F., Sciortino, F., & Russo, J. (2023).
A neural network potential with self-trained atomic fingerprints: A test with the mW water potential.
The Journal of Chemical Physics, 158(10), 104101.
https://doi.org/10.1063/5.0139245
```
