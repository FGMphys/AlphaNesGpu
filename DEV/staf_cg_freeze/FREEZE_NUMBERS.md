# STAF-CG freeze numbers (Sprint 1)

Captured **2026-08-19** from unmodified `DEV/AlphaNesGpu_double_CG_dv_RC/` (tag `pre-staf-cg` = `bcc6a13`).
Do not treat these RMSE values as a target to beat — only to reproduce after the port.

GPU: Tesla V100-PCIE-16GB. Dataset: `/home/francegm/ORIGAMI/INFER_INTRA_TRY2+USCGSITE/`.

## MODEL1896 inference (frames 0–2 of `dataset/training`)

Staging: `DEV/staf_cg_freeze/model1896_infer/` (not committed). `number_of_nn.dat` = 2.
`cutoff_info` was missing from the MD export; reconstructed from origami intra defaults + `DIMERORI16/RUNT400BOX280/nohup.out` (Rc_inter=16, Rs_inter=8, Rc_ang_inter=16; rad blocks=24; ang blocks=276):

```
50 24
50 276
25 0
16 0
16 0
8 0
```

| frame | energy | force RMS |
|------:|-------:|----------:|
| 0 | 356.7282093329031 | 0.19953295780063685 |
| 1 | 360.4664360097036 | 0.20587424087050116 |
| 2 | 360.19864851136845 | 0.35750286849170937 |

Force frame 0, first 6 components:

`-0.48759283769566475, -0.17032375793524646, -0.11479669318908045, -0.4432529512641782, 0.2051048243659687, 0.09622728332782317`

Raw: [`freeze_inference.json`](freeze_inference.json).

**STAF-CG smoke (same frames, `ops_double`):** energies match bit-for-bit; forces agree to ~1e-16. See `staf_cg_smoke_infer.json`.

## MODEL1352 inference (optional, same 3 frames)

Cutoff from the export (`Rc_inter=10`, `Rs_inter=5`):

| frame | energy | force RMS |
|------:|-------:|----------:|
| 0 | -9.16946678601953 | 0.14053312092446482 |
| 1 | -8.78078742698727 | 0.16401051599396088 |
| 2 | -8.929146295532217 | 0.1594336686016678 |

## 1-epoch subsample (Seed 60)

YAML: [`input_epoch1.yaml`](input_epoch1.yaml) — 80 train / 20 test frames, batch 8, `map_NN_layer: {0:[25,25]}`, `type_of_training: energy+force`.

DEV trainer (`tensorgpu`, `lcurve.out`):

```
10 0  RMSE_e=0.6594473693291664  RMSE_f=38.352597573817725  Loss_Tot=13.404372910164412  lr_net=0.0007128896744935718
```

STAF-CG smoke (`.venv`, same YAML; Sprint 2 after `staf_real`):

```
10 0  RMSE_e=0.6592164677271156  RMSE_f=38.35258956886645  Loss_Tot=13.404372303781427  lr_net=0.0007128896744935718
```

Sprint 2 MODEL1896 double infer vs this freeze: max|dE| ≈ 6.8e-6, max|dF0| ≈ 8e-7 (literals `0.5f` promoted to `real`).

Reproduce DEV freeze:

```bash
bash DEV/staf_cg_freeze/run_epoch1.sh
/home/francegm/miniconda3/envs/tensorgpu/bin/python DEV/staf_cg_freeze/run_infer_freeze.py
```
