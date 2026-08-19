# STAF-CG 1-epoch pipeline gate

Reproduces the Sprint 1 freeze (Seed 60, 80/20, batch 8) on official `STAF-CG/`.

```bash
bash run_one_epoch.sh
python export_and_check.py
# expect RMSE_f ≈ 38.3526
```

Full Sprint 3 sequence (GPU: **double then float**, never together):

```bash
bash run_sprint3.sh
```
