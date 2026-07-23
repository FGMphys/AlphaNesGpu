# Leonardo — dati di test e checklist

Il codice STAF vive in **git**. I `.npy` del dataset di acceptance **non** sono in git
(`.gitignore`). Per le prove su Leonardo (multi-GPU / Horovod) usare il bundle:

```text
STAF_leonardo_test_bundle.tar.gz   # generato da pack_leonardo_bundle.sh
```

## Cosa resta in repo (traccia delle verifiche già fatte)

| Traccia | Path |
|--------|------|
| Gate ufficiali | [`ACCEPTANCE.md`](ACCEPTANCE.md) |
| Multi-GPU / Horovod note | [`A3_PREP.md`](A3_PREP.md) |
| Baseline perf V100 | [`test-training-pipeline/comparison/performance_baseline.txt`](test-training-pipeline/comparison/performance_baseline.txt) |
| float↔double training overlay | [`test-training-pipeline/comparison/`](test-training-pipeline/) |
| Parità `none`/`mirrored`/`horovod` (`lcurve_notmean`) | [`test-training-pipeline/parity_distribute/`](test-training-pipeline/parity_distribute/) |
| Compat inference (testo) | [`test-inference-pipeline/comparison_summary.txt`](test-inference-pipeline/comparison_summary.txt) |
| Script regressione force / grad-param | [`test-regression/`](test-regression/) |

I plot `results_*/` delle regressioni restano locali (gitignore); i criteri numerici
sono in `ACCEPTANCE.md` e nei summary di `parity_distribute/`.

## Contenuto del tar (cosa portare su Leonardo)

Dopo `bash test/pack_leonardo_bundle.sh` (dal root del repo):

```text
STAF_leonardo_test_bundle/
  README.md                 # stessi test di sotto
  EXPECTED_BASELINES.md     # numeri di riferimento già visti in casa
  training/
    run_float/dataset/      # .npy + type.dat
    run_float/*.yaml        # input_4test, *_smoke, …
    run_double/dataset/
    run_double/*.yaml
  inference_models/
    model_float/            # SavedModel per force / compat
    model_double/
```

Sul nodo: `git clone` del repo + estrarre il tar **sopra** i path di test
(o copiare `training/run_*` e `inference_models` nei posti giusti).

## Test da fare su Leonardo (ordine consigliato)

Assunzioni: moduli TF/CUDA/MPI/Horovod caricati; ops buildate:

```bash
cd STAF && bash install_path.sh all && cd ..
```

**Non** lanciare float e double in parallelo sulla stessa GPU.

### A — Gate numerici (1 GPU, come in casa)

```bash
# 1) Compat inference
cd test/test-inference-pipeline
# assicurarsi che model_float / model_double ci siano (dal tar → inference_models/)
python analyze_compatibility.py
# expect: Compatible

# 2) Force FD
cd ../test-regression/regression-force
python run_force_regression.py --precision double
python run_force_regression.py --precision float
# expect: corr(δ=0.001) → ~1

# 3) Grad-param (serve model_log1 o symlink a un checkpoint training)
cd ../regression-grad-param
python run_grad_param_regression.py --precision double --n-per-family 100
python run_grad_param_regression.py --precision float  --n-per-family 100
```

### B — Training 1 GPU: parità `distribute`

```bash
cd test/test-training-pipeline
# dataset già in run_float/dataset e run_double/dataset (dal tar)
python compare_distribute_lcurve.py --precision float
python compare_distribute_lcurve.py --precision double
# expect: OVERALL_OK=True (vedi parity_distribute/*/summary.txt in repo)
```

### C — Multi-GPU (obiettivo Leonardo)

Stesso YAML, solo ranks/GPU:

```bash
# Esempio 4 GPU / nodo — Horovod
cd test/test-training-pipeline/run_float
# input: distribute: horovod  (es. input_horovod_smoke.yaml o input_4test.yaml)
mpirun -np 4 python ../../../STAF/staf_train.py input_horovod_smoke.yaml

# Oppure same-node MirroredStrategy (1 processo, N GPU)
# distribute: mirrored
# devices: [0, 1, 2, 3]   # opzionale
python ../../../STAF/staf_train.py input_mirrored_smoke.yaml
```

Criteri multi-GPU (da annotare sul posto):

- Job completa e scrive `model_log*` solo da rank 0 (Horovod).
- Loss non esplode; `lcurve_notmean` confrontabile con run 1-GPU a parità di
  **global batch** (local_batch × n_ranks; LR già × `hvd.size()` nel codice).
- Speed-up vs 1 GPU (wall-time / epoch) da registrare in un note locale o PR.

### D — Produzione (dataset grosso su `$WORK`)

Non è nel tar. Usare `STAF/input_staf.yaml` con `dataset_folder` sul path Leonardo
(es. `/leonardo_work/...`) e `distribute: horovod` + `mpirun` dello job Slurm.

## Rigenerare il tar in casa

```bash
bash test/pack_leonardo_bundle.sh
# scrive: STAF_leonardo_test_bundle.tar.gz  (root repo, gitignored)
```
