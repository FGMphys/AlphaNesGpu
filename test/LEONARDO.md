# Leonardo — dati di test + dataset completo

Il codice STAF vive in **git**. Dataset e SavedModel **non** sono in git
(`.gitignore` sui `.npy`). Per Leonardo (acceptance + training multi-GPU) usare:

```text
STAF_leonardo_test_bundle.tar.gz   # bash test/pack_leonardo_bundle.sh
```

## Cosa resta in repo (traccia delle verifiche)

| Traccia | Path |
|--------|------|
| Gate ufficiali | [`ACCEPTANCE.md`](ACCEPTANCE.md) |
| Multi-GPU / Horovod | [`A3_PREP.md`](A3_PREP.md) |
| Baseline perf V100 | [`test-training-pipeline/comparison/`](test-training-pipeline/comparison/) |
| Parità `none`/`horovod` | [`test-training-pipeline/parity_distribute/`](test-training-pipeline/parity_distribute/) |
| Compat inference (testo) | [`test-inference-pipeline/comparison_summary.txt`](test-inference-pipeline/comparison_summary.txt) |
| Script force / grad-param | [`test-regression/`](test-regression/) |

## Contenuto del tar (trasporto una volta)

```text
STAF_leonardo_test_bundle/
  README.md
  EXPECTED_BASELINES.md
  INSTALL_OVER_CLONE.txt
  datasets/
    DATASET_INFO.txt
    mbpol_full_float/     # FULL float32: 2400 train + 300 test (+ type.dat)
    mbpol_full_double/    # FULL float64: stessa mole di frame
  production/
    afsparam_2/
    input_staf_float.yaml          # distribute: horovod, subsampling: no
    input_staf_double.yaml
    input_staf_float_none.yaml
  training/run_{float,double}/    # YAML acceptance + hardlink allo stesso dataset
  inference_models/model_{float,double}/
  baselines/                      # copie di summary già in git
```

Il dataset **completo** MB-pol STAF è già quello sotto `datasets/mbpol_full_*`
(non un subsample su disco). I YAML di acceptance usano `subsampling: 500 100`
solo a runtime; produzione usa `subsampling: no`.

## Install su Leonardo

```bash
git clone <repo> AlphaNesGpu && cd AlphaNesGpu
tar -xzf /path/to/STAF_leonardo_test_bundle.tar.gz
# overlay path usati dagli script di test:
rsync -a STAF_leonardo_test_bundle/training/run_float/  test/test-training-pipeline/run_float/
rsync -a STAF_leonardo_test_bundle/training/run_double/ test/test-training-pipeline/run_double/
rsync -a STAF_leonardo_test_bundle/inference_models/model_float/  test/test-inference-pipeline/model_float/
rsync -a STAF_leonardo_test_bundle/inference_models/model_double/ test/test-inference-pipeline/model_double/
cd STAF && bash install_path.sh all && cd ..
```

## Test da fare

**Non** lanciare float∥double sulla stessa GPU in parallelo.

### A — Gate numerici (1 GPU)

```bash
cd test/test-inference-pipeline && python analyze_compatibility.py
cd ../test-regression/regression-force
python run_force_regression.py --precision double
python run_force_regression.py --precision float
cd ../regression-grad-param
python run_grad_param_regression.py --precision double --n-per-family 100
python run_grad_param_regression.py --precision float  --n-per-family 100
```

### B — Parità `distribute` (1 GPU, dataset già pieno; YAML subsample)

```bash
cd test/test-training-pipeline
python compare_distribute_lcurve.py --precision float
python compare_distribute_lcurve.py --precision double
```

### C — Multi-GPU / produzione (dataset completo)

```bash
cd STAF_leonardo_test_bundle/production
# 4 GPU / nodo (Horovod only; MirroredStrategy removed)
mpirun -np 4 python ../../STAF/staf_train.py input_staf_float.yaml
```

Annotare: job OK, `model_log*` da rank 0, speed-up vs 1 GPU, loss stabile.

## Rigenerare il tar

```bash
bash test/pack_leonardo_bundle.sh
# → STAF_leonardo_test_bundle.tar.gz  (gitignored)
```
