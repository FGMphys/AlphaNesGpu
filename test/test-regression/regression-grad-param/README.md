# Parameter-gradient regression (`regression-grad-param`)

Finite-difference check of **analytic parameter gradients** on the
**training** graph (`full_train_e_f` chain, no `apply_gradients`), using
checkpoint `model_log1`.

## Method

Separately for MSE `Loss_E` and MSE `Loss_F`:

\[
g_{\mathrm{num}} = \frac{L(w+dw)-L(w)}{dw}
\qquad\text{vs}\qquad
g_{\mathrm{ana}} = \frac{\partial L}{\partial w}
\]

Probes (`--n-per-family`, default **100** each):

| Family | What |
|--------|------|
| dense kernel | random weights across nets |
| dense bias | random biases across nets |
| `alpha2b` | random radial AF params (all types; ≤ available) |
| `alpha3b` β/γ/δ | random **active** angular params (all types) |

Same `--n-frames` batch (default 16, test split) for both losses.
`dw` scan default: `1e-2 1e-3 1e-4`.

Note: `alpha2b` has only **80** unique scalars in this model
(`2 types × (2×20)`), so that family uses all available (<100).
Angular families sample up to 100 slots from 240 (top-|g| pool).
Sporadic GPU FD glitches on `Loss_F` (NaN / `≈L_F/dw`) are filtered
from the correlation metrics (`n_used`/`n_total` in the summary).

## Run (sequentially — do not overlap on GPU)

```bash
source ../../../.venv/bin/activate
cd test/test-regression/regression-grad-param

python run_grad_param_regression.py --precision double
python run_grad_param_regression.py --precision float
```

Uses:

- `test/test-training-pipeline/run_{float,double}/model_log1`
- `input_4test.yaml` + `dataset/test/*.npy`
- `STAF/` with `register_*_grad` loaded (ops under `src/{fingerprint,force,grad_*}/`)

Outputs under `results_{float,double}/`: `summary.txt`, `summary.json`,
`grad_param_regression.png` (CLI flags always recorded in the summaries).

Angular probes (β/γ/δ) are chosen among AF slots with nonzero
`|∂L_E|+|∂L_F|` so dead channels are skipped.

## Baseline (n_frames=16, seed=0)

At `dw=1e-3`, double: abs errors typically ~1e-7–1e-5 for all six probes
on both `Loss_E` and `Loss_F` (truncation shrinks further at `1e-4`).
Float: good around `dw=1e-2`–`1e-3`; `1e-4` is noisier (float energy/force MSE).
