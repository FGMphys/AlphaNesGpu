# Force regression test (`regression-force`)

Verify analytical forces from STAF inference against numerical forces from
energy finite differences on a single frame.

## Method

Forward difference (chosen; central would be similar here):

\[
F_{\mathrm{num}} = -\frac{E_f - E_i}{\delta}
\]

Consistency check:

\[
E_i \approx E_f + F_{\mathrm{ana}}\,\delta
\]

- Models: `../test-inference-pipeline/model_{float,double}`
- Frame: first prepared inference frame (override with `--frame`)
- Probe: `--n-atoms` atoms × 3 Cartesian components (default 40)
- δ scan: `0.1`, `0.01`, `0.001` Å

Expect correlation \(F_{\mathrm{num}}\) vs \(F_{\mathrm{ana}}\) → 1 for a well-chosen δ.

## Run (sequentially — do not overlap on GPU)

```bash
source ../../../.venv/bin/activate
cd test/test-regression/regression-force

python run_force_regression.py --precision double
python run_force_regression.py --precision float
```

Outputs under `results_{float,double}/`: `summary.txt`, `summary.json`,
`force_fd_delta_*.npz`, `force_regression.png`.

`summary.txt` / `summary.json` always record the CLI flags used
(`precision`, `frame`, `dataset_index`, `n_atoms`, `seed`, `deltas`, `model_dir`).

## Baseline (frame 0, 40 atoms × 3, seed 0)

| δ (Å) | double corr | float corr | note |
|------:|------------:|-----------:|------|
| 0.1   | 0.604       | 0.604      | truncation error dominates |
| 0.01  | 0.994       | 0.994      | good for both |
| 0.001 | 0.99994     | 0.99982    | best; float slightly noisier |

Recommendation: use **δ = 0.01** as default regression check (robust for float+double).
Use **δ = 0.001** when validating double carefully.
