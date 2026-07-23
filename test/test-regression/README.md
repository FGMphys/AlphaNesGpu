# Test regression

Finite-difference / analytic consistency checks for STAF.

- `regression-force/` — numerical vs analytical forces (energy FD, inference)
- `regression-grad-param/` — FD vs analytic ∂Loss_E/∂w and ∂Loss_F/∂w
  (dense weights + AF params, training path / `model_log1`)

Acceptance gates for all future changes: [`../ACCEPTANCE.md`](../ACCEPTANCE.md).
