# STAF-CG sprints

Canonical checklist for the port `DEV/AlphaNesGpu_double_CG_dv_RC/` → official [`STAF-CG/`](../STAF-CG/).
Do **not** mix this file into the DEV CG source tree. Do **not** start sprint N+1 until the **Done** box of N is checked.

Plan: Cursor `staf-cg_linea_c`. Living science plan: [`docs/PIANO_ALPHANES.md`](../docs/PIANO_ALPHANES.md) linea C.

**Baseline (do not “improve”):**

| Role | Path |
|------|------|
| Dataset | `/home/francegm/ORIGAMI/INFER_INTRA_TRY2+USCGSITE/` (24 beads, ~91k / 10k) |
| MD model | `/home/francegm/ORIGAMI/ORIGAMI_DYNAMICS/origami_uscgsite/models/MODEL1896/` |
| Last MD | `DIMERORI16/RUNT400BOX280/` (Feb 2026). Inter cutoffs in `nohup.out`: Rc_inter=16, Rs_inter=8, Rc_ang_inter=16; radial blocks=24; angular blocks=276 |
| Freeze numbers | [`staf_cg_freeze/FREEZE_NUMBERS.md`](staf_cg_freeze/FREEZE_NUMBERS.md) |

**Rules:** `DEV/AlphaNesGpu_double_CG_dv_RC/` is read-only. CUDA CG stays a separate `src/` from `STAF/src/`. Float and double GPU tests run **in sequence**, not together.

---

## Sprint 1 — Freeze + scaffold

- [x] Git tag `pre-staf-cg` on repo HEAD (`bcc6a13`, 2026-08-19)
- [x] Capture DEV inference: MODEL1896 (+ MODEL1352) on USCGSITE frames 0–2 → E, F
- [x] Capture DEV 1-epoch subsample (Seed 60) on USCGSITE → `lcurve` / RMSE
- [x] Copy to `STAF-CG/`, flatten `src/mixture/` and Python `mixture/` packages
- [x] `install_path.sh` → `ops_double/` (no `root_path` sed)
- [x] Entry `staf_cg_train.py` / `staf_cg_infer.py`, YAML `input_staf_cg.yaml`
- [x] This file
- [x] Numbers: [`staf_cg_freeze/FREEZE_NUMBERS.md`](staf_cg_freeze/FREEZE_NUMBERS.md)

**Done when:** `bash STAF-CG/install_path.sh double` succeeds and a train/infer smoke starts (numeric parity is Sprint 2).

**Closed 2026-08-19:** compile `ops_double` OK (9 `reforce.so`). Infer smoke matches DEV MODEL1896 E bit-for-bit. 1-epoch smoke starts (`RMSE_f ≈ 38.35`).

---

## Sprint 2 — Precision (A2)

- [x] `real` via shared [`STAF/include/staf_real.h`](../STAF/include/staf_real.h); YAML `precision:`
- [x] Reuse [`STAF/staf/dtype.py`](../STAF/staf/dtype.py); `PYTHONPATH` = `STAF-CG` **then** `STAF` (CG `source_routine` must win)
- [x] Build float + double from the same `STAF-CG/src/`
- [x] Double matches Sprint 1 freeze (same frames / same Seed) within ~1e-6 abs (see FREEZE_NUMBERS)

**Done when:** double = DEV freeze; float compiles and does not crash.

**Closed 2026-08-19:** `bash install_path.sh all` OK. MODEL1896 double infer: max|dE|≈6.8e-6, max|dF|≈8e-7 vs freeze (delta from promoting `0.5f` literals to `real`). 1-epoch Seed 60: RMSE_f 38.35259 vs DEV 38.35260. Float ops compile; float infer of a float64 SavedModel fails at TF signature (CUDA float path constructs).

---

## Sprint 3 — STAF-like regression gates

- [x] `test/test-cg-inference/analyze_compatibility.py` — E,F float↔double Compatible
- [x] `test/test-cg-regression/regression-force` — analytic vs FD
- [x] `test/test-cg-regression/regression-grad-param` — ∂Loss/∂param vs FD
- [x] `test/test-cg-pipeline/` — 1-epoch subsample vs freeze
- [x] GPU sequence: double then float

**Done when:** three gates green (thresholds in ACCEPTANCE CG).

**Closed 2026-08-19:** `bash test/test-cg-pipeline/run_sprint3.sh`. Force FD double (MODEL1896) corr≥0.999 at δ=0.01. 1-epoch RMSE_f=38.352589 vs freeze 38.352598. Float↔double Compatible (max|ΔE|=2.7e-6, max|ΔF|=1.7e-7). Force FD float corr≥0.999. Grad-param: MSE Loss_E corr≈1 on dense/AF; FD forward uses `full_test_e_f` (no nested param grads).

---

## Sprint 4 — Horovod + virial

- [x] `distribute: none | horovod` as [`STAF/staf_train.py`](../STAF/staf_train.py) (no LR × N)
- [x] Smoke `mpirun -np 1`; 2 ranks if hardware allows
- [x] `type_of_training: energy+force+virial` + kernels adapted to intra/inert/sticky
- [x] FD virial on a mini-frame (`pass: true`); origami npy may still lack `virial.npy`

**Done when:** Horovod 1-rank OK; virial compiles; FD virial passes.

**Closed 2026-08-19:** Horovod in `staf_cg_train.py` (no LR×N). `mpirun -np 1` smoke: RMSE_f=38.35259. 1×V100 so np=2 skipped. Virial ops (intra/inert/sticky) in `ops_{float,double}`. Strain FD on MODEL1896: corr(W_ana, W_num)=0.99999948 with `W=-(E⁺−E⁻)/(2ε)`. Inference: `full_test_virial()`; `full_test` still `(E,F)`.

---

## Sprint 5 — `libstaf_cg` + ONNX

- [x] `libstaf_cg/` (ORT MLP, CUDA from STAF-CG, dual-cutoff / color maps)
- [x] `STAF-CG/save_models/export_mlp_grad_onnx.py` + ASCII AF + `map_intra` / color
- [x] Eval E,F on 1-epoch keras export vs Python STAF-CG (1 frame, 1 rank, no LAMMPS)

**Done when:** libstaf_cg matches Python on one frame.

**Closed 2026-08-19:** `cmake --build libstaf_cg/build` links `staf_force_smoke`. Float32 ONNX from `test/test-cg-pipeline/work/staf_cg_freeze_ep10` → `test/test-cg-inference/model_onnx_double/` (MODEL1896 is SavedModel-only; same export is reusable in Sprint 6). Frame 0 (24 beads, box 280): Python float32 vs libstaf_cg float32 `max|ΔE|≈9.5e-7`, `max|ΔF|≈8.3e-8` (`test/test-cg-libstaf/summary.txt` pass true). WCA/LJ not linked (STAF-only). No `DEV/AlphaNesGpu_double_CG_dv_RC` edits; CUDA trees stay separate.

---

## Sprint 6 — LAMMPS `pair_style staf/cg`

- [x] `lammps/USER-STAF-CG/`: `PairStyle(staf/cg, PairSTAFCG)`
- [x] Smoke 1-rank dimer 24 beads
- [x] **Parity required:** same frame (USCGSITE / MODEL1896 export) → LAMMPS vs Python STAF-CG: **energy, forces, configurational pressure** (pair virial only, no kinetic). Fail the sprint if E or F or P disagree. Harness: `test/test-lammps-staf-cg-parity/`
- [x] DD 1 vs 2 vs 4 ranks, same E/F/P trio
- [x] [`test/ACCEPTANCE.md`](../test/ACCEPTANCE.md) CG section; PIANO linea C/B4; `STAF-CG/README.md`

**Done when:** `staf/cg` runs; **E/F/P = Python**; DD parity; DEV remains archive.

**Closed 2026-08-19:** binary `/home/francegm/programmi/lammps-23Jun2022/src/lmp_staf_cg` (`lmp_staf` unchanged). Frame 0 (24 beads, box 280, float32 ONNX): max|ΔE|≈7.0e-7, max|ΔF|≈8.3e-8, |ΔP|/max(|P|,1)≈1.4e-8 (`p_gate: rel_P`; also max|ΔW_diag|≈2.4e-7). DD np=1/2/4 bit-identical E/F/P on 1×V100 (ranks share GPU; dimer sits in one subdomain, empty ranks skip compute). Intra Rc=50 ⇒ `comm_modify cutoff 50`.

---

## Out of scope (after STAF-CG + `staf/cg` are green)

- Fixing inter RMSE_f ~39 (scientific C4)
- oxDNA / linea D
- Merging CG and full-atom CUDA
- Deleting DEV
- Replacing production `jmd_nn` origami before LAMMPS smoke
