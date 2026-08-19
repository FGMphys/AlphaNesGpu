# STAF-CG inference gates

Float vs double energy/force on the same frames. GPU jobs must run **sequentially**.

```bash
python prepare_frames.py
# after 1-epoch export into model_double / model_float:
python run_inference.py --precision double
python run_inference.py --precision float
python analyze_compatibility.py
# expect: Compatible
```

MODEL1896 (float64 SavedModel only) is staged with `stage_model1896.py` for force FD.
Float↔double compatibility uses the 1-epoch export from `test/test-cg-pipeline/`.
