#ifndef STAF_MLP_H
#define STAF_MLP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "staf.h"

/* Pluggable MLP: AF -> energy + dE/dAF.
 * Default ORT path: model_type{k}.onnx from export_mlp_grad_onnx.py
 *   (af → energy, dE_daf) using standard ONNX ops only.
 * Fallback: native Dense from mlp_type{k}.bin with analytical ∂E/∂AF.
 * Convention: energy = 0.5*sum(atomic); dE_daf = ∂sum(atomic)/∂af.
 */

typedef struct StafMlp StafMlp;

typedef struct StafMlpEval {
  int n_type;
  const int* n_atoms; /* [n_type] */
  const int* n_af;    /* [n_type] */
  /* Packed AF per type: type0 atoms, then type1, ... row-major [n_atoms_k, n_af_k] */
  const float* af_f32;
  const double* af_f64;
  float* energy_f32; /* [1] total for all types */
  double* energy_f64;
  float* dE_daf_f32; /* packed like af */
  double* dE_daf_f64;
} StafMlpEval;

StafMlp* staf_mlp_create(StafMlpBackend backend, const char* model_dir,
                         int precision, int device_id);
int staf_mlp_eval(StafMlp* mlp, StafMlpEval* io);
void staf_mlp_destroy(StafMlp* mlp);

/* Number of loaded type nets (after create). */
int staf_mlp_ntypes(const StafMlp* mlp);
int staf_mlp_n_af(const StafMlp* mlp, int type);
/* 0 = float32, 1 = float64 (matches create precision). */
int staf_mlp_precision(const StafMlp* mlp);

#ifdef __cplusplus
}
#endif

#endif /* STAF_MLP_H */
