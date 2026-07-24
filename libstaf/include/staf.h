#ifndef STAF_H
#define STAF_H

#ifdef __cplusplus
extern "C" {
#endif

/* Public C API for LAMMPS pair_staf / standalone MD.
 * Default MLP path: ONNX + ONNX Runtime (see staf_mlp.h, test/B_ARCHITECTURE.md).
 */

typedef struct StafModel StafModel;

typedef enum StafMlpBackend {
  STAF_MLP_ORT = 0,   /* default: ONNX Runtime */
  STAF_MLP_TF_C = 1,  /* legacy SavedModel (optional) */
  STAF_MLP_NATIVE = 2 /* Dense dump, future */
} StafMlpBackend;

typedef struct StafOptions {
  StafMlpBackend mlp_backend; /* default STAF_MLP_ORT */
  int device_id;              /* CUDA device for this MPI rank */
  int precision;              /* 0 = float32, 1 = float64 */
  int reserved;
} StafOptions;

/* Fill defaults: ORT, device 0, float32. */
void staf_options_default(StafOptions* opt);

/* model_dir contains model_type{k}.onnx + type{k}_alpha_*.dat */
StafModel* staf_load(const char* model_dir, const StafOptions* opt);

/* Domain-decomposition contract (Allegro-like):
 *   nall = nlocal + nghost
 *   x, type sized nall; f sized nall*3 (ghost forces for reverse_comm)
 *   No global allgather of the full system inside this call.
 */
int staf_compute(StafModel* m,
                 int nlocal,
                 int nghost,
                 const double* x,     /* [nall*3] */
                 const double* box,   /* 6 or 9 components; see README */
                 const int* type,     /* [nall], 0-based species */
                 double* e_rank,      /* rank energy contribution (scalar) */
                 double* f,           /* [nall*3] */
                 double* virial);     /* [6] xx yy zz xy xz yz */

void staf_free(StafModel* m);

#ifdef __cplusplus
}
#endif

#endif /* STAF_H */
