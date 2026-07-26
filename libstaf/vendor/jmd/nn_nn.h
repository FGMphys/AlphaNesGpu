/*
 * nn_nn.h  —  adapted from neuralmdGPU/full_atom/src/nn_nn.h
 *
 * TF-specific types (nnmodel, model_t) and all TF includes have been removed.
 * The MLP is now provided externally via staf_jmd_set_mlp().
 *
 * Compile unit: nn_nn_mlp.cu
 * Extra include path required: -I<libstaf/include>
 */
#ifndef NN_NN_H
#define NN_NN_H

/* Forward declarations — vector.h and interaction_map.h must be included
 * before this header when declaring function prototypes that use them. */
#include "vector.h"
#include "interaction_map.h"

/* -----------------------------------------------------------------------
 * distsymm helper struct (used inside nn_nn_mlp.cu internally)
 * ----------------------------------------------------------------------- */
typedef struct _distsymm {
    int    index;
    double dist;
    double dx;
    double dy;
    double dz;
} distsymm;

/* -----------------------------------------------------------------------
 * Internal constructor / compute helpers (called from initializenn_)
 * ----------------------------------------------------------------------- */
void Constructor_Descriptors(FILE *pconfig);
void Constructor_AFS(void);

/*
 * Constructor_MLP_Model — replaces Constructor_TensorFlow_Model.
 * Allocates host Gradients[][] and device Gradients_d[][].
 * The actual StafMlp* must already have been set with staf_jmd_set_mlp()
 * before initializenn_() is called (or at latest before the first
 * calculateforces / calculate_energies call).
 */
void Constructor_MLP_Model(void);

void Compute_Descriptors(double *box_d, double *pos_d,
                          int *howmany_d, int *with_d);
void make_typemap(int *type_map, int num_of_particles, char *root_path);
void compute_2bodyAFs(int type);
void compute_3bodyAFs(int type);
void Compute_NNEnergyandGradient(int type, double *energy);
void Compute_NNEnergyandGradient_all(double *energy);
void Compute_Force_2b(int type);
void Compute_Force_3b(int type);
void Compute_Forces_all(vector *force, double *virial, vector *virial_diag);
void Compute_AFs_all(void);

/* deletetensor is a no-op in the MLP variant (kept for ABI compat.) */
void deletetensor(void);

/* -----------------------------------------------------------------------
 * Public C API — same signatures as the original nn_nn.cu
 * ----------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C" {
#endif

/*
 * staf_jmd_set_mlp — must be called once before initializenn_().
 * mlp is a StafMlp* created by the caller (staf_mlp_create).
 * Ownership stays with the caller; call staf_mlp_destroy separately.
 */
void staf_jmd_set_mlp(void *mlp);  /* StafMlp* cast to void* for C callers */

/* Domain decomposition (MPI): external neigh already in JMD-packed order. */
void staf_jmd_set_skip_pbc(int skip);
void staf_jmd_set_external_neigh(const int *howmany_host, const int *with_host,
                                 int n_centers, int maxneigh);
void staf_jmd_clear_external_neigh(void);
/* n_centers = nlocal owned; n_all = nlocal + nghost.
   tipos_owned[NumTypes] per-type owned counts;
   type_map_all[n_all] species index for each packed slot.
   Returns 0 on success. May realloc GPU buffers when sizes change. */
int staf_jmd_resize(int n_centers, int n_all, const int *tipos_owned,
                    const int *type_map_all);
int staf_jmd_num_types(void);
int staf_jmd_radial_buffer(void);

void initializenn_(FILE *config_file, int number_of_particles);

void calculateforces(vector *pos, double *box,
                     interactionmap *ordered_interaction_list,
                     double *energy, vector *force,
                     double *virial, vector *virialxyz);

void calculate_energies(vector *pos, double *box,
                        interactionmap *Ime,
                        double *energy, vector *force,
                        double *virial, vector *virialxyz);

void nnDestructor(void);

#ifdef __cplusplus
}
#endif

/* Debug helpers */
void Alpha_check(int *data, int num);
void Alpha_checkDO(double *data, int num);

#endif /* NN_NN_H */
