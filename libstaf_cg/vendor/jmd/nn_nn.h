/*
 * nn_nn.h — CG origami (dual cutoff + color maps), TF replaced by StafMlp.
 *
 * Source of truth: neuralmdGPU/DEV/CG_and_WCA_LJ2_inter/src/nn_nn.cu
 * Compile unit: nn_nn_mlp.cu
 */
#ifndef NN_NN_H
#define NN_NN_H

#include "vector.h"
#include "interaction_map.h"

typedef struct _distsymm {
    int    index;
    double dist;
    double dx;
    double dy;
    double dz;
} distsymm;

void Constructor_Descriptors(FILE *pconfig);
void Constructor_AFS(void);
void Constructor_MLP_Model(void);

void Compute_Descriptors(double *box_d, double *pos_d,
                         int *howmany_d, int *with_d);
void compute_2bodyAFs(int nnindex);
void compute_3bodyAFs(int nnindex);
void Compute_NNEnergyandGradient(int nnindex, double *energy);
void Compute_NNEnergyandGradient_all(double *energy);
void Compute_Force_2b(int nnindex);
void Compute_Force_3b(int nnindex);
void Compute_Forces_all(vector *force, double *virial, vector *virial_diag);
void Compute_AFs_all(void);
void deletetensor(void);

#ifdef __cplusplus
extern "C" {
#endif

void staf_jmd_set_mlp(void *mlp);
void staf_jmd_set_skip_pbc(int skip);
void staf_jmd_set_external_neigh(const int *howmany_host, const int *with_host,
                                 int n_centers, int maxneigh);
void staf_jmd_clear_external_neigh(void);
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

void Alpha_check(int *data, int num);
void Alpha_checkDO(double *data, int num);

#endif /* NN_NN_H */
