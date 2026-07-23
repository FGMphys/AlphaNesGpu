#pragma once
#include "staf_real.h"
#include <cuda_runtime.h>

void convert_carte_to_int_launcher(real* nowinobox_d, const real* nowpos_d,
                                   real* nowinopos_d, int N, int nf,
                                   cudaStream_t stream);

void celleCompute(int N, const real* box_h, real* inopos_d, real cutoff,
                  int** cells_address, int** cells_howmany_address,
                  int* c_nx, int* c_ny, int* c_nz, int MAX_PARTICLE_CELLS,
                  int* cells_capacity_num, int* cells_capacity_mpc,
                  cudaStream_t stream);

void imeCompute(int N, const real* box_d, real* position_d, real cutoff,
                int* cells, int* cells_howmany, int celle_nx, int celle_ny,
                int celle_nz, int* with, int* howmany, real* with_dist2,
                int MAX_PARTICLE_CELLS, int maxneigh, cudaStream_t stream);
