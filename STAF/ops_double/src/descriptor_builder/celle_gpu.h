#include "staf_real.h"
void celleCompute(int N,real *box_d,real *inopos,real cutoff,int **cells_address,int **cells_howmany_address,int *c_nx,int *c_ny,int *c_nz,int MAX_PARTICLE_CELLS);

void imeCompute(int N,real* box_d,real *position_d,real cutoff,int *cells,int *cells_howmany,int celle_nx,int celle_ny,int celle_nz,int *with,int *howmany,real *with_dist2,int MAX_PARTICLE_CELLS,int maxneigh);
