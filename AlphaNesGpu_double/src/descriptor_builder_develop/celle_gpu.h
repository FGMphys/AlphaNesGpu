void celleCompute(int N,const double *box_d,double *inopos,double cutoff,int **cells_address,int **cells_howmany_address,int *c_nx,int *c_ny,int *c_nz,int MAX_PARTICLE_CELLS);

void imeCompute(int N,const double* box_d,double *position_d,double cutoff,int *cells,int *cells_howmany,int celle_nx,int celle_ny,int celle_nz,int *with,int *howmany,double *with_dist2,int MAX_PARTICLE_CELLS,int maxneigh);
void convert_carte_to_int_launcher(double* nowinobox_d,const double* nowpos_d,double* nowinopos_d,int N,int nf);
