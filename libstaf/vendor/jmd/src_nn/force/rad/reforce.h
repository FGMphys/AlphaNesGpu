void computeforce_doublets_Launcher(const double*  netderiv, const double* des_r,
                    const double* intderiv_r,const int* intmap_r,
                    int nr, int N, int dimbat,int num_alpha_radiale,int num_alpha_ang,
                    const double* alpha_radiale,const double* type_emb2b,int nt,
                    const int* tipos_T,int actual_type,double* forces2b,const int* type_map,int prod,
		    double* virial_diagonal,double* pos_d,double* box_d);
void init_block_dim(int buffdim);
