void computeforce_tripl_Launcher(const double*  netderiv_T_d, const double* desr_T_d, const double* desa_T_d,
                        const double* intderiv_r_T_d, const double* intderiv_a_T_d,
                        const int* intmap_r_T_d,const int* intmap_a_T_d,
                         int nr, int na, int N, int dimbat,int num_finger_a,int num_finger_r,
                         const double* type_emb3b_d,int nt,const int* tipos_T,int actual_type,
                         double* forces3b_T_d,const int *num_triplets_d,const double* smooth_a_T,
                         const int* type_map_T_d,int prod,double* virial_diagonal,double* pos_d,double* box_d);
void init_block_dim_ang(int buffdim);
