void computeforce_tripl_Launcher(const double*  netderiv_T_d, const double* desr_T_d, const double* desa_T_d,
                        const double* intderiv_r_T_d, const double* intderiv_a_T_d,
                        const int* intmap_r_T_d,const int* intmap_a_T_d,
                         int nr, int na, int N, int dimbat,int num_finger_a,int num_finger_r,
                         const double* type_emb3b_d,double* forces3b_T_d,const int *num_triplets_d,
                         const double* smooth_a_T,const int* color_type_map_T_d,int prod,
                         double* virial_diagonal,double* pos_d,double* box_d,const int* map_intra,const int* map_color_interaction,
                         int n_all);
void init_block_dim_ang(int buffdim);
