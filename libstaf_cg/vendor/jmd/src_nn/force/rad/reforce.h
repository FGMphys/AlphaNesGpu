void computeforce_doublets_Launcher(const double*  netderiv, const double* des_r,
                    const double* intderiv_r,const int* intmap_r,
                    int nr, int N, int dimbat,int num_alpha_radiale,int num_alpha_ang,
                    const double* alpha_radiale,const double* type_emb2b,
                    double* forces2b,const int* color_type_map,int prod,
		            double* virial_diagonal,double* pos_d,double* box_d,const int* map_intra,
                    const int* map_color_interaction, int n_all);
void init_block_dim(int buffdim);
