#ifdef __cplusplus
extern "C" {
#endif

void print_gpu_data_launcher(double* data,int dim);
void print_gpu_data_int_launcher(int* data,int dim);
void esplora_gpu_data_launcher(double* data,double* data2,double* data3,int* data4,int dim);
void set_tensor_to_zero_int(int* tensor,int dimten);
void set_tensor_to_zero_double(double* tensor,int dimten);
void fill_angular_launcher(double R_c,int radbuff,double R_a,int angbuff,int N,int N_all,
                      double* inopos_d,const double* box_d,
                      int *howmany_d,int *with_d,
                      double* ang_descr_d,int* intmap3b_d,
                      double* des3bsupp_d,double* der3b_d,
                      double* der3bsupp_d, int nf,int* numtriplet_d);
void fill_radial_launcher(double R_c,int radbuff,double R_a,int angbuff,int N,int N_all,
                      double* inopos_d,const double* box_d,
                      int *howmany_d,int *with_d,
                      double* descriptor_d,int* intmap2b_d,double* der2b_d,
                      double* des3bsupp_d,
                      double* der3bsupp_d, int nf,int* numtriplet_d,
                      double rs, double coeffa,double coeffb,double coeffc,double pow_alpha, double pow_beta);
void staf_descriptor_set_skip_pbc(int skip);

#ifdef __cplusplus
}
#endif
