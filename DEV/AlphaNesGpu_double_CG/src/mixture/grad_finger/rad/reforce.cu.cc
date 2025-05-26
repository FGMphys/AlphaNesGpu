///Implementazione del gradiente di una funzione scalare L(SD), funzione dei SD(alpha).
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"


#define BLOCK_DIM 50

__global__ void alphagrad_dist_kernel(const double* radial_descriptor,int nr,
const double* alpha2b_parameters,int nalpha_r,int dimbat,int N_local,
const int* intmap_r,const double* type_emb2b,const int* color_type_map,
double* nextgrad_alpha2b, double* nextgrad_emb2b,const double* prevgrad,
const int* map_color_interaction,const int* map_intra)
{

  int t=blockIdx.x*blockDim.x+threadIdx.x;
  int b=t/(nr*N_local);
  int reminder=t%(nr*N_local);
  int par=reminder/nr;
  int j=reminder%nr;
  if (t<N_local*dimbat*nr)
  {

      int nr_particle=intmap_r[b*N_local*(nr+1)+par*(nr+1)];
      if (j<nr_particle)
      {
          double accumulate=0.f;
          int actual=b*N_local*nr+par*nr;
          double des_r_el=radial_descriptor[actual+j];
          int neighj=intmap_r[b*(N_local*(nr+1))+(nr+1)*par+1+j];

          int my_mol=map_intra[par];
          int j_mol=map_intra[neighj];
          int row_index=0;
          if (my_mol!=j_mol){
             int my_col=color_type_map[par];
             int j_col=color_type_map[neighj];
             int my_interaction=map_color_interaction[my_col];
             if (my_interaction==j_col){
                  row_index=2;
             }
             else {
                  row_index=1;
             }
          }

          int cht=row_index;
          int i;
          for (i=0;i<nalpha_r;i++){
              double prevgradel=prevgrad[b*nalpha_r*N_local+par*nalpha_r+i];
              double typew=type_emb2b[cht*nalpha_r+i];
              accumulate=des_r_el*des_r_el;
              accumulate*=expf(alpha2b_parameters[cht*nalpha_r+i]*des_r_el)*typew*prevgradel;
              atomicAdd((double*)&nextgrad_alpha2b[cht*nalpha_r+i],accumulate);
              accumulate=des_r_el;
              accumulate*=expf(alpha2b_parameters[cht*nalpha_r+i]*des_r_el)*prevgradel;
              atomicAdd((double*)&nextgrad_emb2b[cht*nalpha_r+i],accumulate);
             }
          }
           }
       }

void alpha_dist_grad_Launcher(const double* radial_descriptor,int nr,
                      const double* alpha2b_parameters,
                      int nalpha_r,double* nextgrad_alpha2b,int dimbat,
                      int N_local,const int* interaction_map_rad,
                      const double* prevgrad,const double* type_emb2b,
                      const int* color_type_map,double* nextgrad_emb2,
                      const int* map_color_interaction,const int* map_intra){

      dim3 dimGrid(ceil(double(dimbat*N_local*nr)/double(BLOCK_DIM)),1,1);
      dim3 dimBlock(BLOCK_DIM,1,1);

      TF_CHECK_OK(
        ::tensorflow::GpuLaunchKernel(alphagrad_dist_kernel,
              dimGrid, dimBlock, 0, nullptr,radial_descriptor,nr,
              alpha2b_parameters,nalpha_r,dimbat,N_local,
              interaction_map_rad,type_emb2b,color_type_map,nextgrad_alpha2b,
              nextgrad_emb2,prevgrad,map_color_interaction,map_intra)
      );

      cudaDeviceSynchronize();

}

__global__ void set_tensor_to_zero_double_kernel(double* tensor,int dim){
          int t=blockIdx.x*blockDim.x+threadIdx.x;

          if (t<dim)
             tensor[t]=0.f;
}

void set_tensor_to_zero_double(double* tensor,int dimten){
     int grids=ceil(double(dimten)/double(300));
     dim3 dimGrid(grids,1,1);
     dim3 dimBlock(300,1,1);
     TF_CHECK_OK(::tensorflow::GpuLaunchKernel(set_tensor_to_zero_double_kernel,
     dimGrid,dimBlock, 0, nullptr,tensor,dimten));
     cudaDeviceSynchronize();
     }
#endif
