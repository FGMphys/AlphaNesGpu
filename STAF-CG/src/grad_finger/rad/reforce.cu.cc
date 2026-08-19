///Implementazione del gradiente di una funzione scalare L(SD), funzione dei SD(alpha).
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "staf_real.h"


#define BLOCK_DIM 50

__global__ void alphagrad_dist_kernel(const real* radial_descriptor,int nr,
const real* alpha2b_parameters,int nalpha_r,int dimbat,int N_local,
const int* intmap_r,const real* type_emb2b,const int* color_type_map,
real* nextgrad_alpha2b, real* nextgrad_emb2b,const real* prevgrad,
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
          real accumulate=real(0.);
          int actual=b*N_local*nr+par*nr;
          real des_r_el=radial_descriptor[actual+j];
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
              real prevgradel=prevgrad[b*nalpha_r*N_local+par*nalpha_r+i];
              real typew=type_emb2b[cht*nalpha_r+i];
              accumulate=des_r_el*des_r_el;
              accumulate*=staf_exp(alpha2b_parameters[cht*nalpha_r+i]*des_r_el)*typew*prevgradel;
              atomicAdd((real*)&nextgrad_alpha2b[cht*nalpha_r+i],accumulate);
              accumulate=des_r_el;
              accumulate*=staf_exp(alpha2b_parameters[cht*nalpha_r+i]*des_r_el)*prevgradel;
              atomicAdd((real*)&nextgrad_emb2b[cht*nalpha_r+i],accumulate);
             }
          }
           }
       }

void alpha_dist_grad_Launcher(const real* radial_descriptor,int nr,
                      const real* alpha2b_parameters,
                      int nalpha_r,real* nextgrad_alpha2b,int dimbat,
                      int N_local,const int* interaction_map_rad,
                      const real* prevgrad,const real* type_emb2b,
                      const int* color_type_map,real* nextgrad_emb2,
                      const int* map_color_interaction,const int* map_intra){

      dim3 dimGrid(ceil(real(dimbat*N_local*nr)/real(BLOCK_DIM)),1,1);
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

__global__ void set_tensor_to_zero_real_kernel(real* tensor,int dim){
          int t=blockIdx.x*blockDim.x+threadIdx.x;

          if (t<dim)
             tensor[t]=real(0.);
}

void set_tensor_to_zero_real(real* tensor,int dimten){
     int grids=ceil(real(dimten)/real(300));
     dim3 dimGrid(grids,1,1);
     dim3 dimBlock(300,1,1);
     TF_CHECK_OK(::tensorflow::GpuLaunchKernel(set_tensor_to_zero_real_kernel,
     dimGrid,dimBlock, 0, nullptr,tensor,dimten));
     cudaDeviceSynchronize();
     }
#endif
