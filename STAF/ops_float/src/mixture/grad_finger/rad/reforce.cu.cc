///Implementazione del gradiente di una funzione scalare L(SD), funzione dei SD(alpha).
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"


#define BLOCK_DIM 50

__global__ void alphagrad_dist_kernel(const float* radial_descriptor,int nr,
const float* alpha2b_parameters,int nalpha_r,int dimbat,int N_local,
const int* intmap_r,const float* type_emb2b,const int* type_map,
float* nextgrad_alpha2b, float* nextgrad_emb2b,const float* prevgrad)
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
          float accumulate=0.f;
          int actual=b*N_local*nr+par*nr;
          float des_r_el=radial_descriptor[actual+j];
          int neighj=intmap_r[b*(N_local*(nr+1))+(nr+1)*par+1+j];
          int cht=type_map[neighj];
          int i;
          for (i=0;i<nalpha_r;i++){
              float prevgradel=prevgrad[b*nalpha_r*N_local+par*nalpha_r+i];
              float typew=type_emb2b[cht*nalpha_r+i];
              accumulate=des_r_el*des_r_el;
              accumulate*=expf(alpha2b_parameters[cht*nalpha_r+i]*des_r_el)*typew*prevgradel;
              atomicAdd((float*)&nextgrad_alpha2b[cht*nalpha_r+i],accumulate);
              accumulate=des_r_el;
              accumulate*=expf(alpha2b_parameters[cht*nalpha_r+i]*des_r_el)*prevgradel;
              atomicAdd((float*)&nextgrad_emb2b[cht*nalpha_r+i],accumulate);
             }
          }
           }
       }

void alpha_dist_grad_Launcher(const float* radial_descriptor,int nr,
                      const float* alpha2b_parameters,
                      int nalpha_r,float* nextgrad_alpha2b,int dimbat,
                      int N_local,const int* interaction_map_rad,
                      const float* prevgrad,const float* type_emb2b,
                      const int* type_map,float* nextgrad_emb2){

      dim3 dimGrid(ceil(float(dimbat*N_local*nr)/float(BLOCK_DIM)),1,1);
      dim3 dimBlock(BLOCK_DIM,1,1);

      TF_CHECK_OK(
        ::tensorflow::GpuLaunchKernel(alphagrad_dist_kernel,
              dimGrid, dimBlock, 0, nullptr,radial_descriptor,nr,
              alpha2b_parameters,nalpha_r,dimbat,N_local,
              interaction_map_rad,type_emb2b,type_map,nextgrad_alpha2b,
              nextgrad_emb2,prevgrad)
      );

      cudaDeviceSynchronize();

}

__global__ void set_tensor_to_zero_float_kernel(float* tensor,int dim){
          int t=blockIdx.x*blockDim.x+threadIdx.x;

          if (t<dim)
             tensor[t]=0.f;
}

void set_tensor_to_zero_float(float* tensor,int dimten){
     int grids=ceil(float(dimten)/float(300));
     dim3 dimGrid(grids,1,1);
     dim3 dimBlock(300,1,1);
     TF_CHECK_OK(::tensorflow::GpuLaunchKernel(set_tensor_to_zero_float_kernel,dimGrid,dimBlock, 0, nullptr,tensor,dimten));
     cudaDeviceSynchronize();
     }
#endif
