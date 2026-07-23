#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "staf_real.h"


#define BLOCK_DIM 80


__global__ void radialAFs_kernel(
        const real* radial_descriptor,const int nr,const real* alpha2b_parameters,
        const int nalpha_r,real* radial_AFs,const int dimbat,const int N_local,
        const int* interaction_map_rad,const real* type_emb2b,const int* type_map)
{
        const int* intmap_r=(const int*) interaction_map_rad;
        const real* alphas=(const real*) alpha2b_parameters;
        const real* ds=(const real*)radial_descriptor;

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
                int actual=b*N_local*nr+par*nr;

                real des_r_el=ds[actual+j];
                int neighj=intmap_r[b*(N_local*(nr+1))+(nr+1)*par+1+j];
                int ch_type=type_map[neighj];

                // costruiamo i descrittori
                for (int i=0; i<nalpha_r;i++){
                    real alpha_now=alphas[nalpha_r*ch_type+i];
                    real chpar=type_emb2b[nalpha_r*ch_type+i];
                    real softmaxweight=staf_exp(alpha_now*des_r_el)*chpar;

                    atomicAdd((real*)&radial_AFs[b*nalpha_r*N_local+par*nalpha_r+i], des_r_el*softmaxweight);
                }
            }
        }
}

void radialAFs_Launcher(
        const real* radial_descriptor,const int nr,const real* alpha2b_parameters,
        const int nalpha_r,real* radial_AFs,const int dimbat,const int N_local,
        const int* interaction_map_rad,const real* type_emb2b,const int* type_map, cudaStream_t stream){
        dim3 dimGrid(ceil(real(dimbat*N_local*nr)/real(BLOCK_DIM)),1,1);
        dim3 dimBlock(BLOCK_DIM,1,1);

        TF_CHECK_OK(
          ::tensorflow::GpuLaunchKernel(
                radialAFs_kernel,
                dimGrid, dimBlock, 0, stream,radial_descriptor,nr,
                alpha2b_parameters,nalpha_r,radial_AFs,dimbat,N_local,
                interaction_map_rad,type_emb2b,type_map
            )
        );

}

__global__ void set_tensor_to_zero_real_kernel(real* tensor,int dim){
          int t=blockIdx.x*blockDim.x+threadIdx.x;

          if (t<dim)
             tensor[t]=real(0.);
}

void set_tensor_to_zero_real(real* tensor,int dimten, cudaStream_t stream){
     int grids=ceil(real(dimten)/real(300));
     dim3 dimGrid(grids,1,1);
     dim3 dimBlock(300,1,1);
     // No DeviceSynchronize: ordered on same stream as subsequent GpuLaunchKernel.
     TF_CHECK_OK(::tensorflow::GpuLaunchKernel(set_tensor_to_zero_real_kernel,dimGrid,dimBlock, 0, stream,tensor,dimten));
}
#endif
