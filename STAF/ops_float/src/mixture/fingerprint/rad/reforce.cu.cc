#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"


#define BLOCK_DIM 80


__global__ void radialAFs_kernel(
        const float* radial_descriptor,const int nr,const float* alpha2b_parameters,
        const int nalpha_r,float* radial_AFs,const int dimbat,const int N_local,
        const int* interaction_map_rad,const float* type_emb2b,const int* type_map)
{
        const int* intmap_r=(const int*) interaction_map_rad;
        const float* alphas=(const float*) alpha2b_parameters;
        const float* ds=(const float*)radial_descriptor;

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

                float des_r_el=ds[actual+j];
                int neighj=intmap_r[b*(N_local*(nr+1))+(nr+1)*par+1+j];
                int ch_type=type_map[neighj];

                // costruiamo i descrittori
                for (int i=0; i<nalpha_r;i++){
                    float alpha_now=alphas[nalpha_r*ch_type+i];
                    float chpar=type_emb2b[nalpha_r*ch_type+i];
                    float softmaxweight=expf(alpha_now*des_r_el)*chpar;

                    atomicAdd((float*)&radial_AFs[b*nalpha_r*N_local+par*nalpha_r+i], des_r_el*softmaxweight);
                }
            }
        }
}


void radialAFs_Launcher(
        const float* radial_descriptor,const int nr,const float* alpha2b_parameters,
        const int nalpha_r,float* radial_AFs,const int dimbat,const int N_local,
        const int* interaction_map_rad,const float* type_emb2b,const int* type_map )
{
        dim3 dimGrid(ceil(float(dimbat*N_local*nr)/float(BLOCK_DIM)),1,1);
        dim3 dimBlock(BLOCK_DIM,1,1);

        TF_CHECK_OK(
          ::tensorflow::GpuLaunchKernel(
                radialAFs_kernel,
                dimGrid, dimBlock, 0, nullptr,radial_descriptor,nr,
                alpha2b_parameters,nalpha_r,radial_AFs,dimbat,N_local,
                interaction_map_rad,type_emb2b,type_map
            )
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
