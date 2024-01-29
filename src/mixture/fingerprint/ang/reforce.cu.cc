#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"


#define BLOCK_DIM 106



__global__ void angularAFs_kernel(const float* radial_descriptor,const float* angular_descriptor,
                           int nr,int na,float* three_body_AFs,int dimbat,int N_local,
                           const int* interaction_map_angular_o,const float* alpha3b_parameters,
                          int nsmooth_a,const float* type_emb3b,
                          const int* type_map,const int* num_triplets)
{

    const int2* intmap_a=(const int2*) interaction_map_angular_o;
    const float3* alphas=(const float3*) alpha3b_parameters;
    const float* ds=(const float*)radial_descriptor;

    int t=blockIdx.x*blockDim.x+threadIdx.x;
    int b=t/(na*N_local);
    int reminder=t%(na*N_local);
    int par=reminder/na;
    int nn=reminder%na;
    if (t<N_local*dimbat*na)
    {
        int na_particle=num_triplets[b*N_local+par];
        int nn_particle=(na_particle*(na_particle-1))/2;
        if (nn<nn_particle)
        {
          int na_dim=na_particle;
          int j=0;
          int prev_row=0;
          int next_row=na_dim-j-1;
          while (nn>=next_row)
          {
              j+=1;
              prev_row=next_row;
              next_row+=na_dim-j-1;
          }
          int k=nn-prev_row+1+j;

        int nne3bod=na_particle;
        int actual=b*N_local*nr+par*nr;
        int actual_ang=b*N_local*na+par*na;
        int aux2=nn;
        int2 neigh=intmap_a[b*N_local*na+par*na+aux2];
	if (neigh.x>767 | neigh.y>767)
           printf("\nneighx %d neighy %d\n",neigh.x,neigh.y);
        if (par>767)
           printf("\nPar %d\n",par);
       int j_type=type_map[neigh.x];
       int k_type=type_map[neigh.y];

       float angulardes=angular_descriptor[actual_ang+aux2];
       int sum=j_type+k_type;
       for (int a1=0;a1<nsmooth_a;a1++){
             float alpha1=alphas[sum*nsmooth_a+a1].x;
             float alpha2=alphas[sum*nsmooth_a+a1].y;
             float betaval=alphas[sum*nsmooth_a+a1].z;

             float chtjy_par=type_emb3b[sum*nsmooth_a+a1];

             float softmaxweight=expf(alpha1*ds[actual+j]+alpha2*ds[actual+k]);
             softmaxweight+=expf(alpha2*ds[actual+j]+alpha1*ds[actual+k]);
             softmaxweight*=expf(betaval*angulardes);
             atomicAdd((float*)&three_body_AFs[b*nsmooth_a*N_local+par*nsmooth_a+a1],angulardes*softmaxweight*chtjy_par/2.f);
            }
              }
      }
	 }

void angularAFs_Launcher(const float* radial_descriptor,const float* angular_descriptor,int nr,int na,
                          float* three_body_AFs,int dimbat,int N_local,
                          const int* interaction_map_angular,const float* alpha3b_parameters,
                          int nsmooth_a,const float* type_emb3b,
                          const int* type_map,const int* num_triplets){

                          dim3 dimGrid(ceil(float(dimbat*N_local*na)/float(BLOCK_DIM)),1,1);
                          dim3 dimBlock(BLOCK_DIM,1,1);

                          TF_CHECK_OK(::tensorflow::GpuLaunchKernel(angularAFs_kernel,                      dimGrid, dimBlock, 0, nullptr,radial_descriptor,angular_descriptor,               nr,na,three_body_AFs,dimbat,N_local,
                           interaction_map_angular,alpha3b_parameters,
                          nsmooth_a,type_emb3b,
                          type_map,num_triplets));

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
