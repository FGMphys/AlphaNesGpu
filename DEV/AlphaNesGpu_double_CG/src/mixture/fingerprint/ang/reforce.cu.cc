#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"


#define BLOCK_DIM 106

__host__ __device__ int get_symmetric_pair_index(int i, int j, int ntypes) {
  int diff = i - j;
  int mask = diff >> 31; // 0 se diff ≥ 0, -1 se diff < 0
  int min = j + (diff & mask);
  int max = i - (diff & mask);
  return min * ntypes - (min * (min + 1)) / 2 + max;
}

__global__ void angularAFs_kernel(const double* radial_descriptor,const double* angular_descriptor,
                           int nr,int na,double* three_body_AFs,int dimbat,int N_local,
                           const int* interaction_map_angular_o,const double* alpha3b_parameters,
                          int nsmooth_a,const double* type_emb3b,
                          const int* color_type_map,const int* num_triplets,const int* map_color_interaction,
                          const int* map_intra)
{

    const int2* intmap_a=(const int2*) interaction_map_angular_o;
    const double3* alphas=(const double3*) alpha3b_parameters;
    const double* ds=(const double*)radial_descriptor;

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

        int type_int_j=0;
        int type_int_k=0;

        int my_mol=map_intra[par];
        int my_col=color_type_map[par];
        int my_interaction=map_color_interaction[my_col];

        int j_mol=map_intra[neigh.x];
        int k_mol=map_intra[neigh.y];

        if (my_mol!=j_mol){
            int j_col=color_type_map[neigh.x];
            if (my_interaction==j_col){
                type_int_j=2; //binding
            }
            else {
                type_int_j=1; //inert
            }
        }

        if (my_mol!=k_mol){
          int k_col=color_type_map[neigh.y];
          if (my_interaction==k_col){
              type_int_k=2; //binding
          }
          else {
              type_int_k=1; //inert
          }
      }




        int sum=get_symmetric_pair_index(type_int_j,type_int_k, 3);

        double angulardes=angular_descriptor[actual_ang+aux2];
        for (int a1=0;a1<nsmooth_a;a1++){
             double alpha1=alphas[sum*nsmooth_a+a1].x;
             double alpha2=alphas[sum*nsmooth_a+a1].y;
             double betaval=alphas[sum*nsmooth_a+a1].z;

             double chtjy_par=type_emb3b[sum*nsmooth_a+a1];

             double softmaxweight=expf(alpha1*ds[actual+j]+alpha2*ds[actual+k]);
             softmaxweight+=expf(alpha2*ds[actual+j]+alpha1*ds[actual+k]);
             softmaxweight*=expf(betaval*angulardes);
             atomicAdd((double*)&three_body_AFs[b*nsmooth_a*N_local+par*nsmooth_a+a1],
                        angulardes*softmaxweight*chtjy_par/2.f);
            }
              }
      }
	 }

void angularAFs_Launcher(const double* radial_descriptor,const double* angular_descriptor,int nr,int na,
                          double* three_body_AFs,int dimbat,int N_local,
                          const int* interaction_map_angular,const double* alpha3b_parameters,
                          int nsmooth_a,const double* type_emb3b,
                          const int* color_type_map,const int* num_triplets,const int* map_color_interaction,
                          const int* map_intra){

                          dim3 dimGrid(ceil(double(dimbat*N_local*na)/double(BLOCK_DIM)),1,1);
                          dim3 dimBlock(BLOCK_DIM,1,1);

                          TF_CHECK_OK(::tensorflow::GpuLaunchKernel(angularAFs_kernel,                      dimGrid, dimBlock, 0, nullptr,radial_descriptor,angular_descriptor,               nr,na,three_body_AFs,dimbat,N_local,
                           interaction_map_angular,alpha3b_parameters,
                          nsmooth_a,type_emb3b,
                          color_type_map,num_triplets,map_color_interaction,map_intra));

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
     TF_CHECK_OK(::tensorflow::GpuLaunchKernel(set_tensor_to_zero_double_kernel,dimGrid,dimBlock, 0, nullptr,tensor,dimten));
     cudaDeviceSynchronize();
     }
#endif
