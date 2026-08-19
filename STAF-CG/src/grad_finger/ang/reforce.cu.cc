///Implementazione del gradiente di una funzione scalare L(SD), funzione dei SD(alpha).
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "staf_real.h"
#define BLOCK_DIM 118

__host__ __device__ int get_symmetric_pair_index(int i, int j, int ntypes) {
  int diff = i - j;
  int mask = diff >> 31; // 0 se diff ≥ 0, -1 se diff < 0
  int min = j + (diff & mask);
  int max = i - (diff & mask);
  return min * ntypes - (min * (min + 1)) / 2 + max;
}

//using namespace tensorflow;

__global__ void alphagrad_ang_kernel(const real* radial_descriptor,const real* angular_descriptor,
                 int nr,int na,const real* prevgrad,int dimbat,
                 int N_local,const int* intmap3b,const real* alpha3b,
                 int nsmooth_a,real* next_alpha3b_grad,
                 const real* type_emb3b,const int* color_type_map,
                 real* next_emb3b_grad,const int* num_triplets,int req_alpha,
		 int req_sum,const int* map_color_interaction,const int* map_intra)
{
  const int2* intmap_a=(const int2*) intmap3b;
  const real3* alphas=(const real3*) alpha3b;
  const real* ds=(const real*)radial_descriptor;

  __shared__ real3 grad_alpha_s[BLOCK_DIM];
  __shared__ real grad_ck_s[BLOCK_DIM];
  grad_alpha_s[threadIdx.x].x=real(0.);
  grad_alpha_s[threadIdx.x].y=real(0.);
  grad_alpha_s[threadIdx.x].z=real(0.);

  grad_ck_s[threadIdx.x]=real(0.);

  real3 local_alpha= {real(0.), real(0.), real(0.)};
  real local_ck= real(0.);




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

     real angulardes=angular_descriptor[actual_ang+aux2];
     int a1=req_alpha;
     if (req_sum==sum){
     	real dj=ds[actual+j];
     	real dy=ds[actual+k];
     	real Tjy=angular_descriptor[actual_ang+aux2];

        real alpha1=alphas[sum*nsmooth_a+a1].x;
        real alpha2=alphas[sum*nsmooth_a+a1].y;
        real betaval=alphas[sum*nsmooth_a+a1].z;

        real chtjy_par=type_emb3b[sum*nsmooth_a+a1];
        real prevgradel=prevgrad[b*nsmooth_a*N_local+par*nsmooth_a+a1];
        real a1dja2dy=staf_exp(alpha1*dj+alpha2*dy);
        real a1dya2dj=staf_exp(alpha1*dy+alpha2*dj);
        real btjy=staf_exp(betaval*Tjy);

	      grad_alpha_s[threadIdx.x].x=prevgradel*chtjy_par*(a1dja2dy*dj+a1dya2dj*dy)*btjy*Tjy/real(2.);
        grad_alpha_s[threadIdx.x].y=prevgradel*chtjy_par*(a1dja2dy*dy+a1dya2dj*dj)*btjy*Tjy/real(2.);
        grad_alpha_s[threadIdx.x].z=prevgradel*chtjy_par*(a1dja2dy+a1dya2dj)*btjy*Tjy*Tjy/real(2.);


	}
      }
     __syncthreads();
     if (threadIdx.x==0){
       for (int dd=0;dd<BLOCK_DIM;dd++){
           local_alpha.x+=grad_alpha_s[dd].x;
           local_alpha.y+=grad_alpha_s[dd].y;
           local_alpha.z+=grad_alpha_s[dd].z;
           }
       atomicAdd((real*)&(next_alpha3b_grad[req_sum*nsmooth_a*3+req_alpha*3+0]),local_alpha.x);
       atomicAdd((real*)&(next_alpha3b_grad[req_sum*nsmooth_a*3+req_alpha*3+1]),local_alpha.y);
       atomicAdd((real*)&(next_alpha3b_grad[req_sum*nsmooth_a*3+req_alpha*3+2]),local_alpha.z);
     }
   }
}


void alphagrad_ang_Launcher(const real* radial_descriptor,const real* angular_descriptor,
                 int nr,int na,const real* prevgrad,int dimbat,
                 int N_local,const int* intmap3b,const real* alpha3b,
                 int nsmooth_a,real* next_alpha3b_grad,
                 const real* type_emb3b,const int* color_type_map,
                 real* next_emb3b_grad,const int* num_triplet,int nt_couple,
                const int* map_color_interaction,const int* map_intra){



                 dim3 dimGrid(ceil(real(dimbat*N_local*na)/real(BLOCK_DIM)),1,1);
                 dim3 dimBlock(BLOCK_DIM,1,1);
                 for (int req_alpha=0;req_alpha<nsmooth_a;req_alpha++){
                     for (int req_sum=0;req_sum<nt_couple;req_sum++){
                         TF_CHECK_OK(::tensorflow::GpuLaunchKernel(alphagrad_ang_kernel,
                           dimGrid, dimBlock, 0, nullptr,radial_descriptor,angular_descriptor,
                           nr,na,prevgrad,dimbat,
                                N_local,intmap3b,alpha3b,
                                nsmooth_a,next_alpha3b_grad,
                                type_emb3b,color_type_map,
                                next_emb3b_grad,num_triplet,req_alpha,req_sum,
                                map_color_interaction,map_intra));
		        }
		 }
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
  TF_CHECK_OK(::tensorflow::GpuLaunchKernel(set_tensor_to_zero_real_kernel,dimGrid,dimBlock, 0, nullptr,tensor,dimten));
  cudaDeviceSynchronize();
  }
#endif
