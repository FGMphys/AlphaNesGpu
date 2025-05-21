///Implementazione del gradiente di una funzione scalare L(SD), funzione dei SD(alpha).
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#define BLOCK_DIM 118


using namespace tensorflow;

__global__ void alphagrad_ang_kernel(const double* radial_descriptor,const double* angular_descriptor,
                 int nr,int na,const double* prevgrad,int dimbat,
                 int N_local,const int* intmap3b,const double* alpha3b,
                 int nsmooth_a,double* next_alpha3b_grad,
                 const double* type_emb3b,const int* type_map,
                 double* next_emb3b_grad,const int* num_triplets,int req_alpha,
		 int req_sum)
{
  const int2* intmap_a=(const int2*) intmap3b;
  const double3* alphas=(const double3*) alpha3b;
  const double* ds=(const double*)radial_descriptor;

  __shared__ double3 grad_alpha_s[BLOCK_DIM];
  __shared__ double grad_ck_s[BLOCK_DIM];
  grad_alpha_s[threadIdx.x].x=0.f;
  grad_alpha_s[threadIdx.x].y=0.f;
  grad_alpha_s[threadIdx.x].z=0.f;

  grad_ck_s[threadIdx.x]=0.f;

  double3 local_alpha= {0.f, 0.f, 0.f};
  double local_ck= 0.f;




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

     int j_type=type_map[neigh.x];
     int k_type=type_map[neigh.y];

     double angulardes=angular_descriptor[actual_ang+aux2];
     int sum=j_type+k_type;
     int a1=req_alpha;
     if (req_sum==sum){
     	double dj=ds[actual+j];
     	double dy=ds[actual+k];
     	double Tjy=angular_descriptor[actual_ang+aux2];

        double alpha1=alphas[sum*nsmooth_a+a1].x;
        double alpha2=alphas[sum*nsmooth_a+a1].y;
        double betaval=alphas[sum*nsmooth_a+a1].z;

        double chtjy_par=type_emb3b[sum*nsmooth_a+a1];
        double prevgradel=prevgrad[b*nsmooth_a*N_local+par*nsmooth_a+a1];
        double a1dja2dy=expf(alpha1*dj+alpha2*dy);
        double a1dya2dj=expf(alpha1*dy+alpha2*dj);
        double btjy=expf(betaval*Tjy);

	grad_alpha_s[threadIdx.x].x=prevgradel*chtjy_par*(a1dja2dy*dj+a1dya2dj*dy)*btjy*Tjy/2.f;
        grad_alpha_s[threadIdx.x].y=prevgradel*chtjy_par*(a1dja2dy*dy+a1dya2dj*dj)*btjy*Tjy/2.f;
        grad_alpha_s[threadIdx.x].z=prevgradel*chtjy_par*(a1dja2dy+a1dya2dj)*btjy*Tjy*Tjy/2.f;


	}
      }
     __syncthreads();
     if (threadIdx.x==0){
       for (int dd=0;dd<BLOCK_DIM;dd++){
           local_alpha.x+=grad_alpha_s[dd].x;
           local_alpha.y+=grad_alpha_s[dd].y;
           local_alpha.z+=grad_alpha_s[dd].z;
           }
       atomicAdd((double*)&(next_alpha3b_grad[req_sum*nsmooth_a*3+req_alpha*3+0]),local_alpha.x);
       atomicAdd((double*)&(next_alpha3b_grad[req_sum*nsmooth_a*3+req_alpha*3+1]),local_alpha.y);
       atomicAdd((double*)&(next_alpha3b_grad[req_sum*nsmooth_a*3+req_alpha*3+2]),local_alpha.z);
     }
   }
}


void alphagrad_ang_Launcher(const double* radial_descriptor,const double* angular_descriptor,
                 int nr,int na,const double* prevgrad,int dimbat,
                 int N_local,const int* intmap3b,const double* alpha3b,
                 int nsmooth_a,double* next_alpha3b_grad,
                 const double* type_emb3b,const int* type_map,
                 double* next_emb3b_grad,const int* num_triplet,int nt_couple){



                 dim3 dimGrid(ceil(double(dimbat*N_local*na)/double(BLOCK_DIM)),1,1);
                 dim3 dimBlock(BLOCK_DIM,1,1);
                 for (int req_alpha=0;req_alpha<nsmooth_a;req_alpha++){
                     for (int req_sum=0;req_sum<nt_couple;req_sum++){
                         TF_CHECK_OK(::tensorflow::GpuLaunchKernel(alphagrad_ang_kernel,                      dimGrid, dimBlock, 0, nullptr,radial_descriptor,angular_descriptor,               nr,na,prevgrad,dimbat,
                                N_local,intmap3b,alpha3b,
                                nsmooth_a,next_alpha3b_grad,
                                type_emb3b,type_map,
                                next_emb3b_grad,num_triplet,req_alpha,req_sum));
		        }
		 }
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
