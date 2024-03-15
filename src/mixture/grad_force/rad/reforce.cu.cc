#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"



static int BLOCK_DIM;

void init_block_dim(int buffdim){
     int i;
     for (i=buffdim;i>0;i--){
         if ((buffdim%i==0) && (i<512)){
            BLOCK_DIM=i;
            i=0;
	    }
     }
     if (i!=-1){
        printf("Alpha_nes: No integer divisor found for the given radial buffer size \n");
        exit(0);
     }
     else{
        printf("Alpha_nes: Blocks for radial forces set to %d\n",BLOCK_DIM);
      }
}

__global__ void back_prop_grad_force2b_kernel(const float* prevgrad,const float* ds,
                           int nr,const float* alpha2b,int num_finger,
                           const float* intderiv_r,const int* intmap_r,
                           int dimbat,int N,int N_local,const float*netderiv,
                           const float* type_emb2b,int nt,const int* type_map,
                           const int* tipos,const int* actual_type_p,float* grad_net,
                           float* grad_alpha2b,float* grad_emb2b)
                            {
       int t=blockIdx.x*blockDim.x+threadIdx.x;
       // from t to b,par,j,k
       int b=t/(nr*N_local);
       int reminder=t%(nr*N_local);
       int par=reminder/nr;
       int j=reminder%nr;
       if (t<N_local*dimbat*nr)
       {

         int b=t/(nr*N_local);
         int reminder=t%(nr*N_local);
         int par=reminder/nr;
         int j=reminder%nr;

	  int nr_particle=intmap_r[b*N_local*(nr+1)+par*(nr+1)];

         if (j<nr_particle)
        {
          int actual_type=actual_type_p[0];
          int tipos_shift=0;

          for (int y=0;y<actual_type;y++){
              tipos_shift=tipos_shift+tipos[y];
          }

          int absolute_par=par+tipos_shift;
          int actual=b*N_local*nr+par*nr;

          int neighj=intmap_r[b*(N_local*(nr+1))+(nr+1)*par+1+j];

          int ch_type=type_map[neighj];


          float ds_el=ds[actual+j];
          for (int i=0;i<num_finger;i++){
          float accumulate1=0.f;
          float accumulate2=0.f;
          float accumulate3=0.f;
	  int index_sup=b*(N_local*num_finger)+par*num_finger+i;
          for (int a =0; a<3; a++){
              float prevgrad_el=prevgrad[b*(N*3)+absolute_par*3+a];
              float prevgrad_neigh=prevgrad[b*(N*3)+neighj*3+a];
              float common = 0.5f*intderiv_r[b*N_local*3*nr+nr*3*par+a*nr+j];


              float alpha_el=alpha2b[num_finger*ch_type+i];
              float chpar=type_emb2b[num_finger*ch_type+i];
              float supp1=expf(alpha_el*ds_el);
              float sds_deriv=supp1*(1.f+alpha_el*ds_el);
              float buff_alpha=chpar*supp1*ds_el*(2.f+alpha_el*ds_el);

              float  NGel=netderiv[b*N_local*num_finger+par*num_finger+i];

              accumulate1-=prevgrad_el*common*chpar*sds_deriv;
              accumulate1+=prevgrad_neigh*common*chpar*sds_deriv;

              accumulate2-=prevgrad_el*NGel*buff_alpha*common;
              accumulate2+=prevgrad_neigh*NGel*buff_alpha*common;

              accumulate3-=prevgrad_el*NGel*sds_deriv*common;
              accumulate3+=prevgrad_neigh*NGel*sds_deriv*common;
            }
            atomicAdd((float*)&grad_net[index_sup],accumulate1);
            atomicAdd((float*)&grad_alpha2b[num_finger*ch_type+i],accumulate2);
            atomicAdd((float*)&grad_emb2b[num_finger*ch_type+i],accumulate3);

           }
         }
      }
  }


void back_prop_grad_force2b_Launcher(const float* prevgrad,const float* radiale,
                           int nr,const float* alpha_radiale,int num_finger,
                           const float* desder,const int* intmap_r,
                           int dimbat,int N,int N_local,const float*netderiv,
                           const float* type_emb2b,int nt,const int* type_map,
                           const int* tipos,const int* actual_type,float* grad_net,
                           float* grad_alpha2b,float* grad_emb2b){

              dim3 dimGrid(ceil(float(dimbat*N_local*nr)/float(BLOCK_DIM)),1,1);
     		      dim3 dimBlock(BLOCK_DIM,1,1);

     		      TF_CHECK_OK(::tensorflow::GpuLaunchKernel(back_prop_grad_force2b_kernel,
                         dimGrid, dimBlock, 0, nullptr,prevgrad,radiale,
                           nr,alpha_radiale,num_finger,
                           desder,intmap_r,
                           dimbat,N,N_local,netderiv,
                           type_emb2b,nt,type_map,
                           tipos,actual_type,grad_net,
                           grad_alpha2b,grad_emb2b));

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
