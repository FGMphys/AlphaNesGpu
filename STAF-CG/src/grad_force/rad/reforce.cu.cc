#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "staf_real.h"
#define STAF_FORCE_PBC_DEFINE
#include "staf_force_pbc.cuh"



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

__global__ void back_prop_grad_force2b_kernel(const real* prevgrad,const real* ds,
                           int nr,const real* alpha2b,int num_finger,
                           const real* intderiv_r,const int* intmap_r,
                           int dimbat,int N,int N_local,const real*netderiv,
                           const real* type_emb2b,int nt,const int* color_type_map,
                           const int* map_color_interaction,const int* actual_type_p,
                           const int* map_intra, real* grad_net,
                           real* grad_alpha2b,real* grad_emb2b)
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

          int absolute_par=par+tipos_shift;
          int actual=b*N_local*nr+par*nr;

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

          int ch_type=row_index;


          real ds_el=ds[actual+j];
          for (int i=0;i<num_finger;i++){
          real accumulate1=real(0.);
          real accumulate2=real(0.);
          real accumulate3=real(0.);
	  int index_sup=b*(N_local*num_finger)+par*num_finger+i;
          for (int a =0; a<3; a++){
              real prevgrad_el=prevgrad[b*(N*3)+absolute_par*3+a];
              real prevgrad_neigh=prevgrad[b*(N*3)+neighj*3+a];
              real common = real(0.5)*intderiv_r[b*N_local*3*nr+nr*3*par+a*nr+j];


              real alpha_el=alpha2b[num_finger*ch_type+i];
              real chpar=type_emb2b[num_finger*ch_type+i];
              real supp1=staf_exp(alpha_el*ds_el);
              real sds_deriv=supp1*(real(1.)+alpha_el*ds_el);
              real buff_alpha=chpar*supp1*ds_el*(real(2.)+alpha_el*ds_el);

              real  NGel=netderiv[b*N_local*num_finger+par*num_finger+i];

              accumulate1-=prevgrad_el*common*chpar*sds_deriv;
              accumulate1+=prevgrad_neigh*common*chpar*sds_deriv;

              accumulate2-=prevgrad_el*NGel*buff_alpha*common;
              accumulate2+=prevgrad_neigh*NGel*buff_alpha*common;

              accumulate3-=prevgrad_el*NGel*sds_deriv*common;
              accumulate3+=prevgrad_neigh*NGel*sds_deriv*common;
            }
            atomicAdd((real*)&grad_net[index_sup],accumulate1);
            atomicAdd((real*)&grad_alpha2b[num_finger*ch_type+i],accumulate2);
            atomicAdd((real*)&grad_emb2b[num_finger*ch_type+i],accumulate3);

           }
         }
      }
  }


void back_prop_grad_force2b_Launcher(const real* prevgrad,const real* radiale,
                           int nr,const real* alpha_radiale,int num_finger,
                           const real* desder,const int* intmap_r,
                           int dimbat,int N,int N_local,const real*netderiv,
                           const real* type_emb2b,int nt,const int* color_type_map,
                           const int* map_color_interaction,const int* actual_type,
                           const int* map_intra,real* grad_net,
                           real* grad_alpha2b,real* grad_emb2b){

              dim3 dimGrid(ceil(real(dimbat*N_local*nr)/real(BLOCK_DIM)),1,1);
     		      dim3 dimBlock(BLOCK_DIM,1,1);

     		      TF_CHECK_OK(::tensorflow::GpuLaunchKernel(back_prop_grad_force2b_kernel,
                         dimGrid, dimBlock, 0, nullptr,prevgrad,radiale,
                           nr,alpha_radiale,num_finger,
                           desder,intmap_r,
                           dimbat,N,N_local,netderiv,
                           type_emb2b,nt,color_type_map,
                           map_color_interaction,actual_type,map_intra,
                           grad_net,grad_alpha2b,grad_emb2b));

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

/* Virial-aware force grad: ∂L/∂g_a = (∂L/∂F_j - ∂L/∂F_i)_a - Σ_b (∂L/∂W_ab) r_b
   with W_ab -= g_a * r_b (MIC Cartesian). */
__global__ void back_prop_grad_force2b_virial_kernel(
                           const real* prevgrad,const real* prevgrad_virial,
                           const real* ds,int nr,const real* alpha2b,int num_finger,
                           const real* intderiv_r,const int* intmap_r,
                           int dimbat,int N,int N_local,const real*netderiv,
                           const real* type_emb2b,int nt,const int* color_type_map,
                           const int* map_color_interaction,const int* actual_type_p,
                           const int* map_intra,
                           const real* pos_d,const real* box_d,
                           real* grad_net,real* grad_alpha2b,real* grad_emb2b)
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
          int actual_type=actual_type_p[0];
          int tipos_shift=0;

          int absolute_par=par+tipos_shift;
          int actual=b*N_local*nr+par*nr;

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

          int ch_type=row_index;
          real ds_el=ds[actual+j];

          const real3* pos_d_l=(const real3*)pos_d;
          real rx,ry,rz;
          staf_min_image_cart_from_cart(
              pos_d_l[b*N+absolute_par].x, pos_d_l[b*N+absolute_par].y, pos_d_l[b*N+absolute_par].z,
              pos_d_l[b*N+neighj].x, pos_d_l[b*N+neighj].y, pos_d_l[b*N+neighj].z,
              box_d + b*6, rx, ry, rz);
          const real* Wg = prevgrad_virial + b*9;
          real rvec[3]={rx,ry,rz};

          for (int i=0;i<num_finger;i++){
          real accumulate1=real(0.);
          real accumulate2=real(0.);
          real accumulate3=real(0.);
	  int index_sup=b*(N_local*num_finger)+par*num_finger+i;
          for (int a =0; a<3; a++){
              real prevgrad_el=prevgrad[b*(N*3)+absolute_par*3+a];
              real prevgrad_neigh=prevgrad[b*(N*3)+neighj*3+a];
              real wdot = Wg[a*3+0]*rvec[0]+Wg[a*3+1]*rvec[1]+Wg[a*3+2]*rvec[2];
              real coeff = prevgrad_neigh - prevgrad_el - wdot;
              real common = real(0.5)*intderiv_r[b*N_local*3*nr+nr*3*par+a*nr+j];

              real alpha_el=alpha2b[num_finger*ch_type+i];
              real chpar=type_emb2b[num_finger*ch_type+i];
              real supp1=staf_exp(alpha_el*ds_el);
              real sds_deriv=supp1*(real(1.)+alpha_el*ds_el);
              real buff_alpha=chpar*supp1*ds_el*(real(2.)+alpha_el*ds_el);
              real  NGel=netderiv[b*N_local*num_finger+par*num_finger+i];

              accumulate1+=coeff*common*chpar*sds_deriv;
              accumulate2+=coeff*NGel*buff_alpha*common;
              accumulate3+=coeff*NGel*sds_deriv*common;
            }
            atomicAdd((real*)&grad_net[index_sup],accumulate1);
            atomicAdd((real*)&grad_alpha2b[num_finger*ch_type+i],accumulate2);
            atomicAdd((real*)&grad_emb2b[num_finger*ch_type+i],accumulate3);

           }
         }
      }
  }


void back_prop_grad_force2b_virial_Launcher(
                           const real* prevgrad,const real* prevgrad_virial,
                           const real* radiale,int nr,const real* alpha_radiale,int num_finger,
                           const real* desder,const int* intmap_r,
                           int dimbat,int N,int N_local,const real*netderiv,
                           const real* type_emb2b,int nt,const int* color_type_map,
                           const int* map_color_interaction,const int* actual_type,
                           const int* map_intra,
                           const real* pos_d,const real* box_d,
                           real* grad_net,real* grad_alpha2b,real* grad_emb2b){

              dim3 dimGrid(ceil(real(dimbat*N_local*nr)/real(BLOCK_DIM)),1,1);
     		      dim3 dimBlock(BLOCK_DIM,1,1);

     		      TF_CHECK_OK(::tensorflow::GpuLaunchKernel(
                         back_prop_grad_force2b_virial_kernel,
                         dimGrid, dimBlock, 0, nullptr,
                         prevgrad,prevgrad_virial,radiale,
                           nr,alpha_radiale,num_finger,
                           desder,intmap_r,
                           dimbat,N,N_local,netderiv,
                           type_emb2b,nt,color_type_map,
                           map_color_interaction,actual_type,map_intra,
                           pos_d,box_d,
                           grad_net,grad_alpha2b,grad_emb2b));

              cudaDeviceSynchronize();

      }

#endif
