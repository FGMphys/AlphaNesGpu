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
         if ((buffdim%i==0) & (i<512)){
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

__global__ void computeforce_doublets_kernel(const real* netderiv,const real* des_r,const real* intderiv_r,const int* intmap_r,
            int nr,int N,int dimbat,
            int num_alpha_radiale,const real* alpha_radiale,
            const real* type_emb2b,const int* actual_type_p,
            const int* color_type_map,real* forces2b_l,
            int BLOCK_DIM,const int* map_color_interaction,const int* map_intra)
{

    int actual_type=actual_type_p[0];
    int N_local=N;

    int tipos_shift=0;

    real3* forces2b=(real3 *)forces2b_l;

    int t=blockIdx.x*blockDim.x+threadIdx.x;

    extern  __shared__ real3 forza_i[];//[BLOCK_DIM];

    forza_i[threadIdx.x].x=0.;
    forza_i[threadIdx.x].y=0.;
    forza_i[threadIdx.x].z=0.;


    real3 local_force = {real(0.), real(0.), real(0.)};
    real3 other_force = {real(0.), real(0.), real(0.)};

    // from t to b,par,j,k
    int b=t/(nr*N_local);
    int reminder=t%(nr*N_local);
    int par=reminder/nr;
    int j=reminder%nr;
    int absolute_par=par+tipos_shift;
    int actual=b*N_local*nr+par*nr;
    if (t<N_local*dimbat*nr)
    {
        int nr_particle=intmap_r[b*N_local*(nr+1)+par*(nr+1)];
        int neighj=intmap_r[b*(N_local*(nr+1))+(nr+1)*par+1+j];

        if (j<nr_particle)
        {



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

          real des_r_el=des_r[actual+j];
          int ch_type=row_index;

          real intder_r_x=intderiv_r[b*N_local*3*nr+nr*3*par+0*nr+j];
          real intder_r_y=intderiv_r[b*N_local*3*nr+nr*3*par+1*nr+j];
          real intder_r_z=intderiv_r[b*N_local*3*nr+nr*3*par+2*nr+j];
          for (int i=0; i<num_alpha_radiale;i++){
                real alpha_now=alpha_radiale[num_alpha_radiale*ch_type+i];
                real chpar=type_emb2b[num_alpha_radiale*ch_type+i];
                real sds_deriv=chpar*staf_exp(alpha_now*des_r_el);
                sds_deriv*=(real(1.)+alpha_now*des_r_el);
                real prevgrad=netderiv[b*N_local*num_alpha_radiale+num_alpha_radiale*par+i];
                real tempx = real(0.5)*sds_deriv*intder_r_x;
                real tempy = real(0.5)*sds_deriv*intder_r_y;
                real tempz = real(0.5)*sds_deriv*intder_r_z;

                forza_i[threadIdx.x].x-=prevgrad*tempx;
                forza_i[threadIdx.x].y-=prevgrad*tempy;
                forza_i[threadIdx.x].z-=prevgrad*tempz;
                other_force.x+=prevgrad*tempx;
                other_force.y+=prevgrad*tempy;
                other_force.z+=prevgrad*tempz;
              }
          }
          atomicAdd((real*)&(forces2b[b*N+neighj].x),other_force.x);
          atomicAdd((real*)&(forces2b[b*N+neighj].y),other_force.y);
          atomicAdd((real*)&(forces2b[b*N+neighj].z),other_force.z);


    __syncthreads();


    if (threadIdx.x==0)
    {
        for (int i=0;i<BLOCK_DIM;i++)
        {
            local_force.x+=forza_i[i].x;
            local_force.y+=forza_i[i].y;
            local_force.z+=forza_i[i].z;
        }

        atomicAdd((real*)&(forces2b[b*N+absolute_par].x),local_force.x);
        atomicAdd((real*)&(forces2b[b*N+absolute_par].y),local_force.y);
        atomicAdd((real*)&(forces2b[b*N+absolute_par].z),local_force.z);

    }

   }
}

void computeforce_doublets_Launcher(const real*  netderiv, const real* des_r,
                    const real* intderiv_r,const int* intmap_r,
                    int nr, int N, int dimbat,int num_alpha_radiale,
                    const real* alpha_radiale,const real* type_emb2b,
                    const int* actual_type,real* forces2b,
                    const int* color_type_map,int prod, const int* map_color_interaction,
                    const int* map_intra)
{
                      dim3 dimGrid(ceil(real(prod)/real(BLOCK_DIM)),1,1);
     		              dim3 dimBlock(BLOCK_DIM,1,1);

     		      TF_CHECK_OK(::tensorflow::GpuLaunchKernel(computeforce_doublets_kernel, dimGrid, dimBlock, BLOCK_DIM*sizeof(real3), nullptr,netderiv,des_r,
                          intderiv_r,intmap_r,
                          nr,N,dimbat,
                          num_alpha_radiale,alpha_radiale,
                          type_emb2b,actual_type,color_type_map,
                          forces2b,BLOCK_DIM,map_color_interaction,map_intra));

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

/* ---- Virial-aware radial force: F + full W (9), W_ab -= f_a * r_b ---- */

__global__ void computeforce_doublets_virial_kernel(const real* netderiv,const real* des_r,const real* intderiv_r,const int* intmap_r,
            int nr,int N,int dimbat,
            int num_alpha_radiale,const real* alpha_radiale,
            const real* type_emb2b,const int* actual_type_p,
            const int* color_type_map,real* forces2b_l,
            int BLOCK_DIM,const int* map_color_interaction,const int* map_intra,
            real* virial_d,const real* pos_d,const real* box_d)
{

    int actual_type=actual_type_p[0];
    int N_local=N;

    int tipos_shift=0;

    real3* forces2b=(real3 *)forces2b_l;
    const real3* pos_d_l=(const real3 *)pos_d;

    int t=blockIdx.x*blockDim.x+threadIdx.x;

    extern __shared__ unsigned char sharedMemory_rad[];
    real3 *forza_i = (real3 *)sharedMemory_rad;
    real *virial_i = (real *)(sharedMemory_rad + sizeof(real3) * BLOCK_DIM);

    forza_i[threadIdx.x].x=real(0.);
    forza_i[threadIdx.x].y=real(0.);
    forza_i[threadIdx.x].z=real(0.);
    for (int c=0;c<9;c++) virial_i[threadIdx.x*9+c]=real(0.);

    real3 local_force = {real(0.), real(0.), real(0.)};
    real3 other_force = {real(0.), real(0.), real(0.)};
    real3 distij = {real(0.), real(0.), real(0.)};

    int b=t/(nr*N_local);
    int reminder=t%(nr*N_local);
    int par=reminder/nr;
    int j=reminder%nr;
    int absolute_par=par+tipos_shift;
    int actual=b*N_local*nr+par*nr;
    if (t<N_local*dimbat*nr)
    {
        int nr_particle=intmap_r[b*N_local*(nr+1)+par*(nr+1)];
        int neighj=intmap_r[b*(N_local*(nr+1))+(nr+1)*par+1+j];

        if (j<nr_particle)
        {
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

          real des_r_el=des_r[actual+j];
          int ch_type=row_index;

          real intder_r_x=intderiv_r[b*N_local*3*nr+nr*3*par+0*nr+j];
          real intder_r_y=intderiv_r[b*N_local*3*nr+nr*3*par+1*nr+j];
          real intder_r_z=intderiv_r[b*N_local*3*nr+nr*3*par+2*nr+j];

          const real3& pos_i = pos_d_l[b*N + absolute_par];
          const real3& pos_j = pos_d_l[b*N + neighj];
          const real* box_b = box_d + b*6;
          staf_min_image_cart_from_cart(pos_i.x, pos_i.y, pos_i.z,
                                        pos_j.x, pos_j.y, pos_j.z, box_b,
                                        distij.x, distij.y, distij.z);

          for (int i=0; i<num_alpha_radiale;i++){
                real alpha_now=alpha_radiale[num_alpha_radiale*ch_type+i];
                real chpar=type_emb2b[num_alpha_radiale*ch_type+i];
                real sds_deriv=chpar*staf_exp(alpha_now*des_r_el);
                sds_deriv*=(real(1.)+alpha_now*des_r_el);
                real prevgrad=netderiv[b*N_local*num_alpha_radiale+num_alpha_radiale*par+i];
                real gx = prevgrad*real(0.5)*sds_deriv*intder_r_x;
                real gy = prevgrad*real(0.5)*sds_deriv*intder_r_y;
                real gz = prevgrad*real(0.5)*sds_deriv*intder_r_z;

                forza_i[threadIdx.x].x-=gx;
                forza_i[threadIdx.x].y-=gy;
                forza_i[threadIdx.x].z-=gz;

                real* vloc = &virial_i[threadIdx.x*9];
                vloc[0]-=gx*distij.x; vloc[1]-=gx*distij.y; vloc[2]-=gx*distij.z;
                vloc[3]-=gy*distij.x; vloc[4]-=gy*distij.y; vloc[5]-=gy*distij.z;
                vloc[6]-=gz*distij.x; vloc[7]-=gz*distij.y; vloc[8]-=gz*distij.z;

                other_force.x+=gx;
                other_force.y+=gy;
                other_force.z+=gz;
              }
          }
          atomicAdd((real*)&(forces2b[b*N+neighj].x),other_force.x);
          atomicAdd((real*)&(forces2b[b*N+neighj].y),other_force.y);
          atomicAdd((real*)&(forces2b[b*N+neighj].z),other_force.z);

    __syncthreads();

    if (threadIdx.x==0)
    {
        real local_vir[9];
        for (int c=0;c<9;c++) local_vir[c]=real(0.);
        for (int i=0;i<BLOCK_DIM;i++)
        {
            local_force.x+=forza_i[i].x;
            local_force.y+=forza_i[i].y;
            local_force.z+=forza_i[i].z;
            for (int c=0;c<9;c++) local_vir[c]+=virial_i[i*9+c];
        }

        atomicAdd((real*)&(forces2b[b*N+absolute_par].x),local_force.x);
        atomicAdd((real*)&(forces2b[b*N+absolute_par].y),local_force.y);
        atomicAdd((real*)&(forces2b[b*N+absolute_par].z),local_force.z);

        for (int c=0;c<9;c++)
            atomicAdd((real*)&(virial_d[b*9+c]),local_vir[c]);
    }

   }
}

void computeforce_doublets_virial_Launcher(const real*  netderiv, const real* des_r,
                    const real* intderiv_r,const int* intmap_r,
                    int nr, int N, int dimbat,int num_alpha_radiale,
                    const real* alpha_radiale,const real* type_emb2b,
                    const int* actual_type,real* forces2b,
                    const int* color_type_map,int prod, const int* map_color_interaction,
                    const int* map_intra,
                    real* virial_d,const real* pos_d,const real* box_d)
{
                      dim3 dimGrid(ceil(real(prod)/real(BLOCK_DIM)),1,1);
     		              dim3 dimBlock(BLOCK_DIM,1,1);
                      const size_t shmem =
                          BLOCK_DIM * sizeof(real3) + BLOCK_DIM * 9 * sizeof(real);

     		      TF_CHECK_OK(::tensorflow::GpuLaunchKernel(computeforce_doublets_virial_kernel, dimGrid, dimBlock, shmem, nullptr,netderiv,des_r,
                          intderiv_r,intmap_r,
                          nr,N,dimbat,
                          num_alpha_radiale,alpha_radiale,
                          type_emb2b,actual_type,color_type_map,
                          forces2b,BLOCK_DIM,map_color_interaction,map_intra,
                          virial_d,pos_d,box_d));

                      cudaDeviceSynchronize();

     }

#endif
