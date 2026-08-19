#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "staf_real.h"
#define STAF_FORCE_PBC_DEFINE
#include "staf_force_pbc.cuh"


static int BLOCK_DIM;

__host__ __device__ int get_symmetric_pair_index(int i, int j, int ntypes) {
  int diff = i - j;
  int mask = diff >> 31; // 0 se diff ≥ 0, -1 se diff < 0
  int min = j + (diff & mask);
  int max = i - (diff & mask);
  return min * ntypes - (min * (min + 1)) / 2 + max;
}

void init_block_dim(int buffdim){
     int i;
     for (i=buffdim;i>0;i--){
         if ((buffdim%i==0) && (i<512)){
            BLOCK_DIM=i;
            i=0;
         }
     }
     if (i!=-1){
        printf("Alpha_nes: No integer divisor found for the given angular buffer size\n");
        exit(0);
     }
     else{
        printf("Alpha_nes: Blocks for angular forces set to %d\n",BLOCK_DIM);
      }
}




__global__ void computeforce_tripl_kernel(const real*  netderiv_T, const real* desr_T, const real* desa_T,
                        const real* intderiv_r_T, const real* intderiv_a_T_l,
                        const int* intmap_r_T,const int* intmap_a_T_l,
                        int nr, const int na, int N, int dimbat , int num_finger,const real* type_emb3b,
                        
                        const int* actual_type_p,real* forces3b_T_l,const int *num_triplets,const real* smooth_a_T_l,
                        const int* color_type_map,int BLOCK_DIM,const int* map_color_interaction,
                        const int* map_intra)
{


    int actual_type=actual_type_p[0];
    int N_local=N; 

    int tipos_shift=0;
    
    const real2* intderiv_a_T=(const real2 *)intderiv_a_T_l;
    const int2* intmap_a_T=(const int2 *) intmap_a_T_l;
    real3* forces3b_T=(real3 *)forces3b_T_l;
    const real3* smooth_a_T=(const real3 *)smooth_a_T_l;

    int t=blockIdx.x*blockDim.x+threadIdx.x;

    extern __shared__ real3 forza_i[];//[BLOCK_DIM];

    forza_i[threadIdx.x].x=0.;
    forza_i[threadIdx.x].y=0.;
    forza_i[threadIdx.x].z=0.;

    // __syncthreads();

    real3 local_force = {real(0.), real(0.), real(0.)};

    // from t to b,par,j,k
    int b=t/(na*N_local);
    int reminder=t%(na*N_local);
    int par=reminder/na;
    int nn=reminder%na;
    int absolute_par=par+tipos_shift;
    if (t<N_local*dimbat*na)
    {



        int na_particle=num_triplets[b*N_local+par];
        int nn_particle=(na_particle*(na_particle-1))/2;
        if (nn<nn_particle)
        {


            real3 other_forcej = {real(0.), real(0.), real(0.)};
            real3 other_forcek = {real(0.), real(0.), real(0.)};

            int na_dim=na_particle;//floorf(real(0.5) + sqrtf(real(0.25) + 2*na));

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



            real delta=real(0.);
            real Bp_j=real(0.);
            real Bp_k=real(0.);



            int actual=b*N_local*nr+par*nr;
            int actual_ang=b*N_local*na+par*na;
            int actgrad=b*N_local*num_finger+par*num_finger;

            int2 neigh=intmap_a_T[b*(N_local*na)+na*par+nn];

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

            real angulardes=desa_T[actual_ang+nn];
            real radialdes_j=desr_T[actual+j];
            real radialdes_k=desr_T[actual+k];



            // loop su alpha
            for (int a1=0; a1<num_finger; a1++)
            {
                real3 alphas=smooth_a_T[sum*num_finger+a1];
		            real chtjk_par=type_emb3b[sum*num_finger+a1];

                real net_der=real(0.5)*netderiv_T[actgrad+a1]*chtjk_par;

		            real expbeta=staf_exp(alphas.z*angulardes);

                real sim1=staf_exp(alphas.y*radialdes_j+alphas.x*radialdes_k);
                real sim2=staf_exp(alphas.x*radialdes_j+alphas.y*radialdes_k);

                delta=expbeta*(1.+alphas.z*angulardes)*(sim1+sim2)*real(0.5);

                real suppj=(alphas.x*sim2+alphas.y*sim1)*expbeta*real(0.5);
                real suppk=(alphas.x*sim1+alphas.y*sim2)*expbeta*real(0.5);
                Bp_j=suppj*angulardes;
                Bp_k=suppk*angulardes;

                // x
                real2 intder = intderiv_a_T[b*(N_local*na)*3+par*na*3+0*na+nn];
                real intder_r_j=intderiv_r_T[b*N_local*3*nr+nr*3*par+0*nr+j];
                real intder_r_k=intderiv_r_T[b*N_local*3*nr+nr*3*par+0*nr+k];

                real fxij=net_der*(delta*intder.x+Bp_j*intder_r_j);
                real fxik=net_der*(delta*intder.y+Bp_k*intder_r_k);

                //local_force.x-=(fxij+fxik);
                forza_i[threadIdx.x].x-=(fxij+fxik);
                other_forcej.x+=fxij;
                other_forcek.x+=fxik;

                // y
                intder = intderiv_a_T[b*(N_local*na)*3+par*na*3+1*na+nn];
                intder_r_j=intderiv_r_T[b*N_local*3*nr+nr*3*par+1*nr+j];
                intder_r_k=intderiv_r_T[b*N_local*3*nr+nr*3*par+1*nr+k];

                fxij=net_der*(delta*intder.x+Bp_j*intder_r_j);
                fxik=net_der*(delta*intder.y+Bp_k*intder_r_k);

                forza_i[threadIdx.x].y-=(fxij+fxik);
                other_forcej.y+=fxij;
                other_forcek.y+=fxik;


                // z
                intder = intderiv_a_T[b*(N_local*na)*3+par*na*3+2*na+nn];
                intder_r_j=intderiv_r_T[b*N_local*3*nr+nr*3*par+2*nr+j];
                intder_r_k=intderiv_r_T[b*N_local*3*nr+nr*3*par+2*nr+k];

                fxij=net_der*(delta*intder.x+Bp_j*intder_r_j);
                fxik=net_der*(delta*intder.y+Bp_k*intder_r_k);

                forza_i[threadIdx.x].z-=(fxij+fxik);
                other_forcej.z+=fxij;
                other_forcek.z+=fxik;

            }

            atomicAdd((real*)&(forces3b_T[b*N+neigh.x].x),other_forcej.x);
            atomicAdd((real*)&(forces3b_T[b*N+neigh.x].y),other_forcej.y);
            atomicAdd((real*)&(forces3b_T[b*N+neigh.x].z),other_forcej.z);

            atomicAdd((real*)&(forces3b_T[b*N+neigh.y].x),other_forcek.x);
            atomicAdd((real*)&(forces3b_T[b*N+neigh.y].y),other_forcek.y);
            atomicAdd((real*)&(forces3b_T[b*N+neigh.y].z),other_forcek.z);



        }
    }

    __syncthreads();

    if (threadIdx.x==0)
    {
        for (int i=0;i<BLOCK_DIM;i++)
        {
            local_force.x+=forza_i[i].x;
            local_force.y+=forza_i[i].y;
            local_force.z+=forza_i[i].z;
        }

        atomicAdd((real*)&(forces3b_T[b*N+absolute_par].x),local_force.x);
        atomicAdd((real*)&(forces3b_T[b*N+absolute_par].y),local_force.y);
        atomicAdd((real*)&(forces3b_T[b*N+absolute_par].z),local_force.z);

    }
}

void computeforce_tripl_Launcher(const real*  netderiv_T_d, const real* desr_T_d, const real* desa_T_d,
                        const real* intderiv_r_T_d, const real* intderiv_a_T_d,
                        const int* intmap_r_T_d,const int* intmap_a_T_d,
                         int nr, int na, int N, int dimbat,int num_finger,
                         const real* type_emb3b_d,
                         const int* actual_type,real* forces3b_T_d,const int *num_triplets_d,
                         const real* smooth_a_T,const int* color_type_map_T_d,int prod,
                         const int* map_color_interaction,const int* map_intra){

    dim3 dimGrid(ceil(real(prod)/real(BLOCK_DIM)),1,1);
    dim3 dimBlock(BLOCK_DIM,1,1);
    TF_CHECK_OK(::tensorflow::GpuLaunchKernel(computeforce_tripl_kernel, dimGrid, dimBlock, BLOCK_DIM*sizeof(real3), nullptr,netderiv_T_d,desr_T_d,desa_T_d,
        intderiv_r_T_d,intderiv_a_T_d,intmap_r_T_d,
        intmap_a_T_d,nr,na,N,dimbat,
        num_finger,
        type_emb3b_d,
        actual_type,forces3b_T_d,num_triplets_d,smooth_a_T,color_type_map_T_d,BLOCK_DIM,
        map_color_interaction,map_intra));

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

/* ---- Virial-aware angular force: F + full W (9), W_ab -= f_a * r_b ---- */

__global__ void computeforce_tripl_virial_kernel(const real*  netderiv_T, const real* desr_T, const real* desa_T,
                        const real* intderiv_r_T, const real* intderiv_a_T_l,
                        const int* intmap_r_T,const int* intmap_a_T_l,
                        int nr, const int na, int N, int dimbat , int num_finger,const real* type_emb3b,
                        
                        const int* actual_type_p,real* forces3b_T_l,const int *num_triplets,const real* smooth_a_T_l,
                        const int* color_type_map,int BLOCK_DIM,const int* map_color_interaction,
                        const int* map_intra,
                        real* virial_d,const real* pos_d,const real* box_d)
{


    int actual_type=actual_type_p[0];
    int N_local=N; 

    int tipos_shift=0;
    
    const real2* intderiv_a_T=(const real2 *)intderiv_a_T_l;
    const int2* intmap_a_T=(const int2 *) intmap_a_T_l;
    real3* forces3b_T=(real3 *)forces3b_T_l;
    const real3* smooth_a_T=(const real3 *)smooth_a_T_l;
    const real3* pos_d_l=(const real3*)pos_d;

    int t=blockIdx.x*blockDim.x+threadIdx.x;

    extern __shared__ unsigned char sharedMemory_ang[];
    real3 *forza_i = (real3 *)sharedMemory_ang;
    real *virial_i = (real *)(sharedMemory_ang + sizeof(real3) * BLOCK_DIM);

    forza_i[threadIdx.x].x=real(0.);
    forza_i[threadIdx.x].y=real(0.);
    forza_i[threadIdx.x].z=real(0.);
    for (int c=0;c<9;c++) virial_i[threadIdx.x*9+c]=real(0.);

    real3 local_force = {real(0.), real(0.), real(0.)};
    real3 distij = {real(0.), real(0.), real(0.)};
    real3 distik = {real(0.), real(0.), real(0.)};

    int b=t/(na*N_local);
    int reminder=t%(na*N_local);
    int par=reminder/na;
    int nn=reminder%na;
    int absolute_par=par+tipos_shift;
    if (t<N_local*dimbat*na)
    {
        int na_particle=num_triplets[b*N_local+par];
        int nn_particle=(na_particle*(na_particle-1))/2;
        if (nn<nn_particle)
        {
            real3 other_forcej = {real(0.), real(0.), real(0.)};
            real3 other_forcek = {real(0.), real(0.), real(0.)};

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

            real delta=real(0.);
            real Bp_j=real(0.);
            real Bp_k=real(0.);

            int actual=b*N_local*nr+par*nr;
            int actual_ang=b*N_local*na+par*na;
            int actgrad=b*N_local*num_finger+par*num_finger;

            int2 neigh=intmap_a_T[b*(N_local*na)+na*par+nn];

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

            real angulardes=desa_T[actual_ang+nn];
            real radialdes_j=desr_T[actual+j];
            real radialdes_k=desr_T[actual+k];

            const real3& pos_i = pos_d_l[b*N + absolute_par];
            const real3& pos_j = pos_d_l[b*N + neigh.x];
            const real3& pos_k = pos_d_l[b*N + neigh.y];
            const real* box_b = box_d + b*6;
            staf_min_image_cart_from_cart(pos_i.x, pos_i.y, pos_i.z,
                                          pos_j.x, pos_j.y, pos_j.z, box_b,
                                          distij.x, distij.y, distij.z);
            staf_min_image_cart_from_cart(pos_i.x, pos_i.y, pos_i.z,
                                          pos_k.x, pos_k.y, pos_k.z, box_b,
                                          distik.x, distik.y, distik.z);

            for (int a1=0; a1<num_finger; a1++)
            {
                real3 alphas=smooth_a_T[sum*num_finger+a1];
		            real chtjk_par=type_emb3b[sum*num_finger+a1];

                real net_der=real(0.5)*netderiv_T[actgrad+a1]*chtjk_par;

		            real expbeta=staf_exp(alphas.z*angulardes);

                real sim1=staf_exp(alphas.y*radialdes_j+alphas.x*radialdes_k);
                real sim2=staf_exp(alphas.x*radialdes_j+alphas.y*radialdes_k);

                delta=expbeta*(1.+alphas.z*angulardes)*(sim1+sim2)*real(0.5);

                real suppj=(alphas.x*sim2+alphas.y*sim1)*expbeta*real(0.5);
                real suppk=(alphas.x*sim1+alphas.y*sim2)*expbeta*real(0.5);
                Bp_j=suppj*angulardes;
                Bp_k=suppk*angulardes;

                real2 intder = intderiv_a_T[b*(N_local*na)*3+par*na*3+0*na+nn];
                real intder_r_j=intderiv_r_T[b*N_local*3*nr+nr*3*par+0*nr+j];
                real intder_r_k=intderiv_r_T[b*N_local*3*nr+nr*3*par+0*nr+k];

                real fxij=net_der*(delta*intder.x+Bp_j*intder_r_j);
                real fxik=net_der*(delta*intder.y+Bp_k*intder_r_k);

                real* vloc = &virial_i[threadIdx.x*9];
                forza_i[threadIdx.x].x-=(fxij+fxik);
                vloc[0]-=fxij*distij.x+fxik*distik.x;
                vloc[1]-=fxij*distij.y+fxik*distik.y;
                vloc[2]-=fxij*distij.z+fxik*distik.z;
                other_forcej.x+=fxij;
                other_forcek.x+=fxik;

                intder = intderiv_a_T[b*(N_local*na)*3+par*na*3+1*na+nn];
                intder_r_j=intderiv_r_T[b*N_local*3*nr+nr*3*par+1*nr+j];
                intder_r_k=intderiv_r_T[b*N_local*3*nr+nr*3*par+1*nr+k];

                fxij=net_der*(delta*intder.x+Bp_j*intder_r_j);
                fxik=net_der*(delta*intder.y+Bp_k*intder_r_k);

                forza_i[threadIdx.x].y-=(fxij+fxik);
                vloc[3]-=fxij*distij.x+fxik*distik.x;
                vloc[4]-=fxij*distij.y+fxik*distik.y;
                vloc[5]-=fxij*distij.z+fxik*distik.z;
                other_forcej.y+=fxij;
                other_forcek.y+=fxik;

                intder = intderiv_a_T[b*(N_local*na)*3+par*na*3+2*na+nn];
                intder_r_j=intderiv_r_T[b*N_local*3*nr+nr*3*par+2*nr+j];
                intder_r_k=intderiv_r_T[b*N_local*3*nr+nr*3*par+2*nr+k];

                fxij=net_der*(delta*intder.x+Bp_j*intder_r_j);
                fxik=net_der*(delta*intder.y+Bp_k*intder_r_k);

                forza_i[threadIdx.x].z-=(fxij+fxik);
                vloc[6]-=fxij*distij.x+fxik*distik.x;
                vloc[7]-=fxij*distij.y+fxik*distik.y;
                vloc[8]-=fxij*distij.z+fxik*distik.z;
                other_forcej.z+=fxij;
                other_forcek.z+=fxik;
            }

            atomicAdd((real*)&(forces3b_T[b*N+neigh.x].x),other_forcej.x);
            atomicAdd((real*)&(forces3b_T[b*N+neigh.x].y),other_forcej.y);
            atomicAdd((real*)&(forces3b_T[b*N+neigh.x].z),other_forcej.z);

            atomicAdd((real*)&(forces3b_T[b*N+neigh.y].x),other_forcek.x);
            atomicAdd((real*)&(forces3b_T[b*N+neigh.y].y),other_forcek.y);
            atomicAdd((real*)&(forces3b_T[b*N+neigh.y].z),other_forcek.z);
        }
    }

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

        atomicAdd((real*)&(forces3b_T[b*N+absolute_par].x),local_force.x);
        atomicAdd((real*)&(forces3b_T[b*N+absolute_par].y),local_force.y);
        atomicAdd((real*)&(forces3b_T[b*N+absolute_par].z),local_force.z);

        for (int c=0;c<9;c++)
            atomicAdd((real*)&(virial_d[b*9+c]),local_vir[c]);
    }
}

void computeforce_tripl_virial_Launcher(const real*  netderiv_T_d, const real* desr_T_d, const real* desa_T_d,
                        const real* intderiv_r_T_d, const real* intderiv_a_T_d,
                        const int* intmap_r_T_d,const int* intmap_a_T_d,
                         int nr, int na, int N, int dimbat,int num_finger,
                         const real* type_emb3b_d,
                         const int* actual_type,real* forces3b_T_d,const int *num_triplets_d,
                         const real* smooth_a_T,const int* color_type_map_T_d,int prod,
                         const int* map_color_interaction,const int* map_intra,
                         real* virial_d,const real* pos_d,const real* box_d){

    dim3 dimGrid(ceil(real(prod)/real(BLOCK_DIM)),1,1);
    dim3 dimBlock(BLOCK_DIM,1,1);
    const size_t shmem = BLOCK_DIM * sizeof(real3) + BLOCK_DIM * 9 * sizeof(real);
    TF_CHECK_OK(::tensorflow::GpuLaunchKernel(computeforce_tripl_virial_kernel, dimGrid, dimBlock, shmem, nullptr,netderiv_T_d,desr_T_d,desa_T_d,
        intderiv_r_T_d,intderiv_a_T_d,intmap_r_T_d,
        intmap_a_T_d,nr,na,N,dimbat,
        num_finger,
        type_emb3b_d,
        actual_type,forces3b_T_d,num_triplets_d,smooth_a_T,color_type_map_T_d,BLOCK_DIM,
        map_color_interaction,map_intra,
        virial_d,pos_d,box_d));

    cudaDeviceSynchronize();

}

#endif
