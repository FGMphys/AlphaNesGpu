#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"


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

__global__ void computeforce_doublets_kernel(const double* netderiv,const double* des_r,const double* intderiv_r,const int* intmap_r,
            int nr,int N,int dimbat,
            int num_alpha_radiale,const double* alpha_radiale,
            const double* type_emb2b,int nt,const int* tipos_T,
            const int* actual_type_p,const int* type_map,double* forces2b_l,int BLOCK_DIM)
{

    int actual_type=actual_type_p[0];
    int N_local=tipos_T[actual_type];

    int tipos_shift=0;
    for (int y=0;y<actual_type;y++){
        tipos_shift=tipos_shift+tipos_T[y];
        }

    double3* forces2b=(double3 *)forces2b_l;

    int t=blockIdx.x*blockDim.x+threadIdx.x;

    extern  __shared__ double3 forza_i[];//[BLOCK_DIM];

    forza_i[threadIdx.x].x=0.;
    forza_i[threadIdx.x].y=0.;
    forza_i[threadIdx.x].z=0.;


    double3 local_force = {0.f, 0.f, 0.f};
    double3 other_force = {0.f, 0.f, 0.f};

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

            double des_r_el=des_r[actual+j];
            int ch_type=type_map[neighj];

            double intder_r_x=intderiv_r[b*N_local*3*nr+nr*3*par+0*nr+j];
            double intder_r_y=intderiv_r[b*N_local*3*nr+nr*3*par+1*nr+j];
            double intder_r_z=intderiv_r[b*N_local*3*nr+nr*3*par+2*nr+j];
            for (int i=0; i<num_alpha_radiale;i++){
                double alpha_now=alpha_radiale[num_alpha_radiale*ch_type+i];
                double chpar=type_emb2b[num_alpha_radiale*ch_type+i];
                double sds_deriv=chpar*exp(alpha_now*des_r_el);
                sds_deriv*=(1.f+alpha_now*des_r_el);
                double prevgrad=netderiv[b*N_local*num_alpha_radiale+num_alpha_radiale*par+i];
                double tempx = 0.5f*sds_deriv*intder_r_x;
                double tempy = 0.5f*sds_deriv*intder_r_y;
                double tempz = 0.5f*sds_deriv*intder_r_z;

                forza_i[threadIdx.x].x-=prevgrad*tempx;
                forza_i[threadIdx.x].y-=prevgrad*tempy;
                forza_i[threadIdx.x].z-=prevgrad*tempz;
                other_force.x+=prevgrad*tempx;
                other_force.y+=prevgrad*tempy;
                other_force.z+=prevgrad*tempz;
              }
          }
          atomicAdd((double*)&(forces2b[b*N+neighj].x),other_force.x);
          atomicAdd((double*)&(forces2b[b*N+neighj].y),other_force.y);
          atomicAdd((double*)&(forces2b[b*N+neighj].z),other_force.z);


    __syncthreads();


    if (threadIdx.x==0)
    {
        for (int i=0;i<BLOCK_DIM;i++)
        {
            local_force.x+=forza_i[i].x;
            local_force.y+=forza_i[i].y;
            local_force.z+=forza_i[i].z;
        }

        atomicAdd((double*)&(forces2b[b*N+absolute_par].x),local_force.x);
        atomicAdd((double*)&(forces2b[b*N+absolute_par].y),local_force.y);
        atomicAdd((double*)&(forces2b[b*N+absolute_par].z),local_force.z);

    }

   }
}

void computeforce_doublets_Launcher(const double*  netderiv, const double* des_r,
                    const double* intderiv_r,const int* intmap_r,
                    int nr, int N, int dimbat,int num_alpha_radiale,
                    const double* alpha_radiale,const double* type_emb2b,int nt,
                    const int* tipos_T,const int* actual_type,double* forces2b,const int* type_map,int prod)
{
                      dim3 dimGrid(ceil(double(prod)/double(BLOCK_DIM)),1,1);
     		      dim3 dimBlock(BLOCK_DIM,1,1);

     		      TF_CHECK_OK(::tensorflow::GpuLaunchKernel(computeforce_doublets_kernel, dimGrid, dimBlock, BLOCK_DIM*sizeof(double3), nullptr,netderiv,des_r,
                          intderiv_r,intmap_r,
                          nr,N,dimbat,
                          num_alpha_radiale,alpha_radiale,
                          type_emb2b,nt,tipos_T,
                          actual_type,type_map,forces2b,BLOCK_DIM));

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
