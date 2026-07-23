#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "staf_real.h"
#include <mutex>
#include <unordered_map>
#include <cuda_runtime.h>

static std::mutex g_bd_mu;
static std::unordered_map<int, int> g_block_dim;

static int choose_block_dim(int buffdim) {
  for (int i = buffdim; i > 0; i--) {
    if ((buffdim % i == 0) & (i < 512)) return i;
  }
  printf("STAF: No integer divisor found for the given radial buffer size \n");
  exit(0);
}

void init_block_dim(int buffdim) {
  int dev = 0;
  cudaGetDevice(&dev);
  int bd = choose_block_dim(buffdim);
  std::lock_guard<std::mutex> lock(g_bd_mu);
  g_block_dim[dev] = bd;
  printf("STAF: Blocks for radial forces set to %d (device %d)\n", bd, dev);
}

static int current_block_dim() {
  int dev = 0;
  cudaGetDevice(&dev);
  std::lock_guard<std::mutex> lock(g_bd_mu);
  auto it = g_block_dim.find(dev);
  if (it == g_block_dim.end()) {
    fprintf(stderr, "STAF: radial BLOCK_DIM not init for device %d\n", dev);
    exit(1);
  }
  return it->second;
}

__global__ void computeforce_doublets_kernel(const real* netderiv,const real* des_r,const real* intderiv_r,const int* intmap_r,
            int nr,int N,int dimbat,
            int num_alpha_radiale,const real* alpha_radiale,
            const real* type_emb2b,int nt,const int* tipos_T,
            const int* actual_type_p,const int* type_map,real* forces2b_l,int BLOCK_DIM)
{

    int actual_type=actual_type_p[0];
    int N_local=tipos_T[actual_type];

    int tipos_shift=0;
    for (int y=0;y<actual_type;y++){
        tipos_shift=tipos_shift+tipos_T[y];
        }

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

            real des_r_el=des_r[actual+j];
            int ch_type=type_map[neighj];

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
                    const real* alpha_radiale,const real* type_emb2b,int nt,
                    const int* tipos_T,const int* actual_type,real* forces2b,const int* type_map,int prod, cudaStream_t stream){
                      const int BLOCK_DIM = current_block_dim();
                      dim3 dimGrid(ceil(real(prod)/real(BLOCK_DIM)),1,1);
     		      dim3 dimBlock(BLOCK_DIM,1,1);

     		      TF_CHECK_OK(::tensorflow::GpuLaunchKernel(computeforce_doublets_kernel, dimGrid, dimBlock, BLOCK_DIM*sizeof(real3), stream,netderiv,des_r,
                          intderiv_r,intmap_r,
                          nr,N,dimbat,
                          num_alpha_radiale,alpha_radiale,
                          type_emb2b,nt,tipos_T,
                          actual_type,type_map,forces2b,BLOCK_DIM));


}


__global__ void set_tensor_to_zero_real_kernel(real* tensor,int dim){
          int t=blockIdx.x*blockDim.x+threadIdx.x;

          if (t<dim)
             tensor[t]=real(0.);
}

void set_tensor_to_zero_real(real* tensor,int dimten, cudaStream_t stream){
     int grids=ceil(real(dimten)/real(300));
     dim3 dimGrid(grids,1,1);
     dim3 dimBlock(300,1,1);
     // No DeviceSynchronize: ordered on same stream as subsequent GpuLaunchKernel.
     TF_CHECK_OK(::tensorflow::GpuLaunchKernel(set_tensor_to_zero_real_kernel,dimGrid,dimBlock, 0, stream,tensor,dimten));
}

#endif
