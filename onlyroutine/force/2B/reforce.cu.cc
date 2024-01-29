#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"


#define BLOCK_DIM 80


__global__ void computeforce_doublets_kernel(const float* netderiv,const float* des_r,const float* intderiv_r,const int* intmap_r,
            int nr,int N,int dimbat,
            int num_alpha_radiale,const float* alpha_radiale,
            const float* type_emb2b,int nt,const int* tipos_T,
            const int* actual_type_p,const int* type_map,float* forces2b_l)
{

    int actual_type=actual_type_p[0];
    int N_local=tipos_T[actual_type];

    int tipos_shift=0;
    for (int y=0;y<actual_type;y++){
        tipos_shift=tipos_shift+tipos_T[y];
        }

    float4* forces2b=(float4 *)forces2b_l;

    int t=blockIdx.x*blockDim.x+threadIdx.x;

    __shared__ float3 forza_i[BLOCK_DIM];

    forza_i[threadIdx.x].x=0.;
    forza_i[threadIdx.x].y=0.;
    forza_i[threadIdx.x].z=0.;

    // __syncthreads();

    float3 local_force = {0.f, 0.f, 0.f};
    float3 other_force = {0.f, 0.f, 0.f};

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

            float des_r_el=des_r[actual+j];
            int ch_type=type_map[neighj];

            float intder_r_x=intderiv_r[b*N_local*3*nr+nr*3*par+0*nr+j];
            float intder_r_y=intderiv_r[b*N_local*3*nr+nr*3*par+1*nr+j];
            float intder_r_z=intderiv_r[b*N_local*3*nr+nr*3*par+2*nr+j];
            for (int i=0; i<num_alpha_radiale;i++){
                float alpha_now=alpha_radiale[num_alpha_radiale*ch_type+i];
                float chpar=type_emb2b[num_alpha_radiale*ch_type+i];
                float sds_deriv=chpar*exp(alpha_now*des_r_el);
                sds_deriv*=(1.+alpha_now*des_r_el);
                float prevgrad=netderiv[b*N_local*num_alpha_radiale+num_alpha_radiale*par+i];
                float tempx = 0.5*sds_deriv*intder_r_x;
                float tempy = 0.5*sds_deriv*intder_r_y;
                float tempz = 0.5*sds_deriv*intder_r_z;

                forza_i[threadIdx.x].x-=prevgrad*tempx;
                forza_i[threadIdx.x].y-=prevgrad*tempy;
                forza_i[threadIdx.x].z-=prevgrad*tempz;
                other_force.x+=prevgrad*tempx;
                other_force.y+=prevgrad*tempy;
                other_force.z+=prevgrad*tempz;
              }
          }
          atomicAdd((float*)&(forces2b[b*N+neighj].x),other_force.x);
          atomicAdd((float*)&(forces2b[b*N+neighj].y),other_force.y);
          atomicAdd((float*)&(forces2b[b*N+neighj].z),other_force.z);

    __syncthreads();

    if (threadIdx.x==0)
    {
        for (int i=0;i<BLOCK_DIM;i++)
        {
            local_force.x+=forza_i[i].x;
            local_force.y+=forza_i[i].y;
            local_force.z+=forza_i[i].z;
        }

        atomicAdd((float*)&(forces2b[b*N+absolute_par].x),local_force.x);
        atomicAdd((float*)&(forces2b[b*N+absolute_par].y),local_force.y);
        atomicAdd((float*)&(forces2b[b*N+absolute_par].z),local_force.z);

    }
  }
}

void computeforce_doublets_Launcher(const float*  netderiv, const float* des_r,
                    const float* intderiv_r,const int* intmap_r,
                    int nr, int N, int dimbat,int num_alpha_radiale,
                    const float* alpha_radiale,const float* type_emb2b,int nt,
                    const int* tipos_T,const int* actual_type,float* forces2b,const int* type_map,int prod)
{
                      dim3 dimGrid(ceil(float(prod)/float(BLOCK_DIM)),1,1);
     		      dim3 dimBlock(BLOCK_DIM,1,1);
     
     		      TF_CHECK_OK(::tensorflow::GpuLaunchKernel(computeforce_doublets_kernel, dimGrid, dimBlock, 0, nullptr,netderiv,des_r,
                          intderiv_r,intmap_r,
                          nr,N,dimbat,
                          num_alpha_radiale,alpha_radiale,
                          type_emb2b,nt,tipos_T,
                          actual_type,type_map,forces2b));

                      cudaDeviceSynchronize();
                    }
#endif
