#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"


#define BLOCK_DIM 118

__global__ void computeforce_tripl_kernel(const float*  netderiv_T, const float* desr_T, const float* desa_T,
                        const float* intderiv_r_T, const float* intderiv_a_T_l,
                        const int* intmap_r_T,const int* intmap_a_T_l,
                        int nr, const int na, int N, int dimbat , int num_finger,const float* type_emb3b,int nt,
                        const int* tipos_T,
                        const int* actual_type_p,float* forces3b_T_l,const int *num_triplets,const float* smooth_a_T_l,const int* type_map_T_d)
{
    

      

    int actual_type=actual_type_p[0];
    int N_local=tipos_T[actual_type];

    int tipos_shift=0;
    for (int y=0;y<actual_type;y++){
        tipos_shift=tipos_shift+tipos_T[y];
    }
    const float2* intderiv_a_T=(const float2 *)intderiv_a_T_l; 
    const int2* intmap_a_T=(const int2 *) intmap_a_T_l;
    float3* forces3b_T=(float3 *)forces3b_T_l;
    const float3* smooth_a_T=(const float3 *)smooth_a_T_l;

    int t=blockIdx.x*blockDim.x+threadIdx.x;
   
    __shared__ float3 forza_i[BLOCK_DIM];

    forza_i[threadIdx.x].x=0.;
    forza_i[threadIdx.x].y=0.;
    forza_i[threadIdx.x].z=0.;

    // __syncthreads();

    float3 local_force = {0.f, 0.f, 0.f};

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

            
            float3 other_forcej = {0.f, 0.f, 0.f};
            float3 other_forcek = {0.f, 0.f, 0.f};

            int na_dim=na_particle;//floorf(0.5f + sqrtf(0.25f + 2*na));
            
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
            
            
            
            float delta=0.f;
            float Bp_j=0.f;
            float Bp_k=0.f;
            

            
            int actual=b*N_local*nr+par*nr;
            int actual_ang=b*N_local*na+par*na;
            int actgrad=b*N_local*num_finger+par*num_finger;

            int2 neigh=intmap_a_T[b*(N_local*na)+na*par+nn];

          

            int j_type=type_map_T_d[neigh.x];
            int k_type=type_map_T_d[neigh.y];
          
            //float chtjk_par=type_emb3b[j_type*nt+k_type];
            
            int sum=j_type+k_type;

            float angulardes=desa_T[actual_ang+nn];
            float radialdes_j=desr_T[actual+j];
            float radialdes_k=desr_T[actual+k];

           

            // loop su alpha
            for (int a1=0; a1<num_finger; a1++)
            {
                float3 alphas=smooth_a_T[sum*num_finger+a1];
		float chtjk_par=type_emb3b[sum*num_finger+a1];

                float net_der=0.5f*netderiv_T[actgrad+a1]*chtjk_par;

		float expbeta=expf(alphas.z*angulardes);

                float sim1=expf(alphas.y*radialdes_j+alphas.x*radialdes_k);
                float sim2=expf(alphas.x*radialdes_j+alphas.y*radialdes_k);

                delta=expbeta*(1.+alphas.z*angulardes)*(sim1+sim2)*0.5f;

                float suppj=(alphas.x*sim2+alphas.y*sim1)*expbeta*0.5f;
                float suppk=(alphas.x*sim1+alphas.y*sim2)*expbeta*0.5f;
                Bp_j=suppj*angulardes;
                Bp_k=suppk*angulardes;

                // x
                float2 intder = intderiv_a_T[b*(N_local*na)*3+par*na*3+0*na+nn];
                float intder_r_j=intderiv_r_T[b*N_local*3*nr+nr*3*par+0*nr+j];
                float intder_r_k=intderiv_r_T[b*N_local*3*nr+nr*3*par+0*nr+k];
                
                float fxij=net_der*(delta*intder.x+Bp_j*intder_r_j);
                float fxik=net_der*(delta*intder.y+Bp_k*intder_r_k);

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
                  
            atomicAdd((float*)&(forces3b_T[b*N+neigh.x].x),other_forcej.x);
            atomicAdd((float*)&(forces3b_T[b*N+neigh.x].y),other_forcej.y);
            atomicAdd((float*)&(forces3b_T[b*N+neigh.x].z),other_forcej.z);
            
            atomicAdd((float*)&(forces3b_T[b*N+neigh.y].x),other_forcek.x);
            atomicAdd((float*)&(forces3b_T[b*N+neigh.y].y),other_forcek.y);
            atomicAdd((float*)&(forces3b_T[b*N+neigh.y].z),other_forcek.z);
        


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

        atomicAdd((float*)&(forces3b_T[b*N+absolute_par].x),local_force.x);
        atomicAdd((float*)&(forces3b_T[b*N+absolute_par].y),local_force.y);
        atomicAdd((float*)&(forces3b_T[b*N+absolute_par].z),local_force.z);

    }
}
    
void computeforce_tripl_Launcher(const float*  netderiv_T_d, const float* desr_T_d, const float* desa_T_d,
                        const float* intderiv_r_T_d, const float* intderiv_a_T_d,
                        const int* intmap_r_T_d,const int* intmap_a_T_d,
                         int nr, int na, int N, int dimbat,int num_finger,const float* type_emb3b_d,int nt,const int* tipos_T,const int* actual_type,float* forces3b_T_d,const int *num_triplets_d,const float* smooth_a_T,const int* type_map_T_d,int prod){

    dim3 dimGrid(ceil(float(prod)/float(BLOCK_DIM)),1,1);
    dim3 dimBlock(BLOCK_DIM,1,1);
    TF_CHECK_OK(::tensorflow::GpuLaunchKernel(computeforce_tripl_kernel, dimGrid, dimBlock, 0, nullptr,netderiv_T_d,desr_T_d,desa_T_d,
        intderiv_r_T_d,intderiv_a_T_d,intmap_r_T_d,
        intmap_a_T_d,nr,na,N,dimbat,
        num_finger,
        type_emb3b_d,nt,tipos_T,
        actual_type,forces3b_T_d,num_triplets_d,smooth_a_T,type_map_T_d));
     
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
