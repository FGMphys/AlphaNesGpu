#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor" 
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
    if ((buffdim % i == 0) && (i < 512)) return i;
  }
  printf("STAF: No integer divisor found for the given angular buffer size\n");
  exit(0);
}

void init_block_dim(int buffdim) {
  int dev = 0;
  cudaGetDevice(&dev);
  int bd = choose_block_dim(buffdim);
  std::lock_guard<std::mutex> lock(g_bd_mu);
  g_block_dim[dev] = bd;
  printf("STAF: Blocks for angular forces set to %d (device %d)\n", bd, dev);
}

static int current_block_dim() {
  int dev = 0;
  cudaGetDevice(&dev);
  std::lock_guard<std::mutex> lock(g_bd_mu);
  auto it = g_block_dim.find(dev);
  if (it == g_block_dim.end()) {
    fprintf(stderr, "STAF: grad angular BLOCK_DIM not init for device %d\n",
            dev);
    exit(1);
  }
  return it->second;
}


__global__ void gradforce_tripl_kernel(const real*  prevgrad_T_d,const real*  netderiv_T,
                                       const real* desr_T, const real* desa_T,
                                       const real* intderiv_r_T, const real* intderiv_a_T_l,
                                       const int* intmap_r_T,const int* intmap_a_T_l,
                                       int nr, const int na, int N, int dimbat , int num_finger,
                                       const real* type_emb3b,int nt,const int* tipos_T,
                                       const int* actual_type_p,
                                       const int *num_triplets,const real* smooth_a_T_l,
                                       const int* type_map_T_d,real* gradnet_3b_T_d,
                                      real* grad_alpha3b_T,real* grad_emb3b_T_d,int nt_couple,int BLOCK_DIM)
{

    int actual_type=actual_type_p[0];
    int N_local=tipos_T[actual_type];

    /* Fused host (alpha,sum) loops: one launch, blockIdx.y selects the pair. */
    const int req_alpha = blockIdx.y / nt_couple;
    const int req_sum = blockIdx.y % nt_couple;

    int tipos_shift=0;
    for (int y=0;y<actual_type;y++){
        tipos_shift=tipos_shift+tipos_T[y];
    }
    const real2* intderiv_a_T=(const real2 *)intderiv_a_T_l;
    const int2* intmap_a_T=(const int2 *) intmap_a_T_l;
    const real3* smooth_a_T=(const real3 *)smooth_a_T_l;
    real3* grad_alpha3b_T_d=(real3*)grad_alpha3b_T;
    int t=blockIdx.x*blockDim.x+threadIdx.x;


    extern __shared__ real4 allgrad[];
    allgrad[threadIdx.x].x=real(0.);
    allgrad[threadIdx.x].y=real(0.);
    allgrad[threadIdx.x].z=real(0.);


    allgrad[threadIdx.x].w=real(0.);
    __syncthreads();


    real3 local_alpha= {real(0.), real(0.), real(0.)};
    real local_ck= real(0.);
    real local_net=real(0.);

    // from t to b,par,j,k
    int b=t/(na*N_local);
    int reminder=t%(na*N_local);
    int par=reminder/na;
    int nn=reminder%na;
    int absolute_par=par+tipos_shift;
    int sum;
    int actgrad=0;
    if (t<N_local*dimbat*na)
    {
        int na_particle=num_triplets[b*N_local+par];
        int nn_particle=(na_particle*(na_particle-1))/2;
	int na_dim=na_particle;

	int actual=b*N_local*nr+par*nr;
        int actual_ang=b*N_local*na+par*na;
        actgrad=b*N_local*num_finger+par*num_finger;
        if (nn<nn_particle)
        {
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


            int2 neigh=intmap_a_T[b*(N_local*na)+na*par+nn];

            int j_type=type_map_T_d[neigh.x];
            int k_type=type_map_T_d[neigh.y];

            sum=j_type+k_type;
	    if (req_sum==sum){

               real angulardes=desa_T[actual_ang+nn];
               real radialdes_j=desr_T[actual+j];
               real radialdes_k=desr_T[actual+k];


	       real accumulate_1=real(0.);
               real accumulate_3=real(0.);
               real accumulate_4=real(0.);
               real accumulate_5=real(0.);
               real NGel=netderiv_T[actgrad+req_alpha];
               real3 alphas=smooth_a_T[sum*num_finger+req_alpha];
               real chtjk_par=type_emb3b[sum*num_finger+req_alpha];

               real expbeta=staf_exp(alphas.z*angulardes);

               real sim1=staf_exp(alphas.y*radialdes_j+alphas.x*radialdes_k);
               real sim2=staf_exp(alphas.x*radialdes_j+alphas.y*radialdes_k);
               real sum_sim=sim1+sim2;

               delta=expbeta*(real(1.0)+alphas.z*angulardes)*sum_sim*real(0.5);

               real suppj=(alphas.x*sim2+alphas.y*sim1)*expbeta;
               real suppk=(alphas.x*sim1+alphas.y*sim2)*expbeta;
               Bp_j=suppj*angulardes*real(0.5);
               Bp_k=suppk*angulardes*real(0.5);

 	       int cor;
               for (cor=0;cor<3;cor++){
                    real2 intder = intderiv_a_T[b*(N_local*na)*3+par*na*3+cor*na+nn];
                    real intder_r_j=intderiv_r_T[b*N_local*3*nr+nr*3*par+cor*nr+j];
                    real intder_r_k=intderiv_r_T[b*N_local*3*nr+nr*3*par+cor*nr+k];
                    real prevgrad_loc=prevgrad_T_d[b*(N*3)+absolute_par*3+cor];
                    real prevgrad_neighj=prevgrad_T_d[b*(N*3)+neigh.x*3+cor];
                    real prevgrad_neighk=prevgrad_T_d[b*(N*3)+neigh.y*3+cor];

                    real gradxij=chtjk_par*delta*intder.x+chtjk_par*Bp_j*intder_r_j;
                    real gradxik=chtjk_par*delta*intder.y+chtjk_par*Bp_k*intder_r_k;
                    accumulate_1+=-prevgrad_loc*real(0.5)*(gradxij+gradxik);
	            accumulate_1+=prevgrad_neighj*real(0.5)*gradxij+prevgrad_neighk*real(0.5)*gradxik;

                    real buff_a1_ang=expbeta*(real(1.)+alphas.z*angulardes)*(sim1*radialdes_k+sim2*radialdes_j)*real(0.5);
                    real buff_a2_ang=expbeta*(real(1.)+alphas.z*angulardes)*(sim1*radialdes_j+sim2*radialdes_k)*real(0.5);
                    real buff_beta_ang=expbeta*angulardes*(real(2.)+alphas.z*angulardes)*sum_sim*real(0.5);

                    real buff_beta_r_j=suppj*angulardes*angulardes*real(0.5);
                    real buff_beta_r_k=suppk*angulardes*angulardes*real(0.5);

                    real buff_a1_r_j=(sim2+alphas.x*sim2*radialdes_j+alphas.y*sim1*radialdes_k)*expbeta*real(0.5)*angulardes;
                    real buff_a2_r_j=(sim1+alphas.y*sim1*radialdes_j+alphas.x*sim2*radialdes_k)*expbeta*real(0.5)*angulardes;

                    real buff_a1_r_k=(sim1+alphas.x*sim1*radialdes_k+alphas.y*sim2*radialdes_j)*expbeta*real(0.5)*angulardes;
                    real buff_a2_r_k=(sim2+alphas.y*sim2*radialdes_k+alphas.x*sim1*radialdes_j)*expbeta*real(0.5)*angulardes;

                    real grad_a1_xij=chtjk_par*buff_a1_ang*intder.x+chtjk_par*buff_a1_r_j*intder_r_j;
                    real grad_a1_xik=chtjk_par*buff_a1_ang*intder.y+chtjk_par*buff_a1_r_k*intder_r_k;

                    real grad_a2_xij=chtjk_par*buff_a2_ang*intder.x+chtjk_par*buff_a2_r_j*intder_r_j;
                    real grad_a2_xik=chtjk_par*buff_a2_ang*intder.y+chtjk_par*buff_a2_r_k*intder_r_k;

                    real grad_beta_xij=chtjk_par*buff_beta_ang*intder.x+chtjk_par*buff_beta_r_j*intder_r_j;
                    real grad_beta_xik=chtjk_par*buff_beta_ang*intder.y+chtjk_par*buff_beta_r_k*intder_r_k;

                    accumulate_3+=-prevgrad_loc*real(0.5)*NGel*(grad_a1_xij+grad_a1_xik)+prevgrad_neighj*real(0.5)*NGel*grad_a1_xij+prevgrad_neighk*real(0.5)*NGel*grad_a1_xik;

                    accumulate_4+=-prevgrad_loc*real(0.5)*NGel*(grad_a2_xij+grad_a2_xik)+prevgrad_neighj*real(0.5)*NGel*grad_a2_xij+prevgrad_neighk*real(0.5)*NGel*grad_a2_xik;

                    accumulate_5+=-prevgrad_loc*real(0.5)*NGel*(grad_beta_xij+grad_beta_xik)+prevgrad_neighj*real(0.5)*NGel*grad_beta_xij+prevgrad_neighk*real(0.5)*NGel*grad_beta_xik;
               }

               allgrad[threadIdx.x].w=accumulate_1;
	       allgrad[threadIdx.x].x=accumulate_3;

	       allgrad[threadIdx.x].y=accumulate_4;
	       allgrad[threadIdx.x].z=accumulate_5;

	     }

            }
    }
    // Must be reached by every thread in the block (no divergent barrier).
    __syncthreads();
    // Thread 0 reduces outside the t/nn/sum filters (padding threads keep zeros).
    if (threadIdx.x==0){
       for (int dd=0;dd<BLOCK_DIM;dd++){
           local_alpha.x+=allgrad[dd].x;
           local_alpha.y+=allgrad[dd].y;
           local_alpha.z+=allgrad[dd].z;
           local_net+=allgrad[dd].w;
           }
       if (t < N_local*dimbat*na){
         atomicAdd((real*)&(gradnet_3b_T_d[actgrad+req_alpha]),local_net);
       }
       atomicAdd((real*)&(grad_alpha3b_T_d[req_sum*num_finger+req_alpha].x),local_alpha.x);
       atomicAdd((real*)&(grad_alpha3b_T_d[req_sum*num_finger+req_alpha].y),local_alpha.y);
       atomicAdd((real*)&(grad_alpha3b_T_d[req_sum*num_finger+req_alpha].z),local_alpha.z);
      }
}

void gradforce_tripl_Launcher(const real*  prevgrad_T_d,const real*  netderiv_T_d, const real* desr_T_d,
                                      const real* desa_T_d,const real* intderiv_r_T_d,
                                      const real* intderiv_a_T_d,const int* intmap_r_T_d,
                                      const int* intmap_a_T_d,int nr, int na, int N,
                                      int dimbat,int num_finger,const real* type_emb3b_d,int nt,
                                      const int* tipos_T,const int* actual_type,
                                      const int *num_triplets_d,const real* smooth_a_T,
                                      const int* type_map_T_d,int prod,real* gradnet_3b_T_d,
                                      real* grad_alpha3b_T_d,real* grad_emb3b_T_d, cudaStream_t stream){

    int nt_couple=nt*(nt+1)/2;
    const int BLOCK_DIM = current_block_dim();
    dim3 dimGrid(ceil(real(prod)/real(BLOCK_DIM)), num_finger * nt_couple, 1);
    dim3 dimBlock(BLOCK_DIM,1,1);
    TF_CHECK_OK(::tensorflow::GpuLaunchKernel(gradforce_tripl_kernel,dimGrid,
                dimBlock, BLOCK_DIM*sizeof(real4), stream,prevgrad_T_d,netderiv_T_d,desr_T_d,desa_T_d,
                intderiv_r_T_d,intderiv_a_T_d,intmap_r_T_d,
                intmap_a_T_d,nr,na,N,dimbat,num_finger,
                type_emb3b_d,nt,tipos_T,actual_type,
                num_triplets_d,smooth_a_T,type_map_T_d,
                gradnet_3b_T_d,grad_alpha3b_T_d,grad_emb3b_T_d,nt_couple,BLOCK_DIM));

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
