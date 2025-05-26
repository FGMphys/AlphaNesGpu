#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"


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


__global__ void gradforce_tripl_kernel(const double*  prevgrad_T_d,const double*  netderiv_T,
                                       const double* desr_T, const double* desa_T,
                                       const double* intderiv_r_T, const double* intderiv_a_T_l,
                                       const int* intmap_r_T,const int* intmap_a_T_l,
                                       int nr, const int na, int N, int dimbat , int num_finger,
                                       const double* type_emb3b,const int* map_color_interaction,
                                       const int* actual_type_p,
                                       const int *num_triplets,const double* smooth_a_T_l,
                                       const int* color_type_map,const int* map_intra,double* gradnet_3b_T_d,
                                      double* grad_alpha3b_T,double* grad_emb3b_T_d,int req_alpha,int req_sum,int BLOCK_DIM)
{

    int actual_type=actual_type_p[0];
    int N_local=N;

    int tipos_shift=0;

    const double2* intderiv_a_T=(const double2 *)intderiv_a_T_l;
    const int2* intmap_a_T=(const int2 *) intmap_a_T_l;
    const double3* smooth_a_T=(const double3 *)smooth_a_T_l;
    double3* grad_alpha3b_T_d=(double3*)grad_alpha3b_T;
    int t=blockIdx.x*blockDim.x+threadIdx.x;


    extern __shared__ double4 allgrad[];
    allgrad[threadIdx.x].x=0.f;
    allgrad[threadIdx.x].y=0.f;
    allgrad[threadIdx.x].z=0.f;


    allgrad[threadIdx.x].w=0.f;


    double3 local_alpha= {0.f, 0.f, 0.f};
    double local_ck= 0.f;
    double local_net=0.f;

    // from t to b,par,j,k
    int b=t/(na*N_local);
    int reminder=t%(na*N_local);
    int par=reminder/na;
    int nn=reminder%na;
    int absolute_par=par+tipos_shift;
    int sum;
    int actgrad;
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


            double delta=0.f;
            double Bp_j=0.f;
            double Bp_k=0.f;


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

	    if (req_sum==sum){

               double angulardes=desa_T[actual_ang+nn];
               double radialdes_j=desr_T[actual+j];
               double radialdes_k=desr_T[actual+k];


	             double accumulate_1=0.f;
               double accumulate_3=0.f;
               double accumulate_4=0.f;
               double accumulate_5=0.f;
               double NGel=netderiv_T[actgrad+req_alpha];
               double3 alphas=smooth_a_T[sum*num_finger+req_alpha];
               double chtjk_par=type_emb3b[sum*num_finger+req_alpha];

               double expbeta=expf(alphas.z*angulardes);

               double sim1=expf(alphas.y*radialdes_j+alphas.x*radialdes_k);
               double sim2=expf(alphas.x*radialdes_j+alphas.y*radialdes_k);
               double sum_sim=sim1+sim2;

               delta=expbeta*(1.0f+alphas.z*angulardes)*sum_sim*0.5f;

               double suppj=(alphas.x*sim2+alphas.y*sim1)*expbeta;
               double suppk=(alphas.x*sim1+alphas.y*sim2)*expbeta;
               Bp_j=suppj*angulardes*0.5f;
               Bp_k=suppk*angulardes*0.5f;

 	       int cor;
               for (cor=0;cor<3;cor++){
                    double2 intder = intderiv_a_T[b*(N_local*na)*3+par*na*3+cor*na+nn];
                    double intder_r_j=intderiv_r_T[b*N_local*3*nr+nr*3*par+cor*nr+j];
                    double intder_r_k=intderiv_r_T[b*N_local*3*nr+nr*3*par+cor*nr+k];
                    double prevgrad_loc=prevgrad_T_d[b*(N*3)+absolute_par*3+cor];
                    double prevgrad_neighj=prevgrad_T_d[b*(N*3)+neigh.x*3+cor];
                    double prevgrad_neighk=prevgrad_T_d[b*(N*3)+neigh.y*3+cor];

                    double gradxij=chtjk_par*delta*intder.x+chtjk_par*Bp_j*intder_r_j;
                    double gradxik=chtjk_par*delta*intder.y+chtjk_par*Bp_k*intder_r_k;
                    accumulate_1+=-prevgrad_loc*0.5f*(gradxij+gradxik);
	            accumulate_1+=prevgrad_neighj*0.5f*gradxij+prevgrad_neighk*0.5f*gradxik;

                    double buff_a1_ang=expbeta*(1.f+alphas.z*angulardes)*(sim1*radialdes_k+sim2*radialdes_j)*0.5f;
                    double buff_a2_ang=expbeta*(1.f+alphas.z*angulardes)*(sim1*radialdes_j+sim2*radialdes_k)*0.5f;
                    double buff_beta_ang=expbeta*angulardes*(2.f+alphas.z*angulardes)*sum_sim*0.5f;

                    double buff_beta_r_j=suppj*angulardes*angulardes*0.5f;
                    double buff_beta_r_k=suppk*angulardes*angulardes*0.5f;

                    double buff_a1_r_j=(sim2+alphas.x*sim2*radialdes_j+alphas.y*sim1*radialdes_k)*expbeta*0.5f*angulardes;
                    double buff_a2_r_j=(sim1+alphas.y*sim1*radialdes_j+alphas.x*sim2*radialdes_k)*expbeta*0.5f*angulardes;

                    double buff_a1_r_k=(sim1+alphas.x*sim1*radialdes_k+alphas.y*sim2*radialdes_j)*expbeta*0.5f*angulardes;
                    double buff_a2_r_k=(sim2+alphas.y*sim2*radialdes_k+alphas.x*sim1*radialdes_j)*expbeta*0.5f*angulardes;

                    double grad_a1_xij=chtjk_par*buff_a1_ang*intder.x+chtjk_par*buff_a1_r_j*intder_r_j;
                    double grad_a1_xik=chtjk_par*buff_a1_ang*intder.y+chtjk_par*buff_a1_r_k*intder_r_k;

                    double grad_a2_xij=chtjk_par*buff_a2_ang*intder.x+chtjk_par*buff_a2_r_j*intder_r_j;
                    double grad_a2_xik=chtjk_par*buff_a2_ang*intder.y+chtjk_par*buff_a2_r_k*intder_r_k;

                    double grad_beta_xij=chtjk_par*buff_beta_ang*intder.x+chtjk_par*buff_beta_r_j*intder_r_j;
                    double grad_beta_xik=chtjk_par*buff_beta_ang*intder.y+chtjk_par*buff_beta_r_k*intder_r_k;

                    accumulate_3+=-prevgrad_loc*0.5f*NGel*(grad_a1_xij+grad_a1_xik)+prevgrad_neighj*0.5f*NGel*grad_a1_xij+prevgrad_neighk*0.5f*NGel*grad_a1_xik;

                    accumulate_4+=-prevgrad_loc*0.5f*NGel*(grad_a2_xij+grad_a2_xik)+prevgrad_neighj*0.5f*NGel*grad_a2_xij+prevgrad_neighk*0.5f*NGel*grad_a2_xik;

                    accumulate_5+=-prevgrad_loc*0.5f*NGel*(grad_beta_xij+grad_beta_xik)+prevgrad_neighj*0.5f*NGel*grad_beta_xij+prevgrad_neighk*0.5f*NGel*grad_beta_xik;
               }

               allgrad[threadIdx.x].w=accumulate_1;
	       allgrad[threadIdx.x].x=accumulate_3;

	       allgrad[threadIdx.x].y=accumulate_4;
	       allgrad[threadIdx.x].z=accumulate_5;

	     }

            }
    __syncthreads();
//Il thread zero deve essere usato fuori dagli if. Infatti se la prima tripletta
//è tipo sum=1 e io sto calcolando req_sum=0 non entra nel loop e non fa la riduzione
    if (threadIdx.x==0){
       for (int dd=0;dd<BLOCK_DIM;dd++){
           local_alpha.x+=allgrad[dd].x;
           local_alpha.y+=allgrad[dd].y;
           local_alpha.z+=allgrad[dd].z;
           local_net+=allgrad[dd].w;
           }
       atomicAdd((double*)&(gradnet_3b_T_d[actgrad+req_alpha]),local_net);
       atomicAdd((double*)&(grad_alpha3b_T_d[req_sum*num_finger+req_alpha].x),local_alpha.x);
       atomicAdd((double*)&(grad_alpha3b_T_d[req_sum*num_finger+req_alpha].y),local_alpha.y);
       atomicAdd((double*)&(grad_alpha3b_T_d[req_sum*num_finger+req_alpha].z),local_alpha.z);
      }
     }
}


void gradforce_tripl_Launcher(const double*  prevgrad_T_d,const double*  netderiv_T_d, const double* desr_T_d,
                                  const double* desa_T_d,const double* intderiv_r_T_d,
                                  const double* intderiv_a_T_d,const int* intmap_r_T_d,
                                  const int* intmap_a_T_d,int nr, int na, int N,int nt_couple,
                                  int dimbat,int num_finger,const double* type_emb3b_d,
                                  const int* map_color_interaction_T,const int* actual_type,
                                  const int *num_triplets_d,const double* smooth_a_T,
                                  const int* color_type_map_T_d,int prod,const int* map_intra_T,
                                  double* gradnet_3b_T_d,double* grad_alpha3b_T_d,double* grad_emb3b_T_d){

    dim3 dimGrid(ceil(double(prod)/double(BLOCK_DIM)),1,1);
    dim3 dimBlock(BLOCK_DIM,1,1);
    for (int req_alpha=0;req_alpha<num_finger;req_alpha++){
	for (int req_sum=0;req_sum<nt_couple;req_sum++){
    TF_CHECK_OK(::tensorflow::GpuLaunchKernel(gradforce_tripl_kernel,dimGrid,
                dimBlock, BLOCK_DIM*sizeof(double4), nullptr,prevgrad_T_d,netderiv_T_d,desr_T_d,desa_T_d,
                intderiv_r_T_d,intderiv_a_T_d,intmap_r_T_d,
                intmap_a_T_d,nr,na,N,dimbat,num_finger,
                type_emb3b_d,map_color_interaction_T,actual_type,
                num_triplets_d,smooth_a_T,color_type_map_T_d,map_intra_T,
                gradnet_3b_T_d,grad_alpha3b_T_d,grad_emb3b_T_d,req_alpha,req_sum,BLOCK_DIM));

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
