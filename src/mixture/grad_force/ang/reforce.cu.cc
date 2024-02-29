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
        printf("Alpha_nes: No integer divisor found for the given angular buffer size\n");
        exit(0);
     }
     else{
        printf("Alpha_nes: Blocks for angular forces set to %d\n",BLOCK_DIM);
      }
}


__global__ void gradforce_tripl_kernel(const float*  prevgrad_T_d,const float*  netderiv_T,
                                       const float* desr_T, const float* desa_T,
                                       const float* intderiv_r_T, const float* intderiv_a_T_l,
                                       const int* intmap_r_T,const int* intmap_a_T_l,
                                       int nr, const int na, int N, int dimbat , int num_finger,
                                       const float* type_emb3b,int nt,const int* tipos_T,
                                       const int* actual_type_p,
                                       const int *num_triplets,const float* smooth_a_T_l,
                                       const int* type_map_T_d,float* gradnet_3b_T_d,
                                      float* grad_alpha3b_T,float* grad_emb3b_T_d,int req_alpha,int req_sum)
{

    int actual_type=actual_type_p[0];
    int N_local=tipos_T[actual_type];

    int tipos_shift=0;
    for (int y=0;y<actual_type;y++){
        tipos_shift=tipos_shift+tipos_T[y];
    }
    const float2* intderiv_a_T=(const float2 *)intderiv_a_T_l;
    const int2* intmap_a_T=(const int2 *) intmap_a_T_l;
    const float3* smooth_a_T=(const float3 *)smooth_a_T_l;
    float3* grad_alpha3b_T_d=(float3*)grad_alpha3b_T;
    int t=blockIdx.x*blockDim.x+threadIdx.x;


    __shared__ float3 grad_alpha_s[BLOCK_DIM];
    //__shared__ float grad_ck_s[BLOCK_DIM];
    __shared__ float grad_net_s[BLOCK_DIM];
    grad_alpha_s[threadIdx.x].x=0.f;
    grad_alpha_s[threadIdx.x].y=0.f;
    grad_alpha_s[threadIdx.x].z=0.f;

    //grad_ck_s[threadIdx.x]=0.f;

    grad_net_s[threadIdx.x]=0.f;

    float3 local_alpha= {0.f, 0.f, 0.f};
    float local_ck= 0.f;
    float local_net=0.f;

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


            float delta=0.f;
            float Bp_j=0.f;
            float Bp_k=0.f;


            int2 neigh=intmap_a_T[b*(N_local*na)+na*par+nn];

            int j_type=type_map_T_d[neigh.x];
            int k_type=type_map_T_d[neigh.y];

            sum=j_type+k_type;
	    if (req_sum==sum){

               float angulardes=desa_T[actual_ang+nn];
               float radialdes_j=desr_T[actual+j];
               float radialdes_k=desr_T[actual+k];


	       float accumulate_1=0.f;
               //float accumulate_2=0.f;
               float accumulate_3=0.f;
               float accumulate_4=0.f;
               float accumulate_5=0.f;
               float NGel=netderiv_T[actgrad+req_alpha];
               float3 alphas=smooth_a_T[sum*num_finger+req_alpha];
               float chtjk_par=type_emb3b[sum*num_finger+req_alpha];

               float expbeta=expf(alphas.z*angulardes);

               float sim1=expf(alphas.y*radialdes_j+alphas.x*radialdes_k);
               float sim2=expf(alphas.x*radialdes_j+alphas.y*radialdes_k);
               float sum_sim=sim1+sim2;

               delta=expbeta*(1.0f+alphas.z*angulardes)*sum_sim*0.5f;

               float suppj=(alphas.x*sim2+alphas.y*sim1)*expbeta;
               float suppk=(alphas.x*sim1+alphas.y*sim2)*expbeta;
               Bp_j=suppj*angulardes*0.5f;
               Bp_k=suppk*angulardes*0.5f;

 	       int cor;
               for (cor=0;cor<3;cor++){
                    float2 intder = intderiv_a_T[b*(N_local*na)*3+par*na*3+cor*na+nn];
                    float intder_r_j=intderiv_r_T[b*N_local*3*nr+nr*3*par+cor*nr+j];
                    float intder_r_k=intderiv_r_T[b*N_local*3*nr+nr*3*par+cor*nr+k];
                    float prevgrad_loc=prevgrad_T_d[b*(N*3)+absolute_par*3+cor];
                    float prevgrad_neighj=prevgrad_T_d[b*(N*3)+neigh.x*3+cor];
                    float prevgrad_neighk=prevgrad_T_d[b*(N*3)+neigh.y*3+cor];

                    float gradxij=chtjk_par*delta*intder.x+chtjk_par*Bp_j*intder_r_j;
                    float gradxik=chtjk_par*delta*intder.y+chtjk_par*Bp_k*intder_r_k;
                    accumulate_1+=-prevgrad_loc*0.5f*(gradxij+gradxik);
	            accumulate_1+=prevgrad_neighj*0.5f*gradxij+prevgrad_neighk*0.5f*gradxik;
		    //float grad_emb_xij=(delta*intder.x+Bp_j*intder_r_j);
                    //float grad_emb_xik=(delta*intder.y+Bp_k*intder_r_k);
                    //accumulate_2+=-prevgrad_loc*0.5f*NGel*(grad_emb_xij+grad_emb_xik);
	            //accumulate_2+=prevgrad_neighj*0.5*NGel*grad_emb_xij+prevgrad_neighk*0.5f*NGel*grad_emb_xik;

                    float buff_a1_ang=expbeta*(1.f+alphas.z*angulardes)*(sim1*radialdes_k+sim2*radialdes_j)*0.5f;
                    float buff_a2_ang=expbeta*(1.f+alphas.z*angulardes)*(sim1*radialdes_j+sim2*radialdes_k)*0.5f;
                    float buff_beta_ang=expbeta*angulardes*(2.f+alphas.z*angulardes)*sum_sim*0.5f;

                    float buff_beta_r_j=suppj*angulardes*angulardes*0.5f;
                    float buff_beta_r_k=suppk*angulardes*angulardes*0.5f;

                    float buff_a1_r_j=(sim2+alphas.x*sim2*radialdes_j+alphas.y*sim1*radialdes_k)*expbeta*0.5f*angulardes;
                    float buff_a2_r_j=(sim1+alphas.y*sim1*radialdes_j+alphas.x*sim2*radialdes_k)*expbeta*0.5f*angulardes;

                    float buff_a1_r_k=(sim1+alphas.x*sim1*radialdes_k+alphas.y*sim2*radialdes_j)*expbeta*0.5f*angulardes;
                    float buff_a2_r_k=(sim2+alphas.y*sim2*radialdes_k+alphas.x*sim1*radialdes_j)*expbeta*0.5f*angulardes;

                    float grad_a1_xij=chtjk_par*buff_a1_ang*intder.x+chtjk_par*buff_a1_r_j*intder_r_j;
                    float grad_a1_xik=chtjk_par*buff_a1_ang*intder.y+chtjk_par*buff_a1_r_k*intder_r_k;

                    float grad_a2_xij=chtjk_par*buff_a2_ang*intder.x+chtjk_par*buff_a2_r_j*intder_r_j;
                    float grad_a2_xik=chtjk_par*buff_a2_ang*intder.y+chtjk_par*buff_a2_r_k*intder_r_k;

                    float grad_beta_xij=chtjk_par*buff_beta_ang*intder.x+chtjk_par*buff_beta_r_j*intder_r_j;
                    float grad_beta_xik=chtjk_par*buff_beta_ang*intder.y+chtjk_par*buff_beta_r_k*intder_r_k;

                    accumulate_3+=-prevgrad_loc*0.5f*NGel*(grad_a1_xij+grad_a1_xik)+prevgrad_neighj*0.5f*NGel*grad_a1_xij+prevgrad_neighk*0.5f*NGel*grad_a1_xik;

                    accumulate_4+=-prevgrad_loc*0.5f*NGel*(grad_a2_xij+grad_a2_xik)+prevgrad_neighj*0.5f*NGel*grad_a2_xij+prevgrad_neighk*0.5f*NGel*grad_a2_xik;

                    accumulate_5+=-prevgrad_loc*0.5f*NGel*(grad_beta_xij+grad_beta_xik)+prevgrad_neighj*0.5f*NGel*grad_beta_xij+prevgrad_neighk*0.5f*NGel*grad_beta_xik;
               }

               grad_net_s[threadIdx.x]=accumulate_1;
	       grad_alpha_s[threadIdx.x].x=accumulate_3;

	       grad_alpha_s[threadIdx.x].y=accumulate_4;
	       grad_alpha_s[threadIdx.x].z=accumulate_5;

	   //    grad_ck_s[threadIdx.x]=accumulate_2;
	     }


    __syncthreads();
//Il thread zero deve essere usato fuori dagli if. Infatti se la prima tripletta
//è tipo sum=1 e io sto calcolando req_sum=0 non entra nel loop e non fa la riduzione
    if (threadIdx.x==0){
       for (int dd=0;dd<BLOCK_DIM;dd++){
           local_alpha.x+=grad_alpha_s[dd].x;
           local_alpha.y+=grad_alpha_s[dd].y;
           local_alpha.z+=grad_alpha_s[dd].z;
         //  local_ck+=grad_ck_s[dd];
           local_net+=grad_net_s[dd];
           }
       atomicAdd((float*)&(gradnet_3b_T_d[actgrad+req_alpha]),local_net);
       //atomicAdd((float*)&(grad_emb3b_T_d[req_sum*num_finger+req_alpha]),local_ck);
       atomicAdd((float*)&(grad_alpha3b_T_d[req_sum*num_finger+req_alpha].x),local_alpha.x);
       atomicAdd((float*)&(grad_alpha3b_T_d[req_sum*num_finger+req_alpha].y),local_alpha.y);
       atomicAdd((float*)&(grad_alpha3b_T_d[req_sum*num_finger+req_alpha].z),local_alpha.z);
      }
     }
   }
}


void gradforce_tripl_Launcher(const float*  prevgrad_T_d,const float*  netderiv_T_d, const float* desr_T_d,
                                      const float* desa_T_d,const float* intderiv_r_T_d,
                                      const float* intderiv_a_T_d,const int* intmap_r_T_d,
                                      const int* intmap_a_T_d,int nr, int na, int N,
                                      int dimbat,int num_finger,const float* type_emb3b_d,int nt,
                                      const int* tipos_T,const int* actual_type,
                                      const int *num_triplets_d,const float* smooth_a_T,
                                      const int* type_map_T_d,int prod,float* gradnet_3b_T_d,
                                      float* grad_alpha3b_T_d,float* grad_emb3b_T_d){

    dim3 dimGrid(ceil(float(prod)/float(BLOCK_DIM)),1,1);
    dim3 dimBlock(BLOCK_DIM,1,1);
    int nt_couple=nt*(nt+1)/2;
    for (int req_alpha=0;req_alpha<num_finger;req_alpha++){
	for (int req_sum=0;req_sum<nt_couple;req_sum++){
    TF_CHECK_OK(::tensorflow::GpuLaunchKernel(gradforce_tripl_kernel,dimGrid,
                dimBlock, 0, nullptr,prevgrad_T_d,netderiv_T_d,desr_T_d,desa_T_d,
                intderiv_r_T_d,intderiv_a_T_d,intmap_r_T_d,
                intmap_a_T_d,nr,na,N,dimbat,num_finger,
                type_emb3b_d,nt,tipos_T,actual_type,
                num_triplets_d,smooth_a_T,type_map_T_d,
                gradnet_3b_T_d,grad_alpha3b_T_d,grad_emb3b_T_d,req_alpha,req_sum));

	}
    }
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
