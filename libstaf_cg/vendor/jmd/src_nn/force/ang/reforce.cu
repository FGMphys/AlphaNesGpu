#include <cuda.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "../../../vector.h"
#include <stdio.h>
#include <stdlib.h>


static int BLOCK_DIM;

__host__ __device__ int get_symmetric_pair_index_f(int i, int j, int ntypes) {
  int diff = i - j;
  int mask = diff >> 31; // 0 se diff ≥ 0, -1 se diff < 0
  int min = j + (diff & mask);
  int max = i - (diff & mask);
  return min * ntypes - (min * (min + 1)) / 2 + max;
}
void init_block_dim_ang(int buffdim){
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




__global__ void computeforce_tripl_kernel(const double*  netderiv_T, const double* desr_T, const double* desa_T,
                        const double* intderiv_r_T, const double* intderiv_a_T_l,
                        const int* intmap_r_T,const int* intmap_a_T_l,
                        int nr, const int na, int N, int dimbat , int num_finger_a,
                        int num_finger_r,const double* type_emb3b,double* forces3b_T_l,const int *num_triplets,
                        const double* smooth_a_T_l,const int* color_type_map,double* virial_diagonal_d,
                        double* pos_d,double* box_d,int BLOCK_DIM,const int* map_intra,const int* map_color_interaction,
                        int n_all)
{


    int N_local=N;
    int N_force = (n_all > 0) ? n_all : N;

    int tipos_shift=0;

    const double2* intderiv_a_T=(const double2 *)intderiv_a_T_l;
    const int2* intmap_a_T=(const int2 *) intmap_a_T_l;
    double3* forces3b_T=(double3 *)forces3b_T_l;
    const double3* smooth_a_T=(const double3 *)smooth_a_T_l;

    int t=blockIdx.x*blockDim.x+threadIdx.x;

    double3* pos_d_l=(double3*)pos_d;

    extern __shared__ unsigned char sharedMemory[];

    // Puntatore all'array di double3
    double3 *forza_i = (double3 *) sharedMemory;

    // Puntatore all'array di int (dopo i double3)
    double3 *virial_diagonal_i = (double3 *)(sharedMemory + sizeof(double3) * BLOCK_DIM);

    forza_i[threadIdx.x].x=0.;
    forza_i[threadIdx.x].y=0.;
    forza_i[threadIdx.x].z=0.;
    
    virial_diagonal_i[threadIdx.x].x=0.;
    virial_diagonal_i[threadIdx.x].y=0.;
    virial_diagonal_i[threadIdx.x].z=0.;
    // __syncthreads();

    double3 local_force = {0., 0., 0.};
    double3 local_virial = {0.,0.,0.};
    double3 distij = {0., 0., 0.};
    double3 rij = {0., 0., 0.};

    double3 distik = {0., 0., 0.};
    double3 rik = {0., 0., 0.};

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


            double3 other_forcej = {0., 0., 0.};
            double3 other_forcek = {0., 0., 0.};

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



            double delta=0.;
            double Bp_j=0.;
            double Bp_k=0.;



            int actual=b*N_local*nr+par*nr;
            int actual_ang=b*N_local*na+par*na;
            //Ho saltato la parte radiale
            int actgrad=b*N_local*(num_finger_r+num_finger_a)+par*(num_finger_r+num_finger_a)+num_finger_r;

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

            int sum=get_symmetric_pair_index_f(type_int_j,type_int_k, 3);


            double angulardes=desa_T[actual_ang+nn];
            double radialdes_j=desr_T[actual+j];
            double radialdes_k=desr_T[actual+k];



            // loop su alpha
            for (int a1=0; a1<num_finger_a; a1++)
            {
                double3 alphas=smooth_a_T[sum*num_finger_a+a1];
		        double chtjk_par=type_emb3b[sum*num_finger_a+a1];

                double net_der=0.5f*netderiv_T[actgrad+a1]*chtjk_par;

		        double expbeta=expf(alphas.z*angulardes);

                double sim1=expf(alphas.y*radialdes_j+alphas.x*radialdes_k);
                double sim2=expf(alphas.x*radialdes_j+alphas.y*radialdes_k);

                delta=expbeta*(1.+alphas.z*angulardes)*(sim1+sim2)*0.5f;

                double suppj=(alphas.x*sim2+alphas.y*sim1)*expbeta*0.5f;
                double suppk=(alphas.x*sim1+alphas.y*sim2)*expbeta*0.5f;
                Bp_j=suppj*angulardes;
                Bp_k=suppk*angulardes;


                //DIST ij
                rij.x=pos_d_l[absolute_par].x-pos_d_l[neigh.x].x;
                rij.y=pos_d_l[absolute_par].y-pos_d_l[neigh.x].y;
                rij.z=pos_d_l[absolute_par].z-pos_d_l[neigh.x].z;

                rij.x-=rint(rij.x);
                rij.y-=rint(rij.y);
                rij.z-=rint(rij.z);

                distij.x=box_d[0]*rij.x+box_d[1]*rij.y+box_d[2]*rij.z;
                distij.y=box_d[3]*rij.y+box_d[4]*rij.z;
                distij.z=box_d[5]*rij.z;
		//DIST ik
                rik.x=pos_d_l[absolute_par].x-pos_d_l[neigh.y].x;
                rik.y=pos_d_l[absolute_par].y-pos_d_l[neigh.y].y;
                rik.z=pos_d_l[absolute_par].z-pos_d_l[neigh.y].z;

                rik.x-=rint(rik.x);
                rik.y-=rint(rik.y);
                rik.z-=rint(rik.z);

                distik.x=box_d[0]*rik.x+box_d[1]*rik.y+box_d[2]*rik.z;
                distik.y=box_d[3]*rik.y+box_d[4]*rik.z;
                distik.z=box_d[5]*rik.z;

                 // x
                double2 intder = intderiv_a_T[b*(N_local*na)*3+par*na*3+0*na+nn];
                double intder_r_j=intderiv_r_T[b*N_local*3*nr+nr*3*par+0*nr+j];
                double intder_r_k=intderiv_r_T[b*N_local*3*nr+nr*3*par+0*nr+k];

                double fxij=net_der*(delta*intder.x+Bp_j*intder_r_j);
                double fxik=net_der*(delta*intder.y+Bp_k*intder_r_k);

                forza_i[threadIdx.x].x-=(fxij+fxik);
                virial_diagonal_i[threadIdx.x].x-=fxij*distij.x;
		virial_diagonal_i[threadIdx.x].x-=fxik*distik.x;

                other_forcej.x+=fxij;
                other_forcek.x+=fxik;

                // y
                intder = intderiv_a_T[b*(N_local*na)*3+par*na*3+1*na+nn];
                intder_r_j=intderiv_r_T[b*N_local*3*nr+nr*3*par+1*nr+j];
                intder_r_k=intderiv_r_T[b*N_local*3*nr+nr*3*par+1*nr+k];

                double fyij=net_der*(delta*intder.x+Bp_j*intder_r_j);
                double fyik=net_der*(delta*intder.y+Bp_k*intder_r_k);

                forza_i[threadIdx.x].y-=(fyij+fyik);
		virial_diagonal_i[threadIdx.x].y-=fyij*distij.y;
                virial_diagonal_i[threadIdx.x].y-=fyik*distik.y;
                other_forcej.y+=fyij;
                other_forcek.y+=fyik;


                // z
                intder = intderiv_a_T[b*(N_local*na)*3+par*na*3+2*na+nn];
                intder_r_j=intderiv_r_T[b*N_local*3*nr+nr*3*par+2*nr+j];
                intder_r_k=intderiv_r_T[b*N_local*3*nr+nr*3*par+2*nr+k];

                double fzij=net_der*(delta*intder.x+Bp_j*intder_r_j);
                double fzik=net_der*(delta*intder.y+Bp_k*intder_r_k);

                forza_i[threadIdx.x].z-=(fzij+fzik);
		virial_diagonal_i[threadIdx.x].z-=fzij*distij.z;
                virial_diagonal_i[threadIdx.x].z-=fzik*distik.z;

                other_forcej.z+=fzij;
                other_forcek.z+=fzik;

            }

            atomicAdd((double*)&(forces3b_T[b*N_force+neigh.x].x),other_forcej.x);
            atomicAdd((double*)&(forces3b_T[b*N_force+neigh.x].y),other_forcej.y);
            atomicAdd((double*)&(forces3b_T[b*N_force+neigh.x].z),other_forcej.z);

            atomicAdd((double*)&(forces3b_T[b*N_force+neigh.y].x),other_forcek.x);
            atomicAdd((double*)&(forces3b_T[b*N_force+neigh.y].y),other_forcek.y);
            atomicAdd((double*)&(forces3b_T[b*N_force+neigh.y].z),other_forcek.z);



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
	    local_virial.x+=virial_diagonal_i[i].x;
            local_virial.y+=virial_diagonal_i[i].y;
            local_virial.z+=virial_diagonal_i[i].z;
        }

        atomicAdd((double*)&(forces3b_T[b*N_force+absolute_par].x),local_force.x);
        atomicAdd((double*)&(forces3b_T[b*N_force+absolute_par].y),local_force.y);
        atomicAdd((double*)&(forces3b_T[b*N_force+absolute_par].z),local_force.z);

	    atomicAdd((double*)&(virial_diagonal_d[0]),local_virial.x);
        atomicAdd((double*)&(virial_diagonal_d[1]),local_virial.y);
        atomicAdd((double*)&(virial_diagonal_d[2]),local_virial.z);

    }
}

void computeforce_tripl_Launcher(const double*  netderiv_T_d, const double* desr_T_d, const double* desa_T_d,
                        const double* intderiv_r_T_d, const double* intderiv_a_T_d,
                        const int* intmap_r_T_d,const int* intmap_a_T_d,
                         int nr, int na, int N, int dimbat,int num_finger_a,int num_finger_r,
                         const double* type_emb3b_d,
                         double* forces3b_T_d,const int *num_triplets_d,const double* smooth_a_T,
                         const int* color_type_map_T_d,int prod,double* virial_diagonal_d,
                         double* pos_d,double* box_d,const int* map_intra,const int* map_color_interaction,
                         int n_all){

    dim3 dimGrid(ceil(double(prod)/double(BLOCK_DIM)),1,1);
    dim3 dimBlock(BLOCK_DIM,1,1);
    computeforce_tripl_kernel<<<dimGrid, dimBlock, 2*BLOCK_DIM*sizeof(double3), nullptr>>>(netderiv_T_d,desr_T_d,desa_T_d,
        intderiv_r_T_d,intderiv_a_T_d,intmap_r_T_d,
        intmap_a_T_d,nr,na,N,dimbat,
        num_finger_a,num_finger_r,
        type_emb3b_d,forces3b_T_d,num_triplets_d,smooth_a_T,color_type_map_T_d,virial_diagonal_d,pos_d,box_d,BLOCK_DIM,
        map_intra,map_color_interaction, n_all);

    cudaDeviceSynchronize();

}
