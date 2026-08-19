#include <cuda.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "../../../vector.h"
#include <stdio.h>
#include <stdlib.h>


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
            int num_alpha_radiale,int num_alpha_ang,const double* alpha_radiale,
            const double* type_emb2b,const int* color_type_map,double* forces2b_l,
            double* virial_diagonal_d,double* pos_d,double* box_d,int BLOCK_DIM,
            const int* map_intra, const int* map_color_interaction, int n_all)
{
    int N_local=N;
    int N_force = (n_all > 0) ? n_all : N;

    int tipos_shift=0;

    double3* forces2b=(double3 *)forces2b_l;
    double3* pos_d_l=(double3 *) pos_d;
    int t=blockIdx.x*blockDim.x+threadIdx.x;


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


    double3 local_force = {0.f, 0.f, 0.f};
    double3 local_virial = {0.f,0.f,0.f};
    double3 other_force = {0.f, 0.f, 0.f};
    double3 distij = {0.f, 0.f, 0.f};
    double3 rij = {0.f, 0.f, 0.f};

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
            int ch_type=row_index;
            double intder_r_x=intderiv_r[b*N_local*3*nr+nr*3*par+0*nr+j];
            double intder_r_y=intderiv_r[b*N_local*3*nr+nr*3*par+1*nr+j];
            double intder_r_z=intderiv_r[b*N_local*3*nr+nr*3*par+2*nr+j];
            for (int i=0; i<num_alpha_radiale;i++){
                double alpha_now=alpha_radiale[num_alpha_radiale*ch_type+i];
                double chpar=type_emb2b[num_alpha_radiale*ch_type+i];
                double sds_deriv=chpar*exp(alpha_now*des_r_el);
                sds_deriv*=(1.f+alpha_now*des_r_el);
                double prevgrad=netderiv[b*N_local*(num_alpha_radiale+num_alpha_ang)+(num_alpha_radiale+num_alpha_ang)*par+i];
                double tempx = 0.5f*sds_deriv*intder_r_x;
                double tempy = 0.5f*sds_deriv*intder_r_y;
                double tempz = 0.5f*sds_deriv*intder_r_z;
               
                forza_i[threadIdx.x].x-=prevgrad*tempx;
                forza_i[threadIdx.x].y-=prevgrad*tempy;
                forza_i[threadIdx.x].z-=prevgrad*tempz;
                //VIRIAL
                rij.x=pos_d_l[absolute_par].x-pos_d_l[neighj].x;
                rij.y=pos_d_l[absolute_par].y-pos_d_l[neighj].y;
                rij.z=pos_d_l[absolute_par].z-pos_d_l[neighj].z;

                rij.x-=rint(rij.x);
                rij.y-=rint(rij.y);
                rij.z-=rint(rij.z);

                distij.x=box_d[0]*rij.x+box_d[1]*rij.y+box_d[2]*rij.z;
                distij.y=box_d[3]*rij.y+box_d[4]*rij.z;
                distij.z=box_d[5]*rij.z;


		virial_diagonal_i[threadIdx.x].x-=prevgrad*tempx*distij.x;
                virial_diagonal_i[threadIdx.x].y-=prevgrad*tempy*distij.y;
		virial_diagonal_i[threadIdx.x].z-=prevgrad*tempz*distij.z;

                other_force.x+=prevgrad*tempx;
                other_force.y+=prevgrad*tempy;
                other_force.z+=prevgrad*tempz;
              }
          }
          atomicAdd((double*)&(forces2b[b*N_force+neighj].x),other_force.x);
          atomicAdd((double*)&(forces2b[b*N_force+neighj].y),other_force.y);
          atomicAdd((double*)&(forces2b[b*N_force+neighj].z),other_force.z);


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


        atomicAdd((double*)&(forces2b[b*N_force+absolute_par].x),local_force.x);
        atomicAdd((double*)&(forces2b[b*N_force+absolute_par].y),local_force.y);
        atomicAdd((double*)&(forces2b[b*N_force+absolute_par].z),local_force.z);

	atomicAdd((double*)&(virial_diagonal_d[0]),local_virial.x);
        atomicAdd((double*)&(virial_diagonal_d[1]),local_virial.y);
        atomicAdd((double*)&(virial_diagonal_d[2]),local_virial.z);

    }

   }
}

void computeforce_doublets_Launcher(const double*  netderiv, const double* des_r,
                    const double* intderiv_r,const int* intmap_r,
                    int nr, int N, int dimbat,int num_alpha_radiale,int num_alpha_ang,
                    const double* alpha_radiale,const double* type_emb2b,double* forces2b,
                    const int* color_type_map,int prod,double* virial_diagonal_d,
                    double* pos_d,double* box_d,const int* map_intra,const int* map_color_interaction,
                    int n_all)
{
                      dim3 dimGrid(ceil(double(prod)/double(BLOCK_DIM)),1,1);
     		      dim3 dimBlock(BLOCK_DIM,1,1);

     		      computeforce_doublets_kernel<<<dimGrid, dimBlock, 2*BLOCK_DIM*sizeof(double3), nullptr>>>(netderiv,des_r,
                          intderiv_r,intmap_r,
                          nr,N,dimbat,
                          num_alpha_radiale,num_alpha_ang,alpha_radiale,
                          type_emb2b,color_type_map,forces2b,virial_diagonal_d,pos_d,box_d,BLOCK_DIM,
                          map_intra,map_color_interaction, n_all);

                      cudaDeviceSynchronize();

     }
