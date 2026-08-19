#include <cuda.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "../../../vector.h"
#include <stdio.h>
#include <stdlib.h>


#define BLOCK_DIM 80


__global__ void radialAFs_kernel(
        const double* radial_descriptor,const int nr,const double* alpha2b_parameters,
         int nalpha_r,int nalpha_a,double* radial_AFs,const int dimbat,const int N_local,
        const int* interaction_map_rad,const double* type_emb2b,const int* color_type_map,const int* map_intra,const int* map_color_interaction)
{
        const int* intmap_r=(const int*) interaction_map_rad;
        const double* alphas=(const double*) alpha2b_parameters;
        const double* ds=(const double*)radial_descriptor;


        int t=blockIdx.x*blockDim.x+threadIdx.x;
        int b=t/(nr*N_local);
        int reminder=t%(nr*N_local);
        int par=reminder/nr;
        int j=reminder%nr;
        if (t<N_local*dimbat*nr)
        {

	    int nr_particle=intmap_r[b*N_local*(nr+1)+par*(nr+1)];
            if (j<nr_particle)
            {
                int actual=b*N_local*nr+par*nr;

                double des_r_el=ds[actual+j];
                int neighj=intmap_r[b*(N_local*(nr+1))+(nr+1)*par+1+j];

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

                // costruiamo i descrittori
                for (int i=0; i<nalpha_r;i++){
                    double alpha_now=alphas[nalpha_r*ch_type+i];
                    double chpar=type_emb2b[nalpha_r*ch_type+i];
                    double softmaxweight=exp(alpha_now*des_r_el)*chpar;

                    atomicAdd((double*)&radial_AFs[b*(nalpha_r+nalpha_a)*N_local+par*(nalpha_r+nalpha_a)+i], des_r_el*softmaxweight);
                }
            }
        }

}


void radialAFs_Launcher(
        const double* radial_descriptor,const int nr,const double* alpha2b_parameters,
        int nalpha_r,int nalpha_a,double* radial_AFs,const int dimbat,const int N_local,
        const int* interaction_map_rad,const double* type_emb2b,const int* color_type_map,const int* map_intra,const int* map_color_interaction)
{
        dim3 dimGrid(ceil(double(dimbat*N_local*nr)/double(BLOCK_DIM)),1,1);
        dim3 dimBlock(BLOCK_DIM,1,1);

        radialAFs_kernel<<<dimGrid, dimBlock, 0, nullptr>>>(radial_descriptor,nr,
                alpha2b_parameters,nalpha_r,nalpha_a,radial_AFs,dimbat,N_local,
                interaction_map_rad,type_emb2b,color_type_map,map_intra,map_color_interaction);
        
        cudaDeviceSynchronize();
}
