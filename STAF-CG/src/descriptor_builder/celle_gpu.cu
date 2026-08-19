#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "src_nn/descriptor_builder/reforce.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include "staf_real.h"
#define BLOCK_DIM 256
#define SQR(x) ((x)*(x))



__global__ void imeBuild(int N,real *box,real *position,int *cells,int *cells_howmany,int celle_nx,int celle_ny,int celle_nz,real cutoff,int *with,int *howmany,real *with_dist2,int MAX_PARTICLE_CELLS,int maxneigh)
{
    real3 *coor=(real3*)position;
    extern __shared__ unsigned char sharedMemory[];  // Dichiarazione generica

    // Puntatore all'array di real3
    real3 *pos_ncella = (real3 *) sharedMemory;

    // Puntatore all'array di int (dopo i real3)
    int *i_ncella = (int *)(sharedMemory + sizeof(real3) * MAX_PARTICLE_CELLS);


    int central_cell=blockIdx.x+blockIdx.y*celle_nx+blockIdx.z*celle_nx*celle_ny;



    real3 p_i;
    int whoami;
    if (threadIdx.x<cells_howmany[central_cell])
    {
    	whoami=cells[central_cell*MAX_PARTICLE_CELLS+threadIdx.x];
    	p_i=coor[whoami];
    }

    
    int i,j,k;
    for (i=-1;i<2;i++)
    {
        int bi=blockIdx.x+i;
        if (bi<0)
            bi=celle_nx-1;
        else if(bi==celle_nx)
            bi=0;

        for (j=-1;j<2;j++)
        {
            int bj=blockIdx.y+j;
            if (bj<0)
                bj=celle_ny-1;
            else if(bj==celle_ny)
                bj=0;

            for (k=-1;k<2;k++)
            {
                int bk=blockIdx.z+k;
                if (bk<0)
                    bk=celle_nz-1;
                else if(bk==celle_nz)
                    bk=0;



                int neighbour_cell=bi+bj*celle_nx+bk*celle_nx*celle_ny;;


                
                if (threadIdx.x<cells_howmany[neighbour_cell])
                {
                    int whoishe=cells[neighbour_cell*MAX_PARTICLE_CELLS+threadIdx.x];
                    pos_ncella[threadIdx.x]=coor[whoishe];
                    i_ncella[threadIdx.x]=whoishe;
                }


                __syncthreads();


		
		
                if (threadIdx.x<cells_howmany[central_cell])
                {
                    int n;
                    for (n=0;n<cells_howmany[neighbour_cell];n++)
                    {
                        real3 olddist,dist;

                        olddist.x=pos_ncella[n].x-p_i.x;
                        olddist.y=pos_ncella[n].y-p_i.y;
                        olddist.z=pos_ncella[n].z-p_i.z;
                        
                        olddist.x-=rint(olddist.x);
                        olddist.y-=rint(olddist.y);
                        olddist.z-=rint(olddist.z);
                        
                        dist.x=box[0]*olddist.x+box[1]*olddist.y+box[2]*olddist.z;
                        dist.y=box[3]*olddist.y+box[4]*olddist.z;
                        dist.z=box[5]*olddist.z;
                        
                        real dist2=SQR(dist.x)+SQR(dist.y)+SQR(dist.z);
                        
                        if (dist2<cutoff*cutoff)
                        {
			    if (whoami!=i_ncella[n])
			    {
                            	with[whoami*maxneigh+howmany[whoami]]=i_ncella[n];
                            	with_dist2[whoami*maxneigh+howmany[whoami]]=dist2;
                            	howmany[whoami]++;
			    }
                        }
                    }
                }

                __syncthreads();
		

		

            }

        }
        
    }

}
__global__ void sort_neighbors(int *with, real *with_r2, int *howmany, int N, int Radial_Buffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int num_neighbors = howmany[i];  // Numero reale di vicini per la particella i
    if (num_neighbors > 1) {  // Sort solo se ci sono almeno 2 vicini

        // Puntatori alla riga della particella i
        int *row_with = with + i * Radial_Buffer;
        real *row_r2 = with_r2 + i * Radial_Buffer;
         
        // Ordina solo i primi num_neighbors elementi
       thrust::sort_by_key(thrust::device, row_r2, row_r2 + num_neighbors, row_with);
    }
}

void sort_in_gpu(int *d_with, real *d_with_r2, int *d_howmany, int N, int Radial_Buffer) {
    int threads_per_block = 128;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    sort_neighbors<<<blocks_per_grid, threads_per_block>>>(d_with, d_with_r2, d_howmany, N, Radial_Buffer);
    cudaDeviceSynchronize();
}

void imeCompute(int N,real *box_d,real *position_d,real cutoff,int *cells,int *cells_howmany,int celle_nx,int celle_ny,int celle_nz,int *with,int *howmany,real *with_dist2,int MAX_PARTICLE_CELLS,int Radial_Buffer)
{

    dim3 dimGrid(celle_nx,celle_ny,celle_nz);
    dim3 dimBlock(BLOCK_DIM,1,1);

    //cudaMemcpyToSymbol(box,box_h,6*sizeof(real));

    cudaMemset(howmany,0,N*sizeof(int));

    imeBuild<<<dimGrid,dimBlock,sizeof(real3)*MAX_PARTICLE_CELLS+sizeof(int)*MAX_PARTICLE_CELLS,nullptr>>>(N,box_d,position_d,cells,cells_howmany,celle_nx,celle_ny,celle_nz,cutoff,with,howmany,with_dist2,MAX_PARTICLE_CELLS,Radial_Buffer);

    cudaDeviceSynchronize();
    sort_in_gpu(with,with_dist2,howmany,N,Radial_Buffer);
}

__global__ void celleBuild(int N,real *inopos,int *cells,int *cells_howmany,real celle_xsize,real celle_ysize,real celle_zsize,int celle_nx,int celle_ny,int celle_nz,int MAX_PARTICLE_CELLS)
{
    real3 *coor=(real3*)inopos;



    int t=blockIdx.x*blockDim.x+threadIdx.x;

    if (t<N)
    {
        
        int posx=(int)floor(coor[t].x/celle_xsize);
        posx=posx%celle_nx;
        while (posx<0)
        {
            posx+=celle_nx;
        }
        
        int posy=(int)floor(coor[t].y/celle_ysize);
        posy=posy%celle_ny;
        while (posy<0)
        {
            posy+=celle_ny;
        }

        int posz=(int)floor(coor[t].z/celle_zsize);
        posz=posz%celle_nz;
        while (posz<0)
        {
            posz+=celle_nz;
        }
        
        int c=posx+posy*celle_nx+posz*(celle_nx*celle_ny);
        int n=atomicAdd(cells_howmany+c,1);
         
        cells[c*MAX_PARTICLE_CELLS+n]=t;
    }
}

void celleCompute(int N,real *box,real *inopos_d,real cutoff,int **cells_address,int **cells_howmany_address,int *c_nx,int *c_ny,int *c_nz,int MAX_PARTICLE_CELLS)
{

    
    real volume=box[0]*box[3]*box[5];
    //volume/Areayz
    int celle_nx=(int)(volume/(staf_sqrt(box[3]*box[3]*box[5]*box[5]+box[5]*box[5]*box[1]*box[1]+box[3]*box[3]*box[2]*box[2]+box[4]*box[4]*box[1]*box[1])*cutoff));
    int celle_ny=(int)(volume/(box[0]*staf_sqrt(box[5]*box[5]+box[4]*box[4])*cutoff));
    int celle_nz=(int)(volume/(box[0]*box[3]*cutoff));

    int *maxncells;
	while ((celle_nx*celle_ny*celle_nz>27) && (celle_nx*celle_ny*celle_nz>N))
	{
		maxncells=(celle_nx>celle_ny ? &(celle_nx) : &(celle_ny) );
		maxncells=( *maxncells > celle_nz ? maxncells : &(celle_nz));
		
		(*maxncells)--;
	}
	

    if (celle_nx<3)
        celle_nx=3;
    if (celle_ny<3)
        celle_ny=3;
    if (celle_nz<3)
        celle_nz=3;


    *c_nx=celle_nx;
    *c_ny=celle_ny;
    *c_nz=celle_nz;

    real celle_xsize=real(1.)/(real)celle_nx;
    real celle_ysize=real(1.)/(real)celle_ny;
    real celle_zsize=real(1.)/(real)celle_nz;

    int num_celle=celle_nx*celle_ny*celle_nz;

    int *cells=(int*) *cells_address;
    int *cells_howmany=(int*) *cells_howmany_address;

    int *new_ptr;
    cudaMalloc((void**)&new_ptr, num_celle*sizeof(int)); 
    if (cells_howmany) { 
        cudaFree(cells_howmany); 
    }
    *cells_howmany_address = new_ptr;
    cudaMemset(*cells_howmany_address,0,num_celle*sizeof(int));
    cells_howmany=(int*) *cells_howmany_address;
    
    int *new_ptr2;
    cudaMalloc((void**)&new_ptr2, num_celle*MAX_PARTICLE_CELLS*sizeof(int)); 

    if (cells) {
        cudaFree(cells);
    }
    *cells_address = new_ptr2;
    cells=(int*) *cells_address;

    // kernel launch
    dim3 dimGrid(ceil(real(N)/real(BLOCK_DIM)),1,1);
    dim3 dimBlock(BLOCK_DIM,1,1);
    celleBuild<<<dimGrid,dimBlock>>>(N,inopos_d,cells,cells_howmany,celle_xsize,celle_ysize,celle_zsize,celle_nx,celle_ny,celle_nz,MAX_PARTICLE_CELLS);

    cudaDeviceSynchronize();
}


