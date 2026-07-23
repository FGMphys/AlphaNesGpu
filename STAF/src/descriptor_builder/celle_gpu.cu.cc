#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "staf_real.h"
#include "celle_gpu.h"

#define BLOCK_DIM 256
#define SQR(x) ((x) * (x))

__global__ void convert_carte_to_int_kernel(real* nowinobox_d,
                                           const real* nowpos_d,
                                           real* nowinopos_d, int N, int nf) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nf * N) return;

  int b = t / N;
  int par = t % N;
  real3* nowinopos3_d = (real3*)nowinopos_d;
  real* Inobox = nowinobox_d + b * 6;
  real px = nowpos_d[b * N * 3 + par * 3 + 0];
  real py = nowpos_d[b * N * 3 + par * 3 + 1];
  real pz = nowpos_d[b * N * 3 + par * 3 + 2];
  nowinopos3_d[b * N + par].x =
      (Inobox[0] * px + Inobox[1] * py + Inobox[2] * pz);
  nowinopos3_d[b * N + par].y = (Inobox[3] * py + Inobox[4] * pz);
  nowinopos3_d[b * N + par].z = (Inobox[5] * pz);
}

void convert_carte_to_int_launcher(real* nowinobox_d, const real* nowpos_d,
                                   real* nowinopos_d, int N, int nf,
                                   cudaStream_t stream) {
  int dimgrid = (N * nf + BLOCK_DIM - 1) / BLOCK_DIM;
  dim3 dimGrid(dimgrid, 1, 1);
  dim3 dimBlock(BLOCK_DIM, 1, 1);
  convert_carte_to_int_kernel<<<dimGrid, dimBlock, 0, stream>>>(
      nowinobox_d, nowpos_d, nowinopos_d, N, nf);
}

__global__ void imeBuild(int N, const real* box, real* position, int* cells,
                         int* cells_howmany, int celle_nx, int celle_ny,
                         int celle_nz, real cutoff, int* with, int* howmany,
                         real* with_dist2, int MAX_PARTICLE_CELLS,
                         int maxneigh) {
  real3* coor = (real3*)position;
  extern __shared__ unsigned char sharedMemory[];
  real3* pos_ncella = (real3*)sharedMemory;
  int* i_ncella =
      (int*)(sharedMemory + sizeof(real3) * MAX_PARTICLE_CELLS);

  int central_cell =
      blockIdx.x + blockIdx.y * celle_nx + blockIdx.z * celle_nx * celle_ny;

  real3 p_i;
  int whoami = -1;
  if (threadIdx.x < cells_howmany[central_cell]) {
    whoami = cells[central_cell * MAX_PARTICLE_CELLS + threadIdx.x];
    p_i = coor[whoami];
  }

  for (int i = -1; i < 2; i++) {
    int bi = blockIdx.x + i;
    if (bi < 0)
      bi = celle_nx - 1;
    else if (bi == celle_nx)
      bi = 0;

    for (int j = -1; j < 2; j++) {
      int bj = blockIdx.y + j;
      if (bj < 0)
        bj = celle_ny - 1;
      else if (bj == celle_ny)
        bj = 0;

      for (int k = -1; k < 2; k++) {
        int bk = blockIdx.z + k;
        if (bk < 0)
          bk = celle_nz - 1;
        else if (bk == celle_nz)
          bk = 0;

        int neighbour_cell = bi + bj * celle_nx + bk * celle_nx * celle_ny;

        if (threadIdx.x < cells_howmany[neighbour_cell]) {
          int whoishe =
              cells[neighbour_cell * MAX_PARTICLE_CELLS + threadIdx.x];
          pos_ncella[threadIdx.x] = coor[whoishe];
          i_ncella[threadIdx.x] = whoishe;
        }

        __syncthreads();

        if (threadIdx.x < cells_howmany[central_cell]) {
          for (int n = 0; n < cells_howmany[neighbour_cell]; n++) {
            real3 olddist, dist;
            olddist.x = pos_ncella[n].x - p_i.x;
            olddist.y = pos_ncella[n].y - p_i.y;
            olddist.z = pos_ncella[n].z - p_i.z;

            olddist.x -= rint(olddist.x);
            olddist.y -= rint(olddist.y);
            olddist.z -= rint(olddist.z);

            dist.x = box[0] * olddist.x + box[1] * olddist.y +
                     box[2] * olddist.z;
            dist.y = box[3] * olddist.y + box[4] * olddist.z;
            dist.z = box[5] * olddist.z;

            real dist2 = SQR(dist.x) + SQR(dist.y) + SQR(dist.z);

            if (dist2 < cutoff * cutoff && whoami != i_ncella[n]) {
              int slot = howmany[whoami]++;
              if (slot < maxneigh) {
                with[whoami * maxneigh + slot] = i_ncella[n];
                with_dist2[whoami * maxneigh + slot] = dist2;
              }
            }
          }
        }

        __syncthreads();
      }
    }
  }
}

/* Per-particle insertion sort by distance — avoids thrust-in-kernel. */
__global__ void sort_neighbors(int* with, real* with_r2, int* howmany, int N,
                               int Radial_Buffer) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  int num_neighbors = howmany[i];
  if (num_neighbors > Radial_Buffer) num_neighbors = Radial_Buffer;
  if (num_neighbors <= 1) return;

  int* row_with = with + i * Radial_Buffer;
  real* row_r2 = with_r2 + i * Radial_Buffer;

  for (int a = 1; a < num_neighbors; ++a) {
    real key_r = row_r2[a];
    int key_w = row_with[a];
    int b = a - 1;
    while (b >= 0 && row_r2[b] > key_r) {
      row_r2[b + 1] = row_r2[b];
      row_with[b + 1] = row_with[b];
      --b;
    }
    row_r2[b + 1] = key_r;
    row_with[b + 1] = key_w;
  }
}

void imeCompute(int N, const real* box_d, real* position_d, real cutoff,
                int* cells, int* cells_howmany, int celle_nx, int celle_ny,
                int celle_nz, int* with, int* howmany, real* with_dist2,
                int MAX_PARTICLE_CELLS, int Radial_Buffer,
                cudaStream_t stream) {
  dim3 dimGrid(celle_nx, celle_ny, celle_nz);
  dim3 dimBlock(BLOCK_DIM, 1, 1);

  cudaMemsetAsync(howmany, 0, N * sizeof(int), stream);

  size_t shmem =
      sizeof(real3) * MAX_PARTICLE_CELLS + sizeof(int) * MAX_PARTICLE_CELLS;
  imeBuild<<<dimGrid, dimBlock, shmem, stream>>>(
      N, box_d, position_d, cells, cells_howmany, celle_nx, celle_ny, celle_nz,
      cutoff, with, howmany, with_dist2, MAX_PARTICLE_CELLS, Radial_Buffer);

  int threads_per_block = 128;
  int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
  sort_neighbors<<<blocks_per_grid, threads_per_block, 0, stream>>>(
      with, with_dist2, howmany, N, Radial_Buffer);
}

__global__ void celleBuild(int N, real* inopos, int* cells, int* cells_howmany,
                           real celle_xsize, real celle_ysize, real celle_zsize,
                           int celle_nx, int celle_ny, int celle_nz,
                           int MAX_PARTICLE_CELLS) {
  real3* coor = (real3*)inopos;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;

  int posx = (int)floor(coor[t].x / celle_xsize);
  posx = posx % celle_nx;
  while (posx < 0) posx += celle_nx;

  int posy = (int)floor(coor[t].y / celle_ysize);
  posy = posy % celle_ny;
  while (posy < 0) posy += celle_ny;

  int posz = (int)floor(coor[t].z / celle_zsize);
  posz = posz % celle_nz;
  while (posz < 0) posz += celle_nz;

  int c = posx + posy * celle_nx + posz * (celle_nx * celle_ny);
  int n = atomicAdd(cells_howmany + c, 1);
  if (n < MAX_PARTICLE_CELLS) {
    cells[c * MAX_PARTICLE_CELLS + n] = t;
  }
}

void celleCompute(int N, const real* box, real* inopos_d, real cutoff,
                  int** cells_address, int** cells_howmany_address, int* c_nx,
                  int* c_ny, int* c_nz, int MAX_PARTICLE_CELLS,
                  int* cells_capacity_num, int* cells_capacity_mpc,
                  cudaStream_t stream) {
  real volume = box[0] * box[3] * box[5];
  int celle_nx = (int)(volume /
                       (sqrt(box[3] * box[3] * box[5] * box[5] +
                             box[5] * box[5] * box[1] * box[1] +
                             box[3] * box[3] * box[2] * box[2] +
                             box[4] * box[4] * box[1] * box[1]) *
                        cutoff));
  int celle_ny =
      (int)(volume / (box[0] * sqrt(box[5] * box[5] + box[4] * box[4]) * cutoff));
  int celle_nz = (int)(volume / (box[0] * box[3] * cutoff));

  while ((celle_nx * celle_ny * celle_nz > 27) &&
         (celle_nx * celle_ny * celle_nz > N)) {
    int* maxncells = (celle_nx > celle_ny ? &(celle_nx) : &(celle_ny));
    maxncells = (*maxncells > celle_nz ? maxncells : &(celle_nz));
    (*maxncells)--;
  }

  if (celle_nx < 3) celle_nx = 3;
  if (celle_ny < 3) celle_ny = 3;
  if (celle_nz < 3) celle_nz = 3;

  *c_nx = celle_nx;
  *c_ny = celle_ny;
  *c_nz = celle_nz;

  real celle_xsize = real(1.) / (real)celle_nx;
  real celle_ysize = real(1.) / (real)celle_ny;
  real celle_zsize = real(1.) / (real)celle_nz;

  int num_celle = celle_nx * celle_ny * celle_nz;

  if (num_celle > *cells_capacity_num ||
      MAX_PARTICLE_CELLS > *cells_capacity_mpc || !*cells_howmany_address ||
      !*cells_address) {
    if (*cells_howmany_address) cudaFree(*cells_howmany_address);
    if (*cells_address) cudaFree(*cells_address);
    cudaMalloc((void**)cells_howmany_address, num_celle * sizeof(int));
    cudaMalloc((void**)cells_address,
               num_celle * MAX_PARTICLE_CELLS * sizeof(int));
    *cells_capacity_num = num_celle;
    *cells_capacity_mpc = MAX_PARTICLE_CELLS;
  }

  cudaMemsetAsync(*cells_howmany_address, 0, num_celle * sizeof(int), stream);

  dim3 dimGrid((N + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
  dim3 dimBlock(BLOCK_DIM, 1, 1);
  celleBuild<<<dimGrid, dimBlock, 0, stream>>>(
      N, inopos_d, *cells_address, *cells_howmany_address, celle_xsize,
      celle_ysize, celle_zsize, celle_nx, celle_ny, celle_nz,
      MAX_PARTICLE_CELLS);
}
