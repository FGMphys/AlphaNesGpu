#ifndef STAF_FORCE_PBC_CUH
#define STAF_FORCE_PBC_CUH

#ifdef STAF_FORCE_PBC_DEFINE
__device__ __constant__ int staf_force_skip_pbc = 0;
#else
extern __device__ __constant__ int staf_force_skip_pbc;
#endif

#endif /* STAF_FORCE_PBC_CUH */
