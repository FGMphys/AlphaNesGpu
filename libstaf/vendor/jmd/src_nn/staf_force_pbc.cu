#define STAF_FORCE_PBC_DEFINE
#include "staf_force_pbc.cuh"
#include "staf_force_pbc.h"
#include <cuda_runtime.h>

extern "C" void staf_force_set_skip_pbc(int skip)
{
  cudaMemcpyToSymbol(staf_force_skip_pbc, &skip, sizeof(int));
}
