#pragma once
/*
 * STAF precision abstraction for CUDA/C++ ops.
 *
 * Compile with -DSTAF_REAL_DOUBLE for float64 ops, otherwise float32.
 * Use `real` / `real3` / `real4` and staf_exp/cos/sin/... so math and
 * allocations (sizeof(real)) stay correct for both trees.
 */
#include <math.h>
#include <cuda_runtime.h>

#if !defined(__CUDACC__)
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

#if defined(STAF_REAL_DOUBLE)
typedef double real;
#if defined(__CUDACC__)
typedef double2 real2;
typedef double3 real3;
typedef double4 real4;
#endif
#define STAF_TF_DTYPE "double"
#else
typedef float real;
#if defined(__CUDACC__)
typedef float2 real2;
typedef float3 real3;
typedef float4 real4;
#endif
#define STAF_TF_DTYPE "float"
#endif

/* Host + device overloads: pick expf vs exp from argument type. */
inline __host__ __device__ float staf_exp(float x) {
#if defined(__CUDA_ARCH__)
  return expf(x);
#else
  return expf(x);
#endif
}
inline __host__ __device__ double staf_exp(double x) {
#if defined(__CUDA_ARCH__)
  return exp(x);
#else
  return exp(x);
#endif
}

inline __host__ __device__ float staf_cos(float x) {
  return cosf(x);
}
inline __host__ __device__ double staf_cos(double x) {
  return cos(x);
}

inline __host__ __device__ float staf_sin(float x) {
  return sinf(x);
}
inline __host__ __device__ double staf_sin(double x) {
  return sin(x);
}

inline __host__ __device__ float staf_sqrt(float x) {
  return sqrtf(x);
}
inline __host__ __device__ double staf_sqrt(double x) {
  return sqrt(x);
}

inline __host__ __device__ float staf_pow(float x, float n) {
  return powf(x, n);
}
inline __host__ __device__ double staf_pow(double x, double n) {
  return pow(x, n);
}

/* Prefer real(0) / real(1) in new code; these help migrate 0.f literals. */
#define STAF_ZERO real(0)
#define STAF_ONE real(1)
