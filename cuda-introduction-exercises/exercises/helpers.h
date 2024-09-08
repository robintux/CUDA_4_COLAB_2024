#pragma once

#include <stdio.h>
#include <iostream>
#include <assert.h>

#include <cuda.h>


/**
 * Helper assert to check for CUDA errors.
 */
#define CUDA_ASSERT(val) { gpuAssert((val), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
   if(code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if(abort) exit(code);
   }
}
