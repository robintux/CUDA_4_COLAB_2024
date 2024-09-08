/**
   Script to check device properties of a CUDA capable device

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 05/2019

 */

#include "helpers.h"

using namespace std;

int main() {
  
  cudaDeviceProp prop;

  int deviceCount = 0;
  CUDA_ASSERT( cudaGetDeviceCount( &deviceCount ) );
  
  if ( deviceCount == 0 ) {
    cout << "No CUDA capable device found" << endl;
    return -1;
  }

  for ( int dev = 0; dev < deviceCount; dev++ ) {
    
    CUDA_ASSERT( cudaGetDeviceProperties( &prop, dev ) );
    cout << "------- General information -------" << endl;
    cout << "Name: \t \t \t" << prop.name << endl;
    cout << "Compute capability: \t" << prop.major << "." << prop.minor << endl;
    cout << "Max clock rate: \t" << prop.clockRate << " kHz" << endl;
    if ( prop.deviceOverlap)
      cout << "Device can concurrently copy memory and execute a kernel" << endl;
    else
      cout << "Device can NOT concurrently copy memory and execute a kernel" << endl;
    if ( prop.concurrentKernels)
      cout << "Device can execute multiple kernels concurrently" << endl;
    else
      cout << "Device can NOT execute multiple kernels concurrently" << endl;

    cout << "Maximum grid dimensions: \t"<< prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << std::endl;
    
    cout << "------- Multiprocessor information -------" << endl;
    cout << "Multiprocessors: \t" << prop.multiProcessorCount << endl;
    cout << "Max size of a block in x: " << prop.maxThreadsDim[0] << ", in y: " << prop.maxThreadsDim[1] << ", in z: " << prop.maxThreadsDim[2] << endl;
    cout << "Max # of threads / block: \t" << prop.maxThreadsPerBlock << endl;
    cout << "Warp size in threads: \t \t" << prop.warpSize << endl;
    cout << "Shared memory per block: \t" << prop.sharedMemPerMultiprocessor / 1000<< " kB" << endl;

    cout << "------- Memory information -------" << endl;
    if ( prop.unifiedAddressing )
      cout << "Device shares a unified address space with host" << endl;
    else
      cout << "Device does NOT share a unified address space with host" << endl;
    cout << "Total global memory: \t" << prop.totalGlobalMem / 1000 << " kB" << endl;
    cout << "Total constant memory: \t" << prop.totalConstMem / 1000 << " kB" << endl;
   
    
  }
  
  return 0;

}
