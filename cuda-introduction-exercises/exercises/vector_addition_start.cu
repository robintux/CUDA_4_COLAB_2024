#include <stdio.h>
#include <iostream>
#include <chrono>

#include "helpers.h"

using namespace std;

__constant__ int vec_size_d;

/* Kernel doing vector addition with one thread per vector entry  */
__global__ void vector_addition_kernel( int *a, int *b, int *c) {
  /* to do: calculate the index correctly from the blockIdx.x and threadIdx.x 
     const int index = ...
   */
  
  if ( index < vec_size_d ) {
    c[ index ] = a[ index ] + b[ index ];
  }
  
}


int main(int argc, char *argv[] ) {

  if ( argc != 4 ) {
    cout << "Need three arguments: size of vector, number of threads / block and device to use" << endl;
    return -1;
  }
  
  const int vec_size_h  = atoi(argv[argc-3]);
  const int n_threads = atoi(argv[argc-2]);
  const int device_id = atoi(argv[argc-1]);
  
  /* Chose device to use */
  CUDA_ASSERT( cudaSetDevice(device_id) );
  
  cout << "Adding vectors of size " <<  vec_size_h << " with " << n_threads << " threads" << endl;
  
  /* Host memory for the two input vectors a and b and the output vector c */
  int *a_h = new int[vec_size_h];
  int *b_h = new int[vec_size_h];
  int *c_h = new int[vec_size_h];

  for ( int i = 0; i < vec_size_h; i++ ) {
    a_h[i] = i;
    b_h[i] = i;
    c_h[i] = 0;
  }
  
  /* Device pointers for the three vectors a, b, c */
  int *a_d, *b_d, *c_d;

  /* to do: Allocate memory on the device for b_d and c_d, following the example for a_d */
  CUDA_ASSERT( cudaMalloc( (void**)&a_d, vec_size_h * sizeof(int) ) );

  
  /* to do: copy the second input vector to the device, following the example of a_d */
  CUDA_ASSERT( cudaMemcpy( a_d, a_h, vec_size_h * sizeof(int), cudaMemcpyHostToDevice ) );

  /* copy vector size to device
     note: vec_size_d is declared as constant memory, 
     therefore the syntax is different than when copying to global memory
  */
  CUDA_ASSERT( cudaMemcpyToSymbol( vec_size_d, &vec_size_h, sizeof(int) ) );
  
  /* Define grid dimensions 
     to do: discuss why the block number is set to this value!
   */
  int n_blocks  = vec_size_h / n_threads + (vec_size_h % n_threads != 0);
  dim3 blocks( n_blocks );
  dim3 threads(n_threads);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  
  /* to do: 
     1) finish kernel definition (see above __global__ void vector_addition_kernel(...) )
     2) call kernel
  */


  
 
    
  /* to do: copy back the result vector */


  cudaDeviceSynchronize();
  
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;

  cout << "Kernel duration: " << elapsed_seconds.count() << " s " << endl;
  cout << "Time per kenel: " << elapsed_seconds.count() / vec_size_h << endl;
  
  for ( int i = 0; i < vec_size_h; i++ ) {
    cout << a_h[i] << " + " << b_h[i] << " = " << c_h[i] << endl;
  }
  
  /* to do: free the remainig device memory */
  CUDA_ASSERT( cudaFree( a_d ) );

  /* free host memory */
  delete [] a_h;
  delete [] b_h;
  delete [] c_h;

  return 0;
}
