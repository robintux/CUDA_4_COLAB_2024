/**
   Vector addition:
   takes vectors a and b as input, computes vector sum 
   and stores output in vector c

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 05/2019

 */

#include <stdio.h>
#include <iostream>
#include <chrono>

#include "helpers.h"

using namespace std;

__constant__ int vec_size_d;

__global__ void vector_addition_kernel( int *a, int *b, int *c) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if ( index < vec_size_d ) {
    c[ index ] = a[ index ] + b[ index ];
  }
  
}

int main(int argc, char *argv[] ) {

  if ( argc != 4 ) {
    cout << "Need two arguments: size of vector, number of threads / block and device to use" << endl;
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
  CUDA_ASSERT( cudaMalloc( (void**)&a_d, vec_size_h * sizeof(int) ) );
  CUDA_ASSERT( cudaMalloc( (void**)&b_d, vec_size_h * sizeof(int) ) );
  CUDA_ASSERT( cudaMalloc( (void**)&c_d, vec_size_h * sizeof(int) ) );

  /* Copy vectors to device */
  CUDA_ASSERT( cudaMemcpy( a_d, a_h, vec_size_h * sizeof(int), cudaMemcpyHostToDevice ) );
  CUDA_ASSERT( cudaMemcpy( b_d, b_h, vec_size_h * sizeof(int), cudaMemcpyHostToDevice ) );
  
  CUDA_ASSERT( cudaMemcpyToSymbol( vec_size_d, &vec_size_h, sizeof(int) ) ); // copy to constant memory on GPU

  /* Define grid dimensions */
  int n_blocks  = vec_size_h / n_threads + (vec_size_h % n_threads != 0);
  dim3 blocks( n_blocks );
  dim3 threads(n_threads);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  /* Call kernel */
  vector_addition_kernel<<<blocks,threads>>>( a_d, b_d, c_d);

  CUDA_ASSERT( cudaMemcpy( c_h, c_d, vec_size_h * sizeof(int), cudaMemcpyDeviceToHost ) );

  /* Make sure GPU work is done */
  cudaDeviceSynchronize();

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  
  for ( int i = 0; i < vec_size_h; i++ ) {
    cout << a_h[i] << " + " << b_h[i] << " = " << c_h[i] << endl;
  }

  cout << "Kernel duration: " << elapsed_seconds.count() << " s " << endl;
  cout << "Time per kenel: " << elapsed_seconds.count() / vec_size_h << endl;
  
  CUDA_ASSERT( cudaFree( a_d ) );
  CUDA_ASSERT( cudaFree( b_d ) );
  CUDA_ASSERT( cudaFree( c_d ) );

  /* free host memory */
  delete [] a_h;
  delete [] b_h;
  delete [] c_h;

  
  return 0;
}
