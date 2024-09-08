#include <stdio.h>
#include <iostream>
#include <chrono>

/**
   Matrix multiplication example using shared memory to save sub-matrices 
   a block works on
   
   Inspired by matrix multiplication example in cuda programming guide 

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 05/2019
*/

#include "helpers.h"
#include "matrix_utils.h"

using namespace std;

#define N_THREADS 8

__global__ void matrixMultiply(int size, double *src_matrix_1, double *src_matrix_2, double *dst_matrix ) {

  // sub-matrix does this block works on
  int blockI = blockIdx.x;
  int blockJ = blockIdx.y;

  // element within sub-matrix this thread works on
  int i = threadIdx.x;
  int j = threadIdx.y;
  
  //Cache in shared memory source sub-matrices needed to compute
  // elements of destination sub-matrix
  __shared__ double cache_1[N_THREADS][N_THREADS];
  __shared__ double cache_2[N_THREADS][N_THREADS];
  
  // Every thread calculates one element of destination sub-matrix
  double sum = 0;
  
  if ( j >= size || i >= size ) return;

  // Loop over sub-sub matrices of size (N_THREADS x N_THREADS) of the input sub-matrices
  for ( int k = 0; k < (size / N_THREADS); ++k ) {

    // Pointer to sub-matrix start
    double *sub_matrix_1 = src_matrix_1 + size * N_THREADS * blockI + N_THREADS * k;
    double *sub_matrix_2 = src_matrix_2 + size * N_THREADS * k + N_THREADS * blockJ;
    
    cache_1[i][j] = sub_matrix_1[i * size + j];
    cache_2[i][j] = sub_matrix_2[i * size + j];

    // Synchronize to make sure all threads have been written to shared memory
    __syncthreads();
    
    // Multiply the two sub matrices
    for ( int e = 0; e < N_THREADS; e++ ) {
      sum += cache_1[i][e] * cache_2[e][j];
    }

    // Synchronize to make sure all threads have computed the sum
    // before the next sub-matrices are loaded
    __syncthreads();

  }

  // Pointer to result sub-matrix
  double *sub_dst_matrix = dst_matrix + size * N_THREADS * blockI + N_THREADS * blockJ;

  // Write result to global memory
  sub_dst_matrix[i * size + j] = sum;
    
}


int main(int argc, char * argv[])
{
/* Comentamos esta parte para poder realizar la ejecucion dentro de google colab

  if ( argc != 3 ) {
    cout << "Need three arguments: number of columns (= number of rows) of matrix and device to use" << endl;
    return -1;
  }


  const int matrix_size = atoi(argv[argc-2]);
  const int device_id = atoi(argv[argc-1]);

*/
const int matrix_size = 10;
const int device_id = 0; 
    
  //Allocate host and device memory for three matrices
  double *host_matrix[3];    //matrix[0] and matrix[1] are the source for the multiplication, result stored in matrix[2]
  double *device_matrix[3];
 
  for (int i = 0;i < 3;i++)
    {
      if ((host_matrix[i] = new double[matrix_size * matrix_size]) == NULL) {
  	printf("Memory allocation error");
  	return -1;
      }
      CUDA_ASSERT( cudaMalloc( (void**)&device_matrix[i], matrix_size * matrix_size * sizeof(double) ) );
    }
  
  //Initialize matrices
  for (int i = 0;i < matrix_size;i++)
    {
      for (int j = 0; j < matrix_size;j++)
	{
	  host_matrix[0][i * matrix_size + j] = i * (j + 1);
	  host_matrix[1][i * matrix_size + j] = 2 * i + j;
	  host_matrix[2][i * matrix_size + j] = 0;
	}
    }
  
  // Copy matrices to device
  for (int i = 0;i < 3;i++)
    {
      CUDA_ASSERT( cudaMemcpy( device_matrix[i], host_matrix[i], matrix_size * matrix_size * sizeof(double), cudaMemcpyHostToDevice )  );
    }

  // Launch kernel
  int size     = matrix_size;
  int n_blocks = size / (N_THREADS ) + (size % (N_THREADS) != 0);
  dim3 grid(n_blocks, n_blocks);
  dim3 block(N_THREADS, N_THREADS);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  
  matrixMultiply<<<grid, block>>>(size, device_matrix[0], device_matrix[1], device_matrix[2] );  

  cudaDeviceSynchronize();
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  
  // Copy back result
  CUDA_ASSERT( cudaMemcpy( host_matrix[2], device_matrix[2], matrix_size * matrix_size * sizeof(double), cudaMemcpyDeviceToHost )  );
  
  //Check and print result
  check_result<double>(matrix_size, host_matrix[2]);

  cout << "Kernel duration: " << elapsed_seconds.count() << " s " << endl;
  
  // Free memory
  for (int i = 0;i < 3;i++)
    {
      delete[] host_matrix[i];
      CUDA_ASSERT( cudaFree( device_matrix[i] ) );
    }
  
}
