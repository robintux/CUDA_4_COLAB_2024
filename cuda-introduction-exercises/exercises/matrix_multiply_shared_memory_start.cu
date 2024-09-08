/**
   Matrix multiplication example

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 05/2019

 */

#include <stdio.h>
#include <iostream>
#include <chrono>

#include "helpers.h"
#include "matrix_utils.h"

using namespace std;

#define N_THREADS 8

typedef double my_float_t;

/* Kernel for matrix multiplication using shared memory
   -> every block works on one sub-matrix of the result matrix 
   -> every thread calculates one element of this sub-matrix */
__global__ void matrixMultiply(const int size, const my_float_t *src_matrix_1, const my_float_t *src_matrix_2, my_float_t *dst_matrix ) {

  /* sub-matrix this block works on */
  int blockI = blockIdx.x;
  int blockJ = blockIdx.y;

  /* element within sub-matrix this thread works on */
  int i = threadIdx.x;
  int j = threadIdx.y;
  
  /* Cache in shared memory source sub-matrices needed to compute
     elements of destination sub-matrix */
  __shared__ double cache_1[N_THREADS][N_THREADS];
  __shared__ double cache_2[N_THREADS][N_THREADS];
  
  /* Every thread calculates one element of destination sub-matrix */
  double sum = 0;
  
  if ( j >= size || i >= size ) return;

  /* Loop over sub-sub matrices of size (N_THREADS x N_THREADS) of the input sub-matrices */
  for ( int k = 0; k < (size / N_THREADS); ++k ) {

    /* Pointer to sub-matrix start 
     to do: understand the two lines below!*/
    double *sub_matrix_1 = src_matrix_1 + size * N_THREADS * blockI + N_THREADS * k;
    double *sub_matrix_2 = src_matrix_2 + size * N_THREADS * k + N_THREADS * blockJ;

    /* to do: read sub-matrix entries from global to shared memory */

    /* to do: synchronize threads to make sure they all finished reading from global memory */

    /* to do: multiply the two submatrices and store the result in sum */

    /* to do: synchronize again to make sure all threads have computed the sum
       before loading the next sub-matrices*/
    
  }
  
  /* Pointer to result sub-matrix */
  double *sub_dst_matrix = dst_matrix + size * N_THREADS * blockI + N_THREADS * blockJ;

  /* Write result to global memory */
  sub_dst_matrix[i * size + j] = sum;
  
}


int main(int argc, char * argv[])
{
  if ( argc != 3 ) {
    cout << "Need three arguments: number of columns (= number of rows) of matrix and device to use" << endl;
    return -1;
  }

  const int matrix_size = atoi(argv[argc-2]);
  const int device_id = atoi(argv[argc-1]);
    
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
