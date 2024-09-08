/**
   Matrix multiplication example using threads and blocks for parallelization

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 05/2019

 */

#include <stdio.h>
#include <iostream>
#include <chrono>

#include "helpers.h"
#include "matrix_utils.h"

using namespace std;

typedef double my_float_t;

__global__ void matrixMultiply(const int size, const my_float_t *src_matrix_1, const my_float_t *src_matrix_2, my_float_t *dst_matrix ) {
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if ( i < size && j < size ) {
    
    my_float_t sum = 0;
    for ( int k = 0; k < size; k++ ) {
      sum += src_matrix_1[i * size + k] * src_matrix_2[k * size + j];
    }
    dst_matrix[i * size + j] = sum;
  }
  
}


int main(int argc, char * argv[])
{

  if ( argc != 4 ) {
    cout << "Need three arguments: number of columns (= number of rows) of matrix, number of threads / block and device to use" << endl;
    return -1;
  }

  const int matrix_size = atoi(argv[argc-3]);
  const int n_threads = atoi(argv[argc-2]);
  const int device_id = atoi(argv[argc-1]);
  
  /* Chose device to use */
  CUDA_ASSERT( cudaSetDevice(device_id) );
    
  //Allocate host and device memory for three matrices
  my_float_t *host_matrix[3];    //matrix[0] and matrix[1] are the source for the multiplication, result stored in matrix[2]
  my_float_t *device_matrix[3];
 
  for (int i = 0;i < 3;i++)
    {
      if ((host_matrix[i] = new my_float_t[matrix_size * matrix_size]) == NULL) {
  	printf("Memory allocation error");
  	return -1;
      }
      CUDA_ASSERT( cudaMalloc( (void**)&device_matrix[i], matrix_size * matrix_size * sizeof(my_float_t) ) );
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
      CUDA_ASSERT( cudaMemcpy( device_matrix[i], host_matrix[i], matrix_size * matrix_size * sizeof(my_float_t), cudaMemcpyHostToDevice )  );
    }

  // Launch kernel
  int size     = matrix_size;
  int n_blocks = size / (n_threads) + (size % (n_threads) != 0);
  dim3 grid(n_blocks, n_blocks);
  dim3 block(n_threads, n_threads);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  
  matrixMultiply<<<grid, block>>>(size, device_matrix[0], device_matrix[1], device_matrix[2] );  

  cudaDeviceSynchronize();
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  
  // Copy back result
  CUDA_ASSERT( cudaMemcpy( host_matrix[2], device_matrix[2], matrix_size * matrix_size * sizeof(my_float_t), cudaMemcpyDeviceToHost )  );
  
  //Check and print result
  check_result<my_float_t>(matrix_size, host_matrix[2]);

  cout << "Kernel duration: " << elapsed_seconds.count() << " s " << endl;
  
  // Free memory
  for (int i = 0;i < 3;i++)
    {
      delete[] host_matrix[i];
      CUDA_ASSERT( cudaFree( device_matrix[i] ) );
    }
  
}
