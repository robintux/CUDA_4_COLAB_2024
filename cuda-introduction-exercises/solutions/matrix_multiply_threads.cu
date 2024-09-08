/**
   Matrix multiplication example using threads for parallelization

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 05/2019

 */

#include <stdio.h>
#include <iostream>
#include <chrono>

#include "helpers.h"
#include "matrix_utils.h"

using namespace std;


__global__ void matrixMultiply(const int size, const double *src_matrix_1, const double *src_matrix_2, double *dst_matrix ) {
  
  for ( int i = threadIdx.x; i < size; i += blockDim.x ) {
    for ( int j = threadIdx.y; j < size; j += blockDim.y ) {
      
      double sum = 0;
      for ( int k = 0; k < size; k++ ) {
        sum += src_matrix_1[i * size + k] * src_matrix_2[k * size + j];
      }
      dst_matrix[i * size + j] = sum;
    }
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
  dim3 grid(1);
  dim3 block(n_threads, n_threads);

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
