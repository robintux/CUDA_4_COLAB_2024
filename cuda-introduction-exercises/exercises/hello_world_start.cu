#include <stdio.h>
#include <iostream>

#include "helpers.h"

using namespace std;

/* Write your own kernel here */


int main( int argc, char *argv[] ) {

  if ( argc != 4 ) {
    cout << "Need three arguments: number of blocks, number of threads and device to use" << endl;
    return -1;
  }

  const int n_blocks  = atoi(argv[argc-3]);
  const int n_threads = atoi(argv[argc-2]);
  const int device_id = atoi(argv[argc-1]);

  /* Chose device to use */
  CUDA_ASSERT( cudaSetDevice(device_id) );

  /* Fill in your own code here for defining the grid 
     and block dimensions and calling the kernel */
  
  

  
  /* cudaDeviceSynchronize() blocks the host thread until all requested 
     tasks on the device were completed;
     needed for printf in kernel to work
  */
  cudaDeviceSynchronize();

  return 0;
}
