#include <stdio.h>

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n Abraham Zamudio");
}

int main()
{
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}