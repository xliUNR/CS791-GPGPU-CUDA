
#include "add.h"

/*
  This is the function that each thread will execute on the GPU. The
  fact that it executes on the device is indicated by the __global__
  modifier in front of the return type of the function. After that,
  the signature of the function isn't special - in particular, the
  pointers we pass in should point to memory on the device, but this
  is not indicated by the function's signature.
 */
__global__ void add(int *a, int *b, int *c) {

  /*
    Each thread knows its identity in the system. This identity is
    made available in code via indices blockIdx and threadIdx. This 
    equation calculates the unique ID for each element in the matrix
    since the memory is stored as a 1D list.
   */
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  /*
    We make sure that the thread_id isn't too large, and then we
    assign c = a + b using the index we calculated above.

    The big picture is that each thread is responsible for adding one
    element from a and one element from b. Each thread is able to run
    in parallel, so we get speedup.
   */
  if (thread_id < N * N ) {

    //c[thread_id] = a[thread_id] + b[thread_id];
    *((*c) + thread_id) = *((*a) + thread_id ) + *((*b) + thread_id);
  }
}

/*//function to populate a matrix
__global__ void mat_init( int N, int *emptyMatrix ) {

  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  //check for valid memory location, then initialize element to 0
  if( thread_id < N * N )
  {
    //commented out one is for array of pointers
    //*((*(emptyMatrix)) + (blockId.x * blockDim.x + threadIdx.x)) = 0;
    *(emptyMatrix + (blockIdx.x * blockDim.x + threadIdx.x)) = 0;
  }
}*/


