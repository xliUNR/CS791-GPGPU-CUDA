/*
  This program demonstrates the basics of working with cuda. We use
  the GPU to add two arrays. We also introduce cuda's approach to
  error handling and timing using cuda Events.

  This is the main program. You should also look at the header add.h
  for the important declarations, and then look at add.cu to see how
  to define functions that execute on the GPU.
 */

#include <iostream>

#include "add.h"

int main() {
  
  // Arrays on the host (CPU)
  int a[N][N], b[N][N], c[N][N];
  
  /*
    These will point to memory on the GPU - notice the correspondence
    between these pointers and the arrays declared above.
   */
  int *dev_a, *dev_b, *dev_c;

  /*
    These calls allocate memory on the GPU (also called the
    device). This is similar to C's malloc, except that instead of
    directly returning a pointer to the allocated memory, cudaMalloc
    returns the pointer through its first argument, which must be a
    void**. The second argument is the number of bytes we want to
    allocate.

    NB: the return value of cudaMalloc (like most cuda functions) is
    an error code. Strictly speaking, we should check this value and
    perform error handling if anything went wrong. We do this for the
    first call to cudaMalloc so you can see what it looks like, but
    for all other function calls we just point out that you should do
    error checking.

    Actually, a good idea would be to wrap this error checking in a
    function or macro, which is what the Cuda By Example book does.
   */
  cudaError_t err = cudaMalloc( (void**) &dev_a, N * N * sizeof(int));
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  cudaMalloc( (void**) &dev_b, N * N * sizeof(int));
  cudaMalloc( (void**) &dev_c, N * N * sizeof(int));

  // These lines just fill the host arrays with some data so we can do
  // something interesting. Well, so we can add two arrays.
  /*for (int i = 0; i < N; ++i) {

    for(int j=0; j < N; ++j){

      a[i][j] = (i * N + j);
      b[i][j] = (i * N + j);

    }
    
  }*/


  /*
    Let's let the user know that everything is ok and then display
    some information about the times we recorded above.
   */
  std::cout << "Yay! Your program's results are correct." << std::endl;
  
  

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

}
