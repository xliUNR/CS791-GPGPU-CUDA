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
  

  /*
    These will point to memory on the GPU - notice the correspondence
    between these pointers and the arrays declared above.
   */
  int *a, *b, *c;

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
  
  cudaMallocManaged( &a, N*sizeof(int));
  cudaMallocManaged( &b, N*sizeof(int));
  cudaMallocManaged( &c, N*sizeof(int));

  // These lines just fill the host arrays with some data so we can do
  // something interesting. Well, so we can add two arrays.
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = i;
  }

  /*
    The following code is responsible for handling timing for code
    that executes on the GPU. The cuda approach to this problem uses
    events. For timing purposes, an event is essentially a point in
    time. We create events for the beginning and end points of the
    process we want to time. When we want to start timing, we call
    cudaEventRecord.

    In this case, we want to record the time it takes to transfer data
    to the GPU, perform some computations, and transfer data back.
  */
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord( start, 0 );


  add<<<N, 1>>>(a,b,c);

  //Need to sync b4 accessing memory locations
  cudaDeviceSynchronize();
  /*
    This is the other end of the timing process. We record an event,
    synchronize on it, and then figure out the difference in time
    between the start and the stop.

    We have to call cudaEventSynchronize before we can safely _read_
    the value of the stop event. This is because the GPU may not have
    actually written to the event until all other work has finished.
   */
  cudaEventRecord( end, 0 );
  cudaEventSynchronize( end );

  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, end );

  /*
    Let's check that the results are what we expect.
   */
  for (int i = 0; i < N; ++i) {
    if (c[i] != a[i] + b[i]) {
      std::cerr << "Oh no! Something went wrong. You should check your cuda install and your GPU. :(" << std::endl;

      // clean up events - we should check for error codes here.
      cudaEventDestroy( start );
      cudaEventDestroy( end );

      // clean up device pointers - just like free in C. We don't have
      // to check error codes for this one.
      cudaFree(a);
      cudaFree(b);
      cudaFree(c);
      exit(1);
    }
  }

  /*
    Let's let the user know that everything is ok and then display
    some information about the times we recorded above.
   */
  std::cout << "Yay! Your program's results are correct." << std::endl;
  std::cout << "Your program took: " << elapsedTime << " ms." << std::endl;
  
  // Cleanup in the event of success.
  cudaEventDestroy( start );
  cudaEventDestroy( end );
  
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

}
