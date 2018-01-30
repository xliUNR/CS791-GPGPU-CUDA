/*
  This program demonstrates the basics of working with cuda. We use
  the GPU to add two arrays. We also introduce cuda's approach to
  error handling and timing using cuda Events.

  This is the main program. You should also look at the header add.h
  for the important declarations, and then look at add.cu to see how
  to define functions that execute on the GPU.
 */

#include <iostream>
#include <cstdio>
#include <time.h>
#include "add.h"

int main() {
  int N;
  std::cout << "Enter square matrix dimension: " << std::endl;
  std::cin >> N;
  
  // Arrays on the host (CPU), dynamically allocated to overcome limited
  // stack size
  int *a = (int*)malloc(N*N*sizeof(int));
  int *b = (int*)malloc(N*N*sizeof(int));
  int *c = (int*)malloc(N*N*sizeof(int));
  int *compare = (int*)malloc(N*N*sizeof(int));
	
  //setup block/thread structure 	
  dim3 grid(N);
  dim3 block(N);

  //setup device matrix pointers
  int *dev_a, *dev_b, *dev_c;

  /*
    Allocate memory for device and check for errors
   */
  
  cudaError_t a_err = cudaMalloc( (void**) &dev_a, N * N * sizeof(int));
  if (a_err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(a_err) << std::endl;
    exit(1);
  }
  
  cudaError_t b_err = cudaMalloc( (void**) &dev_b, N * N * sizeof(int));
  if( b_err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(b_err) << std::endl;
    exit(1);
  }
    
  cudaError_t c_err = cudaMalloc( (void**) &dev_c, N * N * sizeof(int));
  if( c_err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(c_err) << std::endl;
    exit(1);
  }
  
 
 cudaEvent_t hstart, hend;
  cudaEventCreate(&hstart);
  cudaEventCreate(&hend);

  cudaEventRecord( hstart, 0 );


  // Initializes arrays on host.
  for (int i = 0; i < N; i++) {

    for(int j = 0; j < N; j++){

      int offset = i * N +j;
      a[offset] = 1;
      *(b + offset) = 2;
    }
  }

  //CPU addition of arrays

  for (int i = 0; i < N; i++) {

    for(int j = 0; j < N; j++){

      int offset = i * N +j;	
      *(compare + offset) = *(a + offset) + *(b + offset);

    }   
  } 

  cudaEventRecord( hend, 0 );
  cudaEventSynchronize( hend );

  float cpuTime;
  cudaEventElapsedTime( &cpuTime, hstart, hend );

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

  /*
    Once we have host arrays containing data and we have allocated
    memory on the GPU, we have to transfer data from the host to the
    device. Again, notice the similarity to C's memcpy function.

    The first argument is the destination of the copy - in this case a
    pointer to memory allocated on the device. The second argument is
    the source of the copy. The third argument is the number of bytes
    we want to copy. The last argument is a constant that tells
    cudaMemcpy the direction of the transfer.
   */
  cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, N * N * sizeof(int), cudaMemcpyHostToDevice);
  
  /*
    FINALLY we get to run some code on the GPU. At this point, if you
    haven't looked at add.cu (in this folder), you should. The
    comments in that file explain what the add function does, so here
    let's focus on how add is being called. The first thing to notice
    is the <<<...>>>, which you should recognize as _not_ being
    standard C. This syntactic extension tells nvidia's cuda compiler
    how to parallelize the execution of the function. We'll get into
    details as the course progresses, but for we'll say that <<<N,
    1>>> is creating N _blocks_ of 1 _thread_ each. Each of these
    threads is executing add with a different data element (details of
    the indexing are in add.cu). 

    In larger programs, you will typically have many more blocks, and
    each block will have many threads. Each thread will handle a
    different piece of data, and many threads can execute at the same
    time. This is how cuda can get such large speedups.
   */
  add<<<grid, block>>>(N, dev_a, dev_b, dev_c);

  /*
    Unfortunately, the GPU is to some extent a black box. In order to
    print the results of our call to add, we have to transfer the data
    back to the host. We do that with a call to cudaMemcpy, which is
    just like the cudaMemcpy calls above, except that the direction of
    the transfer (given by the last argument) is reversed. In a real
    program we would want to check the error code returned by this
    function.
  */
  cudaError_t err1 = cudaMemcpy(c, dev_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);
  if (err1 != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }

  
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
    for(int j = 0; j < N; ++j){
	
      int offset = i * N +j;
	
      if (*(compare + offset) != (*(a + offset) + *(b + offset))) {
      std::cerr << "Oh no! Something went wrong. You should check your cuda install and your GPU. :(" << std::endl;

      // clean up events - we should check for error codes here.
      cudaEventDestroy( start );
      cudaEventDestroy( end );
      cudaEventDestroy( hstart );
      cudaEventDestroy( hend );

      // clean up device pointers - just like free in C. We don't have
      // to check error codes for this one.
      cudaFree(dev_a);
      cudaFree(dev_b);
      cudaFree(dev_c);
      exit(1);
    }
    } 
    
  }

  /*
    Let's let the user know that everything is ok and then display
    some information about the times we recorded above.
   */
  std::cout << "Yay! Your program's results are correct." << std::endl;
  std::cout << "Your program took: " << elapsedTime << " ms." << std::endl;
  std::cout << "The CPU took: " << cpuTime << "ms " << std::endl;

  // Cleanup in the event of success.
  cudaEventDestroy( start );
  cudaEventDestroy( end );
  cudaEventDestroy( hstart );
  cudaEventDestroy( hend );

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  free(c);
  free(b);
  free(a);
  free(compare);	
  
  return 0;
}
