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
//error checking macro
#define HANDLE_ERROR(func) { GPUAssert((func), __FILE__, __LINE__);}
inline void GPUAssert( cudaError_t errCode, const char *file, int line, bool abort=true)
    {
     if( errCode != cudaSuccess )
         {
          fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(errCode), file, line);
          if (abort) exit(errCode);
         }
    }


int main() {
  //initialize variables
  int N;
  int rowA, colA, rowB, colB, rowP, colP;
  int *matA, *matB, *matC, *partial, *cpuC;
  char userRes;
  bool STRIDEFLAG;
  bool repeat = true;
  std::cout << "Enter # of rows for matrix A: " << std::endl;
  std::cin >> rowA;
  std::cout << "Enter # of columns for matrix A: " << std::endl;
  std::cin >> colA;

  std::cout << "Enter # of rows for matrix B: " << std::endl;
  std::cin >> rowB;
  std::cout << "Enter # of columns for matrix A: " << std::endl;
  std::cin >> colB;

  if( colA != rowB )
    {
      std::cout << "Invalid matrix dimensions (row of A != col of B). Program exiting" << std::endl;
      exit(1);              
    }
  
  //calculate dimensions for partial matrix
  rowP = rowA * colB;
  colP = colA;

  //Ask user if stride enabled
  do{
    std::cout << "stride mode? (Y/N) " << std::endl;
    std::cin >> userRes;
    if( userRes == 'Y' | 'y' ){
      STRIDEFLAG = true;
      repeat = false;
    }
    
    else if( userRes == 'N' | 'n' ){
      STRIDEFLAG = false;
      repeat = false;
    }
    else{
      std::cout << "invalid response. Try again." << std::endl;
    }
  } while( repeat );
  

  
  //Allocated unified memory
  //int *compare = (int*)malloc(N*N*sizeof(int));
	
  HANDLE_ERROR( cudaMallocManaged( &matA, rowA*colA*sizeof(int)) );
  HANDLE_ERROR( cudaMallocManaged( &matB, rowB*colB*sizeof(int)) );
  HANDLE_ERROR( cudaMallocManaged( &partial, rowP*colP*sizeof(int)) );
  HANDLE_ERROR( cudaMallocManaged( &matC, rowA*colB*sizeof(int)) );
  HANDLE_ERROR( cudaMallocManaged( &cpuC, rowA*colB*sizeof(int)) );


  //setup block/thread structure need to change to better method	
  dim3 grid(N);
  dim3 block(N);

 cudaEvent_t hstart, hend;
  cudaEventCreate(&hstart);
  cudaEventCreate(&hend);

  cudaEventRecord( hstart, 0 );


  // Initializes arrays on host
  for (int i = 0; i < N; i++) {

    for(int j = 0; j < N; j++){

      int offset = i * N +j;
      matA[offset] = 1;
      matB[offset] = 2;
    }
  }

  //CPU sequential matrix multiplication
  for(int i=0; i < rowA; i++){
    for(int j=0; j < colB; j++){
      sum = 0;

      for(int k=0; k < colA; k++){
        sum+= matA[i][k] * matB[k][j];
      }
    cpuC[i][j] = sum;  
    }
  }

  std::cout << "Final matrix: " << std::endl;
  //print matrix
  for(int i=0; i < rowA; i++){
    for(int j=0; j < colB; j++){
      std::cout << cpuC[i][j] << ' '; 
    }
    cout << std::endl;
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
  
 //parallel add functions, one for stride mode and one for regular operation
  if( STRIDEFLAG ){
    strideAdd<<<grid, block>>>(N, matA, matB, matC);
  }
  else{128
    add<<<grid, block>>>(N, matA, matB, matC);
  }

//error handling for kernel calls 
HANDLE_ERROR( cudaPeekAtLastError() );
HANDLE_ERROR( cudaDeviceSynchronize() );

  
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
   
  for (int i = 0; i < N; ++i) {
    for(int j = 0; j < N; ++j){
	
      int offset = i * N +j;
	
      if (*(compare + offset) != (*(matA + offset) + *(matB + offset))) {
      std::cerr << "Oh no! Something went wrong. You should check your cuda install and your GPU. :(" << std::endl;

      // clean up events - we should check for error codes here.
      cudaEventDestroy( start );
      cudaEventDestroy( end );
      cudaEventDestroy( hstart );
      cudaEventDestroy( hend );

      // clean up device pointers - just like free in C. We don't have
      // to check error codes for this one.
      cudaFree(matA);
      cudaFree(matB);
      cudaFree(matC);
      exit(1);
    }
    } 
    
  }*/

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

  cudaFree(matA);
  cudaFree(matB);
  cudaFree(matC);
  cudaFree(partial);
  cudaFree(cpuC);

  //free(compare);	
  
  return 0;
}
