///////////////////////////////////////////////////////////////////////////////
////////////// main program for PA2: Matrix Multiplication ////////////////////
////////////////////////// written by Eric Li ////////////////////////////////
///////////////////////////////////////////////////////////////////////////// 
#include <iostream>
#include <cstdio>
#include <time.h>
#include "matrixmult.h"
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
/////////////////////////////////////////////////////////////////////////////
//////////////////// variable declarations //////////////////////////////////  
  //Initialize variables for user specified grid and block structures.   
  int matrixDim, numElements, numThreads, numBlocks;

  //initialize cache size for shared cache among the blocks
  int cacheSize;

  //initialize device properties for checking limitations of GPU
  cudaDeviceProp prop;
  
  /*
    initialize pointers for matrices: A and B are multiplied and stored in
    matrix C. 
    cpuC is for the sequential implementation of matrix multiplication  
  */
  int *matA, *matB, *matC, *cpuC;

///////////////////////////////////////////////////////////////////////////////
/////////////////////////////// start program /////////////////////////////////
  std::cout << std::endl << "Square matrix multiplication program";
  //ask for dimensions of matrices and store in variable matrixDim
  std::cout << std::endl << "Enter dimension of square matrices to be multiplied: ";
  std::cin >> matrixDim;

  /*
    get device properties of nvidia card and test to make sure we are within
    capabilities of the card
  */
  cudaGetDeviceProperties(&prop, 0);

  /*
    test to make sure dimensions aren't larger than shared memory to make sure
    cache has enough room.
  */
  if( ( matrixDim * sizeof(int) ) > prop.sharedMemPerBlock ){
    std::cout << std::endl << "Sorry matrix dimension is too large for shared memory. Program exiting" << std::endl;
    exit(1);
  }

  //ask for threads per block and check if valid
  std::cout << std::endl << "Please specify number of threads per block " 
            << std::endl << "(Max number of threads per block is: " 
            << prop.maxThreadsPerBlock << "): ";

  std::cin >> numThreads;
  if( numThreads > prop.maxThreadsPerBlock ){
    std::cout << std::endl << "Sorry number of threads larger than GPU supports. Program exiting" << std::endl;
    exit(1);
  }

  //ask for grid dimension
  std::cout << std::endl << "Please specify dimension of grid "
            << std::endl << "(Max dimension of grid is: [" 
            << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]: ";
  std::cin >> numBlocks;
  if( numBlocks > prop.maxGridSize[1] ){
    std::cout << std::endl << "Sorry grid dimensions are too larger than GPU supports. Program exiting" << std::endl;
    exit(1);
  }

  //calculate number of elements in each matrix for memory allocation purposes
  numElements = matrixDim * matrixDim; 
   
  //Allocated unified memory
  HANDLE_ERROR( cudaMallocManaged( &matA, numElements*sizeof(int)) );
  HANDLE_ERROR( cudaMallocManaged( &matB, numElements*sizeof(int)) );
  HANDLE_ERROR( cudaMallocManaged( &matC, numElements*sizeof(int)) );
  HANDLE_ERROR( cudaMallocManaged( &cpuC, numElements*sizeof(int)) );


  //setup block/thread structure based on user input	
  dim3 grid(numBlocks, numBlocks);
  dim3 block(numThreads);
  
  // Initializes matrix A
  for (int i = 0; i < matrixDim; i++) {
    for(int j = 0; j < matrixDim; j++){

      //int offset = i * N +j;
      *(matA + i * matrixDim + j) = 1;
      //(i * matrixDim + j);  
    }
  }
  // Initializes matrix B
  for (int i = 0; i < matrixDim; i++) {
    for(int j = 0; j < matrixDim; j++){

      //int offset = i * N +j;
      *(matB + i * matrixDim + j) = 1;
      //(i * matrixDim + j);
    }
  }

  //start timing for sequential portion
  cudaEvent_t hstart, hend;
  cudaEventCreate(&hstart);
  cudaEventCreate(&hend);
  cudaEventRecord( hstart, 0 );

  //CPU sequential matrix multiplication
  for(int i=0; i < matrixDim; i++){
    for(int j=0; j < matrixDim; j++){
      //initialize sum for each multiplication
      int sum = 0;
      
      for(int k=0; k < matrixDim; k++){
        //sum = sum + ( matA[i][k] * matB[k][j] );
        sum+= *(matA + i * matrixDim + k ) * *(matB + k * matrixDim + j);
      }
    //save sum to cpu matrix  
    *( cpuC + i * matrixDim + j ) = sum; 
    }
  }
  //stop time for sequential run.
  cudaEventRecord( hend, 0 );
  cudaEventSynchronize( hend );
  float cpuTime;
  cudaEventElapsedTime( &cpuTime, hstart, hend );
  

  //start event timer for GPU parallel implementation 
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord( start, 0 );
  
  /*
    check to see if matrixDim is even. Will have to pad with 0 if not.
    This is for the reduction step, must have an even number to perform
    reduction algorithm.
  */
  if( matrixDim % 2 != 0 ){
    cacheSize = matrixDim + 1;
  }
  else{
    cacheSize = matrixDim;
  }

  /*
    kernel call for GPU implementation, last part is for dynamic alloc of
    cache
  */  
  matrixMult<<<grid, block, cacheSize*sizeof(int)>>>(matA, matB, matC, matrixDim);

  //error handling for kernel calls 
  HANDLE_ERROR( cudaPeekAtLastError() );
  HANDLE_ERROR( cudaDeviceSynchronize() );

  //end timer for GPU matrix multiplication
  cudaEventRecord( end, 0 );
  cudaEventSynchronize( end );

  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, end );

  
  //check to make sure both matrices match each other
  for (int i = 0; i < matrixDim; ++i){
    for(int j = 0; j < matrixDim; ++j){
	
      int offset = i * matrixDim + j;
	
      if (*(matC + offset) != *(cpuC + offset) ) {
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
      std::cout << *(matC + offset);
    }   
    std::cout << std::endl; 
  }
std::cout << "CPU matrix: " << std::endl;
  for (int i = 0; i < matrixDim; ++i){
    for(int j = 0; j < matrixDim; ++j){
  
      int offset = i * matrixDim + j;
  
      /*if (*(matC + offset) != *(cpuC + offset) ) {
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
      }*/
      std::cout << *(cpuC + offset);
    }   
    std::cout << std::endl; 
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

  cudaFree(matA);
  cudaFree(matB);
  cudaFree(matC);
  cudaFree(cpuC);
  
  return 0;
}
