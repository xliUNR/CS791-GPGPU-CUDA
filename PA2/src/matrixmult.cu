
#include "matrixmult.h"

/*
  This is the GPU kernel for matrix multiplication. Input parameters *a and *b
  are the matrices to be multiplied. *c stores the answer matrix. n is the one 
  side dimension of any of the square matrices
 */
__global__ void matrixMult(int *a, int *b, int *c, int n) {
  //initialize variables used for shared memory cache. set Index to 
  int cacheSize, cacheIndex;

  //Index of matrix a and matrix b used for element by element multiplication 
  int aIndex, bIndex;
  
  //initialize number of threads for reduction step in multiplication 
  int reduceThreads;

  /*
    Calculate the starting block index (x,y). This is used for striding 
    across grid
  */
  int b_x, b_y;
  b_x = blockIdx.x;
  b_y = blockIdx.y;

  /*
    Initialize threadId for thread striding
  */  
  int tid = threadIdx.x;

  //initialize shared cache to store partial results from element by element mult.
  extern __shared__ int cache[];

  //pad if n is odd 
  if( n % 2 != 0 ){
    cacheSize = n + 1;
    cache[cacheSize] = 0;
  }

  else{
    cacheSize = n;
  }
  
  //calculate # of threads for reduction 
  reduceThreads = cacheSize / 2 ;

  //Stride loop.  
  while( b_x * b_y < n*n ){

   //Stride loop for threads.
   while(tid < n){
      //initialize unique indices of input matrix a and b elements
      aIndex = tid + b_y * n;
      bIndex = b_x + tid * n;

      //calculate index of cache for each thread
      cacheIndex = tid;

      //multiply element by element and store in shared cache.
      cache[cacheIndex] = ( *( a + aIndex ) ) * ( *(b + bIndex ) );
      
      /*
      have to synchronize all threads to make sure cache values are all updated
      before accessing them for the summation step.
      */ 
      __syncthreads(); 

      /*
      Reduction loop: will loop until summation complete in cache. Need to sync 
      threads before every addition to avoid race conditions
      */
      while( reduceThreads > 0 ){
       if( threadIdx.x < reduceThreads ){
          cache[threadIdx.x] += cache[ threadIdx.x + reduceThreads ];
        }

      __syncthreads();
      reduceThreads /= 2;
      }

      //stride over number of threads
      tid+=blockDim.x;
   }
    //write results of matrix back to product matrix 
    *(c + b_y * n + b_x) = cache[0];

    /*Increment block x and y indices. This is required b/c of striding. In order
      to multiply the correct elements, the indices are incremented independent 
      of the actual CUDA block indices because those no longer provide the "true"
      indices once striding is used.
    */ 
    b_x+=1;
    b_y+=1; 
  }
}     
