
#include "matrixmult.h"

/*
  This is the GPU kernel for matrix multiplication. Input parameters *a and *b
  are the matrices to be multiplied. *c stores the answer matrix. n is the one 
  side dimension of any of the square matrices
 */
__global__ void matrixMult(int *a, int *b, int *c, int n) {
  //initialize variables used for shared memory cache. set Index to 
  int cacheSize;
  //cacheIndex

  //Index of matrix a and matrix b used for element by element multiplication 
  int aIndex, bIndex;
  
  //initialize number of threads for reduction step in multiplication 
  int reduceThreads;

  /*
    Calculate the starting block index (x,y). This is used for striding 
    across grid
  */
  int blockId = blockIdx.y * n + blockIdx.x;

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
  while( blockId < n*n ){
   //reset threads for every block stride 
   //tid = threadIdx.x; 
   //Stride loop for threads.
   while(tid < n){
      //initialize unique indices of input matrix a and b elements
      aIndex = tid + ( blockId / n ) * n;
      bIndex = ( blockId % n ) + tid * n;

      //multiply element by element and store in shared cache.
      cache[tid] = ( *( a + aIndex ) ) * ( *(b + bIndex ) );
      
      //stride over number of threads
      tid+=blockDim.x;
   }
  /*
    have to synchronize all threads to make sure cache values are all updated
    before accessing them for the summation step.
    */ 
    __syncthreads(); 

    //reset tid to threadIdx.x
    // tid = threadIdx.x;

    /*
      Reduction loop: will loop until summation complete in cache. Need to sync 
      threads before every addition to avoid race conditions
      */
      while( reduceThreads > 0 ){
        //reset tid
        tid = threadIdx.x;
        
       while( tid < reduceThreads ){
          cache[tid] += cache[ tid + reduceThreads ];
          tid+=blockDim.x;
        }
      __syncthreads();
      reduceThreads /= 2;
      }

    //write results of matrix back to product matrix 
    *( c + blockId ) = cache[0];

    /*
      stride over grid
    */ 
    blockId = blockId + ( gridDim.x ); 
  } 
}     
