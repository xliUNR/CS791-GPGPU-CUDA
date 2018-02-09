
#include "add.h"

/*
  This is the function that each thread will execute on the GPU. The
  fact that it executes on the device is indicated by the __global__
  modifier in front of the return type of the function. After that,
  the signature of the function isn't special - in particular, the
  pointers we pass in should point to memory on the device, but this
  is not indicated by the function's signature.
 */
__global__ void add(int n, int *a, int *b, int *c) {
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
  check for even threads per block. Reduction requires even number of threads
  so must pad if odd.
  */  
  if( n % 2 != 0 ){
    cacheSize = n + 1;
  }
  else{
    cacheSize = n;
  }

  //initialize shared cache to store partial results from element by element mult.
  __shared__ cache[cacheSize];

  //pad last element of cache with 0 if cacheSize is odd
  if( blockDim.x % 2 != 0 ){
    cache[cacheSize] = 0;
  }

  //Stride loop. I used for because it seemed safer than while loop.  
  for( int i = (blockIdx.y * gridDim.x + blockIdx.x) ; 
                              i < n*n; i += gridDim.x * gridDim.y ){

   //Stride loop for threads.
   for( int i = threadIdx.x; i < n; i+=blockDim.x){
      //initialize unique indices of input matrix a and b elements
      aIndex = threadIdx.x + b_y * n;
      bIndex = b_x + threadIdx.x * n;

      //calculate index of cache for each thread
      cacheIndex = threadIdx.x;

      //multiply element by element and store in shared cache.
      cache[cacheIndex] = ( *( a + aIndex ) ) * ( *(b + bIndex ) );
      
      //stride over number of threads
      cacheIndex+=blockDim.x;
   }

    /*
      have to synchronize all threads to make sure cache values are all updated
      before accessing them for the summation step.
    */ 
    __syncthreads(); 

    //calculate Index 
    reduceThreads = cacheSize / 2 ;

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

    //write results of matrix back to product matrix 
    *(c + b_y * n + b_x) = cache[0];

    /*Increment block x and y indices. This is required b/c of striding. In order
      to multiply the correct elements, the indices are incremented independent 
      of the actual CUDA block indices because those no longer provide the "true"
      indices once striding is used.
    */ 
    b_x+1;
    b_y++1; 
  }
}     
