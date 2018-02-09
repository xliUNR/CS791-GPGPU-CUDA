
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
/*
  initialize variables, indexes for two arrays being multiplied along with 
  a cache size variable
*/
  int cacheSize, aIndex, bIndex, threadId, blockId, cacheIndex;
  cacheIndex = threadIdx.x;

/*
  check for even threads per block. Reduction requires even number of threads
  so must pad if odd.
*/  
  if( blockDim.x % 2 != 0){
    cacheSize = blockDim.x + 1;
  }
  else{
    cacheSize = blockDim.x;
  }

  __shared__ cache[cacheSize];

  //pad w/ 0 if odd cacheSize
  if( blockDim.x % 2 != 0 ){
    cache[cacheSize] = 0;
  }
/* 
  Loop over n for each thread in a block, One thread corresponds to one element by
  element multiplication. Results are stored in shared memory cache. This way
  all blocks will have independent cache. Requires a synchronization step before
  summing cache for each block. This is to ensure device has finished multiplication 
  step.  
*/
  while( cacheIndex < n ){
    aIndex = threadIdx.x + blockIdx.y * n ;
    bIndex = blockIdx.x + threadIdx.x * n ;
    
    cache[cacheIndex] = *( a + aIndex ) * ( b + bIndex );
    
    //stride to next set of elements
    cacheIndex+=blockDim.x;

 }
 __syncthreads();


/*
  Use reduction to sum values, when it is done, cache[0] will contain sum for each block
  which corresponds to a element in the product matrix.
  
*/
int reduceIndex = cacheSize / 2 ;
while( reduceIndex != 0 ){
  if(threadIdx.x < reduceIndex ){
    cache[threadIdx.x] += cache[ threadIdx.x + reduceIndex ];
  }
  __syncthreads();
  reduceIndex /= 2;
}

//load all values into final matrix c
*(c + (blockIdx.x + blockIdx.y * n) ) = cache[0];

//striding, in cases were N > number of threads    
while( thread_id < n*n ){    
    //c[thread_id] = a[thread_id] + b[thread_id];
    *(c + thread_id) = *(a + thread_id ) * *(b + thread_id);  
    //stride to next grid
    thread_id += blockDim.x * gridDim.x;     
  }
}     


//matrix add function that uses grid-striding
__global__ void strideAdd(int n, int *a, int *b, int *c) {
  //initialize offset AKA unique thread id
   int option = 0;
   int thread_id = 0;

   switch (option ) {
    case 0: 
       thread_id = threadIdx.x + blockIdx.x * blockDim.x;
       break;

    case 1:
       thread_id =  blockIdx.x * blockDim.x * blockDim.y
                          + threadIdx.y * blockDim.x + threadIdx.x;
       break;
       
    case 2:
       int blockId = blockIdx.y * gridDim.x + blockIdx.x;  
       thread_id = blockId * blockDim.x + threadIdx.x;
       break;                     

    }


  //loop over each grid
  for( int i = thread_id; i < n*n; i+= blockDim.x * gridDim.x )
    {
      *(c + thread_id) = *(a + thread_id ) * *(b + thread_id);      
    }

}

