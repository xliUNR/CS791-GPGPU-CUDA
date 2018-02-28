#include "cudafunctions.h"


//extern "C" cudaHello(){}

__global__ void helloThere(int rank, int*a, int*b, int*c){
    
    printf("\n Hello From GPU: %d", rank);
    //do maths
    c[0] = a[0] + b[0];
    printf("\n A = %d |  b = %d | c = %d", a[0], b[0], c[0]);
}

__global__ void matrixMult(int*a, int* b, int* outMat, int arrDim){
   int bidx_y, bidx_x, blockId, tidx, reduceThreads;

   bidx_x = blockIdx.x;
   bidx_y = blockIdx.y;
   blockId = bidx_y * gridDim.x + bidx_x;

   tidx = threadIdx.x;

   //grid stride loop
   for(int i = ( blockId*blockDim.x + tidx ) ; i < arrDim*arrDim*arrDim; 
                                 i+= (gridDim.x * gridDim.y * blockIdx.x) ){
     
      //do multiplication and then store in i of outMat
      outMat[ i ] = a[ bidx_y*arrDim + tidx ] * b[tidx*arrDim + bidx_x ];
      
      //recalculate indices basedon striding.
      tidx+= blockDim.x;
      bidx_y+=gridDim.y;
      bidx_x+=gridDim.x;
      //recalculate 
   }

}

__global__ void reduction(int* inMat, int* outMat, int arrDim){
   for(int i = ( blockId*blockDim.x + tidx ) ; i < arrDim*arrDim*arrDim; 
                                 i+= (gridDim.x * gridDim.y * blockIdx.x) ){
      
   }
}