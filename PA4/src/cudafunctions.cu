#include "cudafunctions.h"


//extern "C" cudaHello(){}

__global__ void helloThere(int rank, int*a, int*b, int*c){
    
    printf("\n Hello From GPU: %d", rank);
    //do maths
    c[0] = a[0] + b[0];
    printf("\n A = %d |  b = %d | c = %d", a[0], b[0], c[0]);
}

//kernel for matrix multiplication
__global__ void matrixMult(int*a, int* b, int* outMat, int arrDim, 
                                                         int partialDim){
   int bidx_y, bidx_x, blockId, tidx;

   bidx_x = blockIdx.x;
   bidx_y = blockIdx.y;
   tidx = threadIdx.x;

   //stride over 3 dimensions  
   while( bidx_y < arrDim ){
      //calculate/reset block index in x of grid 
      bidx_x = blockIdx.x;
      while(bidx_x < arrDim){
         //calculate/reset thread index of each block
         tidx = threadIdx.x;
         while( tidx < arrDim){  
            //calculate index of block in partial results matrix
            blockId = bidx_y * arrDim + bidx_x;   
            //do multiplication and then store in i of outMat
            outMat[ blockId * partialDim + tidx ] = 
                      a[ bidx_y*arrDim + tidx ] * b[tidx*arrDim + bidx_x ];
            //stride to next set of threads      
            tidx+=blockDim.x;
         }
         //stride to next y dimension
         bidx_x+=gridDim.x;
      }
     //stride to next x dimension 
     bidx_y+=gridDim.y; 
   }   
}

//reduction kernel
__global__ void reduction(int* inMat, int* outMat, int arrDim, int partialDim){
   int bidx_y, bidx_x, blockId, tidx, reduceThreads;

   
   bidx_y = blockIdx.y;
   

   //grid stride loop for 3D
   while( bidx_y < arrDim ){
      //calculate/reset x index of block
      bidx_x = blockIdx.x;
      while(bidx_x < arrDim){
         //calculate reduce threads && unique blockId   
         reduceThreads = partialDim / 2;
         blockId = bidx_y*arrDim + bidx_x;
         //thread reduction loop
         while(reduceThreads > 0){
            //calculate/reset tidx
            tidx = threadIdx.x;
            //thread striding loop
            while(tidx + reduceThreads < partialDim ){
            //reduction sum   
            inMat[ blockId * partialDim + tidx ] 
                     += inMat[ blockId * partialDim + tidx + reduceThreads ];
            //stride to next set of threads.
            tidx+= blockDim.x;
            }
          //sync threads before next reduction iteration  
          __syncthreads(); 
          reduceThreads /= 2; 
         }
      //write results of reduction to output matrix
      outMat[ bidx_y*arrDim + bidx_x ] = inMat[ blockId* partialDim + 0 ];   
      bidx_x+=gridDim.x;
      }
   bidx_y+=gridDim.y; 
   }  
}


/*
   kernel for element by element matrix summation. Launch uses 1D grid of 1D blocks. Stores result back in first parameter matrix
*/   
__global__ void matSum( int*a, int*b, int arrDim){
   for(int i=blockIdx.x * blockDim.x + threadIdx.x; i < arrDim*arrDim;
                                                   i+=blockDim.x*gridDim.x ){
      //a[i]+= b[i];
      atomicAdd(&a[i], b[i]);
    printf("\n The result of addition of first and %d is: %d", b[i], a[i]);
   }
  __syncthreads(); 
}