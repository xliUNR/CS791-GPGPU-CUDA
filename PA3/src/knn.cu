///////////////////////////////////////////////////////////////////////////////
////////////// Device code for PA3: kNN data Imputation    ////////////////////
//////////////////////////  written by Eric Li ////////////////////////////////
////////////////////////////////////////////////////////////////////////////// 

#include "knn.c"

/*
  This is the main function that performs the kNN algorithm.
*/

__global__ void kNN( int *inputMat, int *partialMat, int imputIndex, int rows, 
                                                                     int cols){
   //initialize variables
   int blockId, threadId, partialIndex, reduceThreads, reduceIndex;

   blockId = blockIdx.x;
   threadId = threadIdx.x + 2;
   reduceThreads = cols / 2;

   //calculate offset for empty region
   Emptyoffset = ( blockIdx.x * blockDim.x + 2 )*sizeof(int);
   EmptyoffsetIndex = ( blockIdx.x * blockDim.x + 2 );
   

   while( blockId < rows )
      {  
         /*
           test to see if block ( time ) has an empty, if it is empty then threads must idle.
         */
         if( (*inputMat + Emptyoffset) > 0 ){
            while( threadId < cols )
               {  
                  partIndex = blockId * rows + threadId;
                  partialMat[partialIndex] = 
                         square(inputMat[imputIndex] - inputMat[partIndex]);
                  //stride to next
                  threadId = threadId + blockDim.x;
               }
         //sync threads b4 reduction 
         __syncthreads();         
         //do reduction summation   
         while( reduceThreads > 0 )
            {             
               //reset threadId back from striding above
               threadId = threadIdx.x + 2;
               while( threadId < reduceThreads )
                  {  
                     /*
                       caclulate index of partial matrix that the reduction 
                       results are stored in, then sum and stride to next col
                     */  
                     reduceIndex = blockId * rows + threadId;
                     partialMat[ reduceIndex ] += 
                                  partialMat[ reduceIndex + reduceThreads];
                     threadId+=blockDim.x;             
                  }
               __syncthreads();
               reduceThreads /= 2;   
            }

         }
        
         //stride to next set of blocks
         blockId = blockId + gridDim.x;   
      }


//This function performs reduction to find 5 minimum on the input matrix
//Need to do parallel bubble sort, then sum first 5 values and average      
__global__ void reduceMin( int* inMat){
   int reduceThreads, reduceIndex;
   reduceIndex = blockIdx.x * cols + 2;

   reduceThreads = rows / 2;

   while( reduceThreads > 5 )
      {
         atomicMin(&(inMat[reduceIndex]), 
                              inputMat[ reduceIndex + reduceThreads]);
         __syncthreads();
         reduceThreads /= 2;
      }
}  

///////////////////////////////////////////////////////////////////////////////
////////////////////////// helper functions ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
float square( float input ){
   return input * input;
}
