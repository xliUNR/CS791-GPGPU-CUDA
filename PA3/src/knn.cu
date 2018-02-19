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
   int blockId, threadId, partialIndex;

   blockId = blockIdx.x;
   threadId = threadIdx.x;
   //calculate offset for empty region
   Emptyoffset = ( blockIdx.x * blockDim.x + 2 )*sizeof(int);
   EmptyoffsetIndex = ( blockIdx.x * blockDim.x + 2 );
   
   while( blockId < rows )
      {  
         /*
           test to see if block ( time ) has an empty, if it is empty then threads must idle.
         */
         if( (*inputMat + offset) > 0 ){
            while( threadId < cols )
               {  
                  partIndex = blockId * rows + ;
                  partialMat[partialIndex] = 
                         square(inputMat[imputIndex] - inputMat[partIndex]);
                  //stride to next
                  threadId = threadId + blockDim.x;
               }
            //sync threads b4 reduction   
            //do reduction summation   

         }
        
         //stride to next set of blocks
         blockId = blockId + gridDim.x;   
      }


///////////////////////////////////////////////////////////////////////////////
////////////////////////// helper functions ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
float square( float input ){
   return input * input;
}
