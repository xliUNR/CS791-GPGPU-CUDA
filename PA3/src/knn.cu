///////////////////////////////////////////////////////////////////////////////
////////////// Device code for PA3: kNN data Imputation    ////////////////////
//////////////////////////  written by Eric Li ////////////////////////////////
////////////////////////////////////////////////////////////////////////////// 

#include "knn.h"

/*
  This is the main function that performs the kNN algorithm.
*/

__global__ void knnDist( float *inputMat, float *partialMat, int imputRow, 
                                                           int rows, int cols){
   //initialize variables
   int bidx, tidx, reduceThreads, sumIdx, EmptyoffsetIndex, imputIdx;
   float diff;
   /*
     calculate unique index in matrix. This is so that each thread can access
     the correct memory location corresponding to it's data point. This index 
     is calculated from the size of the array and the block index and thread 
     indices of each thread.
   */  
   bidx = blockIdx.x;
   //reduceThreads = cols / 2;

   while( bidx < rows )
      {  
         /*
           calc thread index in partial matrix, offset by 2 since first col is
           id and the second col contains holes 
         */
         tidx = bidx * cols + threadIdx.x + 2;

         //Calculate offset of 2nd col, which tells whether row has hole or not
         EmptyoffsetIndex = ( bidx * cols + 1 );
         /*
           test to see if block ( time ) has an empty, if it is empty then threads must idle because their calculation would be useless.
           Otherwise, this will calculate the partial results of subtraction
           and squaring. Each element is stored in partial matrix which will
           be later summed and square rooted for the Euclidean distance. 
         */
         if( inputMat[ EmptyoffsetIndex ] != -1 ){
            //loop for thread stride
            while( tidx < cols*(bidx+1) )
               {  
                  //calc the column of the row that needs to be imputed
                  imputIdx = imputRow * cols + tidx - (bidx * cols);

                  
                  
                  //Calc difference between elements & square
                  diff = inputMat[imputIdx] - inputMat[tidx];
                  //print impute idx
                  printf("Impute index %d and tidx %d yield %f and %f \n", imputIdx, tidx, 
                                    inputMat[imputIdx], inputMat[tidx]);
                  printf("BID IS: %d \n", bidx);
                  
                  partialMat[tidx] = diff * diff;
                  //stride threads to next set of operations
                  tidx = tidx + blockDim.x;
               }
         //sync threads b4 reduction 
         __syncthreads();         
      
      //do reduction summation  
         //reset tidx from thread striding above
         tidx = bidx*cols + threadIdx.x + 2;  
         /*
           Calculate the index of element to be summed in reduction. 
           This will be a block size over to ensure no threads are summing
           element belonging to other thread. 
         */
         sumIdx = tidx + blockDim.x;
         /*
           stride loop for summing. The first block size number of
           threads will hold the sums. Then this will be reduced.
         */
         while( sumIdx < cols*(bidx+1) )
            {  
               /*
                 caclulate index of partial matrix that the reduction 
                 results are stored in, then sum and stride to next row
               */  
               partialMat[ tidx ] += partialMat[ sumIdx ];
               sumIdx+=blockDim.x;             
            }
            __syncthreads();  

      //thread reduction step
         //reset tidx
         tidx = bidx*cols + threadIdx.x + 2;      
         reduceThreads = blockDim.x / 2;
         while( reduceThreads > 0 )
            { 
               if( threadIdx.x < reduceThreads )
                  {
                     partialMat[ tidx ] += partialMat[ tidx + reduceThreads ];
                  }
               __syncthreads();   
               reduceThreads /= 2;   
            }            
            //Square root results of summation to get distance             
            partialMat[ (bidx * cols + 2) ] = 
                                   sqrt( partialMat[ (bidx * cols + 2) ] );
         }
        
         //stride to next set of blocks
         bidx+=gridDim.x;   
      }
}      

/*
  this function will transfer the second col of each row into an array so
  that sorting can be done on CPU
*/
__global__ void distXfer( float* inMat, float* outArr, int rows, int cols ){
   int bidx;
   bidx = blockIdx.x;
   //grid stride loop
   while( bidx < rows ){
      outArr[ bidx ] = inMat[ (bidx * cols + 2) ];
      bidx += gridDim.x;
   }
}     




