///////////////////////////////////////////////////////////////////////////////
////////////// Device code for PA3: kNN data Imputation    ////////////////////
//////////////////////////  written by Eric Li ////////////////////////////////
////////////////////////////////////////////////////////////////////////////// 

#include "knn.h"

/*
  This is the main function that performs the kNN algorithm. This will check if rows
  has holes, if not it will perform a partial distance calculation with the row that 
  is to be imputed. This calculates the difference between parameters, squares them,
  sums then stores in col 2 of each row.
*/

__global__ void knnDist( float *inputMat, float *partialMat, int imputRow, 
                                                  int rows, int cols, bool oddFlag){
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
           test to see if block ( time ) has an empty, if it is empty then threads 
           must idle because their calculation would be useless.
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
                  //printf("Impute index %d and tidx %d yield %f and %f \n", imputIdx, tidx, 
                                    //inputMat[imputIdx], inputMat[tidx]);
                  //printf("BID IS: %d \n", bidx);
                  
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
         //printf("INIT SUM ID: %d \n", sumIdx);
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
               //printf("loop sum id: %d and tidx %d \n", sumIdx, tidx);
               sumIdx+=blockDim.x;             
            }
            __syncthreads();  

      //thread reduction step
         //reset tidx
         tidx = bidx*cols + threadIdx.x + 2; 
         /*
           test for cases where blockDim is larger than # of cols to be reduced or 
           when cols is larger than blockDim
         */
         if( blockDim.x < (cols - 2) )
           {
              reduceThreads = blockDim.x / 2;
           }     

         else 
           {
              reduceThreads = (cols - 2) / 2;
           }
         //Reduction won't work if odd number of threads, so must add 1 if odd.
         if( oddFlag )
              {
                 reduceThreads+=1;
              }

         while( reduceThreads > 0 )
            { 
               if( threadIdx.x < reduceThreads )
                  {
                     partialMat[ tidx ] += partialMat[ tidx + reduceThreads ];
                  }
               __syncthreads();   
               reduceThreads /= 2;   
            }            
         }
         //stride to next set of blocks
         bidx+=gridDim.x;   
      }
}      

/*
  this function will calculate the square root , thereby finishing the distance calculation 
  and transfer the second col of each row into an array so that sorting can be done on CPU
*/
__global__ void distXfer( float* inMat, float* outArr, int rows, int cols ){
   int tid;
   //bidx = blockIdx.x;
   tid = blockIdx.x*gridDim.x + threadIdx.x;
   
   //grid stride loop
   while( tid < rows ){
      //inMat[ bidx*cols+2 ] = sqrt( inMat[])
      outArr[ tid ] = sqrt( inMat[ (tid * cols + 2) ]);
      //bidx += gridDim.x;
      tid+=gridDim.x*blockDim.x;
   }
}     

__device__ int


