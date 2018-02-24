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

  Parameters are: inputMat: the Input matrix
                  partialMat: partial results matrix that will store partial distance
                  imputRow: The row that contains the hole and needs imputation
                  rows: The number of rows in matrices
                  inCols: number of columns in inputMat
                  pCols: The number of columns in partialMat (could be padded)


*/

__global__ void knnDist( float *inputMat, float *partialMat, int imputRow, 
                                                  int rows, int inCols, int pCols){
   //initialize variables
   int bidx, tidx, pidx, reduceThreads, sumIdx, EmptyoffsetIndex, imputIdx;
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
           calc thread index in input matrix, offset by 2 since first col is
           id and the second col contains holes 
         */
         tidx = bidx * inCols + threadIdx.x + 2;
         

         //Calculate offset of 2nd col, which tells whether row has hole or not
         EmptyoffsetIndex = ( bidx * inCols + 1 );

         /*
           test to see if block ( time ) has an empty, if it is empty then threads 
           must idle because their calculation would be useless.
           Otherwise, this will calculate the partial results of subtraction
           and squaring. Each element is stored in partial matrix which will
           be later summed and square rooted for the Euclidean distance. 
         */
         if( inputMat[ EmptyoffsetIndex ] != -1 ){
            //loop for thread stride
            while( tidx < inCols*(bidx+1) )
               {  
                  /*
                  Calculate an index for the partial matrix. This includes an offset for the
                  padding, if it is there, if not then pidx will just be tidx since 
                  (pCols - inCols) will equal 0.
                  */
                  pidx = bidx * (pCols - inCols ) + tidx;
                  //calc the column of the row that needs to be imputed
                  imputIdx = imputRow * inCols + tidx - (bidx * inCols);  
                  
                  //Calc difference between elements & square
                  diff = inputMat[imputIdx] - inputMat[tidx];
                  //print impute idx
                  //printf("Impute index %d and tidx %d yield %f and %f \n", imputIdx, tidx, 
                                    //inputMat[imputIdx], inputMat[tidx]);
                  //printf("BID IS: %d \n", bidx);
                  
                  //store in partial matrix
                  partialMat[pidx] = diff * diff;
                  //stride threads to next set of operations
                  tidx = tidx + blockDim.x;
               }
         //sync threads b4 reduction 
         __syncthreads();         
      
      //do reduction summation  
         /*
           set pidx to index partial matrix. Now we want to include the padding values
           so we don't need to calculate an offset.
         */
         pidx = bidx * pCols + threadIdx.x + 2;
         /*
           Calculate the index of element to be summed in reduction. 
           This will be a block size over to ensure no threads are summing
           element belonging to other thread. 
         */
         sumIdx = pidx + blockDim.x;
         //printf("INIT SUM ID: %d \n", sumIdx);
         /*
           stride loop for summing. The first block size number of
           threads will hold the sums. Then this will be reduced.
         */
         while( sumIdx < pCols*(bidx+1) )
            {  
               /*
                 caclulate index of partial matrix that the reduction 
                 results are stored in, then sum and stride to next row
               */  
               partialMat[ pidx ] += partialMat[ sumIdx ];
               //printf("loop sum id: %d and tidx %d \n", sumIdx, tidx);
               sumIdx+=blockDim.x;             
            }
            __syncthreads();  

      //thread reduction step
         /*
           test for cases where blockDim is smaller than # of cols to be reduced or 
           when cols is smaller than blockDim. In the former case, the above reduction 
           step has finished and partial sums are stored at each element indexed by 
           a thread Id. So reduceThreads is set to blockDIm so a reduction can be 
           performed only on the threads.

           In the case of the latter where number of threads per block is larger than
           number of columns, a reduction of all the threads would lead to bad results 
           since they would be overextending memory, so reduceThreads must be set to
           columns - 2. This will perform the reduction w/ no striding required. One
           Element is assigned to each thread.
         */
         if( blockDim.x < (pCols - 2) )
           {
              reduceThreads = blockDim.x / 2;
           }     

         else 
           {
              reduceThreads = (pCols - 2) / 2;
           }

         printf("\n Value of reduceThreads= %d", reduceThreads);     
         
         while( reduceThreads > 0 )
            { 
               if( threadIdx.x < reduceThreads )
                  {
                     partialMat[ pidx ] += partialMat[ pidx + reduceThreads ];
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

  Parameters: inMat: The input matrix
              outArr: The array that this function will write to
              rows: Number of rows in input matrix, which is also number of elements in outArr
              cols: The number of columns of input matrix.
*/
__global__ void distXfer( float* inMat, float* outArr, int rows, int cols ){
   int tid;
   tid = blockIdx.x*gridDim.x + threadIdx.x;

   //grid stride loop
   while( tid < rows ){

      outArr[ tid ] = sqrt( inMat[ (tid * cols + 2) ]);
      tid+=gridDim.x*blockDim.x;
   }
}     




