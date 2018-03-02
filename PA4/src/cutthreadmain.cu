///////////////////////////////////////////////////////////////////////////////
////////////   CUTThrad Implementation  //////////////////////////////////////
/////////////////////   by Eric Li ////////////////////////////////////////////

#include <stdlib.h>
#include<stdio.h>
#include <iostream>

#include"cudafunctions.h"
#include "book.h"
 

/*
  declare struct that contains data ID, grid and block structure, as well as 3 pointers that will identify matrices that the kernel will work on. 
  a and b store input matrices
  c is for the results matrix 
*/ 

struct dataStruct
   {
      int deviceID;
      int gridx;
      int gridy;
      int blocks;
      int partialSize;
      int inArrSize;
      int * a;
      int * b;
      int * c;
      int * partial;
      dataStruct * structPtr;
   };
////////////////////////////////////////////////////////////////////////////   
/*
  This routine is called within the start_threads call. This will be run on all threads, each will call kernel on a seperate GPU.
*/
void* routineM(void* dataSPtr)
   {
      dataStruct *data = (dataStruct*)dataSPtr;
      //this pointer is for the whole struct array, one for each GPU
      dataStruct *wStructPtr = data->structPtr;
      int GPUId = data->deviceID;
      dim3 grid(data->gridx, data->gridy);
      dim3 block(data->blocks);
      int arrDim = data->inArrSize;
      int partialDim = data->partialSize;

      HANDLE_ERROR( cudaSetDevice(GPUId) );
      HANDLE_ERROR( cudaDeviceSynchronize() );
      printf("\n GPU ID: %d", data->deviceID);
      printf("\n GPU ID OF NEIGHBOR: %d", data->structPtr[GPUId+1 % 4].deviceID);
      
      //run matrix mult kernel
      matrixMult<<<grid, block>>>
      ( data->a, data->b, data->partial, arrDim, partialDim);
      HANDLE_ERROR( cudaPeekAtLastError() );
      HANDLE_ERROR( cudaDeviceSynchronize() );

      //print partial
      printf("\n partial results for GPU %d: ", GPUId);
      for(int i=0; i < arrDim; i++){
         //std::cout << std::endl;
         for(int j=0; j < arrDim; j++){
            //std::cout << std::endl;
            for(int k=0; k < partialDim; k++){
               std::cout << 
                  data->partial[(i*arrDim + j)*partialDim + k] << ' ';
            }
         }
      }

      //reduction step
      reduction<<<grid,block>>>(data->partial, data->c, 
                                                   arrDim, partialDim);
      HANDLE_ERROR( cudaPeekAtLastError() );
      HANDLE_ERROR( cudaDeviceSynchronize() );


      //Matrix addition step
      //test for even, sum w/ odd and then store in even 
      if( GPUId % 2 == 0)
         {
            printf("GPU ID %d add with GPUID: %d", GPUId, data->structPtr[GPUId+1 % 4].deviceID);
            matSum<<<grid,block>>>
                  (data->c, data->structPtr[GPUId+1 % 4].c, data->c, arrDim);
            HANDLE_ERROR( cudaPeekAtLastError() );
            HANDLE_ERROR( cudaDeviceSynchronize() );
         }
      
      
      /*helloThere<<<grid,block>>>(data[GPUId])   
      HANDLE_ERROR( cudaDeviceSynchronize() );*/
      return 0;
   }  
////////////// free function prototypes ////////////////////////////////////
void seqMatrixMult(int* in1, int* in2, int* output, int arrDim);
void seqMatrixSum(int* in1, int* in2, int* output, int arrDim );

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char const *argv[])
{
   int numGPU, partialSize, gridx, gridy, bdim;
   int N;
   
   //get number of gpus
   cudaGetDeviceCount(&numGPU);
   //initialize struct for data
   dataStruct *runData = new dataStruct[numGPU];
   //initialize thread array, each thread can be accessed by index
   CUTThread *thread = new CUTThread[numGPU];
   //CUTThread threadId[ MAX_GPU_COUNT];

   //ask user for numbers
   std::cout<< std::endl << "Please enter in matrix dimensions: ";
   std::cin >> N;
   std::cout<< std::endl << "Please enter in grid x dim: ";
   std::cin >> gridx;
   std::cout<< std::endl << "Please enter in grid y dim: ";
   std::cin >> gridy;
   std::cout<< std::endl << "Please enter in block dim: ";
   std::cin >> bdim;

   //calculate padding for reduction, needs to be power of 2
   partialSize = N;
   //if odd, add 1 b
   if( partialSize % 2 != 0 ){
     partialSize+=1;
     }
   //check for power of 2, add 2 until it is power of 2
   while( ceil(log2((float)partialSize-2)) 
                                    != floor(log2((float)partialSize-2)) ){
     partialSize+=2;
   }  


   //allocate unified memory and initialize beginning data
   for(int i=0; i < numGPU; i++){
      HANDLE_ERROR( cudaMallocManaged(&(runData[i].a), N*N*sizeof(int)) );
      HANDLE_ERROR( cudaMallocManaged(&(runData[i].b), N*N*sizeof(int)) );
      HANDLE_ERROR( cudaMallocManaged(&(runData[i].c), N*N*sizeof(int)) );
      HANDLE_ERROR( cudaMallocManaged(&(runData[i].partial), 
                                         N*N*partialSize*sizeof(int)) );

      //set grid and block dimensions based on user response
      runData[i].gridx = gridx;
      runData[i].gridy = gridy;
      runData[i].blocks = bdim;
      //set array sizes
      runData[i].inArrSize = N;
      runData[i].partialSize = partialSize;
      //initiate pointer to this array of dataStruct
      runData[i].structPtr = runData;
      //fill array with data including 0 for result matrix
      for( int j=0; j < N*N; j++){
         runData[i].a[j] = 2;
         runData[i].b[j] = 2;
         runData[i].c[j] = 0;
      }
      //fill partial matrix with zeros
      for(int k=0; k < N*N*partialSize; k++){
         runData[i].partial[k] = 0;
      }
      runData[i].deviceID = i;
      //printf(" /n DEVICE ID FROM HOST: %d", runData[i].deviceID);
   }

   /*//sequential portion
   for(int i=0; i < numGPU; i++){
      seqMatrixMult(runData[i].a, runData[i].b, runData[i].c, 
                                             runData[i].inArrSize);
      seqMatrixSum(runData[i].a, runData[i].b, runData[i].c, 
                                             runData[i].inArrSize);
   }*/
   

   //start threads
   for( int i = 0; i < numGPU; i++){
      thread[ i ] = start_thread(routineM, &runData[i]);
   }

   //end threads
   /*for(int i=0; i < numGPU; i++){
      //end_thread( thread[i]);
      wait_for_threads(thread[i], NULL);
   }*/

   //end threads
   for(int i=0; i < numGPU; i++){
      end_thread( thread[i]);
      
   }
   //destroy threads
   for(int i=0; i < numGPU; i++){
      destroy_thread( thread[i]);
   }

   /*dim3 hgrid(runData[0].gridx);
   //do final summation, this one only needs 1 thread
   matSum<<<hgrid,runData[0].blocks>>>(runData[0].c, runData[2].c, runData[0].c, N );
   HANDLE_ERROR( cudaPeekAtLastError() );
   HANDLE_ERROR( cudaDeviceSynchronize() );*/

  
   //print partial results
   std::cout <<std::endl<< " printing final matrix";
   for(int i=0; i< numGPU; i++){
      std::cout << std::endl;
      for(int j=0; j < N; j++){
         std::cout << std::endl;
         for(int k=0; k < N; k++){
            std::cout << runData[i].c[k] << ' ';
         }
      }
      //printf("\n Result from GPU: %d is %d", i, runData[i].c[0]);
   }


   //free memory
   for(int i=0; i<numGPU; i++){
      cudaFree( runData[i].a );
      cudaFree( runData[i].b );
      cudaFree( runData[i].c );
      cudaFree( runData[i].partial);
   }
   
   return 0;
}

//sequential implementation
void seqMatrixMult(int* in1, int* in2, int* output, int arrDim){
   //loop over column and rows for each element of the output matrix
   for(int i = 0; i < arrDim; i++){
      for(int j = 0; j < arrDim; j++){
         //initialize value of 0 for output matrix element
         output[ i*arrDim + j ] = 0;
         for(int k = 0; k < arrDim; k++){
            output[ i*arrDim + k ]+= in1[ i*arrDim + k ] * in2[ k*arrDim + j ];
         }
      }
   }

}

void seqMatrixSum(int* in1, int* in2, int* output, int arrDim ){
   for(int i = 0; i < arrDim; i++){
      for(int j = 0; j < arrDim; j++){
         output[i*arrDim + j] = in1[i*arrDim + j] + in2[i*arrDim + j];
      }
   }
}