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
      int totalGPUs;
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
      int GPUId = data->deviceID;
      dim3 grid(data->gridx, data->gridy);
      dim3 block(data->blocks);
      int arrDim = data->inArrSize;
      int partialDim = data->partialSize;

      HANDLE_ERROR( cudaSetDevice(GPUId) );
      HANDLE_ERROR( cudaDeviceSynchronize() );
      //printf("\n GPU ID: %d", data->deviceID);
      //printf("\n GPU ID OF NEIGHBOR: %d", data->structPtr[GPUId+1 % 4].deviceID);
      
      while( GPUId < 4 ){
            //run matrix mult kernel
         matrixMult<<<grid, block>>>
         ( data->a, data->b, data->partial, arrDim, partialDim);
         HANDLE_ERROR( cudaPeekAtLastError() );
         HANDLE_ERROR( cudaDeviceSynchronize() );

        /* //print partial
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
         }*/

         //reduction step
         reduction<<<grid,block>>>(data->partial, data->c, 
                                                      arrDim, partialDim);
         HANDLE_ERROR( cudaPeekAtLastError() );
         HANDLE_ERROR( cudaDeviceSynchronize() );
      }
      
      return 0;
   } 
/*
   Routine for thread launch of matrix addition
*/
void* routineAdd(void* dataSPtr )
   {
      dataStruct *data = (dataStruct*)dataSPtr;
      //dataStruct *wStructPtr = data->structPtr;
      int GPUId = data->deviceID;
      int arrDim = data->inArrSize;
      //printf("GPU ID %d add with GPUID: %d \n", GPUId, data->structPtr[GPUId+2].deviceID);
      //print array b4 summing
     /* for(int i=0; i < arrDim; i++){
       std::cout << std::endl;
         for(int k=0; k < arrDim; k++ ){
            std::cout << data->structPtr[GPUId+2].c[i*arrDim+k];
         }
      }*/

      matSum<<<data->gridx,data->blocks>>>
            (data->c, data->structPtr[GPUId+2].c, arrDim);
      HANDLE_ERROR( cudaPeekAtLastError() );
      HANDLE_ERROR( cudaDeviceSynchronize() );
      /*//print matrix after sum

       for(int i=0; i < arrDim; i++){
         std::cout << std::endl;
         for(int k=0; k < arrDim; k++ ){
            std::cout << data->c[i*arrDim+k];
         }
      }
      std::cout << std::endl;*/
      return 0;
   }    
////////////// free function prototypes ////////////////////////////////////
void seqMatrixMult(int* in1, int* in2, int* output, int arrDim);
void seqMatrixSum(int* in1, int* in2, int arrDim );

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char const *argv[])
{
   int numGPU, partialSize, gridx, gridy, bdim;
   int N;
   
   //get number of gpus
   cudaGetDeviceCount(&numGPU);
   //initialize struct for data
   dataStruct *runData = new dataStruct[numGPU];
   dataStruct *CPUData = new dataStruct[numGPU];
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
   for(int i=0; i < 4; i++){
      HANDLE_ERROR( cudaMallocManaged(&(runData[i].a), N*N*sizeof(int)) );
      HANDLE_ERROR( cudaMallocManaged(&(runData[i].b), N*N*sizeof(int)) );
      HANDLE_ERROR( cudaMallocManaged(&(runData[i].c), N*N*sizeof(int)) );
      HANDLE_ERROR( cudaMallocManaged(&(runData[i].partial), 
                                         N*N*partialSize*sizeof(int)) );

      //set number of gpus total
      runData[i].totalGPUs = numGPU;
      //set grid and block dimensions based on user response
      runData[i].gridx = gridx;
      runData[i].gridy = gridy;
      runData[i].blocks = bdim;
      //set array sizes
      runData[i].inArrSize = N;
      runData[i].partialSize = partialSize;
      //initiate pointer to this array of dataStruct
      runData[i].structPtr = runData;

      //allocate CPU memory for sequential implementation
      CPUData[i].a = (int*) malloc(N*N*sizeof(int));
      CPUData[i].b = (int*) malloc(N*N*sizeof(int));
      CPUData[i].c = (int*) malloc(N*N*sizeof(int));
      //set array size
      CPUData[i].inArrSize = N;

      //fill array with data including 0 for result matrix
      for( int j=0; j < N*N; j++){
         //GPU data
         runData[i].a[j] = 2;
         runData[i].b[j] = 2;
         runData[i].c[j] = 0;
         //CPU data
         CPUData[i].a[j] = 2;
         CPUData[i].b[j] = 2;
         CPUData[i].c[j] = 0;
      }
 
      //fill partial matrix with zeros
      for(int k=0; k < N*N*partialSize; k++){
         runData[i].partial[k] = 0;
      }
      //set deviceID based on how many GPU there are 
      runData[i].deviceID = i % numGPU;  
      //printf(" /n DEVICE ID FROM HOST: %d", runData[i].deviceID);
   }

/////////////////  Sequential Implementation  ///////////////////////////////
   //make event timing variables for Sequential implementation
   cudaEvent_t hstart, hend;
   cudaEventCreate(&hstart);
   cudaEventCreate(&hend);
   cudaEventRecord( hstart, 0 );

   //sequential portion
   //sequential matrix mult
   for(int i=0; i < numGPU; i++)
      {
         seqMatrixMult(CPUData[i].a, CPUData[i].b, CPUData[i].c, 
                                             CPUData[i].inArrSize);
      }
   //sequential matrix addition  
   for(int i=0; i < numGPU / 2; i++)
      {
         seqMatrixSum(CPUData[i].c, CPUData[i+2].c, CPUData[i].inArrSize);
      }
   //last step of addition   
   seqMatrixSum(CPUData[0].c, CPUData[1].c, CPUData[0].inArrSize);

   //stop timing
   cudaEventRecord( hend, 0 );
   cudaEventSynchronize( hend );
   float cpuTime;
   cudaEventElapsedTime( &cpuTime, hstart, hend);

   /*//print CPU results
   std::cout <<std::endl<< " printing CPU matrix";
   
   std::cout << std::endl;
   for(int j=0; j < N; j++){
      std::cout << std::endl;
      for(int k=0; k < N; k++){
         std::cout << CPUData[0].c[j*N + k] << ' ';
      }
   }*/


////////////////////  GPU Implementation  //////////////////////////////////   
   //start event timer for GPU parallel implementation 
   cudaEvent_t start, end;
   cudaEventCreate(&start);
   cudaEventCreate(&end);
   cudaEventRecord( start, 0 );
   int x = 0;
   //loop over all data points, this is for cases where numGPU < 4
   
          //start threads for matrix multiplication
         for( int i = 0; i < numGPU; i++){
            thread[ i % numGPU ] = start_thread(routineM, &runData[ i ]);
         }
         //end threads
         for(int i=0; i < numGPU; i++){
            end_thread( thread[i % numGPU ]);  
         }
         //destroy threads
         for(int i=0; i < numGPU; i++){
            destroy_thread( thread[i % numGPU ]);
         }
         //increment to next data offsetsz
         x++;
   
  
 
   //start thread for addition
   for( int i = 0; i < 4/ 2; i++){
      thread[ i % numGPU ] = start_thread(routineAdd, &runData[i]);
      //end threads
      for(int i=0; i < numGPU / 2; i++){
         end_thread( thread[i]);    
      }
      //destroy threads
      for(int i=0; i < numGPU / 2; i++){
         destroy_thread( thread[i]);
      }   
   }
   
   //dim3 hgrid(runData[0].gridx);
   //do final summation, this one only needs 1 thread
   matSum<<<runData[0].gridx,runData[0].blocks>>>
                                       (runData[0].c, runData[1].c, N );
   HANDLE_ERROR( cudaPeekAtLastError() );
   HANDLE_ERROR( cudaDeviceSynchronize() );

  
   /*//print results
   std::cout <<std::endl<< " printing GPU matrix";
   
   std::cout << std::endl;
   for(int j=0; j < N; j++){
      std::cout << std::endl;
      for(int k=0; k < N; k++){
         std::cout << runData[0].c[j*N + k] << ' ';
      }
   }*/
      //printf("\n Result from GPU: %d is %d", i, runData[i].c[0]);
   
   //stop GPU timing
   cudaEventRecord( end, 0 );
   cudaEventSynchronize( end );
   float elapsedTime;
   cudaEventElapsedTime( &elapsedTime, start, end );

   //print out program stats
  std::cout << std::endl << "Your program took: " << elapsedTime << " ms." 
                                                                << std::endl;
  std::cout << "The CPU took: " << cpuTime << "ms " << std::endl;

   //free memory
   for(int i=0; i<numGPU; i++){
      cudaFree( runData[i].a );
      cudaFree( runData[i].b );
      cudaFree( runData[i].c );
      cudaFree( runData[i].partial);

      free(CPUData[i].a);
      free(CPUData[i].b);
      free(CPUData[i].c);
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
            output[ i*arrDim + j ]+= in1[ i*arrDim + k ] * in2[ k*arrDim + j ];
         }
      }
   }

}

void seqMatrixSum(int* in1, int* in2, int arrDim ){
   for(int i = 0; i < arrDim; i++){
      for(int j = 0; j < arrDim; j++){
         in1[i*arrDim + j] += in2[i*arrDim + j];
      }
   }
}