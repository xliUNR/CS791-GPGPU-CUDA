///////////////////////////////////////////////////////////////////////////////
////////////   CUTThrad Implementation  //////////////////////////////////////
/////////////////////   by Eric Li ////////////////////////////////////////////

#include <stdlib.h>
#include<stdio.h>


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
      int blocks;
      int * a;
      int * b;
      int * c;
      int * partial;
   };
/*
  This routine is called within the start_threads call. This will be run on all threads, each will call kernel on a seperate GPU.
*/
void* routine(void* dataSPtr)
   {
      dataStruct *data = (dataStruct*)dataSPtr;
      cudaSetDevice(data->deviceID);
      //run kernel?
      helloThere<<<1,1>>>(data->deviceID, data->a, data->b, data->c);
      return 0;
   }

int main(int argc, char const *argv[])
{
   int numGPU, numGPUthreads;
   int N = 1;
   //get number of gpus
   cudaGetDeviceCount(&numGPU);
   //initialize struct for data
   dataStruct *runData = new dataStruct[numGPU];
   //initialize thread array, each thread can be accessed by index
   CUTThread *thread = new CUTThread[numGPU];
   //CUTThread threadId[ MAX_GPU_COUNT];


   std::cout<< "Please enter in matrix dimensions: ";
   std::cin >> N;
   //calculate padding for reduction, needs to be power of 2
   numGPUthreads = N;
   //if odd, add 1 b
   if( numGPUthreads % 2 != 0 ){
     numGPUthreads+=1;
     }
   //check for power of 2, add 2 until it is power of 2
   while( ceil(log2((float)numGPUthreads-2)) 
                                    != floor(log2((float)numGPUthreads-2)) ){
     numGPUthreads+=2;
   }  


   //allocate unified memory and initialize beginning data
   for(int i=0; i < numGPU; i++){
      HANDLE_ERROR( cudaMallocManaged(&(runData[i].a), N*N*sizeof(int)) );
      HANDLE_ERROR( cudaMallocManaged(&(runData[i].b), N*N*sizeof(int)) );
      HANDLE_ERROR( cudaMallocManaged(&(runData[i].c), N*N*sizeof(int)) );
      HANDLE_ERROR( cudaMallocManaged(&(runData[i].partial), 
                                         N*N*numGPUthreads*sizeof(int)) );

      //fill array with data including 0 for result matrix
      for( int j=0; j < N*N; j++){
         runData[i].a[j] = 1;
         runData[i].b[j] = 1;
         runData[i].c[j] = 0;
      }
      //fill partial matrix with zeros
      for(int k=0; k < N*N*numGPUthreads; k++){
         runData[i].partial[k] = 0;
      }
      runData[i].deviceID = i;
   }

   //start threads
   for( int i = 0; i < numGPU; i++){
      thread[ i ] = start_thread(routine, &runData[i]);
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

   //end threads
   for(int i=0; i < numGPU; i++){
      destroy_thread( thread[i]);
   }
   //print results
   for(int i=0; i< numGPU; i++){
      printf("\n Result from GPU: %d is %d", i, runData[i].c[0]);
   }


   //free memory
   for(int i=0; i<numGPU; i++){
      cudaFree( runData[i].a );
      cudaFree( runData[i].b );
      cudaFree( runData[i].c );
      cudaFree( runData[i].partial);
   }
   /* code */
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