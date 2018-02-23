///////////////////////////////////////////////////////////////////////////////
//////////////////// kNN implementation main file /////////////////////////////
///////////////////// Written by Eric Li //////////////////////////////////////

//Includes
#include <cstdio>
#include <iostream>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "knn.h"
//define error macro
#define HANDLE_ERROR(func) { GPUAssert((func), __FILE__, __LINE__);}
inline void GPUAssert( cudaError_t errCode, const char *file, int line, bool abort=true)
    {
     if( errCode != cudaSuccess )
         {
          fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(errCode), file, line);
          if (abort) exit(errCode);
         }
    }
//Define compare function used for qsort
int compareFunc( const void *a, const void *b){
  float *x = (float*)a;
  float *y = (float*)b;
  if( *x < *y ) return -1;
  else if(*x > *y) return 1; return 0;
}
///////////////////////////////////////////////////////////////////////////////
//main function
int main(int argc, char const *argv[])
{
   //initialize variables
   //file pointer for reading data from file
   FILE * fp;
   int rows, cols, numEmpty, knnCtr, knnIdx;
   float *inData, *partial, *GPUsortArr, *CPUsortArr;
   float accum, partResult, avg; 
   char* charBuffer;
   char* str;
   char* endlineBuffer;
   size_t len;
   
   
   //ask user for dimension of input data matrix
   std::cout << " Please enter amount of rows desired to read in: ";
   std::cin >> rows;
   
   std::cout << " Please enter amount of columns desired to read in: ";
   std::cin >> cols;

   //declare grid structure
   dim3 grid(32);
   dim3 block((cols+32/32));

   //allocate Unified memory for input data storage
   HANDLE_ERROR( cudaMallocManaged( &inData, rows*cols*sizeof(float)) );
   HANDLE_ERROR( cudaMallocManaged( &partial, rows*cols*sizeof(float)) );
   HANDLE_ERROR( cudaMallocManaged( &GPUsortArr, rows*sizeof(float)) );
   
   //allocate CPU memory
   charBuffer = (char*) malloc(20*sizeof(double));
   endlineBuffer = (char*) malloc(100*sizeof(double));
   CPUsortArr = (float*) malloc(rows*sizeof(float));
   //open file and read in data
   fp = fopen("../src/PA3_nrdc_data.csv", "r");
   
   //test for successful file opening
   if(fp){
      std::cout << std::endl << "Printing buffer vals: ";
      for(int i = 0; i < rows; i++){
         //read in first value, discard and put index i instead as the first column
         getdelim(&charBuffer, &len, ',' ,fp);
         str = strtok( charBuffer, ",");
         inData[ i*cols ] = (float)i;

         //loop over all columns and input value into 1D array
         for(int j = 1; j < cols; j++){
            getdelim(&charBuffer, &len, ',',fp);
            str = strtok( charBuffer, ",");
            inData[ i*cols+j ] = std::strtod(str,NULL);
           }
         //skip until endline  
         getdelim(&endlineBuffer, &len, '\n', fp); 
        }
     }
   //else print error message and exit 
   else{
      std::cout << std::endl << "File opening error, please try again";
      exit(1);    
    }

  //close file 
  fclose(fp); 

   //make some missing values (10%), the first 10% of rows
   numEmpty = (rows <= 10) ? 1: (rows/10);

   for(int i = 0; i < numEmpty; i++){
       inData[ i*rows+1] = -1;
   }   
  //////////////////////////////////////////////////////////////////////////
  //////////////////// sequential Implementation  //////////////////////////
  //make event timing variables
  cudaEvent_t hstart, hend;
  cudaEventCreate(&hstart);
  cudaEventCreate(&hend);
  cudaEventRecord( hstart, 0 );

  //outermost loop is to loop over all rows
  for(int i=0; i < rows; i++){
    //look for columns that are missing value, which is denoted by a -1
    if( inData[ i*cols + 1] == -1 ){
      //loop over all rows again for nearest neighbor calc
      for(int j=0; j < rows; j++){
        //set accumulator to 0. This will store partial results from dist
        accum = 0;
        //This time checking for nonempty rows to calculate the
        if( inData[ j*cols +1 ] != -1){
          //loop over columns and calculate partial distance then sum into
          //accumulator
          for(int k = 2; k < cols; k++){
            partResult = inData[ i*cols + k ] - inData[ j*cols + k ];
            partResult *= partResult;
            accum += partResult;
          }
          //square root accumulator to get distance
          accum = sqrt(accum);
        }
        //store accum value. 0 for rows w/ holes. Distance for other
        CPUsortArr[ j ] = accum;
      }
      //printing CPUsort Arr
      std::cout << "CPUsortArr: ";
      for(int m = 0; m < rows; m++){
        std::cout << CPUsortArr[m] << std::endl; 
      }
      //use qsort from stdlib. 
      qsort(CPUsortArr, rows, sizeof(float), compareFunc);
      //Then find k = 5 nearest neighbors. Average then
      //deposit back into inMat.
      knnCtr = 0;
      knnIdx = 0;
      avg = 0;
      while( knnCtr < 5 && knnIdx < rows ){
        if( CPUsortArr[ knnIdx ] != 0 ){
          avg+=CPUsortArr[ knnIdx ];
          knnCtr++;
        }
        knnIdx++;
      }
      //divide by 5 to get average
      avg /=5;
      //write back into array
      std::cout << std::endl << "Imputed Index: " << i; 
      std::cout << "  Imputed Value: " << avg; 
    }
  }
  //stop timing
  cudaEventRecord( hend, 0 );
  cudaEventSynchronize( hend );
  float cpuTime;
  cudaEventElapsedTime( &cpuTime, hstart, hend);
  //////////////////////////////////////////////////////////////////////////
  /////////////// parallel Implementation /////////////////////////////////   
  //start event timer for GPU parallel implementation 
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord( start, 0 );

  //loop over all rows
  for(int i=0; i < rows; i++){
    //If row needs to be imputed, will execute GPU kernel
    if( inData[ i*cols + 1] == -1){
      /*
        kernel call to knnDist which calculates the distance between the row 
        to be imputed with every other row and returns a partial matrix with 
        distances stored in the second col of each row 
      */  
      knnDist<<<grid,block>>>(inData, partial, i, rows, cols);
      //error checking for kernel call
      HANDLE_ERROR( cudaPeekAtLastError() );
      HANDLE_ERROR( cudaDeviceSynchronize() );

      //this kernel transfers distance into 1D array for sorting on CPU
      distXfer<<<grid,1>>>(partial, GPUsortArr, rows, cols);
      //error checking for kernel call
      HANDLE_ERROR( cudaPeekAtLastError() );
      HANDLE_ERROR( cudaDeviceSynchronize() );
      //print GPU sort array
      std::cout << "GPUsortArr: ";
      for(int m = 0; m < rows; m++){
        std::cout << GPUsortArr[m] << std::endl; 
      }
      //sort array
      qsort(GPUsortArr, rows, sizeof(float), compareFunc);
      //Then find k = 5 nearest neighbors. Average then
      //deposit back into inMat.
      knnCtr = 0;
      knnIdx = 0;
      avg = 0;
      while( knnCtr < 5 && knnIdx < rows ){
        if( GPUsortArr[ knnIdx ] != 0 ){
          avg+=GPUsortArr[ knnIdx ];
          knnCtr++;
        }
        knnIdx++;
      }
      //divide by 5 to get average
      avg /=5;
      //write back into array
      std::cout << std::endl << "GPU Imputed Index: " << i; 
      std::cout << "  GPU Imputed Value: " << avg; 
    }
  }
  cudaEventRecord( end, 0 );
  cudaEventSynchronize( end );

  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, end );

  //print out program stats
  std::cout << "Your program took: " << elapsedTime << " ms." << std::endl;
  std::cout << "The CPU took: " << cpuTime << "ms " << std::endl;

   //free memory
   cudaFree(inData);
   cudaFree(partial);
   cudaFree(GPUsortArr);
   free(charBuffer);
   free(endlineBuffer);
   free(CPUsortArr);
   return 0;
}



