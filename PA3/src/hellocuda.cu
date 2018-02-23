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
   FILE * fp;
   int rows, cols, numEmpty, knnCtr, knnIdx;
   float *inData, *partial, *sortArray, *CPUsortArr;
   float accum, partResult, avg; 
   char* buffer;
   char* charBuffer;
   char* str;
   size_t len;
   
   //ask user for dimension of input data matrix
   std::cout << " Please enter amount of rows desired to read in: ";
   std::cin >> rows;
   
   std::cout << " Please enter amount of columns desired to read in: ";
   std::cin >> cols;

   //allocate Unified memory for input data storage
   HANDLE_ERROR( cudaMallocManaged( &inData, rows*cols*sizeof(float)) );
   HANDLE_ERROR( cudaMallocManaged( &partial, rows*cols*sizeof(float)) );
   HANDLE_ERROR( cudaMallocManaged( &sortArray, rows*sizeof(float)) );
   
   //allocate CPU memory
   buffer = (char*) malloc(cols*sizeof(double));
   charBuffer = (char*) malloc(20*sizeof(double));
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
        fclose(fp); 
      }
   }
   //else print error message and exit 
   else{
      std::cout << std::endl << "File opening error, please try again";
      exit(1);
      fclose(fp);
   }


   //make some missing values (10%), the first 10% of rows
   numEmpty = (rows <= 10) ? 1: (rows/10);

   for(int i = 0; i < numEmpty; i++){
       inData[ i*rows+1] = -1;
   }   

    for(int i = 0; i < rows; i++){

      for(int j= 0; j < cols; j++){
        std::cout << inData[ i*cols +j] << ' ';
      }
      std::cout << std::endl;
    }  
      
//////////////////////////////////////////////////////////////////////////
//////////////////// sequential Implementation  //////////////////////////
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
    std::cout << std::endl << 'Imputed Value: ' << avg; 
  }
}

//////////////////////////////////////////////////////////////////////////
/////////////// parallel Implementation  /////////////////////////////////     
//loop over all rows
/*for(int i=0; i < rows; i++){
  if( inData[ i*cols + 2] == -1){

  }
}*/


   //free memory
   cudaFree(inData);
   cudaFree(partial);
   cudaFree(sortArray);
   free(buffer);
   free(charBuffer);
   free(CPUsortArr);
   return 0;
}

///////////////////////////////////////////////////////////////////////////////
////////////////////////  free functions //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
