///////////////////////////////////////////////////////////////////////////////
//////////////////// kNN implementation main file /////////////////////////////
///////////////////// Written by Eric Li //////////////////////////////////////

//Includes
#include <cstdio>
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

//main function
int main(int argc, char const *argv[])
{
   //initialize variables
   FILE * fp;
   int rows, cols;
   float *inData, *partial, *sortArray; 
   char* buffer;
   char* str;
   //ask user for dimension of input data matrix
   std::cout << " Please enter amount of rows desired to read in: ";
   std::cin >> rows;
   
   std::cout << " Please enter amount of columns desired to read in: ";
   std::cin >> cols;

   //allocate Unified memory for input data storage
   HANDLE_ERROR( cudaMallocManaged( &inData, row*cols*sizeof(float)) );
   HANDLE_ERROR( cudaMallocManaged( &partial, row*cols*sizeof(float)) );
   HANDLE_ERROR( cudaMallocManaged( &sortArray, rows*sizeof(float)) );
   
   //allocate memory for read buffer
   malloc(&buffer, rows*sizeof(float));
   //open file and read in data
   fp = fopen("PA3_nrdc_data.csv", "r");
   fgets(buffer, rows*sizeof(float), fp);
   str = strtok(buffer, " ,");
   std::cout << std::endl << "This is the string printed: " << str ;
   //test for successful file opening
   /*if(fp){
      for(int i = 0; i < rows; i++){
         fgets(buffer, rows*sizeof(float), fp);

         for(int j = 0; j < cols; j++){

         }
      }
   }*/

   else{
      std::cout << std::endl << "File opening error, please try again";
   }
   //read in data from file




   //free memory
   cudaFree(inData);
   cudaFree(partial);
   cudaFree(sortArray);
   free(buffer);
   return 0;
}