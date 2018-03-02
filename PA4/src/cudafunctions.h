#ifndef CUDAFUNCTIONS_H_
#define CUDAFUNCTIONS_H_

#include <stdio.h>
#include <stdlib.h>

__global__ void helloThere(int rank, int*, int*, int*);
__global__ void matrixMult(int*, int* , int* , int arrDim, int partialDim);
__global__ void reduction(int* , int*, int arrDim, int partialDim);
__global__ void matSum( int*, int*, int*, int arrDim);

#endif