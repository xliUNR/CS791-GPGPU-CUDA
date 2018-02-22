//This is the header file for kNN
#ifndef KNN_H_
#define KNN_H_

#include <math.h>

//function declarations
__global__ void kNN( float *, float *, int imputIndex, int rows, int cols);

__global__ void distXfer( float* , float* , int rows, int cols );


#endif KNN_H_