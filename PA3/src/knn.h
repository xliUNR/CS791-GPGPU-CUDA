//This is the header file for kNN
#ifndef KNN_H_
#define KNN_H_

#include <math.h>
#include <cstdio>
#include <iostream>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
//function declarations
__global__ void knnDist( float *, float *, int imputIndex, int rows, int inCols, int pCols);

__global__ void distXfer( float* , float* , int rows, int cols );


#endif 