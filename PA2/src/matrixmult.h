/*
  This header demonstrates how we build cuda programs spanning
  multiple files. 
 */

#ifndef MATRIXMULT_H_
#define MATRIXMULT_H_

// This is the number of elements we want to process.


// This is the declaration of the function that will execute on the GPU.
__global__ void matrixMult(int *, int *, int *, int n);


#endif // MATRIXMULT_H_
