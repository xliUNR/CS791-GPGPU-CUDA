#ifndef CUDAFUNCTIONS_H_
#define CUDAFUNCTIONS_H_

#include <stdio.h>
#include <stdlib.h>


__global__ void helloThere(int rank, int*, int*, int*);

#endif