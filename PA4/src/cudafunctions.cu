#include <stdio>
#include <cstdlib>


//extern "C" cudaHello(){}

__global__ void helloThere(int rank){
    printf("\n Hello From GPU: %d", rank);
}
