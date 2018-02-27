#include <stdio>
#include <cstdlib>


//extern "C" cudaHello(){}

__global__ void helloThere(int rank, int*a, int*b, int*c){
    int result;
    printf("\n Hello From GPU: %d", rank);
    //do maths
    c[0] = a[0] + b[0];
}
