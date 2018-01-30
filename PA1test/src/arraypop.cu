#include <iostream>


#define N 2

//CUDA error handler provided by text
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
};

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void add(int *a, int *b, int *c);

__global__ void matpop(int N, int* );







///////////////////// main function /////////////////////////////////////

int main(int argc, char const *argv[])
{
   int *dev_a;
   int host_a[N][N];
   //allocate memory


   HANDLE_ERROR( cudaMalloc( (void**) &dev_a, N * N * sizeof(int)));

   //initialize matrix on device using parallel and copy over
   matpop<<<N,N>>>(N, dev_a);

   cudaMemcpy(host_a, dev_a, N * N * sizeof(int), cudaMemcpyDeviceToHost);

   printf( "\n Matrix: \n");
   //prints matrix elements
    for (int i = 0; i < N; i++){

          for (int j = 0; j < N; j++){
	  
	      printf ("%i ", *((host_a) + (i * N + j)));

	      }

      printf ("\n");
    }
   
	return 0;
}




//////////////////// function declarations ////////////////////////////

//populates a matrix on device
__global__ void matpop( int N, int *emptyMatrix ){
  
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  //check for valid memory location, then initialize element to 0
  if( thread_id < N * N )
  {
    //commented out one is for array of pointers
    //*((*(emptyMatrix)) + (blockId.x * blockDim.x + threadIdx.x)) = 0;
    *(emptyMatrix + (blockIdx.x * blockDim.x + threadIdx.x)) = 0;
  }
  
}

__global__ void add(int *a, int *b, int *c) {

  /*
    Each thread knows its identity in the system. This identity is
    made available in code via indices blockIdx and threadIdx. We
    write blockIdx.x because block indices are multidimensional. In
    this case, we have linear arrays of data, so we only need one
    dimension. If this doesn't make sense, don't worry - the important
    thing is that the first step in the function is converting the
    thread's indentity into an index into the data.
   */
  int thread_id = blockIdx.x;

  /*
    We make sure that the thread_id isn't too large, and then we
    assign c = a + b using the index we calculated above.

    The big picture is that each thread is responsible for adding one
    element from a and one element from b. Each thread is able to run
    in parallel, so we get speedup.
   */
  if (thread_id < N) {
    c[thread_id] = a[thread_id] + b[thread_id];
  }
}

