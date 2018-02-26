///////////////////////////////////////////////////////////////////////////////
///////////////////   MultiGPU PA4: MPI implementation ////////////////////////
/////////////////////////     by Eric Li     //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#include <mpi.h>
//#include <cudafunctions.cu>


//define error macro
/*#define HANDLE_ERROR(func) { GPUAssert((func), __FILE__, __LINE__);}
inline void GPUAssert( cudaError_t errCode, const char *file, int line, bool abort=true)
    {
     if( errCode != cudaSuccess )
         {
          fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(errCode), file, line);
          if (abort) exit(errCode);
         }
    }*/



int main(int argc, char const *argv[])
{
   int rank, size;
   MPI_Init(argc, argv);
   dim3 grid = 1;
   dim3 block = 1;

   //get number of MPI processes and rank 
   MPI_comm_rank(MPI_COMM_WORLD, rank);
   MPI_comm_size(MPI_COMM_WORLD, size);
   printf("\n Hello from process %d", rank);
   //code to run on each process
   //HANDLE_ERROR(cudaSetDevice(rank));
   //CUDA KERNEL
   //helloThere<<<grid, block>>>;

   MPI_Finalize();
   return 0;
}