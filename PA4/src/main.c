#include <mpi.h>

int main(int argc, char const *argv[])
{
   int rank, size;
   MPI_Init(argc, argv);

   //get number of MPI processes and rank 
   MPI_comm_rank(MPI_COMM_WORLD, rank);
   MPI_comm_size(MPI_COMM_WORLD, size);

   //code to run on each process
   cudaGetDeviceProp();

   MPI_Finalize();
   return 0;
}