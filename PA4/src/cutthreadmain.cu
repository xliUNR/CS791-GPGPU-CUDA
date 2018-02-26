///////////////////////////////////////////////////////////////////////////////
////////////   CUTThrad Implementation  //////////////////////////////////////
/////////////////////   by Eric Li ////////////////////////////////////////////



int main(int argc, char const *argv[])
{
   int numGPU;
   //get number of gpus
   cudaGetDeviceCount(&numGPU);
   
   //initialize threads
   CUTThread *thread = new CUTThread[numGPU];

   CUTThread threadId[ MAX_GPU_COUNT];

   for( gpuIdx = 0; gpuIdx < numGPU; gpuIdx++){
      threadId[ gpuIdx-1 ] = cutStartThread(routine, &dataStruct[gpuIdx-1]);
   }

   
   /* code */
   return 0;
}