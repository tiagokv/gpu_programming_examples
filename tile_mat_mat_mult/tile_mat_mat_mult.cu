#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  int n = numAColumns; // or numBRows
  float sum = 0.0;

  for(int i=0;i<(n-1)/TILE_WIDTH+1;i++){

    // the row is within the boundaries and the thread within the tile is loading a value within the matrix
    if( row < numARows && (i*TILE_WIDTH+threadIdx.x) < numAColumns )
      s_A[threadIdx.y][threadIdx.x] = A[row*n + i*TILE_WIDTH + threadIdx.x];
    else
      s_A[threadIdx.y][threadIdx.x] = 0.0;

    //the same mental exercise has to be done, should this line be written? is this tile value ok?
    if( (i*TILE_WIDTH+threadIdx.y) < numBRows && col < numBColumns )
      s_B[threadIdx.y][threadIdx.x] = B[col + (i*TILE_WIDTH+threadIdx.y)*numCColumns];
    else
      s_B[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    // Now, the product is partial, only calculated to what has been loaded from shared memory
    for(int pos = 0; pos < TILE_WIDTH; ++pos){
        sum += s_A[threadIdx.y][pos] * s_B[pos][threadIdx.x];
    }
    __syncthreads();
  }

  if( (row < numCRows) && (col < numCColumns) ){
    C[row*numCColumns + col] = sum;
  }
}

void loadBestLaunchKernelConfig(dim3* dimGrid, dim3* DimBlock){

  

  
}



int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix

  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here

  wbCheck(cudaMalloc((void**) &deviceA, numARows * numAColumns * sizeof(float)));
  wbCheck(cudaMalloc((void**) &deviceB, numBRows * numBColumns * sizeof(float)));
  wbCheck(cudaMalloc((void**) &deviceC, numCRows * numCColumns * sizeof(float)));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  cudaMemcpy(deviceA, hostA , numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB , numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((numCColumns-1)/TILE_WIDTH + 1, 
               (numCRows-1)/TILE_WIDTH + 1, 1); //nro of blocks in grid (at least 1 blocks)
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);     //nro of threads in blocks  

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  matrixMultiplyShared<<<DimGrid,DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns,
                                                                        numBRows, numBColumns,
                                                                        numCRows, numCColumns );

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float) , cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
