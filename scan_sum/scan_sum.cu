// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
    
__device__ void reduction(int threadPos, float * input){

    // Reduction simulates a binary tree (balanced, actually) hence the 2 by 2 mult.
    // Until block size because it can be a exact multiplier or not
    for(int stride = 1; stride <= BLOCK_SIZE; stride *= 2){
        int i = (threadPos+1) * stride * 2 - 1;
        if( i < 2 * BLOCK_SIZE ){ // ensure boundary condition
            input[i] += input[i-stride];
        }
        __syncthreads();
    }
}

__device__ void reverse(int threadPos, float * input){

    //Runs the binary tree from the bottom to the top
    for(int stride = BLOCK_SIZE/2; stride > 0; stride /= 2){
        __syncthreads();

        int i = (threadPos+1) * stride * 2 - 1;
        if( i + stride < 2 * BLOCK_SIZE ){
            input[i+stride] += input[i];
        }
    }

}

__global__ void scan(float * input, float * output, float * interm, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

    __shared__ float sh_input[2*BLOCK_SIZE];

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if( i + threadIdx.x < len ){
        sh_input[threadIdx.x*2] = input[i + threadIdx.x];
    }else{
        sh_input[threadIdx.x*2] = 0;
    }

    if( i + threadIdx.x + 1 < len ){
        sh_input[threadIdx.x*2+1] = input[i + threadIdx.x + 1];
    }else{
        sh_input[threadIdx.x*2+1] = 0;
    }

    __syncthreads();

    reduction(threadIdx.x, sh_input);

    reverse(threadIdx.x, sh_input);

    __syncthreads();

    if( i < len ) output[i] = sh_input[threadIdx.x];

    //Store the last number
    //Maybe this could lead to a control divergence, not clear
    if( interm != NULL && threadIdx.x == blockDim.x - 1 ) interm[blockIdx.x] = sh_input[threadIdx.x];
}

__global__ void sumByBlocks(float * sum, float* values, int length){
    int i = threadIdx.x + blockDim.x * (blockIdx.x+1);
    if(i < length) sum[i] += values[blockIdx.x];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list
    float * intermScan;
    float * intermScanOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    int quantity_blocks = (numElements-1)/BLOCK_SIZE + 1;

    wbCheck(cudaMalloc((void**)&intermScan, quantity_blocks*sizeof(float)));
    wbCheck(cudaMalloc((void**)&intermScanOutput, quantity_blocks*sizeof(float)));

    dim3 DimGrid(quantity_blocks, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);                   

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    scan<<<DimGrid,DimBlock>>>(deviceInput,deviceOutput,intermScan,numElements);

    scan<<<DimGrid,DimBlock>>>(intermScan,intermScanOutput, NULL, quantity_blocks);

    DimGrid.x -= 1;
    sumByBlocks<<<DimGrid,DimBlock>>>(deviceOutput,intermScanOutput,numElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

