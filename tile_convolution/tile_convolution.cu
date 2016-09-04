#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define MASK_WIDTH  5
#define MASK_RADIUS MASK_WIDTH/2
#define O_TILE_WIDTH 12 // Size of the tile that will be computed
// The size of the whole block, different from the tile
// MASK_WIDTH-1 because you consider the radius twice 
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH-1) 
#define CHANNEL_RGB 3 // This is assumed by the WP
#define clamp(x,start,end) min(max(x, start), end)

//@@ INSERT CODE HERE
__global__ void applyConvolution(float * inputImage, const float * __restrict__ convKernel, float* outputImage, int inputImageRows, int inputImageColumns, int inputImageChannels,
                                                                                                           int outputImageRows, int outputImageColumns){

  // There are differences to take care for convolutions:
  // Now, there are input and output mappings due to padding
  // why there's padding? to easily deal with convoluting the borders!

  int output_row = threadIdx.y + O_TILE_WIDTH * blockIdx.y;
  int output_col = threadIdx.x + O_TILE_WIDTH * blockIdx.x;

  int input_row = output_row - MASK_RADIUS;
  int input_col = output_col - MASK_RADIUS;

  __shared__ float s_inpImage[BLOCK_WIDTH][BLOCK_WIDTH][CHANNEL_RGB];

  if( (input_row >= 0 && input_row < inputImageRows) && (input_col >= 0 && input_col <= inputImageColumns) )
    for(int channel=0;channel<CHANNEL_RGB;channel++){
      s_inpImage[threadIdx.y][threadIdx.x][channel] = inputImage[input_row*inputImageColumns*CHANNEL_RGB + input_col*CHANNEL_RGB + channel];
    }
  else{
    for(int channel=0;channel<CHANNEL_RGB;channel++){
      s_inpImage[threadIdx.y][threadIdx.x][channel] = 0.0;
    }
  }

  __syncthreads();

  float sum[CHANNEL_RGB];
  for(int channel=0;channel<CHANNEL_RGB;channel++){
    sum[channel] = 0.0;
  }

  if( threadIdx.y < O_TILE_WIDTH && threadIdx.x < O_TILE_WIDTH ){
    for(int channel=0; channel < CHANNEL_RGB; channel++){
      //sum of neighbors in a radius
      for(int y=0;y<MASK_WIDTH;y++){
        for(int x=0;x<MASK_WIDTH;x++){
            sum[channel] += s_inpImage[y+threadIdx.y][x+threadIdx.x][channel]*convKernel[y * MASK_WIDTH + x];
        }
      }
    }

    __syncthreads();
    if( (output_row < outputImageRows) && (output_col < outputImageColumns) ){
      for(int channel=0;channel<CHANNEL_RGB;channel++){
        outputImage[output_row*outputImageColumns*CHANNEL_RGB + output_col*CHANNEL_RGB + channel] = sum[channel];//clamp(sum[channel],0.0,1.0);
      }
    }

  }

} 

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE

    dim3 DimGrid((imageWidth-1)/O_TILE_WIDTH+1,
                 (imageHeight-1)/O_TILE_WIDTH+1,
                 1);
    dim3 DimBlock(BLOCK_WIDTH,BLOCK_WIDTH,1);

    applyConvolution<<<DimGrid,DimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData, imageHeight, imageWidth, imageChannels,
                                                                                                        imageHeight, imageWidth);
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
