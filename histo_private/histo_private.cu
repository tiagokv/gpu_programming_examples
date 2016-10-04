// Histogram Equalization

#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16
#define CHANNEL_SIZE 3

#define clamp(x,start,end) min(max(x, start), end)
#define correct_color(val, cdf, cdfmin) 
#define rgbToGreyscale(r,g,b) (0.21*r + 0.71*g + 0.07*b)

#define CHECKPOINT //useful for a "unit test like"

//@@ insert code here
__global__ void floatToUnsignedChar(float * vector, unsigned int * unsignedVector,  int width, int height, int channels){
    // I modeled this to be used with blocks with 3d threads, to cover the 3d image space
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = 0; i < channels; ++i)
    {
        unsignedVector[row*width*channels + col*channels + i]  = (unsigned int) (255 * vector[row*width*channels + col*channels + i] );
    }
    
}

__global__ void unsignedCharToFloat(float * vector, unsigned int * unsignedVector, int width, int height, int channels){
    // I modeled this to be used with blocks with 3d threads, to cover the 3d image space
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = 0; i < channels; ++i)
    {
        vector[row*width*channels + col*channels + i] = (float) (unsignedVector[row*width*channels + col*channels + i]/255.0);
    }
}

__global__ void correctColor(unsigned int * unsignedVector, float * cdf, float * mincdf, int width, int height, int channels){
    // I modeled this to be used with blocks with 3d threads, to cover the 3d image space
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if( row < height && col < width){
        for (int i = 0; i < channels; ++i){
            int pos = row*width*channels + col*channels + i;
            unsignedVector[pos] = (unsigned int) clamp(255*(cdf[unsignedVector[pos]] - mincdf[0])/(1-mincdf[0]), 0.0 , 255.0);
        }
    }
}

__global__ void convertToGrayscale(unsigned int * grayScale, unsigned int * unsignedVector, int width, int height, int channels){

    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ float s_image[BLOCK_SIZE][BLOCK_SIZE][CHANNEL_SIZE];

    if( (row >= 0 && row < height) && (col >= 0 && col < width) )
        for(int channel=0; channel < channels; channel++)
            s_image[threadIdx.y][threadIdx.x][channel] = unsignedVector[row*width*channels + col*channels + channel];
    else{
        for(int channel=0; channel < channels; channel++)
            s_image[threadIdx.y][threadIdx.x][channel] = 0.0;
    }

    __syncthreads();

    if( col < width && row < height ){
        grayScale[row*width + col] = (unsigned int) rgbToGreyscale(s_image[threadIdx.y][threadIdx.x][0],
                                                                   s_image[threadIdx.y][threadIdx.x][1],
                                                                   s_image[threadIdx.y][threadIdx.x][2]);
    }

}

__global__ void computeHistogram(unsigned int * image, unsigned int * histogram, int size){

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ unsigned int s_priv_hist[HISTOGRAM_LENGTH];

    if( threadIdx.x < HISTOGRAM_LENGTH ) s_priv_hist[threadIdx.x] = 0;

    __syncthreads();

    int stride = blockDim.x * gridDim.x;
    while( i < size ){
        atomicAdd(&(s_priv_hist[image[i]]), 1);
        i += stride;
    }

    __syncthreads();

    if( threadIdx.x < HISTOGRAM_LENGTH ) atomicAdd(&(histogram[threadIdx.x]), s_priv_hist[threadIdx.x]);

}

__device__ void sum_reduction(int threadPos, float * input){

    // Reduction simulates a binary tree (balanced, actually) hence the 2 by 2 mult.
    // Until block size because it can be a exact multiplier or not
    for(int stride = 1; stride <= HISTOGRAM_LENGTH; stride *= 2){
        int i = (threadPos+1) * stride * 2 - 1;
        if( i < 2 * HISTOGRAM_LENGTH ){ // ensure boundary condition
            input[i] =  input[i] + input[i-stride];
        }
        __syncthreads();
    }
}

__device__ void sum_reverse(int threadPos, float * input){

    //Runs the binary tree from the bottom to the top
    for(int stride = HISTOGRAM_LENGTH/2; stride > 0; stride /= 2){
        __syncthreads();

        int i = (threadPos+1) * stride * 2 - 1;
        if( i + stride < 2 * HISTOGRAM_LENGTH ){
            input[i+stride] = input[i+stride] + input[i];
        }
    }

}

//quick and dirty solution
__global__ void cdf(unsigned int * histogram, float * cdf, int len, int total_size) {

    __shared__ float sh_input[2*HISTOGRAM_LENGTH];

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if( i + threadIdx.x < len ){
        sh_input[threadIdx.x*2] = histogram[i + threadIdx.x] / (float) total_size;
    }else{
        sh_input[threadIdx.x*2] = 0;
    }

    if( i + threadIdx.x + 1 < len ){
        sh_input[threadIdx.x*2+1] = histogram[i + threadIdx.x + 1] / (float) total_size;
    }else{
        sh_input[threadIdx.x*2+1] = 0;
    }

    __syncthreads();

    sum_reduction(threadIdx.x, sh_input);

    sum_reverse(threadIdx.x, sh_input);

    __syncthreads();

    if( i < len ) cdf[i] = sh_input[threadIdx.x];
}

__device__ void min_reduction(int threadPos, float * input){

    // Reduction simulates a binary tree (balanced, actually) hence the 2 by 2 mult.
    // Until block size because it can be a exact multiplier or not
    for(int stride = 1; stride <= HISTOGRAM_LENGTH; stride *= 2){
        int i = (threadPos+1) * stride * 2 - 1;
        if( i < 2 * HISTOGRAM_LENGTH ){ // ensure boundary condition
            input[i] =  min(input[i], input[i-stride]);
        }
        __syncthreads();
    }
}

__device__ void min_reverse(int threadPos, float * input){

    //Runs the binary tree from the bottom to the top
    for(int stride = HISTOGRAM_LENGTH/2; stride > 0; stride /= 2){
        __syncthreads();

        int i = (threadPos+1) * stride * 2 - 1;
        if( i + stride < 2 * HISTOGRAM_LENGTH ){
            input[i+stride] = min(input[i+stride], input[i]);
        }
    }

}

//quick and dirty solution
__global__ void mincdf(float * cdf, float * mincdf, int len) {

    __shared__ float sh_input[2*HISTOGRAM_LENGTH];

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if( i + threadIdx.x < len ){
        sh_input[threadIdx.x*2] = cdf[i + threadIdx.x];
    }else{
        sh_input[threadIdx.x*2] = 0;
    }

    if( i + threadIdx.x + 1 < len ){
        sh_input[threadIdx.x*2+1] = cdf[i + threadIdx.x + 1];
    }else{
        sh_input[threadIdx.x*2+1] = 0;
    }

    __syncthreads();

    min_reduction(threadIdx.x, sh_input);

    min_reverse(threadIdx.x, sh_input);

    __syncthreads();

    if( i < len ) mincdf[i] = sh_input[threadIdx.x];
}


int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here
    unsigned int * deviceHistogram;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    unsigned int * deviceUnsignedOutputImageData;
    unsigned int * deviceGrayImage;
    float * deviceCdf;
    float * deviceMinCdf;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here   
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU memory allocation");
    wbCheck(cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceUnsignedOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned int)));
    wbCheck(cudaMalloc((void **) &deviceGrayImage, imageWidth * imageHeight * sizeof(unsigned int)));
    wbCheck(cudaMalloc((void **) &deviceCdf, HISTOGRAM_LENGTH  * sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceMinCdf, HISTOGRAM_LENGTH  *sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH  * sizeof(unsigned int)));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    //fill with zeroes the histogram
    cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH  *sizeof(float));

    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    dim3 dimGrid((imageWidth-1)/BLOCK_SIZE+1, (imageHeight-1)/BLOCK_SIZE+1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); //Note the 3, this is valid for the copies from one type to others

    // Transform float to unsigned char
    floatToUnsignedChar<<<dimGrid,dimBlock>>>(deviceInputImageData,
                                              deviceUnsignedOutputImageData,
                                              imageWidth, imageHeight, imageChannels);

    #ifdef CHECKPOINT

        unsigned int * hostUnsignedOutputVect = (unsigned int *) malloc(imageWidth * imageHeight * imageChannels * sizeof(unsigned int));
        cudaMemcpy(hostUnsignedOutputVect,
                   deviceUnsignedOutputImageData,
                   imageWidth * imageHeight * imageChannels * sizeof(float),
                   cudaMemcpyDeviceToHost);

        for( int i = 0; i < imageWidth * imageHeight * imageChannels; i++ ){
            unsigned int val = (unsigned int) (255 * hostInputImageData[i] );
            if( val != hostUnsignedOutputVect[i] ){
                wbLog(ERROR, "Failed to analyze float to unsigned int conversion in position ", i);
                return 1;
            }
        }
    #endif

    // // Convert to Grayscale
    convertToGrayscale<<<dimGrid,dimBlock>>>(deviceGrayImage, deviceUnsignedOutputImageData, imageWidth, imageHeight, imageChannels);

    #ifdef CHECKPOINT
        unsigned int * hostGrayImage = (unsigned int *) malloc(imageWidth * imageHeight * sizeof(unsigned int));
        cudaMemcpy(hostGrayImage,
                   deviceGrayImage,
                   imageWidth * imageHeight * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);

        for( int row = 0; row < imageHeight; row++ ){
            for( int col = 0; col < imageWidth; col++ ){
                int pos = row * imageWidth + col;
                unsigned int gray = (unsigned int) (0.21*hostUnsignedOutputVect[3*pos] + 0.71*hostUnsignedOutputVect[3*pos + 1] + 0.07*hostUnsignedOutputVect[3*pos + 2]);

                if( hostGrayImage[pos] != gray ){
                    wbLog(ERROR, "Failed to analyze RGB to grayScale ", pos, "with values: device ", hostGrayImage[pos], " calculated ", gray );
                    return 1;
                }
            }
        }
    #endif

    dimGrid.x = (imageWidth*imageHeight-1)/HISTOGRAM_LENGTH+1;
    dimGrid.y = 1;

    dimBlock.x = HISTOGRAM_LENGTH;
    dimBlock.y = 1;
    //Compute histogram
    computeHistogram<<<dimGrid,dimBlock>>>(deviceGrayImage, deviceHistogram, imageWidth*imageHeight);

    #ifdef CHECKPOINT
        unsigned int * hostHistogram = (unsigned int *) malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));

        cudaMemcpy(hostHistogram,
                   deviceHistogram,
                   HISTOGRAM_LENGTH * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);

        unsigned int * hostHistogramCalculated = (unsigned int *) malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
        for(int i = 0; i < HISTOGRAM_LENGTH; i++) hostHistogramCalculated[i] = 0;

        for( int row = 0; row < imageHeight; row++ ){
            for( int col = 0; col < imageWidth; col++ ){
                int pos = row * imageWidth + col;
                hostHistogramCalculated[hostGrayImage[pos]]++;
            }
        }

        for(int i = 0; i < HISTOGRAM_LENGTH; i++){
            if( hostHistogramCalculated[i] != hostHistogram[i] ){
                wbLog(ERROR, "Failed to analyze histogram ", i );
                return 1;
            }
        }

        free(hostHistogramCalculated);

    #endif

    dimGrid.x = 1;

    // Calculate CDF from the histogram
    cdf<<<dimGrid,dimBlock>>>(deviceHistogram, deviceCdf, HISTOGRAM_LENGTH, imageWidth * imageHeight);   

    #ifdef CHECKPOINT

        float * hostCdf = (float *) malloc(HISTOGRAM_LENGTH * sizeof(float));
        cudaMemcpy(hostCdf,
                   deviceCdf,
                   HISTOGRAM_LENGTH * sizeof(float),
                   cudaMemcpyDeviceToHost);

        float * hostCdfCalculated = (float *) malloc(HISTOGRAM_LENGTH * sizeof(float));
        hostCdfCalculated[0] = hostHistogram[0] / (float) (imageWidth * imageHeight);
        for (int i = 1; i < HISTOGRAM_LENGTH; ++i){
            hostCdfCalculated[i] = hostCdfCalculated[i-1] + hostHistogram[i] / (float)(imageWidth * imageHeight);
        }

        for (int i = 0; i < HISTOGRAM_LENGTH; ++i){
            if( wbUnequalQ(hostCdfCalculated[i],hostCdf[i]) ){
                wbLog(ERROR, "Failed to analyze CDF in position ", i , " real value was ", hostCdfCalculated[i], " but instead got ", hostCdf[i]);
                return 1;
            }
        }

        free(hostCdfCalculated);

    #endif

    // // Compute minimum value of CDF
    mincdf<<<dimGrid,dimBlock>>>(deviceCdf, deviceMinCdf, HISTOGRAM_LENGTH);

    #ifdef CHECKPOINT

        float * hostMinCdf = (float *) malloc(HISTOGRAM_LENGTH * sizeof(float));
        cudaMemcpy(hostMinCdf,
                   deviceMinCdf,
                   HISTOGRAM_LENGTH * sizeof(float),
                   cudaMemcpyDeviceToHost);

        float hostMinCdfCalculated = hostCdf[0];
        for (int i = 1; i < HISTOGRAM_LENGTH; ++i){
            hostMinCdfCalculated = min(hostCdf[i], hostMinCdfCalculated);
        }

        if(wbUnequalQ(hostMinCdfCalculated, hostMinCdf[0])){
            wbLog(ERROR, "Failed to analyze Min CDF ", " real value was ", hostMinCdfCalculated, " but instead got ", hostMinCdf[0]);
            return 1;
        }

    #endif

    dimGrid.x = (imageWidth-1)/BLOCK_SIZE+1;
    dimGrid.y = (imageHeight-1)/BLOCK_SIZE+1;

    dimBlock.x = BLOCK_SIZE;
    dimBlock.y = BLOCK_SIZE;

    correctColor<<<dimGrid,dimBlock>>>(deviceUnsignedOutputImageData, deviceCdf, deviceMinCdf, imageWidth,imageHeight,imageChannels);

    unsignedCharToFloat<<<dimGrid,dimBlock>>>(deviceOutputImageData, deviceUnsignedOutputImageData, imageWidth, imageHeight, imageChannels);
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbSolution(args, outputImage);

    //@@ insert code here

    #ifdef CHECKPOINT

        free(hostUnsignedOutputVect);
        free(hostGrayImage);
        free(hostHistogram);
        free(hostCdf);
        free(hostMinCdf);

    #endif

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceUnsignedOutputImageData);
    cudaFree(deviceCdf);
    cudaFree(deviceMinCdf);
    cudaFree(deviceHistogram);
    cudaFree(deviceGrayImage);

    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

