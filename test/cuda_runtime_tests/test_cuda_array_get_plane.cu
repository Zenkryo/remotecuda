#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaArrayGetPlane){
    const int width = 4;
    const int height = 4;
    const int numChannels = 4;
    const int channelSize = width * height;
    const int totalSize = channelSize * numChannels;
    unsigned char h_data[totalSize];
    for(int i = 0; i < channelSize; i++) {
        h_data[i + 0 * channelSize] = (unsigned char)(i % 256);         // R channel
        h_data[i + 1 * channelSize] = (unsigned char)((i + 64) % 256);  // G channel
        h_data[i + 2 * channelSize] = (unsigned char)((i + 128) % 256); // B channel
        h_data[i + 3 * channelSize] = (unsigned char)((i + 192) % 256); // A channel
    }
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray;
    cudaError_t err = cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArrayDefault);
    CHECK_CUDA_ERROR(err, "Failed to allocate CUDA array");
    err = cudaMemcpy2DToArray(cuArray, 0, 0, h_data, width * numChannels, width * numChannels, height, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err, "Failed to copy data to CUDA array");
    unsigned char *d_linear;
    err = cudaMalloc(&d_linear, totalSize);
    CHECK_CUDA_ERROR(err, "Failed to allocate linear device memory");
    err = cudaMemcpy2DFromArray(d_linear, width * numChannels, cuArray, 0, 0, width * numChannels, height, cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR(err, "Failed to copy array to linear memory");
    unsigned char *d_channel;
    err = cudaMalloc(&d_channel, channelSize);
    CHECK_CUDA_ERROR(err, "Failed to allocate device memory for channel");
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    extractChannelKernel<<<gridDim, blockDim>>>(d_linear, d_channel, width, height, 0);
    err = cudaGetLastError();
    CHECK_CUDA_ERROR(err, "Kernel launch failed");
    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err, "Failed to synchronize device");
    unsigned char h_channel[channelSize];
    err = cudaMemcpy(h_channel, d_channel, channelSize, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err, "Failed to copy extracted channel back to host");
    err = cudaFree(d_channel);
    CHECK_CUDA_ERROR(err, "Failed to free device memory for channel");
    err = cudaFree(d_linear);
    CHECK_CUDA_ERROR(err, "Failed to free linear device memory");
    err = cudaFreeArray(cuArray);
    CHECK_CUDA_ERROR(err, "Failed to free CUDA array");
}
