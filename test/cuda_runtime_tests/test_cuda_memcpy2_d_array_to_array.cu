#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemcpy2DArrayToArray){
    const int width = 32;
    const int height = 32;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t srcArray, dstArray;
    cudaError_t err = cudaMallocArray(&srcArray, &channelDesc, width, height);
    CHECK_CUDA_ERROR(err, "Failed to allocate source array");
    err = cudaMallocArray(&dstArray, &channelDesc, width, height);
    CHECK_CUDA_ERROR(err, "Failed to allocate destination array");
    float *hostData = new float[width * height];
    for(int i = 0; i < width * height; i++) {
        hostData[i] = static_cast<float>(i);
    }
    err = cudaMemcpy2DToArray(srcArray, 0, 0, hostData, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err, "Failed to copy from host to source array");
    err = cudaMemcpy2DArrayToArray(dstArray, 0, 0, srcArray, 0, 0, width * sizeof(float), height, cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR(err, "Failed to copy from source array to destination array");
    delete[] hostData;
    err = cudaFreeArray(srcArray);
    CHECK_CUDA_ERROR(err, "Failed to free source array");
    err = cudaFreeArray(dstArray);
    CHECK_CUDA_ERROR(err, "Failed to free destination array");
}
