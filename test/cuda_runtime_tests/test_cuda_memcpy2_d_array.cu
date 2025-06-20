#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemcpy2DArray){
    const int width = 32;
    const int height = 32;
    const int pitch = width * sizeof(float);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    cudaError_t err = cudaMallocArray(&array, &channelDesc, width, height);
    CHECK_CUDA_ERROR(err, "Failed to allocate CUDA array");
    float *hostSrc = new float[width * height];
    float *hostDst = new float[width * height];
    for(int i = 0; i < width * height; i++) {
        hostSrc[i] = static_cast<float>(i);
    }
    err = cudaMemcpy2DToArray(array, 0, 0, hostSrc, pitch, width * sizeof(float), height, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err, "Failed to copy from host to array");
    err = cudaMemcpy2DFromArray(hostDst, pitch, array, 0, 0, width * sizeof(float), height, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err, "Failed to copy from array to host");
    for(int i = 0; i < width * height; i++) {
        ASSERT_EQ(hostDst[i], hostSrc[i]) << "Memory copy failed at index " << i;
    }
    delete[] hostSrc;
    delete[] hostDst;
    err = cudaFreeArray(array);
    CHECK_CUDA_ERROR(err, "Failed to free CUDA array");
}
