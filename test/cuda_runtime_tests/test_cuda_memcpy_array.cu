#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemcpyArray){
    const int size = 1024;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    cudaError_t err = cudaMallocArray(&array, &channelDesc, size);
    CHECK_CUDA_ERROR(err, "Failed to allocate CUDA array");
    float *hostSrc = new float[size];
    float *hostDst = new float[size];
    for(int i = 0; i < size; i++) {
        hostSrc[i] = static_cast<float>(i);
    }
    err = cudaMemcpyToArray(array, 0, 0, hostSrc, size * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err, "Failed to copy from host to array");
    err = cudaMemcpyFromArray(hostDst, array, 0, 0, size * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err, "Failed to copy from array to host");
    for(int i = 0; i < size; i++) {
        ASSERT_EQ(hostDst[i], hostSrc[i]) << "Memory copy failed at index " << i;
    }
    delete[] hostSrc;
    delete[] hostDst;
    err = cudaFreeArray(array);
    CHECK_CUDA_ERROR(err, "Failed to free CUDA array");
}
