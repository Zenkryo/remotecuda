#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemcpyAsync){
    const int size = 1024;
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    float *hostSrc = new float[size];
    float *hostDst = new float[size];
    float *devPtr;
    err = cudaMalloc(&devPtr, size * sizeof(float));
    CHECK_CUDA_ERROR(err, "Failed to allocate device memory");
    for(int i = 0; i < size; i++) {
        hostSrc[i] = static_cast<float>(i);
    }
    err = cudaMemcpyAsync(devPtr, hostSrc, size * sizeof(float), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA_ERROR(err, "Failed to copy from host to device");
    err = cudaMemcpyAsync(hostDst, devPtr, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    CHECK_CUDA_ERROR(err, "Failed to copy from device to host");
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA_ERROR(err, "Failed to synchronize stream");
    for(int i = 0; i < size; i++) {
        ASSERT_EQ(hostDst[i], hostSrc[i]) << "Memory copy failed at index " << i;
    }
    delete[] hostSrc;
    delete[] hostDst;
    err = cudaFree(devPtr);
    CHECK_CUDA_ERROR(err, "Failed to free device memory");
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
}
