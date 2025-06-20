#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemPrefetchAsync){
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    void *devPtr;
    err = cudaMalloc(&devPtr, 1024);
    CHECK_CUDA_ERROR(err, "Failed to allocate device memory");
    err = cudaMemPrefetchAsync(devPtr, 1024, 0, stream);
    if(err != cudaSuccess) {
        SUCCEED() << "Memory prefetch not supported on this device, skipping test";
    }
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
    err = cudaFree(devPtr);
    CHECK_CUDA_ERROR(err, "Failed to free device memory");
}
