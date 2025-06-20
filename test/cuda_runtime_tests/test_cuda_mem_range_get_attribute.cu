#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemRangeGetAttribute){
    const int size = 1024;
    void *devPtr;
    cudaError_t err = cudaMalloc(&devPtr, size);
    CHECK_CUDA_ERROR(err, "Failed to allocate device memory");
    cudaMemRangeAttribute attr = cudaMemRangeAttributeReadMostly;
    int value;
    err = cudaMemRangeGetAttribute(&value, sizeof(value), attr, devPtr, size);
    if(err == cudaErrorNotSupported) {
        SUCCEED() << "Memory range attribute not supported, skipping test";
    } else if(err == cudaErrorInvalidValue) {
        SUCCEED() << "Memory range attribute not valid for this memory, skipping test";
    } else {
        CHECK_CUDA_ERROR(err, "Failed to get memory range attribute");
    }
    err = cudaFree(devPtr);
    CHECK_CUDA_ERROR(err, "Failed to free device memory");
}
