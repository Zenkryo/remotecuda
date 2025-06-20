#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaPointerGetAttributes){
    const int size = 1024;
    void *devPtr;
    cudaError_t err = cudaMalloc(&devPtr, size);
    CHECK_CUDA_ERROR(err, "Failed to allocate device memory");
    cudaPointerAttributes attr;
    err = cudaPointerGetAttributes(&attr, devPtr);
    CHECK_CUDA_ERROR(err, "Failed to get pointer attributes");
    ASSERT_EQ(attr.type, cudaMemoryTypeDevice) << "Invalid memory type";
    err = cudaFree(devPtr);
    CHECK_CUDA_ERROR(err, "Failed to free device memory");
}
