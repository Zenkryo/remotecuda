#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaHostMemory){
    void *hostPtr;
    cudaError_t err = cudaMallocHost(&hostPtr, 1024);
    CHECK_CUDA_ERROR(err, "Failed to allocate pinned host memory");
    void *devPtr;
    err = cudaHostGetDevicePointer(&devPtr, hostPtr, 0);
    if(err == cudaSuccess) {
        unsigned int flags;
        err = cudaHostGetFlags(&flags, hostPtr);
        CHECK_CUDA_ERROR(err, "Failed to get host memory flags");
        ASSERT_NE(flags, 0) << "Invalid host memory flags";
    } else {
        SUCCEED() << "Host device pointer not supported, skipping test";
    }
    err = cudaFreeHost(hostPtr);
    CHECK_CUDA_ERROR(err, "Failed to free pinned host memory");
}
