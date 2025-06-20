#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemset){
    void *devPtr;
    cudaError_t err = cudaMalloc(&devPtr, 1024);
    CHECK_CUDA_ERROR(err, "Failed to allocate device memory");
    err = cudaMemset(devPtr, 0x42, 1024);
    CHECK_CUDA_ERROR(err, "Failed to set device memory");
    char *hostPtr = new char[1024];
    err = cudaMemcpy(hostPtr, devPtr, 1024, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err, "Failed to copy from device to host");
    for(int i = 0; i < 1024; i++) {
        ASSERT_EQ(hostPtr[i], 0x42) << "Memory not set correctly at index " << i;
    }
    delete[] hostPtr;
    err = cudaFree(devPtr);
    CHECK_CUDA_ERROR(err, "Failed to free device memory");
}
