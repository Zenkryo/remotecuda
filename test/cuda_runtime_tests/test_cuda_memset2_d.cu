#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemset2D){
    size_t pitch;
    void *devPtr;
    cudaError_t err = cudaMallocPitch(&devPtr, &pitch, 32, 32);
    CHECK_CUDA_ERROR(err, "Failed to allocate pitched device memory");
    err = cudaMemset2D(devPtr, pitch, 0x42, 32, 32);
    CHECK_CUDA_ERROR(err, "Failed to set 2D device memory");
    char *hostPtr = new char[32 * 32];
    err = cudaMemcpy2D(hostPtr, 32, devPtr, pitch, 32, 32, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err, "Failed to copy from device to host");
    for(int i = 0; i < 32 * 32; i++) {
        ASSERT_EQ(hostPtr[i], 0x42) << "Memory not set correctly at index " << i;
    }
    delete[] hostPtr;
    err = cudaFree(devPtr);
    CHECK_CUDA_ERROR(err, "Failed to free device memory");
}
