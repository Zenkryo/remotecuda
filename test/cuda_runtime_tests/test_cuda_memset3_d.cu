#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemset3D){
    cudaPitchedPtr devPtr;
    cudaExtent extent = make_cudaExtent(32, 32, 32);
    cudaError_t err = cudaMalloc3D(&devPtr, extent);
    CHECK_CUDA_ERROR(err, "Failed to allocate 3D device memory");
    err = cudaMemset3D(devPtr, 0x42, extent);
    CHECK_CUDA_ERROR(err, "Failed to set 3D device memory");
    err = cudaFree(devPtr.ptr);
    CHECK_CUDA_ERROR(err, "Failed to free device memory");
}
