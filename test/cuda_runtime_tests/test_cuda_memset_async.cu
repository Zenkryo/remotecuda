#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemsetAsync){
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    void *devPtr;
    err = cudaMalloc(&devPtr, 1024);
    CHECK_CUDA_ERROR(err, "Failed to allocate device memory");
    err = cudaMemsetAsync(devPtr, 0x42, 1024, stream);
    CHECK_CUDA_ERROR(err, "Failed to set device memory asynchronously");
    size_t pitch;
    void *devPtr2D;
    err = cudaMallocPitch(&devPtr2D, &pitch, 32, 32);
    CHECK_CUDA_ERROR(err, "Failed to allocate pitched device memory");
    err = cudaMemset2DAsync(devPtr2D, pitch, 0x42, 32, 32, stream);
    CHECK_CUDA_ERROR(err, "Failed to set 2D device memory asynchronously");
    cudaPitchedPtr devPtr3D;
    cudaExtent extent = make_cudaExtent(32, 32, 32);
    err = cudaMalloc3D(&devPtr3D, extent);
    CHECK_CUDA_ERROR(err, "Failed to allocate 3D device memory");
    err = cudaMemset3DAsync(devPtr3D, 0x42, extent, stream);
    CHECK_CUDA_ERROR(err, "Failed to set 3D device memory asynchronously");
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA_ERROR(err, "Failed to synchronize stream");
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
    err = cudaFree(devPtr);
    CHECK_CUDA_ERROR(err, "Failed to free device memory");
    err = cudaFree(devPtr2D);
    CHECK_CUDA_ERROR(err, "Failed to free 2D device memory");
    err = cudaFree(devPtr3D.ptr);
    CHECK_CUDA_ERROR(err, "Failed to free 3D device memory");
}
