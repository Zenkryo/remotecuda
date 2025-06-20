#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMallocAsync){
    const int size = 1024;
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    void *devPtr;
    err = cudaMallocAsync(&devPtr, size, stream);
    if(err != cudaErrorNotSupported) {
        CHECK_CUDA_ERROR(err, "Failed to allocate memory asynchronously");
        err = cudaFreeAsync(devPtr, stream);
        CHECK_CUDA_ERROR(err, "Failed to free memory asynchronously");
    }
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
}
