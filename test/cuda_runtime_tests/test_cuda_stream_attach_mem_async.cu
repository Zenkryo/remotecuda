#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaStreamAttachMemAsync){
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    void *ptr;
    err = cudaMallocManaged(&ptr, 1024, cudaMemAttachGlobal);
    if(err == cudaSuccess) {
        err = cudaStreamAttachMemAsync(stream, ptr, 0, cudaMemAttachGlobal);
        CHECK_CUDA_ERROR(err, "Failed to attach memory to stream");
        err = cudaFree(ptr);
        CHECK_CUDA_ERROR(err, "Failed to free managed memory");
    } else {
        SUCCEED() << "Managed memory not supported, skipping test";
    }
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
}
