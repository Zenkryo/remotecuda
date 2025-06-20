#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaStreamCreate){
    cudaStream_t stream;
    cudaError_t err;
    err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    CHECK_CUDA_ERROR(err, "Failed to create stream with flags");
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
    int leastPriority, greatestPriority;
    err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    CHECK_CUDA_ERROR(err, "Failed to get stream priority range");
    err = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority);
    CHECK_CUDA_ERROR(err, "Failed to create stream with priority");
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
}
