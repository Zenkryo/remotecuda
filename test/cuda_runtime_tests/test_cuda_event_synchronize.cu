#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaEventSynchronize){
    cudaEvent_t start, stop;
    cudaStream_t stream;
    cudaError_t err;
    err = cudaEventCreate(&start);
    CHECK_CUDA_ERROR(err, "Failed to create start event");
    err = cudaEventCreate(&stop);
    CHECK_CUDA_ERROR(err, "Failed to create stop event");
    err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    err = cudaEventRecord(start, stream);
    CHECK_CUDA_ERROR(err, "Failed to record start event");
    test_kernel<<<1, 1, 0, stream>>>();
    err = cudaEventRecord(stop, stream);
    CHECK_CUDA_ERROR(err, "Failed to record stop event");
    err = cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR(err, "Failed to synchronize stop event");
    float elapsedTime;
    err = cudaEventElapsedTime(&elapsedTime, start, stop);
    CHECK_CUDA_ERROR(err, "Failed to get elapsed time");
    ASSERT_GT(elapsedTime, 0.0f) << "Invalid elapsed time";
    err = cudaEventDestroy(start);
    CHECK_CUDA_ERROR(err, "Failed to destroy start event");
    err = cudaEventDestroy(stop);
    CHECK_CUDA_ERROR(err, "Failed to destroy stop event");
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
}
