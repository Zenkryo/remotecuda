#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaStreamWaitEvent){
    cudaStream_t stream;
    cudaEvent_t event;
    cudaError_t err;
    err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    err = cudaEventCreate(&event);
    CHECK_CUDA_ERROR(err, "Failed to create event");
    err = cudaEventRecord(event, stream);
    CHECK_CUDA_ERROR(err, "Failed to record event");
    err = cudaStreamWaitEvent(stream, event, 0);
    CHECK_CUDA_ERROR(err, "Failed to wait for event");
    err = cudaEventDestroy(event);
    CHECK_CUDA_ERROR(err, "Failed to destroy event");
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
}
