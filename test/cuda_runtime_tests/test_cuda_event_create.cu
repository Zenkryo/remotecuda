#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaEventCreate){
    cudaEvent_t event;
    cudaError_t err;
    err = cudaEventCreate(&event);
    CHECK_CUDA_ERROR(err, "Failed to create event");
    err = cudaEventDestroy(event);
    CHECK_CUDA_ERROR(err, "Failed to destroy event");
    err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    CHECK_CUDA_ERROR(err, "Failed to create event with flags");
    err = cudaEventDestroy(event);
    CHECK_CUDA_ERROR(err, "Failed to destroy event");
}
