#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaEventDestroy){
    cudaEvent_t event;
    cudaError_t err = cudaEventCreate(&event);
    CHECK_CUDA_ERROR(err, "Failed to create event");
    err = cudaEventDestroy(event);
    CHECK_CUDA_ERROR(err, "Failed to destroy event");
}
