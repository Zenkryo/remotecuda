#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaThreadExit){
    cudaError_t err = cudaThreadExit();
    CHECK_CUDA_ERROR(err, "Failed to exit thread");
}
