#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaThreadSynchronize){
    cudaError_t err = cudaThreadSynchronize();
    CHECK_CUDA_ERROR(err, "Failed to synchronize thread");
}
