#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDeviceSynchronize){
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err, "Failed to synchronize device");
}
