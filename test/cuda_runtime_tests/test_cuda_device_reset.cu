#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDeviceReset){
    cudaError_t err = cudaDeviceReset();
    CHECK_CUDA_ERROR(err, "Failed to reset device");
}
