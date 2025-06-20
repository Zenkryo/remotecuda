#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemGetInfo){
    size_t free, total;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    CHECK_CUDA_ERROR(err, "Failed to get memory info");
    ASSERT_GT(total, 0) << "Invalid total memory";
    ASSERT_LE(free, total) << "Free memory exceeds total memory";
}
