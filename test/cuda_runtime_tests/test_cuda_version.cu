#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaVersion){
    int driverVersion;
    cudaError_t err = cudaDriverGetVersion(&driverVersion);
    CHECK_CUDA_ERROR(err, "Failed to get driver version");
    ASSERT_GT(driverVersion, 0) << "Invalid driver version";
    int runtimeVersion;
    err = cudaRuntimeGetVersion(&runtimeVersion);
    CHECK_CUDA_ERROR(err, "Failed to get runtime version");
    ASSERT_GT(runtimeVersion, 0) << "Invalid runtime version";
}
