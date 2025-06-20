#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGetDeviceInfo){
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CHECK_CUDA_ERROR(err, "Failed to get device count");
    ASSERT_GT(deviceCount, 0) << "No CUDA devices found";
    for(int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        CHECK_CUDA_ERROR(err, "Failed to get device properties");
        ASSERT_GT(prop.major, 0) << "Invalid compute capability major version";
        ASSERT_GE(prop.minor, 0) << "Invalid compute capability minor version";
        ASSERT_GT(prop.totalGlobalMem, 0) << "Invalid total global memory";
        ASSERT_GT(prop.multiProcessorCount, 0) << "Invalid multiprocessor count";
    }
}
