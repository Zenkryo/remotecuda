#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDeviceGraphMem){
    cudaError_t err = cudaDeviceGraphMemTrim(0);
    CHECK_CUDA_ERROR(err, "Failed to trim graph memory");
    int value;
    err = cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, 0);
    if(err == cudaSuccess) {
        ASSERT_GT(value, 0) << "Invalid max threads per block";
    }
}
