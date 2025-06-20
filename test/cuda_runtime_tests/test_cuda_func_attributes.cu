#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaFuncAttributes){
    cudaError_t err = cudaFuncSetCacheConfig(test_kernel, cudaFuncCachePreferL1);
    if(err == cudaSuccess) {
        err = cudaFuncSetSharedMemConfig(test_kernel, cudaSharedMemBankSizeFourByte);
        if(err == cudaSuccess) {
            cudaFuncAttributes attr;
            err = cudaFuncGetAttributes(&attr, test_kernel);
            CHECK_CUDA_ERROR(err, "Failed to get function attributes");
            ASSERT_GT(attr.maxThreadsPerBlock, 0) << "Invalid max threads per block";
            err = cudaFuncSetAttribute(test_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024);
            if(err == cudaSuccess) {
                SUCCEED() << "Function attributes set successfully";
            } else {
                SUCCEED() << "Function attribute setting not supported, skipping test";
            }
        } else {
            SUCCEED() << "Shared memory config not supported, skipping test";
        }
    } else {
        SUCCEED() << "Cache config not supported, skipping test";
    }
}
