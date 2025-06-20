#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaThreadCacheConfig){
    cudaFuncCache currentConfig;
    cudaError_t err = cudaThreadGetCacheConfig(&currentConfig);
    CHECK_CUDA_ERROR(err, "Failed to get cache config");
    cudaFuncCache configs[] = {cudaFuncCachePreferNone, cudaFuncCachePreferShared, cudaFuncCachePreferL1, cudaFuncCachePreferEqual};
    for(auto config : configs) {
        err = cudaThreadSetCacheConfig(config);
        CHECK_CUDA_ERROR(err, "Failed to set cache config");
        cudaFuncCache newConfig;
        err = cudaThreadGetCacheConfig(&newConfig);
        CHECK_CUDA_ERROR(err, "Failed to verify cache config");
        ASSERT_EQ(newConfig, config) << "Cache config not set correctly";
    }
    err = cudaThreadSetCacheConfig(currentConfig);
    CHECK_CUDA_ERROR(err, "Failed to restore cache config");
}
