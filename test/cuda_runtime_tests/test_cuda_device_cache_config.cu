#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDeviceCacheConfig){
    cudaFuncCache currentConfig;
    cudaError_t err = cudaDeviceGetCacheConfig(&currentConfig);
    CHECK_CUDA_ERROR(err, "Failed to get cache config");
    cudaFuncCache configs[] = {cudaFuncCachePreferNone, cudaFuncCachePreferShared, cudaFuncCachePreferL1, cudaFuncCachePreferEqual};
    for(auto config : configs) {
        err = cudaDeviceSetCacheConfig(config);
        CHECK_CUDA_ERROR(err, "Failed to set cache config");
        cudaFuncCache newConfig;
        err = cudaDeviceGetCacheConfig(&newConfig);
        CHECK_CUDA_ERROR(err, "Failed to verify cache config");
        ASSERT_EQ(newConfig, config) << "Cache config not set correctly";
    }
    err = cudaDeviceSetCacheConfig(currentConfig);
    CHECK_CUDA_ERROR(err, "Failed to restore cache config");
}
