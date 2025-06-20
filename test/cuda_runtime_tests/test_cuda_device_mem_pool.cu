#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDeviceMemPool){
    cudaMemPool_t defaultPool;
    cudaError_t err = cudaDeviceGetDefaultMemPool(&defaultPool, 0);
    if(err == cudaErrorNotSupported) {
        GTEST_SKIP() << "Memory pools not supported on this device";
    }
    CHECK_CUDA_ERROR(err, "Failed to get default memory pool");
    ASSERT_NE(defaultPool, nullptr);
    cudaMemPool_t newPool = nullptr;
    cudaMemPoolProps poolProps = {};
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.location.id = 0;
    poolProps.location.type = cudaMemLocationTypeDevice;
    err = cudaMemPoolCreate(&newPool, &poolProps);
    CHECK_CUDA_ERROR(err, "Failed to create memory pool");
    ASSERT_NE(newPool, nullptr);
    err = cudaDeviceSetMemPool(0, newPool);
    CHECK_CUDA_ERROR(err, "Failed to set memory pool");
    cudaMemPool_t currentPool;
    err = cudaDeviceGetMemPool(&currentPool, 0);
    CHECK_CUDA_ERROR(err, "Failed to get current memory pool");
    ASSERT_EQ(currentPool, newPool) << "Memory pool not set correctly";
    err = cudaDeviceSetMemPool(0, defaultPool);
    CHECK_CUDA_ERROR(err, "Failed to restore default memory pool");
    err = cudaMemPoolDestroy(newPool);
    CHECK_CUDA_ERROR(err, "Failed to destroy memory pool");
}
