#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemPoolAttributes){
    cudaMemPool_t memPool = nullptr;
    cudaMemPoolProps poolProps = {};
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.location.id = 0; // Current device
    poolProps.location.type = cudaMemLocationTypeDevice;
    cudaError_t err = cudaMemPoolCreate(&memPool, &poolProps);
    if(err == cudaErrorNotSupported) {
        GTEST_SKIP() << "Memory pools not supported on this device";
    }
    CHECK_CUDA_ERROR(err, "Failed to create memory pool");
    uint64_t releaseThreshold = 1024 * 1024; // 1MB
    err = cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold);
    if(err == cudaErrorNotSupported) {
        GTEST_SKIP() << "Memory pool attributes not supported on this device";
    }
    CHECK_CUDA_ERROR(err, "Failed to set release threshold");
    uint64_t retrievedThreshold;
    err = cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &retrievedThreshold);
    CHECK_CUDA_ERROR(err, "Failed to get release threshold");
    ASSERT_EQ(retrievedThreshold, releaseThreshold) << "Release threshold not set correctly";
    if(memPool != nullptr) {
        cudaMemPoolDestroy(memPool);
    }
}
