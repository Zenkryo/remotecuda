#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemPoolCreate){
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
    ASSERT_NE(memPool, nullptr);
    err = cudaMemPoolDestroy(memPool);
    CHECK_CUDA_ERROR(err, "Failed to destroy memory pool");
}
