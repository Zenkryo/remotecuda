#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemPoolAccess){
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
    int deviceCount;
    err = cudaGetDeviceCount(&deviceCount);
    CHECK_CUDA_ERROR(err, "Failed to get device count");
    if(deviceCount > 1) {
        cudaMemAccessDesc accessDesc = {};
        accessDesc.location.type = cudaMemLocationTypeDevice;
        accessDesc.location.id = 1;
        accessDesc.flags = cudaMemAccessFlagsProtReadWrite;
        err = cudaMemPoolSetAccess(memPool, &accessDesc, 1);
        if(err == cudaErrorNotSupported) {
            GTEST_SKIP() << "Memory pool access not supported on this device";
        }
        CHECK_CUDA_ERROR(err, "Failed to set memory pool access");
        cudaMemAccessFlags accessFlags;
        err = cudaMemPoolGetAccess(&accessFlags, memPool, &accessDesc.location);
        CHECK_CUDA_ERROR(err, "Failed to get memory pool access");
        ASSERT_EQ(accessFlags, cudaMemAccessFlagsProtReadWrite) << "Access flags not set correctly";
    }
    if(memPool != nullptr) {
        cudaMemPoolDestroy(memPool);
    }
}
