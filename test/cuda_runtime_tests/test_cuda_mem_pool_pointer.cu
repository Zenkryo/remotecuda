#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemPoolPointer){
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
    void *devPtr;
    err = cudaMallocFromPoolAsync(&devPtr, 1024, memPool, 0);
    if(err == cudaErrorNotSupported) {
        GTEST_SKIP() << "Memory pool allocation not supported on this device";
    }
    CHECK_CUDA_ERROR(err, "Failed to allocate memory from pool");
    cudaMemPoolPtrExportData exportData;
    err = cudaMemPoolExportPointer(&exportData, devPtr);
    if(err == cudaSuccess) {
        void *importedPtr;
        err = cudaMemPoolImportPointer(&importedPtr, memPool, &exportData);
        if(err == cudaSuccess) {
            ASSERT_EQ(importedPtr, devPtr) << "Imported pointer does not match original pointer";
        }
    }
    err = cudaFree(devPtr);
    CHECK_CUDA_ERROR(err, "Failed to free device memory");
    err = cudaMemPoolDestroy(memPool);
    CHECK_CUDA_ERROR(err, "Failed to destroy memory pool");
}
