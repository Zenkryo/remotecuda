#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMallocFromPoolAsync){
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
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    void *devPtr;
    err = cudaMallocFromPoolAsync(&devPtr, 1024, memPool, stream);
    if(err == cudaErrorNotSupported) {
        GTEST_SKIP() << "Memory pool allocation not supported on this device";
    }
    CHECK_CUDA_ERROR(err, "Failed to allocate memory from pool");
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA_ERROR(err, "Failed to synchronize stream");
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
    err = cudaFree(devPtr);
    CHECK_CUDA_ERROR(err, "Failed to free device memory");
    err = cudaMemPoolDestroy(memPool);
    CHECK_CUDA_ERROR(err, "Failed to destroy memory pool");
}
