#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDeviceGraphMemAttributes){
    cudaError_t err;
    int deviceCount;
    err = cudaGetDeviceCount(&deviceCount);
    CHECK_CUDA_ERROR(err, "Failed to get device count");
    for(int device = 0; device < deviceCount; ++device) {
        cudaSetDevice(device);
        int computeCapabilityMajor;
        int computeCapabilityMinor;
        int maxThreadsPerBlock;
        int sharedMemPerBlock;
        int maxThreadsPerMultiProcessor;
        int multiProcessorCount;
        int maxGridSize[3];
        int maxThreadsDim[3];
        int warpSize;
        err = cudaDeviceGetAttribute(&computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device);
        CHECK_CUDA_ERROR(err, "Failed to get compute capability major");
        err = cudaDeviceGetAttribute(&computeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device);
        CHECK_CUDA_ERROR(err, "Failed to get compute capability minor");
        err = cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device);
        CHECK_CUDA_ERROR(err, "Failed to get max threads per block");
        err = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device);
        CHECK_CUDA_ERROR(err, "Failed to get warp size");
        err = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_CUDA_ERROR(err, "Failed to get multi processor count");
        err = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device);
        CHECK_CUDA_ERROR(err, "Failed to get max threads per multi processor");
        err = cudaDeviceGetAttribute(&maxGridSize[0], cudaDevAttrMaxGridDimX, device);
        CHECK_CUDA_ERROR(err, "Failed to get max grid size x");
        err = cudaDeviceGetAttribute(&maxGridSize[2], cudaDevAttrMaxGridDimZ, device);
        CHECK_CUDA_ERROR(err, "Failed to get max grid size z");
        err = cudaDeviceGetAttribute(&maxThreadsDim[0], cudaDevAttrMaxBlockDimX, device);
        CHECK_CUDA_ERROR(err, "Failed to get max block dim x");
        err = cudaDeviceGetAttribute(&maxThreadsDim[1], cudaDevAttrMaxBlockDimY, device);
        CHECK_CUDA_ERROR(err, "Failed to get max block dim y");
        err = cudaDeviceGetAttribute(&maxThreadsDim[2], cudaDevAttrMaxBlockDimZ, device);
        CHECK_CUDA_ERROR(err, "Failed to get max block dim z");
        err = cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);
        CHECK_CUDA_ERROR(err, "Failed to get shared memory per block");
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, device);
        CHECK_CUDA_ERROR(err, "Failed to get device properties");
        cudaMemPool_t memPool;
        cudaError_t poolErr = cudaDeviceGetDefaultMemPool(&memPool, device);
        if(poolErr != cudaSuccess) {
            GTEST_SKIP() << "Default memory pool not supported on this device";
        }
        uint64_t poolLowWatermark = 1024 * 1024 * 512; // 512MB
        cudaError_t attrErr = cudaDeviceSetGraphMemAttribute(device, cudaGraphMemAttrReservedMemCurrent, &poolLowWatermark);
        if(attrErr == cudaSuccess) {
            uint64_t retrievedLowWatermark;
            err = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrReservedMemCurrent, &retrievedLowWatermark);
            CHECK_CUDA_ERROR(err, "Failed to get graph memory attribute");
            uint64_t usedMem;
            uint64_t reservedMem;
            err = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrUsedMemCurrent, &usedMem);
            CHECK_CUDA_ERROR(err, "Failed to get graph memory attribute");
            err = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrReservedMemCurrent, &reservedMem);
            CHECK_CUDA_ERROR(err, "Failed to get graph memory attribute");
        } else {
            GTEST_SKIP() << "Graph memory attributes not supported on this device";
        }
    }
    err = cudaDeviceReset();
    CHECK_CUDA_ERROR(err, "Failed to reset device");
}
