#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
    do {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
        cudaError_t err = call;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
        if(err != cudaSuccess) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                                                                                                                                                                                                                                                                                                                                                                                                                 \
            exit(EXIT_FAILURE);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
        }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
    } while(0)

int main() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    printf("Found %d CUDA device(s)\n", deviceCount);

    for(int device = 0; device < deviceCount; ++device) {
        cudaSetDevice(device);

        // Get device attributes
        int computeCapabilityMajor;
        int computeCapabilityMinor;
        int maxThreadsPerBlock;
        size_t totalGlobalMem;
        int sharedMemPerBlock;
        int maxThreadsPerMultiProcessor;
        int multiProcessorCount;
        int maxGridSize[3];
        int maxThreadsDim[3];
        int warpSize;

        CUDA_CHECK(cudaDeviceGetAttribute(&computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&computeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxGridSize[0], cudaDevAttrMaxGridDimX, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxGridSize[1], cudaDevAttrMaxGridDimY, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxGridSize[2], cudaDevAttrMaxGridDimZ, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsDim[0], cudaDevAttrMaxBlockDimX, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsDim[1], cudaDevAttrMaxBlockDimY, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsDim[2], cudaDevAttrMaxBlockDimZ, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        totalGlobalMem = prop.totalGlobalMem;

        printf("\nDevice %d:\n", device);
        printf("  Compute Capability: %d.%d\n", computeCapabilityMajor, computeCapabilityMinor);
        printf("  Device Name: %s\n", prop.name);
        printf("  Total Global Memory: %lu bytes\n", (unsigned long)totalGlobalMem);
        printf("  Shared Memory per Block: %lu bytes\n", (unsigned long)sharedMemPerBlock);
        printf("  Max Threads per Block: %d\n", maxThreadsPerBlock);
        printf("  Warp Size: %d\n", warpSize);
        printf("  Number of Multiprocessors: %d\n", multiProcessorCount);
        printf("  Max Threads per Multiprocessor: %d\n", maxThreadsPerMultiProcessor);
        printf("  Max Grid Size: [%d, %d, %d]\n", maxGridSize[0], maxGridSize[1], maxGridSize[2]);
        printf("  Max Thread Block Size: [%d, %d, %d]\n", maxThreadsDim[0], maxThreadsDim[1], maxThreadsDim[2]);
        printf("  Clock Rate: %.2f GHz\n", prop.clockRate * 1e-6f);
        printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate * 1e-6f);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 Cache Size: %d bytes\n", prop.l2CacheSize);

        // Get default memory pool
        cudaMemPool_t memPool;
        cudaError_t poolErr = cudaDeviceGetDefaultMemPool(&memPool, device);
        if(poolErr == cudaSuccess) {
            printf("  Default Memory Pool retrieved successfully\n");
        } else {
            printf("  Warning: Default Memory Pool not supported on this device\n");
        }

        // Set and get graph memory attributes
        uint64_t poolLowWatermark = 1024 * 1024 * 512; // 512MB
        cudaError_t attrErr = cudaDeviceSetGraphMemAttribute(device, cudaGraphMemAttrReservedMemCurrent, &poolLowWatermark);

        if(attrErr == cudaSuccess) {
            uint64_t retrievedLowWatermark;
            CUDA_CHECK(cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrReservedMemCurrent, &retrievedLowWatermark));

            printf("  Graph Memory Low Watermark: Set to %lu bytes, Retrieved %lu bytes\n", (unsigned long)poolLowWatermark, (unsigned long)retrievedLowWatermark);

            // Get used and reserved memory attributes
            uint64_t usedMem;
            uint64_t reservedMem;
            CUDA_CHECK(cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrUsedMemCurrent, &usedMem));
            CUDA_CHECK(cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrReservedMemCurrent, &reservedMem));

            printf("  Current Used Graph Memory: %lu bytes\n", (unsigned long)usedMem);
            printf("  Current Reserved Graph Memory: %lu bytes\n", (unsigned long)reservedMem);
        } else {
            printf("  Warning: Graph Memory Attributes not supported on this device\n");
        }
    }

    // Reset device
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
