#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDeviceFlushGPUDirectRDMAWrites){
    cudaError_t err = cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice, cudaFlushGPUDirectRDMAWritesToOwner);
    if(err != cudaErrorNotSupported) {
        CHECK_CUDA_ERROR(err, "Failed to flush GPU Direct RDMA writes");
    }
}
