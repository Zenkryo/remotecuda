#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDevicePCIBusId){
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CHECK_CUDA_ERROR(err, "Failed to get device count");
    for(int i = 0; i < deviceCount; i++) {
        char pciBusId[32];
        err = cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), i);
        CHECK_CUDA_ERROR(err, "Failed to get PCI bus ID");
        int device;
        err = cudaDeviceGetByPCIBusId(&device, pciBusId);
        CHECK_CUDA_ERROR(err, "Failed to get device by PCI bus ID");
        ASSERT_EQ(device, i) << "Device ID mismatch";
    }
}
