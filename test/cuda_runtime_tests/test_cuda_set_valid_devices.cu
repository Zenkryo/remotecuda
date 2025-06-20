#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaSetValidDevices){
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CHECK_CUDA_ERROR(err, "Failed to get device count");
    if(deviceCount > 0) {
        int *devices = new int[deviceCount];
        for(int i = 0; i < deviceCount; i++) {
            devices[i] = i;
        }
        err = cudaSetValidDevices(devices, deviceCount);
        if(err != cudaErrorNotSupported) {
            CHECK_CUDA_ERROR(err, "Failed to set valid devices");
        }
        delete[] devices;
    }
}
