#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaChooseDevice){
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CHECK_CUDA_ERROR(err, "Failed to get device count");
    ASSERT_GT(deviceCount, 0) << "No CUDA devices found";
    int currentDevice;
    err = cudaGetDevice(&currentDevice);
    CHECK_CUDA_ERROR(err, "Failed to get current device");
    for(int i = 0; i < deviceCount; i++) {
        err = cudaSetDevice(i);
        CHECK_CUDA_ERROR(err, "Failed to set device");
        int newDevice;
        err = cudaGetDevice(&newDevice);
        CHECK_CUDA_ERROR(err, "Failed to verify device setting");
        ASSERT_EQ(newDevice, i) << "Device not set correctly";
    }
    err = cudaSetDevice(currentDevice);
    CHECK_CUDA_ERROR(err, "Failed to restore original device");
}
