#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDevicePeerAccess){
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CHECK_CUDA_ERROR(err, "Failed to get device count");
    if(deviceCount > 1) {
        int canAccessPeer;
        err = cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
        CHECK_CUDA_ERROR(err, "Failed to check peer access");
        if(canAccessPeer) {
            err = cudaDeviceEnablePeerAccess(1, 0);
            if(err == cudaSuccess) {
                err = cudaDeviceDisablePeerAccess(1);
                CHECK_CUDA_ERROR(err, "Failed to disable peer access");
            }
        }
    }
}
