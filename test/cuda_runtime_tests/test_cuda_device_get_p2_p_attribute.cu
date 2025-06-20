#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDeviceGetP2PAttribute){
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CHECK_CUDA_ERROR(err, "Failed to get device count");
    if(deviceCount > 1) {
        int value;
        err = cudaDeviceGetP2PAttribute(&value, cudaDevP2PAttrPerformanceRank, 0, 1);
        if(err == cudaSuccess) {
            SUCCEED() << "P2P attributes retrieved successfully";
        } else if(err == cudaErrorInvalidDevice) {
            SUCCEED() << "P2P not supported between these devices";
        } else {
            CHECK_CUDA_ERROR(err, "Failed to get P2P attribute");
        }
    } else {
        SUCCEED() << "Skipping test - requires multiple devices";
    }
}
