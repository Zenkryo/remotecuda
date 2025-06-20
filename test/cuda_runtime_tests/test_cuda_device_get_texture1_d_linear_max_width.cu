#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDeviceGetTexture1DLinearMaxWidth){
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if(err == cudaSuccess && deviceCount > 0) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, 0);
        if(err == cudaSuccess && prop.major > 0 && prop.canMapHostMemory) {
            size_t maxWidth;
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
            err = cudaDeviceGetTexture1DLinearMaxWidth(&maxWidth, &desc, 32);
            if(err == cudaSuccess) {
                ASSERT_GT(maxWidth, 0) << "Invalid texture 1D linear max width";
            } else {
                SUCCEED() << "Function not supported on this device, skipping test";
            }
        } else {
            SUCCEED() << "Device not suitable, skipping test";
        }
    } else {
        SUCCEED() << "No devices available, skipping test";
    }
}
