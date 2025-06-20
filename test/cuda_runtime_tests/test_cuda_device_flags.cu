#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDeviceFlags){
    unsigned int flags;
    cudaError_t err = cudaGetDeviceFlags(&flags);
    if(err == cudaSuccess) {
        unsigned int flagCombinations[] = {cudaDeviceScheduleAuto, cudaDeviceScheduleSpin, cudaDeviceScheduleYield, cudaDeviceScheduleBlockingSync, cudaDeviceMapHost, cudaDeviceLmemResizeToMax};
        for(auto newFlags : flagCombinations) {
            err = cudaSetDeviceFlags(newFlags);
            if(err == cudaSuccess) {
                unsigned int currentFlags;
                err = cudaGetDeviceFlags(&currentFlags);
                if(err == cudaSuccess) {
                    ASSERT_EQ(currentFlags & newFlags, newFlags) << "Device flags not set correctly";
                }
            } else {
                if(err == cudaErrorInvalidValue) {
                    SUCCEED() << "Device flags not supported, skipping test";
                }
            }
        }
        err = cudaSetDeviceFlags(flags);
        if(err != cudaSuccess) {
            SUCCEED() << "Failed to restore flags, but skipping";
        }
    } else {
        SUCCEED() << "Failed to get device flags, skipping test";
    }
}
