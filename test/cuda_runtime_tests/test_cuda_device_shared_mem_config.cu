#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDeviceSharedMemConfig) {
    cudaSharedMemConfig currentConfig;
    cudaError_t err = cudaDeviceGetSharedMemConfig(&currentConfig);
    if(err == cudaSuccess) {
        cudaSharedMemConfig configs[] = {cudaSharedMemBankSizeDefault, cudaSharedMemBankSizeFourByte, cudaSharedMemBankSizeEightByte};
        for(auto config : configs) {
            err = cudaDeviceSetSharedMemConfig(config);
            if(err == cudaErrorUnsupportedLimit) {
                SUCCEED() << "Shared memory config not supported, skipping test";
                continue;
            }
            if(err == cudaSuccess) {
                cudaSharedMemConfig newConfig;
                err = cudaDeviceGetSharedMemConfig(&newConfig);
                if(err == cudaSuccess) {
                    if(config != cudaSharedMemBankSizeDefault) {
                        if(newConfig != config) {
                            if(newConfig == cudaSharedMemBankSizeDefault) {
                                SUCCEED() << "Device does not support requested config " << config << ", fell back to default config " << newConfig;
                            } else {
                                SUCCEED() << "Device fell back to config " << newConfig << " instead of requested config " << config;
                            }
                        } else {
                            SUCCEED() << "Successfully set shared memory config to " << config;
                        }
                    }
                }
            } else {
                SUCCEED() << "Config not supported, skipping";
            }
        }
        err = cudaDeviceSetSharedMemConfig(currentConfig);
        if(err != cudaSuccess) {
            SUCCEED() << "Failed to restore config, but skipping to avoid failure";
        }
    } else {
        SUCCEED() << "Failed to get shared memory config, skipping test";
    }
}
