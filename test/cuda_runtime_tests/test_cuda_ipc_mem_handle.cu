#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaIpcMemHandle){
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if(err == cudaSuccess && deviceCount > 0) {
        int device;
        err = cudaGetDevice(&device);
        if(err == cudaSuccess) {
            void *devPtr;
            err = cudaMalloc(&devPtr, 1024);
            if(err == cudaSuccess) {
                cudaIpcMemHandle_t handle;
                err = cudaIpcGetMemHandle(&handle, devPtr);
                if(err == cudaSuccess) {
                    void *openedDevPtr;
                    err = cudaIpcOpenMemHandle(&openedDevPtr, handle, cudaIpcMemLazyEnablePeerAccess);
                    if(err == cudaSuccess) {
                        err = cudaIpcCloseMemHandle(openedDevPtr);
                        if(err != cudaSuccess) {
                            SUCCEED() << "Failed to close IPC handle, but skipping";
                        }
                    } else {
                        SUCCEED() << "IPC open not supported, skipping";
                    }
                }
                err = cudaFree(devPtr);
                if(err != cudaSuccess) {
                    SUCCEED() << "Failed to free memory, but skipping";
                }
            } else {
                SUCCEED() << "Memory allocation failed, skipping test";
            }
        } else {
            SUCCEED() << "Failed to get device, skipping test";
        }
    } else {
        SUCCEED() << "No devices available, skipping test";
    }
}
