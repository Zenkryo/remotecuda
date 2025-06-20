#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaIpcEventHandle){
    cudaEvent_t event;
    cudaError_t err = cudaEventCreate(&event);
    CHECK_CUDA_ERROR(err, "Failed to create event");
    cudaIpcEventHandle_t handle;
    err = cudaIpcGetEventHandle(&handle, event);
    if(err == cudaSuccess) { // Only proceed if IPC is supported
        cudaEvent_t openedEvent;
        err = cudaIpcOpenEventHandle(&openedEvent, handle);
        CHECK_CUDA_ERROR(err, "Failed to open IPC event handle");
        err = cudaEventDestroy(openedEvent);
        CHECK_CUDA_ERROR(err, "Failed to destroy opened event");
    }
    err = cudaEventDestroy(event);
    CHECK_CUDA_ERROR(err, "Failed to destroy event");
}
