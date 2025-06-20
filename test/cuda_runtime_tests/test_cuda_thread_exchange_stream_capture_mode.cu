#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaThreadExchangeStreamCaptureMode){
    cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal;
    cudaError_t err = cudaThreadExchangeStreamCaptureMode(&mode);
    if(err != cudaErrorNotSupported) {
        CHECK_CUDA_ERROR(err, "Failed to exchange stream capture mode");
    }
}
