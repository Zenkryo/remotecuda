#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaStreamCapture){
    cudaError_t err;
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    CHECK_CUDA_ERROR(err, "Failed to begin stream capture");
    cudaStreamCaptureStatus captureStatus;
    unsigned long long graphHandle = 0;
    err = cudaStreamGetCaptureInfo(stream, &captureStatus, &graphHandle);
    CHECK_CUDA_ERROR(err, "Failed to get stream capture info");
    EXPECT_EQ(captureStatus, cudaStreamCaptureStatusActive);
    cudaGraph_t graph = (cudaGraph_t)graphHandle;
    err = cudaStreamEndCapture(stream, &graph);
    CHECK_CUDA_ERROR(err, "Failed to end stream capture");
    if(graph != nullptr) {
        err = cudaGraphDestroy(graph);
        CHECK_CUDA_ERROR(err, "Failed to destroy graph");
    }
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
}
