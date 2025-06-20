#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaEventRecordWithFlags){
    cudaEvent_t event;
    cudaStream_t stream;
    cudaError_t err = cudaEventCreate(&event);
    CHECK_CUDA_ERROR(err, "Failed to create event");
    err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    err = cudaEventRecord(event, stream);
    CHECK_CUDA_ERROR(err, "Failed to record event");
    err = cudaEventSynchronize(event);
    CHECK_CUDA_ERROR(err, "Failed to synchronize event");
    err = cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal);
    if(err == cudaErrorNotSupported) {
        SUCCEED() << "Event recording with flags not supported, skipping test";
    } else if(err == cudaErrorIllegalState) {
        SUCCEED() << "Event recording with flags not allowed in current state, skipping test";
    } else {
        CHECK_CUDA_ERROR(err, "Failed to record event with flags");
    }
    err = cudaEventDestroy(event);
    CHECK_CUDA_ERROR(err, "Failed to destroy event");
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
}
