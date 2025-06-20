#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaEventRecord){
    cudaEvent_t event;
    cudaStream_t stream;
    cudaError_t err;
    err = cudaEventCreate(&event);
    CHECK_CUDA_ERROR(err, "Failed to create event");
    err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    test_kernel<<<1, 1, 0, stream>>>();
    err = cudaEventRecord(event, stream);
    CHECK_CUDA_ERROR(err, "Failed to record event");
    err = cudaEventQuery(event);
    if(err == cudaSuccess) {
        SUCCEED() << "Event completed faster than expected, but this is acceptable";
    } else {
        ASSERT_EQ(err, cudaErrorNotReady) << "Unexpected error from cudaEventQuery";
    }
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA_ERROR(err, "Failed to synchronize stream");
    err = cudaEventQuery(event);
    ASSERT_EQ(err, cudaSuccess) << "Event should be complete after stream synchronization";
    err = cudaEventDestroy(event);
    CHECK_CUDA_ERROR(err, "Failed to destroy event");
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
}
