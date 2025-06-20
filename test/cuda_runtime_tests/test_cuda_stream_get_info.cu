#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaStreamGetInfo){
    cudaStream_t stream;
    cudaError_t err;
    int leastPriority, greatestPriority;
    err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    CHECK_CUDA_ERROR(err, "Failed to get stream priority range");
    err = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority);
    CHECK_CUDA_ERROR(err, "Failed to create stream with priority");
    int priority;
    err = cudaStreamGetPriority(stream, &priority);
    CHECK_CUDA_ERROR(err, "Failed to get stream priority");
    ASSERT_EQ(priority, greatestPriority) << "Stream priority not set correctly";
    unsigned int flags;
    err = cudaStreamGetFlags(stream, &flags);
    CHECK_CUDA_ERROR(err, "Failed to get stream flags");
    ASSERT_EQ(flags, cudaStreamNonBlocking) << "Stream flags not set correctly";
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
}
