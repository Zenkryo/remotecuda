#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaStreamSynchronize){
    cudaStream_t stream;
    cudaError_t err;
    err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    test_kernel<<<1, 1, 0, stream>>>();
    err = cudaStreamQuery(stream);
    if(err == cudaSuccess) {
        SUCCEED() << "Stream completed faster than expected, but this is acceptable";
    } else {
        ASSERT_EQ(err, cudaErrorNotReady) << "Unexpected error from cudaStreamQuery";
    }
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA_ERROR(err, "Failed to synchronize stream");
    err = cudaStreamQuery(stream);
    ASSERT_EQ(err, cudaSuccess) << "Stream should be complete after synchronization";
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
}
