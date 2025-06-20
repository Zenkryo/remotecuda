#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaArray){
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    cudaError_t err = cudaMallocArray(&array, &channelDesc, 32, 32);
    CHECK_CUDA_ERROR(err, "Failed to allocate array");
    ASSERT_NE(array, nullptr);
    err = cudaFreeArray(array);
    CHECK_CUDA_ERROR(err, "Failed to free array");
}
