#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGetChannelDesc){
    const int width = 32;
    const int height = 32;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    cudaError_t err = cudaMallocArray(&array, &channelDesc, width, height);
    CHECK_CUDA_ERROR(err, "Failed to allocate CUDA array");
    cudaChannelFormatDesc retrievedDesc;
    err = cudaGetChannelDesc(&retrievedDesc, array);
    CHECK_CUDA_ERROR(err, "Failed to get channel description");
    ASSERT_EQ(retrievedDesc.x, channelDesc.x) << "Channel format mismatch";
    ASSERT_EQ(retrievedDesc.y, channelDesc.y) << "Channel format mismatch";
    ASSERT_EQ(retrievedDesc.z, channelDesc.z) << "Channel format mismatch";
    ASSERT_EQ(retrievedDesc.w, channelDesc.w) << "Channel format mismatch";
    ASSERT_EQ(retrievedDesc.f, channelDesc.f) << "Channel format mismatch";
    err = cudaFreeArray(array);
    CHECK_CUDA_ERROR(err, "Failed to free CUDA array");
}
