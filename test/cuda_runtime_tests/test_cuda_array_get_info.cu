#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaArrayGetInfo){
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    cudaError_t err = cudaMallocArray(&array, &channelDesc, 32, 32);
    CHECK_CUDA_ERROR(err, "Failed to allocate array");
    cudaChannelFormatDesc retrievedDesc;
    cudaExtent extent;
    unsigned int flags;
    err = cudaArrayGetInfo(&retrievedDesc, &extent, &flags, array);
    CHECK_CUDA_ERROR(err, "Failed to get array info");
    ASSERT_EQ(retrievedDesc.x, channelDesc.x) << "Channel format mismatch";
    ASSERT_EQ(retrievedDesc.y, channelDesc.y) << "Channel format mismatch";
    ASSERT_EQ(retrievedDesc.z, channelDesc.z) << "Channel format mismatch";
    ASSERT_EQ(retrievedDesc.w, channelDesc.w) << "Channel format mismatch";
    ASSERT_EQ(retrievedDesc.f, channelDesc.f) << "Channel format mismatch";
    ASSERT_EQ(extent.width, 32) << "Width mismatch";
    ASSERT_EQ(extent.height, 32) << "Height mismatch";
    ASSERT_EQ(extent.depth, 0) << "Depth mismatch";
    err = cudaFreeArray(array);
    CHECK_CUDA_ERROR(err, "Failed to free array");
}
