#include "common.h"

TEST_F(CudaRuntimeApiTest, Cuda3DArray){
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(32, 32, 32);
    cudaArray_t array;
    cudaError_t err = cudaMalloc3DArray(&array, &channelDesc, extent);
    CHECK_CUDA_ERROR(err, "Failed to allocate 3D array");
    ASSERT_NE(array, nullptr);
    err = cudaFreeArray(array);
    CHECK_CUDA_ERROR(err, "Failed to free 3D array");
}
