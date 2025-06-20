#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMipmappedArray){
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(32, 32, 0);
    unsigned int numLevels = 5;
    cudaMipmappedArray_t mipmappedArray;
    cudaError_t err = cudaMallocMipmappedArray(&mipmappedArray, &channelDesc, extent, numLevels);
    CHECK_CUDA_ERROR(err, "Failed to allocate mipmapped array");
    ASSERT_NE(mipmappedArray, nullptr);
    for(unsigned int level = 0; level < numLevels; level++) {
        cudaArray_t levelArray;
        err = cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, level);
        CHECK_CUDA_ERROR(err, (std::string("Failed to get mipmap level ") + std::to_string(level)).c_str());
        ASSERT_NE(levelArray, nullptr);
    }
    err = cudaFreeMipmappedArray(mipmappedArray);
    CHECK_CUDA_ERROR(err, "Failed to free mipmapped array");
}
