#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaArrayGetSparseProperties){
    int supportsSparseArrays = 0;
    cudaError_t err = cudaDeviceGetAttribute(&supportsSparseArrays, cudaDevAttrSparseCudaArraySupported, 0);
    CHECK_CUDA_ERROR(err, "Failed to get device attribute");
    if(!supportsSparseArrays) {
        GTEST_SKIP() << "Device does not support sparse arrays";
    }
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(32, 32, 1);
    cudaArray_t array;
    err = cudaMalloc3DArray(&array, &channelDesc, extent, cudaArraySparse);
    CHECK_CUDA_ERROR(err, "Failed to allocate sparse 3D array");
    cudaArraySparseProperties sparseProps;
    err = cudaArrayGetSparseProperties(&sparseProps, array);
    CHECK_CUDA_ERROR(err, "Failed to get sparse properties");
    err = cudaFreeArray(array);
    CHECK_CUDA_ERROR(err, "Failed to free array");
}
