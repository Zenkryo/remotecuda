#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemcpy3DAsync){
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(32, 32, 32);
    cudaArray_t srcArray, dstArray;
    cudaError_t err = cudaMalloc3DArray(&srcArray, &channelDesc, extent);
    CHECK_CUDA_ERROR(err, "Failed to allocate source array");
    err = cudaMalloc3DArray(&dstArray, &channelDesc, extent);
    CHECK_CUDA_ERROR(err, "Failed to allocate destination array");
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcArray = srcArray;
    copyParams.dstArray = dstArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    err = cudaMemcpy3DAsync(&copyParams, stream);
    CHECK_CUDA_ERROR(err, "Failed to perform async 3D memory copy");
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA_ERROR(err, "Failed to synchronize stream");
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
    err = cudaFreeArray(srcArray);
    CHECK_CUDA_ERROR(err, "Failed to free source array");
    err = cudaFreeArray(dstArray);
    CHECK_CUDA_ERROR(err, "Failed to free destination array");
}
