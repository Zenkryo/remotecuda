#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemcpy3DPeer){
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CHECK_CUDA_ERROR(err, "Failed to get device count");
    if(deviceCount > 1) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaExtent extent = make_cudaExtent(32, 32, 32);
        cudaArray_t srcArray, dstArray;
        err = cudaSetDevice(0);
        CHECK_CUDA_ERROR(err, "Failed to set source device");
        err = cudaMalloc3DArray(&srcArray, &channelDesc, extent);
        CHECK_CUDA_ERROR(err, "Failed to allocate source array");
        err = cudaSetDevice(1);
        CHECK_CUDA_ERROR(err, "Failed to set destination device");
        err = cudaMalloc3DArray(&dstArray, &channelDesc, extent);
        CHECK_CUDA_ERROR(err, "Failed to allocate destination array");
        err = cudaDeviceEnablePeerAccess(0, 0);
        if(err == cudaSuccess) {
            cudaMemcpy3DPeerParms copyParams = {0};
            copyParams.srcArray = srcArray;
            copyParams.srcDevice = 0;
            copyParams.dstArray = dstArray;
            copyParams.dstDevice = 1;
            copyParams.extent = extent;
            err = cudaMemcpy3DPeer(&copyParams);
            CHECK_CUDA_ERROR(err, "Failed to perform 3D peer memory copy");
            err = cudaDeviceDisablePeerAccess(0);
            CHECK_CUDA_ERROR(err, "Failed to disable peer access");
        }
        err = cudaSetDevice(0);
        CHECK_CUDA_ERROR(err, "Failed to set device 0");
        err = cudaFreeArray(srcArray);
        CHECK_CUDA_ERROR(err, "Failed to free source array");
        err = cudaSetDevice(1);
        CHECK_CUDA_ERROR(err, "Failed to set device 1");
        err = cudaFreeArray(dstArray);
        CHECK_CUDA_ERROR(err, "Failed to free destination array");
    } else {
        SUCCEED() << "Skipping test - requires multiple devices";
    }
}
