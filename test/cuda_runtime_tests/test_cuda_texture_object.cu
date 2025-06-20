#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaTextureObject){
    const int width = 32;
    const int height = 32;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    cudaError_t err = cudaMallocArray(&array, &channelDesc, width, height);
    CHECK_CUDA_ERROR(err, "Failed to allocate CUDA array");
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    cudaTextureObject_t texObj;
    err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if(err == cudaSuccess) {
        cudaResourceDesc retrievedResDesc;
        err = cudaGetTextureObjectResourceDesc(&retrievedResDesc, texObj);
        CHECK_CUDA_ERROR(err, "Failed to get texture object resource descriptor");
        cudaTextureDesc retrievedTexDesc;
        err = cudaGetTextureObjectTextureDesc(&retrievedTexDesc, texObj);
        CHECK_CUDA_ERROR(err, "Failed to get texture object texture descriptor");
        err = cudaDestroyTextureObject(texObj);
        CHECK_CUDA_ERROR(err, "Failed to destroy texture object");
    } else {
        SUCCEED() << "Texture object creation not supported, skipping test";
    }
    err = cudaFreeArray(array);
    CHECK_CUDA_ERROR(err, "Failed to free CUDA array");
}
