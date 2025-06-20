#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaSurfaceObject){
    const int width = 32;
    const int height = 32;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    cudaError_t err = cudaMallocArray(&array, &channelDesc, width, height);
    CHECK_CUDA_ERROR(err, "Failed to allocate CUDA array");
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;
    cudaSurfaceObject_t surfObj;
    err = cudaCreateSurfaceObject(&surfObj, &resDesc);
    if(err == cudaSuccess) {
        cudaResourceDesc retrievedResDesc;
        err = cudaGetSurfaceObjectResourceDesc(&retrievedResDesc, surfObj);
        CHECK_CUDA_ERROR(err, "Failed to get surface object resource descriptor");
        err = cudaDestroySurfaceObject(surfObj);
        CHECK_CUDA_ERROR(err, "Failed to destroy surface object");
    } else {
        SUCCEED() << "Surface object creation not supported, skipping test";
    }
    err = cudaFreeArray(array);
    CHECK_CUDA_ERROR(err, "Failed to free CUDA array");
}
