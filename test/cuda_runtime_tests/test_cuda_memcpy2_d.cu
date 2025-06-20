#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemcpy2D){
    const int width = 32;
    const int height = 32;
    const int pitch = width * sizeof(float);
    float *hostSrc = new float[width * height];
    float *hostDst = new float[width * height];
    void *devSrc, *devDst;
    size_t devPitch;
    cudaError_t err = cudaMallocPitch(&devSrc, &devPitch, width * sizeof(float), height);
    CHECK_CUDA_ERROR(err, "Failed to allocate source device memory");
    err = cudaMallocPitch(&devDst, &devPitch, width * sizeof(float), height);
    CHECK_CUDA_ERROR(err, "Failed to allocate destination device memory");
    for(int i = 0; i < width * height; i++) {
        hostSrc[i] = static_cast<float>(i);
    }
    err = cudaMemcpy2D(devSrc, devPitch, hostSrc, pitch, width * sizeof(float), height, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err, "Failed to copy from host to device");
    err = cudaMemcpy2D(devDst, devPitch, devSrc, devPitch, width * sizeof(float), height, cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR(err, "Failed to copy from device to device");
    err = cudaMemcpy2D(hostDst, pitch, devDst, devPitch, width * sizeof(float), height, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err, "Failed to copy from device to host");
    for(int i = 0; i < width * height; i++) {
        ASSERT_EQ(hostDst[i], hostSrc[i]) << "Memory copy failed at index " << i;
    }
    delete[] hostSrc;
    delete[] hostDst;
    err = cudaFree(devSrc);
    CHECK_CUDA_ERROR(err, "Failed to free source device memory");
    err = cudaFree(devDst);
    CHECK_CUDA_ERROR(err, "Failed to free destination device memory");
}
