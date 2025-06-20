#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemcpyArrayAsync){
    const int size = 1024;
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err, "Failed to create stream");
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    err = cudaMallocArray(&array, &channelDesc, size);
    CHECK_CUDA_ERROR(err, "Failed to allocate CUDA array");
    float *hostSrc = new float[size];
    float *hostDst = new float[size];
    for(int i = 0; i < size; i++) {
        hostSrc[i] = static_cast<float>(i);
    }
    err = cudaMemcpyToArrayAsync(array, 0, 0, hostSrc, size * sizeof(float), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA_ERROR(err, "Failed to copy from host to array");
    err = cudaMemcpyFromArrayAsync(hostDst, array, 0, 0, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    CHECK_CUDA_ERROR(err, "Failed to copy from array to host");
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA_ERROR(err, "Failed to synchronize stream");
    for(int i = 0; i < size; i++) {
        ASSERT_EQ(hostDst[i], hostSrc[i]) << "Memory copy failed at index " << i;
    }
    delete[] hostSrc;
    delete[] hostDst;
    err = cudaFreeArray(array);
    CHECK_CUDA_ERROR(err, "Failed to free CUDA array");
    err = cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(err, "Failed to destroy stream");
}
