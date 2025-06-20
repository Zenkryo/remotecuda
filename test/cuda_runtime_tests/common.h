#ifndef TEST_COMMON_H_
#define TEST_COMMON_H_

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <iostream>

// Type definitions for CUDA runtime API
typedef CUgraph_st *cudaGraph_t;
typedef void (*cudaHostFn_t)(void *);

// CUDA 核函数声明
__global__ void test_kernel();
__global__ void extractChannelKernel(unsigned char *input, unsigned char *output, int width, int height, int channelIdx);
__global__ void vectorAdd(const float *A, const float *B, float *C, int N);

// 辅助函数声明
void checkCudaError(cudaError_t error, const char *message, const char *file, int line);

class CudaRuntimeApiTest : public ::testing::Test {
  protected:
    void SetUp() override {
        cudaDeviceReset();
        cudaError_t err = cudaSetDevice(0);
        if(err != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA device");
        }
    }

    void TearDown() override {
        cudaDeviceReset();
        cudaDeviceSynchronize();
    }
};

// 宏定义用于简化错误检查调用
#define CHECK_CUDA_ERROR(err, msg) checkCudaError(err, msg, __FILE__, __LINE__)

#endif // TEST_COMMON_H_
