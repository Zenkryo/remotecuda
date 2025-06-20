#include "common.h"

// 设备端的全局变量定义
static __device__ int dev_data;

// 简单的核函数，用于测试
__global__ void test_kernel() {
    // 执行一些计算密集型操作
    float sum = 0.0f;
    for(int i = 0; i < 1000000; i++) {
        sum += sinf(i) * cosf(i);
    }
    // 将结果写入全局内存，防止编译器优化掉循环
    dev_data = (int)sum;
}

// CUDA kernel to extract a specific channel from a 4-channel array
__global__ void extractChannelKernel(unsigned char *input, unsigned char *output, int width, int height, int channelIdx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height) {
        int idx = y * width + x;
        int inputIdx = idx * 4 + channelIdx; // 4 channels (RGBA)
        output[idx] = input[inputIdx];
    }
}

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        C[i] = A[i] + B[i];
    }
}

// 辅助函数用于检查CUDA错误
void checkCudaError(cudaError_t error, const char *message, const char *file, int line) {
    if(error != cudaSuccess) {
        const char *errorName = cudaGetErrorName(error);
        const char *errorString = cudaGetErrorString(error);
        FAIL() << "Error at " << file << ":" << line << " - " << message << ": " << errorName << " - " << errorString;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
