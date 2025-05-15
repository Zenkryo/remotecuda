#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

// 向量加法的CUDA内核函数
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        c[i] = a[i] + b[i];
    }
}

// 检查CUDA错误
void checkCudaError(cudaError_t error, const char *msg) {
    if(error != cudaSuccess) {
        fprintf(stderr, "CUDA错误: %s: %s\n", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 32; // 向量大小
    size_t size = N * sizeof(float);

    // 使用统一内存分配内存
    int *a, *b, *c;
    a = nullptr;
    b = nullptr;
    c = nullptr;

    checkCudaError(cudaMallocManaged(&a, size), "分配a失败");
    checkCudaError(cudaMallocManaged(&b, size), "分配b失败");
    checkCudaError(cudaMallocManaged(&c, size), "分配c失败");

    // 初始化输入数据
    for(int i = 0; i < N; i++) {
        a[i] = 1;
        b[i] = 2;
    }
    printf("a: %p\n", a);
    printf("b: %p\n", b);
    printf("c: %p\n", c);

    // 设置CUDA内核参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 启动内核
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, N);

    sleep(1);
    // 等待内核完成
    checkCudaError(cudaDeviceSynchronize(), "内核执行失败");

    for(int i = 0; i < N; i++) {
        printf("c[%d] = %d\n", i, c[i]);
    }
    // 验证结果
    bool success = true;
    for(int i = 0; i < N; i++) {
        if(fabs(c[i] - 3) > 1e-5) {
            printf("验证失败: c[%d] = %d\n", i, c[i]);
            success = false;
            break;
        }
    }

    if(success) {
        printf("向量加法成功完成！\n");
    }

    // 释放统一内存
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
