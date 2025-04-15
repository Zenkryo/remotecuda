#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// 协作核函数，计算数组元素的和
__global__ void cooperativeSumKernel(int *data, int *result, int size) {
    // 创建网格级别的协作组
    cg::grid_group grid = cg::this_grid();

    // 每个线程处理一个元素
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= size)
        return;

    // 网格范围内的同步
    grid.sync();

    // 归约求和
    for(int stride = grid.size() / 2; stride > 0; stride >>= 1) {
        if(tid < stride && tid + stride < size) {
            data[tid] += data[tid + stride];
        }
        grid.sync();
    }

    // 第一个线程写入结果
    if(tid == 0) {
        *result = data[0];
    }
}

int main() {
    int size = 1024;
    int *d_data = NULL, *d_result = NULL;
    int h_data[size], h_result = 0;

    // 初始化数据
    for(int i = 0; i < size; i++) {
        h_data[i] = i + 1;
    }

    // 分配设备内存
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    // 拷贝数据到设备
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    // 设置启动参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // 检查设备是否支持协作组
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if(prop.cooperativeLaunch == 0) {
        printf("Device does not support cooperative kernels!\n");
        return 1;
    }

    // 准备核函数参数
    void *kernelArgs[] = {&d_data, &d_result, &size};

    // 启动协作核函数
    cudaError_t err = cudaLaunchCooperativeKernel((void *)cooperativeSumKernel, blocksPerGrid, threadsPerBlock, kernelArgs);

    if(err != cudaSuccess) {
        printf("Failed to launch cooperative kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 拷贝结果回主机
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // 验证结果 (1+2+...+1024 = 1024*1025/2 = 524800)
    printf("Computed sum: %d\n", h_result);
    printf("Expected sum: %d\n", size * (size + 1) / 2);

    // 清理
    cudaFree(d_data);
    cudaFree(d_result);
    cudaDeviceReset();

    return 0;
}
