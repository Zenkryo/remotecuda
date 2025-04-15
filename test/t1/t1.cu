#include <stdio.h>
#include <cuda_runtime.h>

// 定义设备端的全局变量
__device__ int dev_data;

// 简单的核函数，用于修改设备端全局变量
__global__ void modifyData(int value) {
    dev_data = value;
}

int main() {
    int *d_ptr = NULL;
    int value = 42;
    int host_data = 0;
    
    // 获取设备端符号的地址
    cudaError_t err = cudaGetSymbolAddress((void**)&d_ptr, dev_data);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get symbol address: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("Device symbol address: %p\n", d_ptr);
    
    // 启动核函数修改设备端全局变量
    modifyData<<<1, 1>>>(value);
    
    // 将设备端数据拷贝回主机
    err = cudaMemcpy(&host_data, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from device: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("Data from device: %d\n", host_data);
    
    // 清理
    cudaDeviceReset();
    
    return 0;
}
