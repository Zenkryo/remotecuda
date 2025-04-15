#include <stdio.h>
#include <cuda.h>

#define DRIVER_API_CALL(apiFuncCall)                                   \
do {                                                                   \
    CUresult _status = apiFuncCall;                                    \
    if (_status != CUDA_SUCCESS) {                                     \
        const char *errName;                                           \
        cuGetErrorName(_status, &errName);                            \
        fprintf(stderr, "CUDA driver API error %d: %s\n", _status, errName); \
        exit(1);                                                       \
    }                                                                  \
} while (0)

int main() {
    // 初始化CUDA驱动API
    DRIVER_API_CALL(cuInit(0));

    // 获取设备
    CUdevice device;
    DRIVER_API_CALL(cuDeviceGet(&device, 0));

    // 创建上下文
    CUcontext context;
    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));

    // 设置虚拟内存分配参数
    size_t allocationSize = 1 << 24; // 16MB
    size_t alignment = 1 << 16;     // 64KB对齐

    // 保留虚拟地址范围
    CUdeviceptr ptr;
    DRIVER_API_CALL(cuMemAddressReserve(&ptr, allocationSize, alignment, 0, 0));

    printf("Reserved virtual address range: 0x%llx - 0x%llx (%zu bytes)\n",
           (unsigned long long)ptr, 
           (unsigned long long)ptr + allocationSize - 1,
           allocationSize);

    // 分配物理内存
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;

    CUmemGenericAllocationHandle handle;
    DRIVER_API_CALL(cuMemCreate(&handle, allocationSize, &prop, 0));

    // 将物理内存映射到保留的虚拟地址范围
    DRIVER_API_CALL(cuMemMap(ptr, allocationSize, 0, handle, 0));

    // 设置访问权限
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    DRIVER_API_CALL(cuMemSetAccess(ptr, allocationSize, &accessDesc, 1));

    // 使用分配的内存（示例：填充数据）
    int *d_data = (int*)ptr;
    int numElements = allocationSize / sizeof(int);
    
    // 启动核函数填充数据（这里简化处理，实际使用时需要编写核函数）
    printf("Memory successfully reserved and mapped. Ready for use.\n");

    // 清理资源
    DRIVER_API_CALL(cuMemUnmap(ptr, allocationSize));
    DRIVER_API_CALL(cuMemRelease(handle));
    DRIVER_API_CALL(cuMemAddressFree(ptr, allocationSize));
    DRIVER_API_CALL(cuCtxDestroy(context));

    return 0;
}
