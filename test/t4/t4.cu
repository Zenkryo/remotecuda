#include <stdio.h>
#include <cuda.h>

#define CHECK_CUDA_ERROR(call)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
    do {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
        CUresult err = call;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   \
        if(err != CUDA_SUCCESS) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              \
            const char *errStr;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
            cuGetErrorString(err, &errStr);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, errStr);                                                                                                                                                                                                                                                                                                                                                                                                                                         \
            exit(EXIT_FAILURE);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
        }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
    } while(0)

int main() {
    // 初始化CUDA驱动API
    CHECK_CUDA_ERROR(cuInit(0));

    // 获取设备
    CUdevice device;
    CHECK_CUDA_ERROR(cuDeviceGet(&device, 0));

    // 创建上下文
    CUcontext context;
    CHECK_CUDA_ERROR(cuCtxCreate(&context, 0, device));

    // 查询设备属性
    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;

    CHECK_CUDA_ERROR(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    printf("Minimum allocation granularity: %zu bytes\n", granularity);

    // 确保大小是粒度的整数倍
    size_t size = ((1024 * 1024) + granularity - 1) & ~(granularity - 1); // 1MB对齐

    // 创建内存分配
    CUmemGenericAllocationHandle handle;
    CHECK_CUDA_ERROR(cuMemCreate(&handle, size, &prop, 0));

    printf("Successfully created memory allocation of size %zu bytes\n", size);

    // 获取分配的内存物理地址
    CUdeviceptr ptr;
    CHECK_CUDA_ERROR(cuMemAddressReserve(&ptr, size, 0, 0, 0));
    CHECK_CUDA_ERROR(cuMemMap(ptr, size, 0, handle, 0));

    // 设置访问权限
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA_ERROR(cuMemSetAccess(ptr, size, &accessDesc, 1));

    // 使用分配的内存...
    printf("Memory successfully allocated and mapped at address: %p\n", (void *)ptr);

    // 清理资源
    CHECK_CUDA_ERROR(cuMemUnmap(ptr, size));
    CHECK_CUDA_ERROR(cuMemAddressFree(ptr, size));
    CHECK_CUDA_ERROR(cuMemRelease(handle));
    CHECK_CUDA_ERROR(cuCtxDestroy(context));

    return 0;
}
