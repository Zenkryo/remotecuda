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

    // 分配设备内存
    size_t size = 1024 * 1024; // 1MB
    CUdeviceptr d_ptr;
    CHECK_CUDA_ERROR(cuMemAlloc(&d_ptr, size));

    // 检查设备是否支持所需功能
    int major = 0, minor = 0;
    CHECK_CUDA_ERROR(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CUDA_ERROR(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    printf("Device compute capability: %d.%d\n", major, minor);

    if(major < 6) {
        fprintf(stderr, "This program requires compute capability 6.0 or higher\n");
        CHECK_CUDA_ERROR(cuMemFree(d_ptr));
        CHECK_CUDA_ERROR(cuCtxDestroy(context));
        return 1;
    }

    // 准备查询属性
    CUmem_range_attribute attr = CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION;
    size_t data_size = sizeof(CUmemLocation);
    CUmemLocation pref_loc;
    void *data = &pref_loc;

    // 正确调用cuMemRangeGetAttributes
    CHECK_CUDA_ERROR(cuMemRangeGetAttributes(&data,      // void** 指针数组
                                             &data_size, // size_t* 大小数组
                                             &attr,      // CUmem_range_attribute* 属性数组
                                             1,          // 属性数量
                                             d_ptr,      // 起始地址
                                             size        // 范围大小
                                             ));

    // 打印结果
    printf("Memory range preferred location:\n");
    printf("  Type: %d\n", pref_loc.type);
    printf("  ID: %d\n", pref_loc.id);

    // 清理资源
    CHECK_CUDA_ERROR(cuMemFree(d_ptr));
    CHECK_CUDA_ERROR(cuCtxDestroy(context));

    return 0;
}
