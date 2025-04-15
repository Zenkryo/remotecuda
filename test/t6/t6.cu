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

    // 准备查询指针属性
    CUpointer_attribute attributes[3] = {CU_POINTER_ATTRIBUTE_CONTEXT, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL};

    // 查询指针属性
    void *data[3];
    CUcontext ctx;
    unsigned int mem_type;
    int device_ordinal;

    data[0] = &ctx;
    data[1] = &mem_type;
    data[2] = &device_ordinal;

    CHECK_CUDA_ERROR(cuPointerGetAttributes(3,          // 属性数量
                                            attributes, // 属性数组
                                            data,       // 结果数据数组
                                            d_ptr       // 要查询的指针
                                            ));

    // 打印查询结果
    printf("Pointer attributes:\n");
    printf("  Context: %p\n", ctx);
    printf("  Memory type: %u\n", mem_type);
    printf("  Device ordinal: %d\n", device_ordinal);

    // 清理资源
    CHECK_CUDA_ERROR(cuMemFree(d_ptr));
    CHECK_CUDA_ERROR(cuCtxDestroy(context));

    return 0;
}
