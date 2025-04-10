#include <iostream>
#include <unordered_map>
#include "gen/hook_api.h"
#include "gen/handle_server.h"
#include "rpc.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "hidden_api.h"
#include "nvml.h"

cudaMemoryType checkPointer(void *ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if(err != cudaSuccess) {
        perror(cudaGetErrorString(err));
        return cudaMemoryTypeUnregistered;
    }

    return attributes.type;
}

int handle_mem2server(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function mem2server called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args0;
    void *ptrs[32];
    size_t size;
    int i = 0;
    int j = 0;
    while(i++ < 32) {
        ptrs[i] = nullptr;
        // 读取服务器端指针
        if(read_one_now(client, &ptrs[i], sizeof(ptrs[i]), false) < 0) {
            goto ERROR;
        }
        if(ptrs[i] == (void *)0xffffffff) {
            break;
        }
    }
    while(j++ < i - 1) {
        // 读取数据大小
        size = 0;
        if(read_one_now(client, &size, sizeof(size), false) < 0) {
            goto ERROR;
        }
        if(ptrs[j] == nullptr && size > 0) {
            ptrs[j] = malloc(size);
            if(ptrs[j] == nullptr) {
                std::cerr << "Failed to malloc" << std::endl;
                goto ERROR;
            }
            client->tmpbufs.push(ptrs[j]);                // 保存临时内存地址
            rpc_write(client, &ptrs[j], sizeof(ptrs[j])); // 返回服务器侧申请的内存地址
        }
        if(read_one_now(client, ptrs[j], size, false) < 0) {
            goto ERROR;
        }
    }
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        goto ERROR;
    }
    return 0;
ERROR:
    while(!client->tmpbufs.empty()) {
        void *ptr = client->tmpbufs.front();
        client->tmpbufs.pop();
        free(ptr);
    }
    return 1;
}

int handle_mem2client(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function mem2client called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args0;
    void *ptrs[32];
    size_t sizes[32];
    std::vector<void *> ptrs2free;
    int i = 0;
    int j = 0;
    while(i++ < 32) {
        ptrs[i] = nullptr;
        sizes[i] = 0;
        // 读取服务器端指针
        if(read_one_now(client, &ptrs[i], sizeof(ptrs[i]), false) < 0) {
            goto ERROR;
        }
        if(ptrs[i] == (void *)0xffffffff) {
            break;
        }
        if(read_one_now(client, &sizes[i], sizeof(sizes[i]), false) < 0) {
            goto ERROR;
        }
    }
    while(j++ < i - 1) {
        if(sizes[j] <= 0) {
            continue;
        }
        if(ptrs[j] == nullptr) {
            void *ptr = client->tmpbufs.front();
            client->tmpbufs.pop();
            rpc_write(client, ptr, sizes[j], true);
            ptrs2free.push_back(ptr);
        } else {
            rpc_write(client, ptrs[j], sizes[j], true);
        }
    }
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        goto ERROR;
    }
    return 0;
ERROR:
    for(auto ptr : ptrs2free) {
        free(ptr);
    }
    return 1;
}

// #region CUDA Runtime API (cuda*)

int handle_cudaFree(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cudaFree called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args;
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaFree(devPtr);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cudaFreeHost(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cudaFreeHost called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args;
    void *ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaFreeHost(ptr);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cudaGetErrorName(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetErrorName called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args0;
    cudaError_t error;
    rpc_read(client, &error, sizeof(error));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    const char *_result = cudaGetErrorName(error);
    rpc_write(client, _result, strlen(_result) + 1, true);
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle_cudaGetErrorString(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetErrorString called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args0;
    cudaError_t error;
    rpc_read(client, &error, sizeof(error));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    const char *_result = cudaGetErrorString(error);
    rpc_write(client, _result, strlen(_result) + 1, true);
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle_cudaGetSymbolAddress(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cudaGetSymbolAddress called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args;
    void *devPtr;
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return cudaErrorUnknown;
    }
    cudaError_t _result = cudaGetSymbolAddress(&devPtr, symbol);
    rpc_write(client, &devPtr, sizeof(devPtr));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return cudaErrorUnknown;
    }
    return 0;
}

int handle_cudaHostAlloc(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cudaHostAlloc called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args;
    void *pHost;
    size_t size;
    unsigned int flags;
    rpc_write(client, &pHost, sizeof(pHost));
    rpc_read(client, &size, sizeof(size));
    rpc_read(client, &flags, sizeof(flags));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return cudaErrorUnknown;
    }
    cudaError_t _result = cudaHostAlloc(&pHost, size, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

int handle_cudaLaunchKernel(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaLaunchKernel called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args0;
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;
    int arg_count;
    rpc_read(client, &func, sizeof(func));
    rpc_read(client, &gridDim, sizeof(gridDim));
    rpc_read(client, &blockDim, sizeof(blockDim));
    rpc_read(client, &sharedMem, sizeof(sharedMem));
    rpc_read(client, &stream, sizeof(stream));
    rpc_read(client, &arg_count, sizeof(arg_count));

    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    args = (void **)malloc(sizeof(void *) * arg_count);
    if(args == nullptr) {
        std::cerr << "Failed to allocate args" << std::endl;
        return 1;
    }
    if(read_all_now(client, args, nullptr, arg_count) == -1) {
        std::cerr << "Failed to read args" << std::endl;
        return 1;
    }

    cudaError_t _result = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    for(int i = 0; i < arg_count; i++) {
        free(args[i]);
    }
    free(args);
    cudaDeviceSynchronize();
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle_cudaLaunchCooperativeKernel(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaLaunchCooperativeKernel called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args0;
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;
    int arg_count;
    rpc_read(client, &func, sizeof(func));
    rpc_read(client, &gridDim, sizeof(gridDim));
    rpc_read(client, &blockDim, sizeof(blockDim));
    rpc_read(client, &sharedMem, sizeof(sharedMem));
    rpc_read(client, &stream, sizeof(stream));
    rpc_read(client, &arg_count, sizeof(arg_count));

    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    args = (void **)malloc(sizeof(void *) * arg_count);
    if(args == nullptr) {
        std::cerr << "Failed to allocate args" << std::endl;
        return 1;
    }
    if(read_all_now(client, args, nullptr, arg_count) == -1) {
        std::cerr << "Failed to read args" << std::endl;
        return 1;
    }

    cudaError_t _result = cudaLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, stream);
    for(int i = 0; i < arg_count; i++) {
        free(args[i]);
    }
    free(args);
    cudaDeviceSynchronize();
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle_cudaMalloc(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cudaMalloc called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args;
    size_t size;
    rpc_read(client, &size, sizeof(size));
    void *devPtr;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaMalloc(&devPtr, size);
    rpc_write(client, &devPtr, sizeof(devPtr));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cudaMalloc3D(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cudaMalloc3D called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args;
    struct cudaPitchedPtr pitchedDevPtr;
    struct cudaExtent extent;
    rpc_read(client, &extent, sizeof(extent));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaMalloc3D(&pitchedDevPtr, extent);
    rpc_write(client, &pitchedDevPtr, sizeof(pitchedDevPtr));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cudaMallocHost(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cudaMallocHost called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args;
    void *ptr;
    size_t size;
    rpc_write(client, &ptr, sizeof(ptr));
    rpc_read(client, &size, sizeof(size));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaMallocHost(&ptr, size);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cudaMallocManaged(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cudaMallocManaged called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args;
    void *devPtr;
    size_t size;
    unsigned int flags;
    rpc_write(client, &devPtr, sizeof(devPtr));
    rpc_read(client, &size, sizeof(size));
    rpc_read(client, &flags, sizeof(flags));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaMallocManaged(&devPtr, size, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cudaMallocPitch(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cudaMallocPitch called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args;
    void *devPtr;
    size_t pitch;
    size_t width;
    size_t height;
    rpc_read(client, &width, sizeof(width));
    rpc_read(client, &height, sizeof(height));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaMallocPitch(&devPtr, &pitch, width, height);
    rpc_write(client, &devPtr, sizeof(devPtr));
    rpc_write(client, &pitch, sizeof(pitch));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}
int handle___cudaPopCallConfiguration(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function __cudaPopCallConfiguration called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args0;
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = __cudaPopCallConfiguration(&gridDim, &blockDim, &sharedMem, &stream);
    rpc_write(client, &gridDim, sizeof(gridDim));
    rpc_write(client, &blockDim, sizeof(blockDim));
    rpc_write(client, &sharedMem, sizeof(sharedMem));
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle___cudaPushCallConfiguration(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function __cudaPushCallConfiguration called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args0;
    dim3 gridDim;
    rpc_read(client, &gridDim, sizeof(gridDim));
    dim3 blockDim;
    rpc_read(client, &blockDim, sizeof(blockDim));
    size_t sharedMem;
    rpc_read(client, &sharedMem, sizeof(sharedMem));
    struct CUstream_st *stream;
    rpc_read(client, &stream, sizeof(stream));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    unsigned _result = __cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle___cudaRegisterFatBinary(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function __cudaRegisterFatBinary called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args0;
    __cudaFatCudaBinary2 *fatCubin = (__cudaFatCudaBinary2 *)malloc(sizeof(__cudaFatCudaBinary2));
    if(fatCubin == nullptr) {
        std::cerr << "Failed to allocate fatCubin" << std::endl;
        return 1;
    }
    rpc_read(client, fatCubin, sizeof(__cudaFatCudaBinary2));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    void *cubin = nullptr;
    int len;
    read_one_now(client, &cubin, 0, true);
    fatCubin->text = (uint64_t)cubin;
    void **_result = __cudaRegisterFatBinary(fatCubin);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle___cudaRegisterFatBinaryEnd(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function __cudaRegisterFatBinaryEnd called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args0;
    void **fatCubinHandle;
    rpc_read(client, &fatCubinHandle, sizeof(fatCubinHandle));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    __cudaRegisterFatBinaryEnd(fatCubinHandle);
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle___cudaRegisterFunction(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function __cudaRegisterFunction called" << std::endl;
#endif

    int rtn = 0;
    RpcClient *client = (RpcClient *)args0;
    void **fatCubinHandle;
    char *hostFun;
    char *deviceName = nullptr;
    int thread_limit;
    uint3 tid;
    uint3 bid;
    dim3 bDim;
    dim3 gDim;
    int wSize;
    uint8_t mask;
    rpc_read(client, &fatCubinHandle, sizeof(fatCubinHandle));
    rpc_read(client, &hostFun, sizeof(hostFun));
    rpc_read(client, &deviceName, 0, true);
    rpc_read(client, &thread_limit, sizeof(thread_limit));
    rpc_read(client, &mask, sizeof(mask));

    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    if(mask & 1 << 0) {
        read_one_now(client, &tid, sizeof(uint3), false);
    }
    if(mask & 1 << 1) {
        read_one_now(client, &bid, sizeof(uint3), false);
    }
    if(mask & 1 << 2) {
        read_one_now(client, &bDim, sizeof(dim3), false);
    }
    if(mask & 1 << 3) {
        read_one_now(client, &gDim, sizeof(dim3), false);
    }
    if(mask & 1 << 4) {
        read_one_now(client, &wSize, sizeof(wSize), false);
    }
    __cudaRegisterFunction(fatCubinHandle, hostFun, deviceName, deviceName, thread_limit, mask & 1 << 0 ? &tid : nullptr, mask & 1 << 1 ? &bid : nullptr, mask & 1 << 2 ? &bDim : nullptr, mask & 1 << 3 ? &gDim : nullptr, mask & 1 << 4 ? &wSize : nullptr);
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
#ifdef DEBUG
    if(mask & 1 << 0) {
        printf("tid: %u %u %u\n", tid.x, tid.y, tid.z);
    }
    if(mask & 1 << 1) {
        printf("bid: %u %u %u\n", bid.x, bid.y, bid.z);
    }
    if(mask & 1 << 2) {
        printf("bDim: %u %u %u\n", bDim.x, bDim.y, bDim.z);
    }
    if(mask & 1 << 3) {
        printf("gDim: %u %u %u\n", gDim.x, gDim.y, gDim.z);
    }
    if(mask & 1 << 4) {
        printf("wSize: %d\n", wSize);
    }
#endif
_RTN_:
    return rtn;
}

int handle___cudaRegisterManagedVar(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function __cudaRegisterManagedVar called" << std::endl;
#endif

    int rtn = 0;
    RpcClient *client = (RpcClient *)args0;
    void **fatCubinHandle;
    void *hostVarPtrAddress = nullptr;
    char *deviceName = nullptr;
    rpc_read(client, &fatCubinHandle, sizeof(fatCubinHandle));
    rpc_read(client, &deviceName, 0, true);
    int ext;
    rpc_read(client, &ext, sizeof(ext));
    size_t size;
    rpc_read(client, &size, sizeof(size));
    int constant;
    rpc_read(client, &constant, sizeof(constant));
    int global;
    rpc_read(client, &global, sizeof(global));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    __cudaRegisterManagedVar(fatCubinHandle, &hostVarPtrAddress, deviceName, deviceName, ext, size, constant, global);
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
_RTN_:
    return rtn;
}

int handle___cudaRegisterVar(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function __cudaRegisterVar called" << std::endl;
#endif

    int rtn = 0;
    RpcClient *client = (RpcClient *)args0;
    void **fatCubinHandle;
    char *hostVar;
    char *deviceName = nullptr;
    rpc_read(client, &fatCubinHandle, sizeof(fatCubinHandle));
    rpc_read(client, &hostVar, sizeof(hostVar));
    rpc_read(client, &deviceName, 0, true);
    int ext;
    rpc_read(client, &ext, sizeof(ext));
    size_t size;
    rpc_read(client, &size, sizeof(size));
    int constant;
    rpc_read(client, &constant, sizeof(constant));
    int global;
    rpc_read(client, &global, sizeof(global));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    __cudaRegisterVar(fatCubinHandle, hostVar, deviceName, deviceName, ext, size, constant, global);
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
_RTN_:
    return rtn;
}

int handle___cudaUnregisterFatBinary(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function __cudaUnregisterFatBinary called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args0;
    void **fatCubinHandle;
    rpc_read(client, &fatCubinHandle, sizeof(fatCubinHandle));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    __cudaUnregisterFatBinary(fatCubinHandle);
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle___cudaInitModule(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function __cudaInitModule called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args0;
    void *fatCubinHandle;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    char _result = __cudaInitModule(&fatCubinHandle);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

// #endregion
// #region CUDA Driver API (cu*)

int handle_cuExternalMemoryGetMappedBuffer(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuExternalMemoryGetMappedBuffer called" << std::endl;
#endif

    return 0;
}

int handle_cuGetErrorName(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGetErrorName called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args0;
    CUresult error;
    rpc_read(client, &error, sizeof(error));
    const char *pStr;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuGetErrorName(error, &pStr);
    rpc_write(client, pStr, strlen(pStr) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuGetErrorString(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGetErrorString called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args0;
    CUresult error;
    rpc_read(client, &error, sizeof(error));
    const char *pStr;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuGetErrorString(error, &pStr);
    rpc_write(client, pStr, strlen(pStr) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

#if CUDA_VERSION <= 11040
int handle_cuGetProcAddress(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGetProcAddress called" << std::endl;
#endif

    return 0;
}
#endif

int handle_cuGraphicsResourceGetMappedPointer_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuGraphicsResourceGetMappedPointer_v2 called" << std::endl;
#endif

    return 0;
}

int handle_cuImportExternalMemory(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuImportExternalMemory called" << std::endl;
#endif

    return 0;
}

int handle_cuIpcOpenMemHandle_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuIpcOpenMemHandle_v2 called" << std::endl;
#endif

    return 0;
}

#if CUDA_VERSION > 11040
int handle_cuLibraryGetGlobal(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuLibraryGetGlobal called" << std::endl;
#endif

    return 0;
}

int handle_cuLibraryGetManaged(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuLibraryGetManaged called" << std::endl;
#endif

    return 0;
}
#endif

int handle_cuMemAddressReserve(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemAddressReserve called" << std::endl;
#endif

    // TODO: Implement the function logic
    return 0;
}

int handle_cuMemAlloc_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemAlloc_v2 called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args;
    CUdeviceptr dptr;
    size_t bytesize;
    rpc_read(client, &bytesize, sizeof(bytesize));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemAlloc_v2(&dptr, bytesize);
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemAllocHost_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemAllocHost_v2 called" << std::endl;
#endif

    return 0;
}

int handle_cuMemAllocManaged(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemAllocManaged called" << std::endl;
#endif

    return 0;
}

int handle_cuMemAllocPitch_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemAllocPitch_v2 called" << std::endl;
#endif

    return 0;
}

int handle_cuMemCreate(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemCreate called" << std::endl;
#endif

    // TODO: Implement the function logic
    return 0;
}

int handle_cuMemFreeHost(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemFreeHost called" << std::endl;
#endif

    // TODO: Implement the function logic
    return 0;
}

int handle_cuMemGetAddressRange_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemGetAddressRange_v2 called" << std::endl;
#endif

    return 0;
}

int handle_cuMemHostAlloc(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemHostAlloc called" << std::endl;
#endif

    // TODO: Implement the function logic
    return 0;
}

int handle_cuMemHostGetDevicePointer_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemHostGetDevicePointer_v2 called" << std::endl;
#endif

    return 0;
}

int handle_cuMemMap(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemMap called" << std::endl;
#endif

    // TODO: Implement the function logic
    return 0;
}

int handle_cuMemPoolImportPointer(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemPoolImportPointer called" << std::endl;
#endif

    return 0;
}

int handle_cuMemRelease(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemRelease called" << std::endl;
#endif

    // TODO: Implement the function logic
    return 0;
}

int handle_cuModuleGetGlobal_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuModuleGetGlobal_v2 called" << std::endl;
#endif

    return 0;
}

int handle_cuTexRefGetAddress_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuTexRefGetAddress_v2 called" << std::endl;
#endif

    return 0;
}

int handle_cuGraphMemFreeNodeGetParams(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuGraphMemFreeNodeGetParams called" << std::endl;
#endif

    return 0;
}

// #endregion
// #region NVML (nvml*)

int handle_nvmlErrorString(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlErrorString called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args0;
    nvmlReturn_t result;
    rpc_read(client, &result, sizeof(result));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    const char *_result = nvmlErrorString(result);
    rpc_write(client, _result, strlen(_result) + 1, true);
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

// #endregion
