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
    int ret = 0;
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
        // 读取数据大小
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
        ret = 1;
    }
ERROR:
    for(auto ptr : ptrs2free) {
        free(ptr);
    }
    return ret;
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

int handle_cudaHostRegister(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaHostRegister called" << std::endl;
#endif
    int rtn = 0;
    RpcClient *client = (RpcClient *)args0;
    void *ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    size_t size;
    rpc_read(client, &size, sizeof(size));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    if(ptr == nullptr) {
        ptr = malloc(size);
        if(ptr == nullptr) {
            std::cerr << "Failed to malloc" << std::endl;
            rtn = 1;
            goto _RTN_;
        }
        // TODO: 需要释放ptr
        client->server_host_mems.insert(ptr);
    }
    read_one_now(client, ptr, size, false);
    _result = cudaHostRegister(ptr, size, flags);
    rpc_write(client, &ptr, sizeof(ptr));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    return rtn;
}

int handle_cudaHostUnregister(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaHostUnregister called" << std::endl;
#endif
    int rtn = 0;
    RpcClient *client = (RpcClient *)args0;
    void *ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaHostUnregister(ptr);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    return rtn;
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

int handle_cudaMemRangeGetAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemRangeGetAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t *dataSizes;
    enum cudaMemRangeAttribute *attributes;
    void **data;
    size_t numAttributes;
    rpc_read(client, &numAttributes, sizeof(numAttributes));
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    dataSizes = (size_t *)malloc(sizeof(size_t) * numAttributes);
    if(dataSizes == nullptr) {
        goto _RTN_;
    }
    buffers.insert(dataSizes);
    read_one_now(client, dataSizes, sizeof(size_t) * numAttributes, false);

    attributes = (enum cudaMemRangeAttribute *)malloc(sizeof(enum cudaMemRangeAttribute) * numAttributes);
    if(attributes == nullptr) {
        goto _RTN_;
    }
    buffers.insert(attributes);
    read_one_now(client, attributes, sizeof(enum cudaMemRangeAttribute) * numAttributes, false);

    data = (void **)malloc(sizeof(void *) * numAttributes);
    if(data == nullptr) {
        goto _RTN_;
    }
    buffers.insert(data);
    for(size_t i = 0; i < numAttributes; i++) {
        data[i] = malloc(dataSizes[i]);
        if(data[i] == nullptr) {
            goto _RTN_;
        }
        buffers.insert(data[i]);
    }
    _result = cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count);
    for(size_t i = 0; i < numAttributes; i++) {
        rpc_write(client, data[i], dataSizes[i], false);
    }
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
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

    RpcClient *client = (RpcClient *)args;
    CUdeviceptr devPtr;
    CUexternalMemory extMem;
    rpc_read(client, &extMem, sizeof(extMem));
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc;
    rpc_read(client, &bufferDesc, sizeof(bufferDesc));

    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }

    CUresult _result = cuExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufferDesc);
    rpc_write(client, &devPtr, sizeof(devPtr));
    rpc_write(client, &_result, sizeof(_result));

    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

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

int handle_cuGraphicsResourceGetMappedPointer_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuGraphicsResourceGetMappedPointer_v2 called" << std::endl;
#endif

    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr devPtr;
    size_t size;
    CUgraphicsResource resource;
    rpc_read(client, &resource, sizeof(resource));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuGraphicsResourceGetMappedPointer_v2(&devPtr, &size, resource);
    rpc_write(client, &devPtr, sizeof(devPtr));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuLaunchCooperativeKernel(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLaunchCooperativeKernel called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args0;
    CUfunction func;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void **kernelParams;
    int arg_count;
    rpc_read(client, &func, sizeof(func));
    rpc_read(client, &gridDimX, sizeof(gridDimX));
    rpc_read(client, &gridDimY, sizeof(gridDimY));
    rpc_read(client, &gridDimZ, sizeof(gridDimZ));
    rpc_read(client, &blockDimX, sizeof(blockDimX));
    rpc_read(client, &blockDimY, sizeof(blockDimY));
    rpc_read(client, &blockDimZ, sizeof(blockDimZ));
    rpc_read(client, &sharedMemBytes, sizeof(sharedMemBytes));
    rpc_read(client, &hStream, sizeof(hStream));
    rpc_read(client, &arg_count, sizeof(arg_count));

    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    kernelParams = (void **)malloc(sizeof(void *) * arg_count);
    if(kernelParams == nullptr) {
        std::cerr << "Failed to allocate args" << std::endl;
        return 1;
    }
    if(read_all_now(client, kernelParams, nullptr, arg_count) == -1) {
        std::cerr << "Failed to read args" << std::endl;
        return 1;
    }
    CUresult _result = cuLaunchCooperativeKernel(func, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
    for(int i = 0; i < arg_count; i++) {
        free(kernelParams[i]);
    }
    free(kernelParams);
    cudaDeviceSynchronize();
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle_cuImportExternalMemory(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuImportExternalMemory called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args0;
    CUexternalMemory extMem_out;
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memHandleDesc;
    rpc_read(client, &memHandleDesc, sizeof(memHandleDesc));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuImportExternalMemory(&extMem_out, &memHandleDesc);
    rpc_write(client, &extMem_out, sizeof(extMem_out));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuIpcOpenMemHandle_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuIpcOpenMemHandle_v2 called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dptr;
    CUipcMemHandle handle;
    unsigned int Flags;
    rpc_read(client, &handle, sizeof(handle));
    rpc_read(client, &Flags, sizeof(Flags));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuIpcOpenMemHandle_v2(&dptr, handle, Flags);
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

#if CUDA_VERSION > 11040
int handle_cuLibraryGetGlobal(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuLibraryGetGlobal called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dptr;
    size_t bytes;
    CUlibrary library;
    char *name = nullptr;
    rpc_read(client, &library, sizeof(library));
    rpc_read(client, &name, 0, true);
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        if(name != nullptr) {
            free(name);
        }
        return 1;
    }
    CUresult _result = cuLibraryGetGlobal(&dptr, &bytes, library, name);
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_write(client, &bytes, sizeof(bytes));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        if(name != nullptr) {
            free(name);
        }
        return 1;
    }
    if(name != nullptr) {
        free(name);
    }
    return 0;
}

int handle_cuLibraryGetManaged(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuLibraryGetManaged called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dptr;
    size_t bytes;
    CUlibrary library;
    char *name = nullptr;
    rpc_read(client, &library, sizeof(library));
    rpc_read(client, &name, 0, true);
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        if(name != nullptr) {
            free(name);
        }
        return 1;
    }
    CUresult _result = cuLibraryGetManaged(&dptr, &bytes, library, name);
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_write(client, &bytes, sizeof(bytes));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        if(name != nullptr) {
            free(name);
        }
        return 1;
    }
    if(name != nullptr) {
        free(name);
    }
    return 0;
}
#endif

int handle_cuMemAddressReserve(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemAddressReserve called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr ptr;
    size_t size;
    size_t alignment;
    CUdeviceptr addr;
    unsigned long long flags;
    rpc_read(client, &size, sizeof(size));
    rpc_read(client, &alignment, sizeof(alignment));
    rpc_read(client, &addr, sizeof(addr));
    rpc_read(client, &flags, sizeof(flags));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemAddressReserve(&ptr, size, alignment, addr, flags);
    rpc_write(client, &ptr, sizeof(ptr));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
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
    RpcClient *client = (RpcClient *)args;
    void *pp;
    size_t bytesize;
    rpc_read(client, &bytesize, sizeof(bytesize));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemAllocHost_v2(&pp, bytesize);
    printf("cuMemAllocHost_v2 called with pointer: %p\n", pp);
    rpc_write(client, &pp, sizeof(pp));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemAllocManaged(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemAllocManaged called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args;
    CUdeviceptr ptr;
    size_t bytesize;
    unsigned int flags;
    rpc_read(client, &bytesize, sizeof(bytesize));
    rpc_read(client, &flags, sizeof(flags));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemAllocManaged(&ptr, bytesize, flags);
    rpc_write(client, &ptr, sizeof(ptr));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemAllocPitch_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemAllocPitch_v2 called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args;
    CUdeviceptr dptr;
    size_t pitch;
    size_t WidthInBytes;
    size_t Height;
    unsigned int ElementSizeBytes;
    rpc_read(client, &WidthInBytes, sizeof(WidthInBytes));
    rpc_read(client, &Height, sizeof(Height));
    rpc_read(client, &ElementSizeBytes, sizeof(ElementSizeBytes));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemAllocPitch_v2(&dptr, &pitch, WidthInBytes, Height, ElementSizeBytes);
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_write(client, &pitch, sizeof(pitch));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemCreate(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemCreate called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args;
    CUmemGenericAllocationHandle handle;
    size_t size;
    CUmemAllocationProp prop;
    unsigned long long flags;
    rpc_read(client, &size, sizeof(size));
    rpc_read(client, &prop, sizeof(prop));
    rpc_read(client, &flags, sizeof(flags));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemCreate(&handle, size, &prop, flags);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemFreeHost(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemFreeHost called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args;
    void *p;
    rpc_read(client, &p, sizeof(p));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    printf("cuMemFreeHost called with pointer: %p\n", p);
    CUresult _result = cuMemFreeHost(p);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemGetAddressRange_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemGetAddressRange_v2 called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args;
    CUdeviceptr base;
    size_t size;
    CUdeviceptr dptr;
    rpc_read(client, &dptr, sizeof(dptr));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemGetAddressRange_v2(&base, &size, dptr);
    rpc_write(client, &base, sizeof(base));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemRangeGetAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemRangeGetAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t *dataSizes;
    CUmem_range_attribute *attributes;
    void **data;
    size_t numAttributes;
    rpc_read(client, &numAttributes, sizeof(numAttributes));
    CUdeviceptr devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    dataSizes = (size_t *)malloc(sizeof(size_t) * numAttributes);
    if(dataSizes == nullptr) {
        goto _RTN_;
    }
    buffers.insert(dataSizes);
    read_one_now(client, dataSizes, sizeof(size_t) * numAttributes, false);

    attributes = (CUmem_range_attribute *)malloc(sizeof(CUmem_range_attribute) * numAttributes);
    if(attributes == nullptr) {
        goto _RTN_;
    }
    buffers.insert(attributes);
    read_one_now(client, attributes, sizeof(enum cudaMemRangeAttribute) * numAttributes, false);

    data = (void **)malloc(sizeof(void *) * numAttributes);
    if(data == nullptr) {
        goto _RTN_;
    }
    buffers.insert(data);
    for(size_t i = 0; i < numAttributes; i++) {
        data[i] = malloc(dataSizes[i]);
        if(data[i] == nullptr) {
            goto _RTN_;
        }
        buffers.insert(data[i]);
    }
    _result = cuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count);
    for(size_t i = 0; i < numAttributes; i++) {
        rpc_write(client, data[i], dataSizes[i], false);
    }
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemHostAlloc(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemHostAlloc called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args;
    void *p;
    size_t bytesize;
    unsigned int Flags;
    rpc_read(client, &bytesize, sizeof(bytesize));
    rpc_read(client, &Flags, sizeof(Flags));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemHostAlloc(&p, bytesize, Flags);
    rpc_write(client, &p, sizeof(p));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemHostGetDevicePointer_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemHostGetDevicePointer_v2 called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args;
    CUdeviceptr dptr;
    void *p;
    unsigned int Flags;
    rpc_read(client, &p, sizeof(p));
    rpc_read(client, &Flags, sizeof(Flags));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemHostGetDevicePointer_v2(&dptr, p, Flags);
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemMap(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemMap called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args;
    CUdeviceptr ptr;
    size_t size;
    size_t offset;
    CUmemGenericAllocationHandle handle;
    unsigned long long flags;
    rpc_read(client, &ptr, sizeof(ptr));
    rpc_read(client, &size, sizeof(size));
    rpc_read(client, &offset, sizeof(offset));
    rpc_read(client, &handle, sizeof(handle));
    rpc_read(client, &flags, sizeof(flags));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemMap(ptr, size, offset, handle, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemPoolImportPointer(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemPoolImportPointer called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args;
    CUdeviceptr ptr_out;
    CUmemoryPool pool;
    CUmemPoolPtrExportData shareData;
    rpc_read(client, &pool, sizeof(pool));
    rpc_read(client, &shareData, sizeof(shareData));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemPoolImportPointer(&ptr_out, pool, &shareData);
    rpc_write(client, &ptr_out, sizeof(ptr_out));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemRelease(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuMemRelease called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args;
    CUmemGenericAllocationHandle handle;
    rpc_read(client, &handle, sizeof(handle));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemRelease(handle);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuModuleGetGlobal_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuModuleGetGlobal_v2 called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args;
    CUdeviceptr dptr;
    size_t bytes;
    CUmodule hmod;
    char *name = nullptr;
    rpc_read(client, &hmod, sizeof(hmod));
    rpc_read(client, &name, 0, true);
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        if(name != nullptr) {
            free(name);
        }
        return 1;
    }
    CUresult _result = cuModuleGetGlobal_v2(&dptr, &bytes, hmod, name);
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_write(client, &bytes, sizeof(bytes));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        if(name != nullptr) {
            free(name);
        }
        return 1;
    }
    if(name != nullptr) {
        free(name);
    }
    return 0;
}

static size_t getAttributeSize(CUpointer_attribute attribute) {
    switch(attribute) {
    // 4-byte attributes
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
    case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
    case CU_POINTER_ATTRIBUTE_IS_MANAGED:
    case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
    case CU_POINTER_ATTRIBUTE_MAPPED:
    case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
    case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
        return sizeof(unsigned int); // 4 bytes

    // 8-byte attributes
    case CU_POINTER_ATTRIBUTE_CONTEXT:
        return sizeof(CUcontext); // 8 bytes (pointer-sized)
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
        return sizeof(CUdeviceptr); // 8 bytes
    case CU_POINTER_ATTRIBUTE_HOST_POINTER:
        return sizeof(void *); // 8 bytes
    case CU_POINTER_ATTRIBUTE_BUFFER_ID:
        return sizeof(unsigned long long); // 8 bytes
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
        return sizeof(void *); // 8 bytes
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
        return sizeof(size_t); // 8 bytes
    case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
        return sizeof(CUmemoryPool); // 8 bytes

    // 4-byte (int)
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
        return sizeof(int); // 4 bytes

    // 16-byte (P2P tokens)
    case CU_POINTER_ATTRIBUTE_P2P_TOKENS:
        return 2 * sizeof(unsigned long long); // 16 bytes

    default:
        // Unknown attribute, return 0 or handle error
        return 0;
    }
}

int handle_cuPointerGetAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuPointerGetAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int numAttributes;
    rpc_read(client, &numAttributes, sizeof(numAttributes));
    CUdeviceptr ptr;
    rpc_read(client, &ptr, sizeof(ptr));

    CUpointer_attribute *attributes;
    void **data;
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    attributes = (CUpointer_attribute *)malloc(sizeof(CUpointer_attribute) * numAttributes);
    if(attributes == nullptr) {
        goto _RTN_;
    }
    buffers.insert(attributes);
    read_one_now(client, attributes, sizeof(CUpointer_attribute) * numAttributes, false);
    data = (void **)malloc(sizeof(void *) * numAttributes);
    if(data == nullptr) {
        goto _RTN_;
    }
    buffers.insert(data);
    for(size_t i = 0; i < numAttributes; i++) {
        size_t dataSize = getAttributeSize(attributes[i]);
        data[i] = malloc(dataSize);
        if(data[i] == nullptr) {
            goto _RTN_;
        }
        buffers.insert(data[i]);
    }
    _result = cuPointerGetAttributes(numAttributes, attributes, data, ptr);
    for(size_t i = 0; i < numAttributes; i++) {
        rpc_write(client, data[i], getAttributeSize(attributes[i]), false);
    }
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefGetAddress_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuTexRefGetAddress_v2 called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args;
    CUdeviceptr pdptr;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuTexRefGetAddress_v2(&pdptr, hTexRef);
    rpc_write(client, &pdptr, sizeof(pdptr));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuGraphMemFreeNodeGetParams(void *args) {
#ifdef DEBUG
    std::cout << "Handle function handle_cuGraphMemFreeNodeGetParams called" << std::endl;
#endif
    RpcClient *client = (RpcClient *)args;
    CUgraphNode hNode;
    CUdeviceptr dptr_out;
    rpc_read(client, &hNode, sizeof(hNode));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuGraphMemFreeNodeGetParams(hNode, &dptr_out);
    rpc_write(client, &dptr_out, sizeof(dptr_out));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
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
