#include <iostream>
#include <map>
#include <string.h>
#include "gen/hook_api.h"
#include "gen/handle_server.h"
#include "rpc/rpc_core.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "hidden_api.h"
#include "nvml.h"

typedef struct Async2ClientInfo {
    std::string client_id;
    std::vector<Async2Client> async2clients;
} Async2ClientInfo;

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
    RpcConn *conn = (RpcConn *)args0;
    void *ptrs[32];
    size_t size;
    int i = 0;
    int j = 0;
    for(i = 0; i < 32; i++) {
        ptrs[i] = nullptr;
        // 读取服务器端指针
        if(conn->read_one_now(&ptrs[i], sizeof(ptrs[i]), false) != RpcError::OK) {
            return 1;
        }
        if(ptrs[i] == (void *)0xffffffff) {
            break;
        }
    }
    for(j = 0; j < i; j++) {
        // 读取数据大小
        size = 0;
        if(conn->read_one_now(&size, sizeof(size), false) != RpcError::OK) {
            return 1;
        }
        if(ptrs[j] == nullptr && size > 0) {
            void *tmp_buffer = conn->alloc_host_buffer(size);
            if(tmp_buffer == nullptr) {
                std::cerr << "Failed to get tmp buffer" << std::endl;
                return 1;
            }
            ptrs[j] = tmp_buffer;
            conn->write(&ptrs[j], sizeof(tmp_buffer)); // 返回服务器侧申请的内存地址
        }
        if(conn->read_one_now(ptrs[j], size, false) != RpcError::OK) {
            return 1;
        }
    }
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_mem2client(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function mem2client called" << std::endl;
#endif
    int ret = 1;
    RpcConn *conn = (RpcConn *)args0;
    void *ptrs[32];
    size_t sizes[32];
    int del_tmp_ptrs[32];
    std::vector<void *> ptrs2free;
    int i = 0;
    int j = 0;
    for(i = 0; i < 32; i++) {
        ptrs[i] = nullptr;
        sizes[i] = 0;
        del_tmp_ptrs[i] = false;
        // 读取服务器端指针
        if(conn->read_one_now(&ptrs[i], sizeof(ptrs[i]), false) != RpcError::OK) {
            goto ERROR;
        }

        if(ptrs[i] == (void *)0xffffffff) {
            break;
        }
        // 读取是否删除临时指针
        if(conn->read_one_now(&del_tmp_ptrs[i], sizeof(del_tmp_ptrs[i]), false) != RpcError::OK) {
            goto ERROR;
        }
        // 读取数据大小
        if(conn->read_one_now(&sizes[i], sizeof(sizes[i]), false) != RpcError::OK) {
            goto ERROR;
        }
    }
    for(j = 0; j < i; j++) {
        if(sizes[j] <= 0) {
            continue;
        }
        if(ptrs[j] == nullptr) {
            std::cerr << "WARNING: unknown server side host memory for conn pointer: 0x" << std::hex << ptrs[j] << std::endl;
            goto ERROR;
        }
        conn->write(ptrs[j], sizes[j], true);
        if(del_tmp_ptrs[j]) {
            ptrs2free.push_back(ptrs[j]);
        }
    }
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        goto ERROR;
    }
    ret = 0;
ERROR:
    for(auto ptr : ptrs2free) {
        conn->free_host_buffer(ptr);
    }
    return ret;
}

void mem2client_async_task(cudaStream_t stream, cudaError_t status, void *userData) {
#ifdef DEBUG
    std::cout << "Callback function mem2client_async_task triggered" << std::endl;
#endif
    Async2ClientInfo *async2clients = (Async2ClientInfo *)userData;
    RpcConn *conn = RpcServer::getInstance().get_async_conn(async2clients->client_id);
    if(conn == nullptr) {
        std::cerr << "Failed to get conn" << std::endl;
        if(async2clients != nullptr) {
            delete async2clients;
        }
        return;
    }
    conn->prepare_request(RPC_async_mem2client);
    for(auto &async2client : async2clients->async2clients) {
        // 写入客户端指针
        conn->write(&async2client.clientPtr, sizeof(async2client.clientPtr));
        // 写入服务器端数据
        conn->write(async2client.serverPtr, async2client.size, true);
    }
    void *end_flag = (void *)0xffffffff;
    conn->write(&end_flag, sizeof(end_flag));
    int result = 0;
    conn->read(&result, sizeof(result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        RpcServer::getInstance().release_async_conn(conn, true);
        if(async2clients != nullptr) {
            delete async2clients;
        }
        return;
    }
    RpcServer::getInstance().release_async_conn(conn);
    if(async2clients != nullptr) {
        delete async2clients;
    }
}

int handle_mem2client_async(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function mem2client_async called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    int i = 0;
    int j = 0;
    Async2ClientInfo *async2clients = new(std::nothrow) Async2ClientInfo();
    if(async2clients == nullptr) {
        std::cerr << "Failed to allocate async2clients" << std::endl;
        return 1;
    }
    async2clients->client_id = conn->get_client_id();
    while(true) {
        Async2Client info;
        if(conn->read_one_now(&info, sizeof(info), true) != RpcError::OK) {
            return 1;
        }

        if(info.clientPtr == (void *)0xffffffff) {
            if(conn->read_one_now(&stream, sizeof(stream), true) != RpcError::OK) {
                return 1;
            }
            break;
        }
        async2clients->async2clients.push_back(info);
    }
    cudaStreamAddCallback(stream, mem2client_async_task, (void *)async2clients, 0);
    return 0;
}

// #region CUDA Runtime API (cuda*)

int handle_cudaFree(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cudaFree called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args;
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaFree(devPtr);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cudaFreeHost(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cudaFreeHost called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args;
    void *ptr;
    conn->read(&ptr, sizeof(ptr));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaFreeHost(ptr);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cudaGetErrorName(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetErrorName called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args0;
    cudaError_t error;
    conn->read(&error, sizeof(error));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    const char *_result = cudaGetErrorName(error);
    conn->write(_result, strlen(_result) + 1, true);
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle_cudaGetErrorString(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetErrorString called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args0;
    cudaError_t error;
    conn->read(&error, sizeof(error));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    const char *_result = cudaGetErrorString(error);
    conn->write(_result, strlen(_result) + 1, true);
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle_cudaGetSymbolAddress(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetSymbolAddress called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args;
    void *devPtr;
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return cudaErrorUnknown;
    }
    cudaError_t _result = cudaGetSymbolAddress(&devPtr, symbol);
    conn->write(&devPtr, sizeof(devPtr));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return cudaErrorUnknown;
    }
    return 0;
}

int handle_cudaGraphMemcpyNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemcpyNodeGetParams called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    struct cudaMemcpy3DParms pNodeParams;
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    _result = cudaGraphMemcpyNodeGetParams(node, &pNodeParams);
    conn->write(&pNodeParams, sizeof(pNodeParams));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cudaHostAlloc(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cudaHostAlloc called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args;
    void *pHost;
    size_t size;
    unsigned int flags;
    conn->write(&pHost, sizeof(pHost));
    conn->read(&size, sizeof(size));
    conn->read(&flags, sizeof(flags));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return cudaErrorUnknown;
    }
    cudaError_t _result = cudaHostAlloc(&pHost, size, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    RpcConn *conn = (RpcConn *)args0;
    void *ptr;
    conn->read(&ptr, sizeof(ptr));
    size_t size;
    conn->read(&size, sizeof(size));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
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
        ptr = conn->alloc_host_buffer(size);
        if(ptr == nullptr) {
            std::cerr << "Failed to malloc" << std::endl;
            rtn = 1;
            goto _RTN_;
        }
    }
    conn->read_one_now(ptr, size, false);
    _result = cudaHostRegister(ptr, size, flags);
    conn->write(&ptr, sizeof(ptr));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    RpcConn *conn = (RpcConn *)args0;
    void *ptr;
    conn->read(&ptr, sizeof(ptr));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaHostUnregister(ptr);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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

    RpcConn *conn = (RpcConn *)args0;
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;
    int arg_count;
    conn->read(&func, sizeof(func));
    conn->read(&gridDim, sizeof(gridDim));
    conn->read(&blockDim, sizeof(blockDim));
    conn->read(&sharedMem, sizeof(sharedMem));
    conn->read(&stream, sizeof(stream));
    conn->read(&arg_count, sizeof(arg_count));

    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    args = (void **)malloc(sizeof(void *) * arg_count);
    if(args == nullptr) {
        std::cerr << "Failed to allocate args" << std::endl;
        return 1;
    }
    memset(args, 0, sizeof(void *) * arg_count);
    if(conn->read_all_now(args, nullptr, arg_count) != RpcError::OK) {
        std::cerr << "Failed to read args" << std::endl;
        free(args);
        return 1;
    }

    cudaError_t _result = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    for(int i = 0; i < arg_count; i++) {
        free(args[i]);
    }
    free(args);

    handle_mem2client_async(args0);

    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle_cudaLaunchCooperativeKernel(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaLaunchCooperativeKernel called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args0;
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;
    int arg_count;
    conn->read(&func, sizeof(func));
    conn->read(&gridDim, sizeof(gridDim));
    conn->read(&blockDim, sizeof(blockDim));
    conn->read(&sharedMem, sizeof(sharedMem));
    conn->read(&stream, sizeof(stream));
    conn->read(&arg_count, sizeof(arg_count));

    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    args = (void **)malloc(sizeof(void *) * arg_count);
    if(args == nullptr) {
        std::cerr << "Failed to allocate args" << std::endl;
        return 1;
    }
    memset(args, 0, sizeof(void *) * arg_count);
    if(conn->read_all_now(args, nullptr, arg_count) != RpcError::OK) {
        std::cerr << "Failed to read args" << std::endl;
        return 1;
    }

    cudaError_t _result = cudaLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, stream);
    for(int i = 0; i < arg_count; i++) {
        free(args[i]);
    }
    free(args);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle_cudaMalloc(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cudaMalloc called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args;
    size_t size;
    conn->read(&size, sizeof(size));
    void *devPtr;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaMalloc(&devPtr, size);
    conn->write(&devPtr, sizeof(devPtr));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cudaMalloc3D(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cudaMalloc3D called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args;
    struct cudaPitchedPtr pitchedDevPtr;
    struct cudaExtent extent;
    conn->read(&extent, sizeof(extent));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaMalloc3D(&pitchedDevPtr, extent);
    conn->write(&pitchedDevPtr, sizeof(pitchedDevPtr));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cudaMallocHost(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cudaMallocHost called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args;
    void *ptr;
    size_t size;
    conn->write(&ptr, sizeof(ptr));
    conn->read(&size, sizeof(size));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaMallocHost(&ptr, size);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cudaMallocManaged(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cudaMallocManaged called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args;
    void *devPtr;
    size_t size;
    unsigned int flags;
    conn->write(&devPtr, sizeof(devPtr));
    conn->read(&size, sizeof(size));
    conn->read(&flags, sizeof(flags));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaMallocManaged(&devPtr, size, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cudaMallocPitch(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cudaMallocPitch called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args;
    void *devPtr;
    size_t pitch;
    size_t width;
    size_t height;
    conn->read(&width, sizeof(width));
    conn->read(&height, sizeof(height));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaMallocPitch(&devPtr, &pitch, width, height);
    conn->write(&devPtr, sizeof(devPtr));
    conn->write(&pitch, sizeof(pitch));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    RpcConn *conn = (RpcConn *)args0;
    size_t *dataSizes;
    enum cudaMemRangeAttribute *attributes;
    void **data;
    size_t numAttributes;
    conn->read(&numAttributes, sizeof(numAttributes));
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    size_t count;
    conn->read(&count, sizeof(count));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    dataSizes = (size_t *)malloc(sizeof(size_t) * numAttributes);
    if(dataSizes == nullptr) {
        goto _RTN_;
    }
    buffers.insert(dataSizes);
    conn->read_one_now(dataSizes, sizeof(size_t) * numAttributes, false);

    attributes = (enum cudaMemRangeAttribute *)malloc(sizeof(enum cudaMemRangeAttribute) * numAttributes);
    if(attributes == nullptr) {
        goto _RTN_;
    }
    buffers.insert(attributes);
    conn->read_one_now(attributes, sizeof(enum cudaMemRangeAttribute) * numAttributes, false);

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
        conn->write(data[i], dataSizes[i], false);
    }
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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

    RpcConn *conn = (RpcConn *)args0;
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = __cudaPopCallConfiguration(&gridDim, &blockDim, &sharedMem, &stream);
    conn->write(&gridDim, sizeof(gridDim));
    conn->write(&blockDim, sizeof(blockDim));
    conn->write(&sharedMem, sizeof(sharedMem));
    conn->write(&stream, sizeof(stream));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle___cudaPushCallConfiguration(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function __cudaPushCallConfiguration called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args0;
    dim3 gridDim;
    conn->read(&gridDim, sizeof(gridDim));
    dim3 blockDim;
    conn->read(&blockDim, sizeof(blockDim));
    size_t sharedMem;
    conn->read(&sharedMem, sizeof(sharedMem));
    struct CUstream_st *stream;
    conn->read(&stream, sizeof(stream));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    unsigned _result = __cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle___cudaRegisterFatBinary(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function __cudaRegisterFatBinary called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args0;
    __cudaFatCudaBinary2 *fatCubin = (__cudaFatCudaBinary2 *)malloc(sizeof(__cudaFatCudaBinary2));
    if(fatCubin == nullptr) {
        std::cerr << "Failed to allocate fatCubin" << std::endl;
        return 1;
    }
    conn->read(fatCubin, sizeof(__cudaFatCudaBinary2));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    void *cubin = nullptr;
    int len;
    conn->read_one_now(&cubin, 0, true);
    fatCubin->text = (uint64_t)cubin;
    void **_result = __cudaRegisterFatBinary(fatCubin);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle___cudaRegisterFatBinaryEnd(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function __cudaRegisterFatBinaryEnd called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args0;
    void **fatCubinHandle;
    conn->read(&fatCubinHandle, sizeof(fatCubinHandle));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    __cudaRegisterFatBinaryEnd(fatCubinHandle);
    if(conn->submit_response() != RpcError::OK) {
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
    RpcConn *conn = (RpcConn *)args0;
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
    conn->read(&fatCubinHandle, sizeof(fatCubinHandle));
    conn->read(&hostFun, sizeof(hostFun));
    conn->read(&deviceName, 0, true);
    conn->read(&thread_limit, sizeof(thread_limit));
    conn->read(&mask, sizeof(mask));

    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    if(mask & 1 << 0) {
        conn->read_one_now(&tid, sizeof(uint3), false);
    }
    if(mask & 1 << 1) {
        conn->read_one_now(&bid, sizeof(uint3), false);
    }
    if(mask & 1 << 2) {
        conn->read_one_now(&bDim, sizeof(dim3), false);
    }
    if(mask & 1 << 3) {
        conn->read_one_now(&gDim, sizeof(dim3), false);
    }
    if(mask & 1 << 4) {
        conn->read_one_now(&wSize, sizeof(wSize), false);
    }
    __cudaRegisterFunction(fatCubinHandle, hostFun, deviceName, deviceName, thread_limit, mask & 1 << 0 ? &tid : nullptr, mask & 1 << 1 ? &bid : nullptr, mask & 1 << 2 ? &bDim : nullptr, mask & 1 << 3 ? &gDim : nullptr, mask & 1 << 4 ? &wSize : nullptr);
    if(conn->submit_response() != RpcError::OK) {
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
    RpcConn *conn = (RpcConn *)args0;
    void **fatCubinHandle;
    void *hostVarPtrAddress = nullptr;
    char *deviceName = nullptr;
    conn->read(&fatCubinHandle, sizeof(fatCubinHandle));
    conn->read(&deviceName, 0, true);
    int ext;
    conn->read(&ext, sizeof(ext));
    size_t size;
    conn->read(&size, sizeof(size));
    int constant;
    conn->read(&constant, sizeof(constant));
    int global;
    conn->read(&global, sizeof(global));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    __cudaRegisterManagedVar(fatCubinHandle, &hostVarPtrAddress, deviceName, deviceName, ext, size, constant, global);
    if(conn->submit_response() != RpcError::OK) {
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
    RpcConn *conn = (RpcConn *)args0;
    void **fatCubinHandle;
    char *hostVar;
    char *deviceName = nullptr;
    conn->read(&fatCubinHandle, sizeof(fatCubinHandle));
    conn->read(&hostVar, sizeof(hostVar));
    conn->read(&deviceName, 0, true);
    int ext;
    conn->read(&ext, sizeof(ext));
    size_t size;
    conn->read(&size, sizeof(size));
    int constant;
    conn->read(&constant, sizeof(constant));
    int global;
    conn->read(&global, sizeof(global));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    __cudaRegisterVar(fatCubinHandle, hostVar, deviceName, deviceName, ext, size, constant, global);
    if(conn->submit_response() != RpcError::OK) {
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

    RpcConn *conn = (RpcConn *)args0;
    void **fatCubinHandle;
    conn->read(&fatCubinHandle, sizeof(fatCubinHandle));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    __cudaUnregisterFatBinary(fatCubinHandle);
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle___cudaInitModule(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function __cudaInitModule called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args0;
    void *fatCubinHandle;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    char _result = __cudaInitModule(&fatCubinHandle);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

// #endregion
// #region CUDA Driver API (cu*)

int handle_cuExternalMemoryGetMappedBuffer(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuExternalMemoryGetMappedBuffer called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args;
    CUdeviceptr devPtr;
    CUexternalMemory extMem;
    conn->read(&extMem, sizeof(extMem));
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc;
    conn->read(&bufferDesc, sizeof(bufferDesc));

    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }

    CUresult _result = cuExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufferDesc);
    conn->write(&devPtr, sizeof(devPtr));
    conn->write(&_result, sizeof(_result));

    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle_cuGetErrorName(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGetErrorName called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args0;
    CUresult error;
    conn->read(&error, sizeof(error));
    const char *pStr;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuGetErrorName(error, &pStr);
    conn->write(pStr, strlen(pStr) + 1, true);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuGetErrorString(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGetErrorString called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args0;
    CUresult error;
    conn->read(&error, sizeof(error));
    const char *pStr;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuGetErrorString(error, &pStr);
    conn->write(pStr, strlen(pStr) + 1, true);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    std::cout << "Handle function cuGraphicsResourceGetMappedPointer_v2 called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr devPtr;
    size_t size;
    CUgraphicsResource resource;
    conn->read(&resource, sizeof(resource));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuGraphicsResourceGetMappedPointer_v2(&devPtr, &size, resource);
    conn->write(&devPtr, sizeof(devPtr));
    conn->write(&size, sizeof(size));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuLaunchCooperativeKernel(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLaunchCooperativeKernel called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args0;
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
    conn->read(&func, sizeof(func));
    conn->read(&gridDimX, sizeof(gridDimX));
    conn->read(&gridDimY, sizeof(gridDimY));
    conn->read(&gridDimZ, sizeof(gridDimZ));
    conn->read(&blockDimX, sizeof(blockDimX));
    conn->read(&blockDimY, sizeof(blockDimY));
    conn->read(&blockDimZ, sizeof(blockDimZ));
    conn->read(&sharedMemBytes, sizeof(sharedMemBytes));
    conn->read(&hStream, sizeof(hStream));
    conn->read(&arg_count, sizeof(arg_count));

    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    kernelParams = (void **)malloc(sizeof(void *) * arg_count);
    if(kernelParams == nullptr) {
        std::cerr << "Failed to allocate args" << std::endl;
        return 1;
    }
    if(conn->read_all_now(kernelParams, nullptr, arg_count) != RpcError::OK) {
        std::cerr << "Failed to read args" << std::endl;
        return 1;
    }
    CUresult _result = cuLaunchCooperativeKernel(func, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
    for(int i = 0; i < arg_count; i++) {
        free(kernelParams[i]);
    }
    free(kernelParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle_cuImportExternalMemory(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuImportExternalMemory called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args0;
    CUexternalMemory extMem_out;
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memHandleDesc;
    conn->read(&memHandleDesc, sizeof(memHandleDesc));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuImportExternalMemory(&extMem_out, &memHandleDesc);
    conn->write(&extMem_out, sizeof(extMem_out));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuIpcOpenMemHandle_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuIpcOpenMemHandle_v2 called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dptr;
    CUipcMemHandle handle;
    unsigned int Flags;
    conn->read(&handle, sizeof(handle));
    conn->read(&Flags, sizeof(Flags));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuIpcOpenMemHandle_v2(&dptr, handle, Flags);
    conn->write(&dptr, sizeof(dptr));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

// #if CUDA_VERSION > 11040
int handle_cuLibraryGetGlobal(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLibraryGetGlobal called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dptr;
    size_t bytes;
    CUlibrary library;
    char *name = nullptr;
    conn->read(&library, sizeof(library));
    conn->read(&name, 0, true);
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        if(name != nullptr) {
            free(name);
        }
        return 1;
    }
    CUresult _result = cuLibraryGetGlobal(&dptr, &bytes, library, name);
    conn->write(&dptr, sizeof(dptr));
    conn->write(&bytes, sizeof(bytes));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    std::cout << "Handle function cuLibraryGetManaged called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dptr;
    size_t bytes;
    CUlibrary library;
    char *name = nullptr;
    conn->read(&library, sizeof(library));
    conn->read(&name, 0, true);
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        if(name != nullptr) {
            free(name);
        }
        return 1;
    }
    CUresult _result = cuLibraryGetManaged(&dptr, &bytes, library, name);
    conn->write(&dptr, sizeof(dptr));
    conn->write(&bytes, sizeof(bytes));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
// #endif

int handle_cuMemAddressReserve(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemAddressReserve called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr ptr;
    size_t size;
    size_t alignment;
    CUdeviceptr addr;
    unsigned long long flags;
    conn->read(&size, sizeof(size));
    conn->read(&alignment, sizeof(alignment));
    conn->read(&addr, sizeof(addr));
    conn->read(&flags, sizeof(flags));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemAddressReserve(&ptr, size, alignment, addr, flags);
    conn->write(&ptr, sizeof(ptr));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemAlloc_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuMemAlloc_v2 called" << std::endl;
#endif

    RpcConn *conn = (RpcConn *)args;
    CUdeviceptr dptr;
    size_t bytesize;
    conn->read(&bytesize, sizeof(bytesize));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemAlloc_v2(&dptr, bytesize);
    conn->write(&dptr, sizeof(dptr));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemAllocHost_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuMemAllocHost_v2 called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    void *pp;
    size_t bytesize;
    conn->read(&bytesize, sizeof(bytesize));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemAllocHost_v2(&pp, bytesize);
    conn->write(&pp, sizeof(pp));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemAllocManaged(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuMemAllocManaged called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    CUdeviceptr ptr;
    size_t bytesize;
    unsigned int flags;
    conn->read(&bytesize, sizeof(bytesize));
    conn->read(&flags, sizeof(flags));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemAllocManaged(&ptr, bytesize, flags);
    conn->write(&ptr, sizeof(ptr));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemAllocPitch_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuMemAllocPitch_v2 called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    CUdeviceptr dptr;
    size_t pitch;
    size_t WidthInBytes;
    size_t Height;
    unsigned int ElementSizeBytes;
    conn->read(&WidthInBytes, sizeof(WidthInBytes));
    conn->read(&Height, sizeof(Height));
    conn->read(&ElementSizeBytes, sizeof(ElementSizeBytes));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemAllocPitch_v2(&dptr, &pitch, WidthInBytes, Height, ElementSizeBytes);
    conn->write(&dptr, sizeof(dptr));
    conn->write(&pitch, sizeof(pitch));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemCreate(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuMemCreate called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    CUmemGenericAllocationHandle handle;
    size_t size;
    CUmemAllocationProp prop;
    unsigned long long flags;
    conn->read(&size, sizeof(size));
    conn->read(&prop, sizeof(prop));
    conn->read(&flags, sizeof(flags));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemCreate(&handle, size, &prop, flags);
    conn->write(&handle, sizeof(handle));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemFreeHost(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuMemFreeHost called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    void *p;
    conn->read(&p, sizeof(p));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemFreeHost(p);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemGetAddressRange_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuMemGetAddressRange_v2 called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    CUdeviceptr base;
    size_t size;
    CUdeviceptr dptr;
    conn->read(&dptr, sizeof(dptr));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemGetAddressRange_v2(&base, &size, dptr);
    conn->write(&base, sizeof(base));
    conn->write(&size, sizeof(size));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    RpcConn *conn = (RpcConn *)args0;
    size_t *dataSizes;
    CUmem_range_attribute *attributes;
    void **data;
    size_t numAttributes;
    conn->read(&numAttributes, sizeof(numAttributes));
    CUdeviceptr devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    size_t count;
    conn->read(&count, sizeof(count));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    dataSizes = (size_t *)malloc(sizeof(size_t) * numAttributes);
    if(dataSizes == nullptr) {
        goto _RTN_;
    }
    buffers.insert(dataSizes);
    conn->read_one_now(dataSizes, sizeof(size_t) * numAttributes, false);

    attributes = (CUmem_range_attribute *)malloc(sizeof(CUmem_range_attribute) * numAttributes);
    if(attributes == nullptr) {
        goto _RTN_;
    }
    buffers.insert(attributes);
    conn->read_one_now(attributes, sizeof(enum cudaMemRangeAttribute) * numAttributes, false);

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
        conn->write(data[i], dataSizes[i], false);
    }
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    std::cout << "Handle function cuMemHostAlloc called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    void *p;
    size_t bytesize;
    unsigned int Flags;
    conn->read(&bytesize, sizeof(bytesize));
    conn->read(&Flags, sizeof(Flags));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemHostAlloc(&p, bytesize, Flags);
    conn->write(&p, sizeof(p));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemHostGetDevicePointer_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuMemHostGetDevicePointer_v2 called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    CUdeviceptr dptr;
    void *p;
    unsigned int Flags;
    conn->read(&p, sizeof(p));
    conn->read(&Flags, sizeof(Flags));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemHostGetDevicePointer_v2(&dptr, p, Flags);
    conn->write(&dptr, sizeof(dptr));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemMap(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuMemMap called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    CUdeviceptr ptr;
    size_t size;
    size_t offset;
    CUmemGenericAllocationHandle handle;
    unsigned long long flags;
    conn->read(&ptr, sizeof(ptr));
    conn->read(&size, sizeof(size));
    conn->read(&offset, sizeof(offset));
    conn->read(&handle, sizeof(handle));
    conn->read(&flags, sizeof(flags));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemMap(ptr, size, offset, handle, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemPoolImportPointer(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuMemPoolImportPointer called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    CUdeviceptr ptr_out;
    CUmemoryPool pool;
    CUmemPoolPtrExportData shareData;
    conn->read(&pool, sizeof(pool));
    conn->read(&shareData, sizeof(shareData));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemPoolImportPointer(&ptr_out, pool, &shareData);
    conn->write(&ptr_out, sizeof(ptr_out));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuMemRelease(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuMemRelease called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    CUmemGenericAllocationHandle handle;
    conn->read(&handle, sizeof(handle));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuMemRelease(handle);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuModuleGetGlobal_v2(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuModuleGetGlobal_v2 called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    CUdeviceptr dptr;
    size_t bytes;
    CUmodule hmod;
    char *name = nullptr;
    conn->read(&hmod, sizeof(hmod));
    conn->read(&name, 0, true);
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        if(name != nullptr) {
            free(name);
        }
        return 1;
    }
    CUresult _result = cuModuleGetGlobal_v2(&dptr, &bytes, hmod, name);
    conn->write(&dptr, sizeof(dptr));
    conn->write(&bytes, sizeof(bytes));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int numAttributes;
    conn->read(&numAttributes, sizeof(numAttributes));
    CUdeviceptr ptr;
    conn->read(&ptr, sizeof(ptr));

    CUpointer_attribute *attributes;
    void **data;
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    attributes = (CUpointer_attribute *)malloc(sizeof(CUpointer_attribute) * numAttributes);
    if(attributes == nullptr) {
        goto _RTN_;
    }
    buffers.insert(attributes);
    conn->read_one_now(attributes, sizeof(CUpointer_attribute) * numAttributes, false);
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
        conn->write(data[i], getAttributeSize(attributes[i]), false);
    }
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    std::cout << "Handle function cuTexRefGetAddress_v2 called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    CUdeviceptr pdptr;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuTexRefGetAddress_v2(&pdptr, hTexRef);
    conn->write(&pdptr, sizeof(pdptr));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}

int handle_cuGraphMemFreeNodeGetParams(void *args) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphMemFreeNodeGetParams called" << std::endl;
#endif
    RpcConn *conn = (RpcConn *)args;
    CUgraphNode hNode;
    CUdeviceptr dptr_out;
    conn->read(&hNode, sizeof(hNode));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    CUresult _result = cuGraphMemFreeNodeGetParams(hNode, &dptr_out);
    conn->write(&dptr_out, sizeof(dptr_out));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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

    RpcConn *conn = (RpcConn *)args0;
    nvmlReturn_t result;
    conn->read(&result, sizeof(result));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    const char *_result = nvmlErrorString(result);
    conn->write(_result, strlen(_result) + 1, true);
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

// #endregion
