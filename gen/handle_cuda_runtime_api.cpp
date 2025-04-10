#include <iostream>
#include <unordered_map>
#include "hook_api.h"
#include "handle_server.h"
#include "../rpc.h"
#include "cuda_runtime_api.h"

int handle_cudaDeviceReset(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceReset called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceReset();
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

int handle_cudaDeviceSynchronize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceSynchronize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceSynchronize();
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

int handle_cudaDeviceSetLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceSetLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    enum cudaLimit limit;
    rpc_read(client, &limit, sizeof(limit));
    size_t value;
    rpc_read(client, &value, sizeof(value));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceSetLimit(limit, value);
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

int handle_cudaDeviceGetLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t *pValue;
    rpc_read(client, &pValue, sizeof(pValue));
    enum cudaLimit limit;
    rpc_read(client, &limit, sizeof(limit));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetLimit(pValue, limit);
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

int handle_cudaDeviceGetTexture1DLinearMaxWidth(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetTexture1DLinearMaxWidth called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t *maxWidthInElements;
    rpc_read(client, &maxWidthInElements, sizeof(maxWidthInElements));
    struct cudaChannelFormatDesc *fmtDesc;
    rpc_read(client, &fmtDesc, sizeof(fmtDesc));
    int device;
    rpc_read(client, &device, sizeof(device));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device);
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

int handle_cudaDeviceGetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    enum cudaFuncCache *pCacheConfig;
    rpc_read(client, &pCacheConfig, sizeof(pCacheConfig));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetCacheConfig(pCacheConfig);
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

int handle_cudaDeviceGetStreamPriorityRange(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetStreamPriorityRange called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int *leastPriority;
    rpc_read(client, &leastPriority, sizeof(leastPriority));
    int *greatestPriority;
    rpc_read(client, &greatestPriority, sizeof(greatestPriority));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
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

int handle_cudaDeviceSetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceSetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    enum cudaFuncCache cacheConfig;
    rpc_read(client, &cacheConfig, sizeof(cacheConfig));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceSetCacheConfig(cacheConfig);
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

int handle_cudaDeviceGetSharedMemConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetSharedMemConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    enum cudaSharedMemConfig *pConfig;
    rpc_read(client, &pConfig, sizeof(pConfig));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetSharedMemConfig(pConfig);
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

int handle_cudaDeviceSetSharedMemConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceSetSharedMemConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    enum cudaSharedMemConfig config;
    rpc_read(client, &config, sizeof(config));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceSetSharedMemConfig(config);
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

int handle_cudaDeviceGetByPCIBusId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetByPCIBusId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int *device;
    rpc_read(client, &device, sizeof(device));
    char *pciBusId = nullptr;
    rpc_read(client, &pciBusId, 0, true);
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(pciBusId);
    _result = cudaDeviceGetByPCIBusId(device, pciBusId);
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

int handle_cudaDeviceGetPCIBusId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetPCIBusId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    char pciBusId[1024];
    int len;
    rpc_read(client, &len, sizeof(len));
    int device;
    rpc_read(client, &device, sizeof(device));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetPCIBusId(pciBusId, len, device);
    rpc_write(client, pciBusId, strlen(pciBusId) + 1, true);
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

int handle_cudaIpcGetEventHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaIpcGetEventHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaIpcEventHandle_t *handle;
    rpc_read(client, &handle, sizeof(handle));
    cudaEvent_t event;
    rpc_read(client, &event, sizeof(event));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaIpcGetEventHandle(handle, event);
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

int handle_cudaIpcOpenEventHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaIpcOpenEventHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaEvent_t *event;
    rpc_read(client, &event, sizeof(event));
    cudaIpcEventHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaIpcOpenEventHandle(event, handle);
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

int handle_cudaIpcGetMemHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaIpcGetMemHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaIpcMemHandle_t *handle;
    rpc_read(client, &handle, sizeof(handle));
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaIpcGetMemHandle(handle, devPtr);
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

int handle_cudaIpcOpenMemHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaIpcOpenMemHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *devPtr;
    cudaIpcMemHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaIpcOpenMemHandle(&devPtr, handle, flags);
    rpc_write(client, &devPtr, sizeof(devPtr));
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

int handle_cudaIpcCloseMemHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaIpcCloseMemHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaIpcCloseMemHandle(devPtr);
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

int handle_cudaDeviceFlushGPUDirectRDMAWrites(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceFlushGPUDirectRDMAWrites called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    enum cudaFlushGPUDirectRDMAWritesTarget target;
    rpc_read(client, &target, sizeof(target));
    enum cudaFlushGPUDirectRDMAWritesScope scope;
    rpc_read(client, &scope, sizeof(scope));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceFlushGPUDirectRDMAWrites(target, scope);
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

int handle_cudaThreadExit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadExit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadExit();
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

int handle_cudaThreadSynchronize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadSynchronize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadSynchronize();
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

int handle_cudaThreadSetLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadSetLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    enum cudaLimit limit;
    rpc_read(client, &limit, sizeof(limit));
    size_t value;
    rpc_read(client, &value, sizeof(value));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadSetLimit(limit, value);
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

int handle_cudaThreadGetLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadGetLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t *pValue;
    rpc_read(client, &pValue, sizeof(pValue));
    enum cudaLimit limit;
    rpc_read(client, &limit, sizeof(limit));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadGetLimit(pValue, limit);
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

int handle_cudaThreadGetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadGetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    enum cudaFuncCache *pCacheConfig;
    rpc_read(client, &pCacheConfig, sizeof(pCacheConfig));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadGetCacheConfig(pCacheConfig);
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

int handle_cudaThreadSetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadSetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    enum cudaFuncCache cacheConfig;
    rpc_read(client, &cacheConfig, sizeof(cacheConfig));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadSetCacheConfig(cacheConfig);
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

int handle_cudaGetLastError(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetLastError called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetLastError();
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

int handle_cudaPeekAtLastError(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaPeekAtLastError called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaPeekAtLastError();
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

int handle_cudaGetDeviceCount(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetDeviceCount called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int *count;
    rpc_read(client, &count, sizeof(count));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetDeviceCount(count);
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

int handle_cudaGetDeviceProperties(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetDeviceProperties called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaDeviceProp *prop;
    rpc_read(client, &prop, sizeof(prop));
    int device;
    rpc_read(client, &device, sizeof(device));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetDeviceProperties(prop, device);
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

int handle_cudaDeviceGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int *value;
    rpc_read(client, &value, sizeof(value));
    enum cudaDeviceAttr attr;
    rpc_read(client, &attr, sizeof(attr));
    int device;
    rpc_read(client, &device, sizeof(device));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetAttribute(value, attr, device);
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

int handle_cudaDeviceGetDefaultMemPool(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetDefaultMemPool called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaMemPool_t *memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    int device;
    rpc_read(client, &device, sizeof(device));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetDefaultMemPool(memPool, device);
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

int handle_cudaDeviceSetMemPool(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceSetMemPool called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int device;
    rpc_read(client, &device, sizeof(device));
    cudaMemPool_t memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceSetMemPool(device, memPool);
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

int handle_cudaDeviceGetMemPool(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetMemPool called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaMemPool_t *memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    int device;
    rpc_read(client, &device, sizeof(device));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetMemPool(memPool, device);
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

int handle_cudaDeviceGetNvSciSyncAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetNvSciSyncAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *nvSciSyncAttrList;
    rpc_read(client, &nvSciSyncAttrList, sizeof(nvSciSyncAttrList));
    int device;
    rpc_read(client, &device, sizeof(device));
    int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, device, flags);
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

int handle_cudaDeviceGetP2PAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetP2PAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int *value;
    rpc_read(client, &value, sizeof(value));
    enum cudaDeviceP2PAttr attr;
    rpc_read(client, &attr, sizeof(attr));
    int srcDevice;
    rpc_read(client, &srcDevice, sizeof(srcDevice));
    int dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice);
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

int handle_cudaChooseDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaChooseDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int *device;
    rpc_read(client, &device, sizeof(device));
    struct cudaDeviceProp *prop;
    rpc_read(client, &prop, sizeof(prop));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaChooseDevice(device, prop);
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

int handle_cudaSetDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaSetDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int device;
    rpc_read(client, &device, sizeof(device));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaSetDevice(device);
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

int handle_cudaGetDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int *device;
    rpc_read(client, &device, sizeof(device));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetDevice(device);
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

int handle_cudaSetValidDevices(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaSetValidDevices called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int *device_arr;
    rpc_read(client, &device_arr, sizeof(device_arr));
    int len;
    rpc_read(client, &len, sizeof(len));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaSetValidDevices(device_arr, len);
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

int handle_cudaSetDeviceFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaSetDeviceFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaSetDeviceFlags(flags);
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

int handle_cudaGetDeviceFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetDeviceFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int *flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetDeviceFlags(flags);
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

int handle_cudaStreamCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t *pStream;
    rpc_read(client, &pStream, sizeof(pStream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamCreate(pStream);
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

int handle_cudaStreamCreateWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamCreateWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t *pStream;
    rpc_read(client, &pStream, sizeof(pStream));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamCreateWithFlags(pStream, flags);
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

int handle_cudaStreamCreateWithPriority(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamCreateWithPriority called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t *pStream;
    rpc_read(client, &pStream, sizeof(pStream));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    int priority;
    rpc_read(client, &priority, sizeof(priority));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamCreateWithPriority(pStream, flags, priority);
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

int handle_cudaStreamGetPriority(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamGetPriority called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    int *priority;
    rpc_read(client, &priority, sizeof(priority));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamGetPriority(hStream, priority);
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

int handle_cudaStreamGetFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamGetFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    unsigned int *flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamGetFlags(hStream, flags);
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

int handle_cudaCtxResetPersistingL2Cache(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaCtxResetPersistingL2Cache called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaCtxResetPersistingL2Cache();
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

int handle_cudaStreamCopyAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamCopyAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t dst;
    rpc_read(client, &dst, sizeof(dst));
    cudaStream_t src;
    rpc_read(client, &src, sizeof(src));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamCopyAttributes(dst, src);
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

int handle_cudaStreamGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    enum cudaStreamAttrID attr;
    rpc_read(client, &attr, sizeof(attr));
    union cudaStreamAttrValue *value_out;
    rpc_read(client, &value_out, sizeof(value_out));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamGetAttribute(hStream, attr, value_out);
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

int handle_cudaStreamSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    enum cudaStreamAttrID attr;
    rpc_read(client, &attr, sizeof(attr));
    union cudaStreamAttrValue *value;
    rpc_read(client, &value, sizeof(value));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamSetAttribute(hStream, attr, value);
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

int handle_cudaStreamDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamDestroy(stream);
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

int handle_cudaStreamWaitEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamWaitEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaEvent_t event;
    rpc_read(client, &event, sizeof(event));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamWaitEvent(stream, event, flags);
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

int handle_cudaStreamAddCallback(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamAddCallback called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaStreamCallback_t callback;
    rpc_read(client, &callback, sizeof(callback));
    void *userData;
    rpc_read(client, &userData, sizeof(userData));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamAddCallback(stream, callback, userData, flags);
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

int handle_cudaStreamSynchronize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamSynchronize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamSynchronize(stream);
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

int handle_cudaStreamQuery(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamQuery called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamQuery(stream);
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

int handle_cudaStreamAttachMemAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamAttachMemAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    size_t length;
    rpc_read(client, &length, sizeof(length));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamAttachMemAsync(stream, devPtr, length, flags);
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

int handle_cudaStreamBeginCapture(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamBeginCapture called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    enum cudaStreamCaptureMode mode;
    rpc_read(client, &mode, sizeof(mode));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamBeginCapture(stream, mode);
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

int handle_cudaThreadExchangeStreamCaptureMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadExchangeStreamCaptureMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    enum cudaStreamCaptureMode *mode;
    rpc_read(client, &mode, sizeof(mode));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadExchangeStreamCaptureMode(mode);
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

int handle_cudaStreamEndCapture(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamEndCapture called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaGraph_t *pGraph;
    rpc_read(client, &pGraph, sizeof(pGraph));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamEndCapture(stream, pGraph);
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

int handle_cudaStreamIsCapturing(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamIsCapturing called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    enum cudaStreamCaptureStatus *pCaptureStatus;
    rpc_read(client, &pCaptureStatus, sizeof(pCaptureStatus));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamIsCapturing(stream, pCaptureStatus);
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

int handle_cudaStreamGetCaptureInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamGetCaptureInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    enum cudaStreamCaptureStatus *pCaptureStatus;
    rpc_read(client, &pCaptureStatus, sizeof(pCaptureStatus));
    unsigned long long *pId;
    rpc_read(client, &pId, sizeof(pId));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamGetCaptureInfo(stream, pCaptureStatus, pId);
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

int handle_cudaStreamGetCaptureInfo_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamGetCaptureInfo_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    enum cudaStreamCaptureStatus *captureStatus_out;
    rpc_read(client, &captureStatus_out, sizeof(captureStatus_out));
    unsigned long long *id_out;
    rpc_read(client, &id_out, sizeof(id_out));
    cudaGraph_t *graph_out;
    rpc_read(client, &graph_out, sizeof(graph_out));
    const cudaGraphNode_t *dependencies_out;
    size_t *numDependencies_out;
    rpc_read(client, &numDependencies_out, sizeof(numDependencies_out));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, &dependencies_out, numDependencies_out);
    rpc_write(client, dependencies_out, sizeof(cudaGraphNode_t));
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

int handle_cudaStreamUpdateCaptureDependencies(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamUpdateCaptureDependencies called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaGraphNode_t *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags);
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

int handle_cudaEventCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaEvent_t *event;
    rpc_read(client, &event, sizeof(event));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventCreate(event);
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

int handle_cudaEventCreateWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventCreateWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaEvent_t *event;
    rpc_read(client, &event, sizeof(event));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventCreateWithFlags(event, flags);
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

int handle_cudaEventRecord(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventRecord called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaEvent_t event;
    rpc_read(client, &event, sizeof(event));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventRecord(event, stream);
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

int handle_cudaEventRecordWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventRecordWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaEvent_t event;
    rpc_read(client, &event, sizeof(event));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventRecordWithFlags(event, stream, flags);
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

int handle_cudaEventQuery(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventQuery called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaEvent_t event;
    rpc_read(client, &event, sizeof(event));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventQuery(event);
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

int handle_cudaEventSynchronize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventSynchronize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaEvent_t event;
    rpc_read(client, &event, sizeof(event));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventSynchronize(event);
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

int handle_cudaEventDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaEvent_t event;
    rpc_read(client, &event, sizeof(event));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventDestroy(event);
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

int handle_cudaEventElapsedTime(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventElapsedTime called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    float *ms;
    rpc_read(client, &ms, sizeof(ms));
    cudaEvent_t start;
    rpc_read(client, &start, sizeof(start));
    cudaEvent_t end;
    rpc_read(client, &end, sizeof(end));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventElapsedTime(ms, start, end);
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

int handle_cudaImportExternalMemory(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaImportExternalMemory called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaExternalMemory_t *extMem_out;
    rpc_read(client, &extMem_out, sizeof(extMem_out));
    struct cudaExternalMemoryHandleDesc *memHandleDesc;
    rpc_read(client, &memHandleDesc, sizeof(memHandleDesc));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaImportExternalMemory(extMem_out, memHandleDesc);
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

int handle_cudaExternalMemoryGetMappedBuffer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaExternalMemoryGetMappedBuffer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *devPtr;
    cudaExternalMemory_t extMem;
    rpc_read(client, &extMem, sizeof(extMem));
    struct cudaExternalMemoryBufferDesc *bufferDesc;
    rpc_read(client, &bufferDesc, sizeof(bufferDesc));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, bufferDesc);
    rpc_write(client, &devPtr, sizeof(devPtr));
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

int handle_cudaExternalMemoryGetMappedMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaExternalMemoryGetMappedMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaMipmappedArray_t *mipmap;
    rpc_read(client, &mipmap, sizeof(mipmap));
    cudaExternalMemory_t extMem;
    rpc_read(client, &extMem, sizeof(extMem));
    struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc;
    rpc_read(client, &mipmapDesc, sizeof(mipmapDesc));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc);
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

int handle_cudaDestroyExternalMemory(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDestroyExternalMemory called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaExternalMemory_t extMem;
    rpc_read(client, &extMem, sizeof(extMem));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDestroyExternalMemory(extMem);
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

int handle_cudaImportExternalSemaphore(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaImportExternalSemaphore called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaExternalSemaphore_t *extSem_out;
    rpc_read(client, &extSem_out, sizeof(extSem_out));
    struct cudaExternalSemaphoreHandleDesc *semHandleDesc;
    rpc_read(client, &semHandleDesc, sizeof(semHandleDesc));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaImportExternalSemaphore(extSem_out, semHandleDesc);
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

int handle_cudaSignalExternalSemaphoresAsync_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaSignalExternalSemaphoresAsync_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaExternalSemaphore_t *extSemArray;
    rpc_read(client, &extSemArray, sizeof(extSemArray));
    struct cudaExternalSemaphoreSignalParams *paramsArray;
    rpc_read(client, &paramsArray, sizeof(paramsArray));
    unsigned int numExtSems;
    rpc_read(client, &numExtSems, sizeof(numExtSems));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaSignalExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream);
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

int handle_cudaWaitExternalSemaphoresAsync_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaWaitExternalSemaphoresAsync_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaExternalSemaphore_t *extSemArray;
    rpc_read(client, &extSemArray, sizeof(extSemArray));
    struct cudaExternalSemaphoreWaitParams *paramsArray;
    rpc_read(client, &paramsArray, sizeof(paramsArray));
    unsigned int numExtSems;
    rpc_read(client, &numExtSems, sizeof(numExtSems));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaWaitExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream);
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

int handle_cudaDestroyExternalSemaphore(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDestroyExternalSemaphore called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaExternalSemaphore_t extSem;
    rpc_read(client, &extSem, sizeof(extSem));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDestroyExternalSemaphore(extSem);
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

int handle_cudaLaunchCooperativeKernelMultiDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaLaunchCooperativeKernelMultiDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaLaunchParams *launchParamsList;
    rpc_read(client, &launchParamsList, sizeof(launchParamsList));
    unsigned int numDevices;
    rpc_read(client, &numDevices, sizeof(numDevices));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
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

int handle_cudaFuncSetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFuncSetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *func;
    rpc_read(client, &func, sizeof(func));
    enum cudaFuncCache cacheConfig;
    rpc_read(client, &cacheConfig, sizeof(cacheConfig));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFuncSetCacheConfig(func, cacheConfig);
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

int handle_cudaFuncSetSharedMemConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFuncSetSharedMemConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *func;
    rpc_read(client, &func, sizeof(func));
    enum cudaSharedMemConfig config;
    rpc_read(client, &config, sizeof(config));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFuncSetSharedMemConfig(func, config);
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

int handle_cudaFuncGetAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFuncGetAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaFuncAttributes *attr;
    rpc_read(client, &attr, sizeof(attr));
    void *func;
    rpc_read(client, &func, sizeof(func));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFuncGetAttributes(attr, func);
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

int handle_cudaFuncSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFuncSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *func;
    rpc_read(client, &func, sizeof(func));
    enum cudaFuncAttribute attr;
    rpc_read(client, &attr, sizeof(attr));
    int value;
    rpc_read(client, &value, sizeof(value));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFuncSetAttribute(func, attr, value);
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

int handle_cudaSetDoubleForDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaSetDoubleForDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    double *d;
    rpc_read(client, &d, sizeof(d));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaSetDoubleForDevice(d);
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

int handle_cudaSetDoubleForHost(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaSetDoubleForHost called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    double *d;
    rpc_read(client, &d, sizeof(d));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaSetDoubleForHost(d);
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

int handle_cudaLaunchHostFunc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaLaunchHostFunc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaHostFn_t fn;
    rpc_read(client, &fn, sizeof(fn));
    void *userData;
    rpc_read(client, &userData, sizeof(userData));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaLaunchHostFunc(stream, fn, userData);
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

int handle_cudaOccupancyMaxActiveBlocksPerMultiprocessor(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaOccupancyMaxActiveBlocksPerMultiprocessor called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int *numBlocks;
    rpc_read(client, &numBlocks, sizeof(numBlocks));
    void *func;
    rpc_read(client, &func, sizeof(func));
    int blockSize;
    rpc_read(client, &blockSize, sizeof(blockSize));
    size_t dynamicSMemSize;
    rpc_read(client, &dynamicSMemSize, sizeof(dynamicSMemSize));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
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

int handle_cudaOccupancyAvailableDynamicSMemPerBlock(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaOccupancyAvailableDynamicSMemPerBlock called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t *dynamicSmemSize;
    rpc_read(client, &dynamicSmemSize, sizeof(dynamicSmemSize));
    void *func;
    rpc_read(client, &func, sizeof(func));
    int numBlocks;
    rpc_read(client, &numBlocks, sizeof(numBlocks));
    int blockSize;
    rpc_read(client, &blockSize, sizeof(blockSize));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize);
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

int handle_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int *numBlocks;
    rpc_read(client, &numBlocks, sizeof(numBlocks));
    void *func;
    rpc_read(client, &func, sizeof(func));
    int blockSize;
    rpc_read(client, &blockSize, sizeof(blockSize));
    size_t dynamicSMemSize;
    rpc_read(client, &dynamicSMemSize, sizeof(dynamicSMemSize));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
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

int handle_cudaMallocArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMallocArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaArray_t *array;
    rpc_read(client, &array, sizeof(array));
    struct cudaChannelFormatDesc *desc;
    rpc_read(client, &desc, sizeof(desc));
    size_t width;
    rpc_read(client, &width, sizeof(width));
    size_t height;
    rpc_read(client, &height, sizeof(height));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMallocArray(array, desc, width, height, flags);
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

int handle_cudaFreeArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFreeArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaArray_t array;
    rpc_read(client, &array, sizeof(array));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFreeArray(array);
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

int handle_cudaFreeMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFreeMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaMipmappedArray_t mipmappedArray;
    rpc_read(client, &mipmappedArray, sizeof(mipmappedArray));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFreeMipmappedArray(mipmappedArray);
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

int handle_cudaHostRegister(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaHostRegister called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
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
    _result = cudaHostRegister(ptr, size, flags);
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

int handle_cudaHostUnregister(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaHostUnregister called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
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
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cudaHostGetDevicePointer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaHostGetDevicePointer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *pDevice;
    void *pHost;
    rpc_read(client, &pHost, sizeof(pHost));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaHostGetDevicePointer(&pDevice, pHost, flags);
    rpc_write(client, &pDevice, sizeof(pDevice));
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

int handle_cudaHostGetFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaHostGetFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int *pFlags;
    rpc_read(client, &pFlags, sizeof(pFlags));
    void *pHost;
    rpc_read(client, &pHost, sizeof(pHost));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaHostGetFlags(pFlags, pHost);
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

int handle_cudaMalloc3DArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMalloc3DArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaArray_t *array;
    rpc_read(client, &array, sizeof(array));
    struct cudaChannelFormatDesc *desc;
    rpc_read(client, &desc, sizeof(desc));
    struct cudaExtent extent;
    rpc_read(client, &extent, sizeof(extent));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMalloc3DArray(array, desc, extent, flags);
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

int handle_cudaMallocMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMallocMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaMipmappedArray_t *mipmappedArray;
    rpc_read(client, &mipmappedArray, sizeof(mipmappedArray));
    struct cudaChannelFormatDesc *desc;
    rpc_read(client, &desc, sizeof(desc));
    struct cudaExtent extent;
    rpc_read(client, &extent, sizeof(extent));
    unsigned int numLevels;
    rpc_read(client, &numLevels, sizeof(numLevels));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags);
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

int handle_cudaGetMipmappedArrayLevel(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetMipmappedArrayLevel called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaArray_t *levelArray;
    rpc_read(client, &levelArray, sizeof(levelArray));
    cudaMipmappedArray_const_t mipmappedArray;
    rpc_read(client, &mipmappedArray, sizeof(mipmappedArray));
    unsigned int level;
    rpc_read(client, &level, sizeof(level));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level);
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

int handle_cudaMemcpy3D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy3D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaMemcpy3DParms *p;
    rpc_read(client, &p, sizeof(p));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy3D(p);
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

int handle_cudaMemcpy3DPeer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy3DPeer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaMemcpy3DPeerParms *p;
    rpc_read(client, &p, sizeof(p));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy3DPeer(p);
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

int handle_cudaMemcpy3DAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy3DAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaMemcpy3DParms *p;
    rpc_read(client, &p, sizeof(p));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy3DAsync(p, stream);
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

int handle_cudaMemcpy3DPeerAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy3DPeerAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaMemcpy3DPeerParms *p;
    rpc_read(client, &p, sizeof(p));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy3DPeerAsync(p, stream);
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

int handle_cudaMemGetInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemGetInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t *free;
    rpc_read(client, &free, sizeof(free));
    size_t *total;
    rpc_read(client, &total, sizeof(total));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemGetInfo(free, total);
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

int handle_cudaArrayGetInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaArrayGetInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaChannelFormatDesc *desc;
    rpc_read(client, &desc, sizeof(desc));
    struct cudaExtent *extent;
    rpc_read(client, &extent, sizeof(extent));
    unsigned int *flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaArray_t array;
    rpc_read(client, &array, sizeof(array));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaArrayGetInfo(desc, extent, flags, array);
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

int handle_cudaArrayGetPlane(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaArrayGetPlane called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaArray_t *pPlaneArray;
    rpc_read(client, &pPlaneArray, sizeof(pPlaneArray));
    cudaArray_t hArray;
    rpc_read(client, &hArray, sizeof(hArray));
    unsigned int planeIdx;
    rpc_read(client, &planeIdx, sizeof(planeIdx));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaArrayGetPlane(pPlaneArray, hArray, planeIdx);
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

int handle_cudaArrayGetSparseProperties(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaArrayGetSparseProperties called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaArraySparseProperties *sparseProperties;
    rpc_read(client, &sparseProperties, sizeof(sparseProperties));
    cudaArray_t array;
    rpc_read(client, &array, sizeof(array));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaArrayGetSparseProperties(sparseProperties, array);
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

int handle_cudaMipmappedArrayGetSparseProperties(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMipmappedArrayGetSparseProperties called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaArraySparseProperties *sparseProperties;
    rpc_read(client, &sparseProperties, sizeof(sparseProperties));
    cudaMipmappedArray_t mipmap;
    rpc_read(client, &mipmap, sizeof(mipmap));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMipmappedArrayGetSparseProperties(sparseProperties, mipmap);
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

int handle_cudaMemcpy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy(dst, src, count, kind);
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

int handle_cudaMemcpyPeer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyPeer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    int dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    void *src;
    rpc_read(client, &src, sizeof(src));
    int srcDevice;
    rpc_read(client, &srcDevice, sizeof(srcDevice));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
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

int handle_cudaMemcpy2D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    size_t dpitch;
    rpc_read(client, &dpitch, sizeof(dpitch));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t spitch;
    rpc_read(client, &spitch, sizeof(spitch));
    size_t width;
    rpc_read(client, &width, sizeof(width));
    size_t height;
    rpc_read(client, &height, sizeof(height));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
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

int handle_cudaMemcpy2DToArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2DToArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaArray_t dst;
    rpc_read(client, &dst, sizeof(dst));
    size_t wOffset;
    rpc_read(client, &wOffset, sizeof(wOffset));
    size_t hOffset;
    rpc_read(client, &hOffset, sizeof(hOffset));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t spitch;
    rpc_read(client, &spitch, sizeof(spitch));
    size_t width;
    rpc_read(client, &width, sizeof(width));
    size_t height;
    rpc_read(client, &height, sizeof(height));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
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

int handle_cudaMemcpy2DFromArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2DFromArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    size_t dpitch;
    rpc_read(client, &dpitch, sizeof(dpitch));
    cudaArray_const_t src;
    rpc_read(client, &src, sizeof(src));
    size_t wOffset;
    rpc_read(client, &wOffset, sizeof(wOffset));
    size_t hOffset;
    rpc_read(client, &hOffset, sizeof(hOffset));
    size_t width;
    rpc_read(client, &width, sizeof(width));
    size_t height;
    rpc_read(client, &height, sizeof(height));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
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

int handle_cudaMemcpy2DArrayToArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2DArrayToArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaArray_t dst;
    rpc_read(client, &dst, sizeof(dst));
    size_t wOffsetDst;
    rpc_read(client, &wOffsetDst, sizeof(wOffsetDst));
    size_t hOffsetDst;
    rpc_read(client, &hOffsetDst, sizeof(hOffsetDst));
    cudaArray_const_t src;
    rpc_read(client, &src, sizeof(src));
    size_t wOffsetSrc;
    rpc_read(client, &wOffsetSrc, sizeof(wOffsetSrc));
    size_t hOffsetSrc;
    rpc_read(client, &hOffsetSrc, sizeof(hOffsetSrc));
    size_t width;
    rpc_read(client, &width, sizeof(width));
    size_t height;
    rpc_read(client, &height, sizeof(height));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
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

int handle_cudaMemcpyToSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyToSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    size_t offset;
    rpc_read(client, &offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyToSymbol(symbol, src, count, offset, kind);
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

int handle_cudaMemcpyFromSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyFromSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    size_t offset;
    rpc_read(client, &offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
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

int handle_cudaMemcpyAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyAsync(dst, src, count, kind, stream);
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

int handle_cudaMemcpyPeerAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyPeerAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    int dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    void *src;
    rpc_read(client, &src, sizeof(src));
    int srcDevice;
    rpc_read(client, &srcDevice, sizeof(srcDevice));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
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

int handle_cudaMemcpy2DAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2DAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    size_t dpitch;
    rpc_read(client, &dpitch, sizeof(dpitch));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t spitch;
    rpc_read(client, &spitch, sizeof(spitch));
    size_t width;
    rpc_read(client, &width, sizeof(width));
    size_t height;
    rpc_read(client, &height, sizeof(height));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
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

int handle_cudaMemcpy2DToArrayAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2DToArrayAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaArray_t dst;
    rpc_read(client, &dst, sizeof(dst));
    size_t wOffset;
    rpc_read(client, &wOffset, sizeof(wOffset));
    size_t hOffset;
    rpc_read(client, &hOffset, sizeof(hOffset));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t spitch;
    rpc_read(client, &spitch, sizeof(spitch));
    size_t width;
    rpc_read(client, &width, sizeof(width));
    size_t height;
    rpc_read(client, &height, sizeof(height));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
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

int handle_cudaMemcpy2DFromArrayAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2DFromArrayAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    size_t dpitch;
    rpc_read(client, &dpitch, sizeof(dpitch));
    cudaArray_const_t src;
    rpc_read(client, &src, sizeof(src));
    size_t wOffset;
    rpc_read(client, &wOffset, sizeof(wOffset));
    size_t hOffset;
    rpc_read(client, &hOffset, sizeof(hOffset));
    size_t width;
    rpc_read(client, &width, sizeof(width));
    size_t height;
    rpc_read(client, &height, sizeof(height));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
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

int handle_cudaMemcpyToSymbolAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyToSymbolAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    size_t offset;
    rpc_read(client, &offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);
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

int handle_cudaMemcpyFromSymbolAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyFromSymbolAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    size_t offset;
    rpc_read(client, &offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, stream);
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

int handle_cudaMemset(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemset called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    int value;
    rpc_read(client, &value, sizeof(value));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemset(devPtr, value, count);
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

int handle_cudaMemset2D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemset2D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    size_t pitch;
    rpc_read(client, &pitch, sizeof(pitch));
    int value;
    rpc_read(client, &value, sizeof(value));
    size_t width;
    rpc_read(client, &width, sizeof(width));
    size_t height;
    rpc_read(client, &height, sizeof(height));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemset2D(devPtr, pitch, value, width, height);
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

int handle_cudaMemset3D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemset3D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaPitchedPtr pitchedDevPtr;
    rpc_read(client, &pitchedDevPtr, sizeof(pitchedDevPtr));
    int value;
    rpc_read(client, &value, sizeof(value));
    struct cudaExtent extent;
    rpc_read(client, &extent, sizeof(extent));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemset3D(pitchedDevPtr, value, extent);
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

int handle_cudaMemsetAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemsetAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    int value;
    rpc_read(client, &value, sizeof(value));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemsetAsync(devPtr, value, count, stream);
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

int handle_cudaMemset2DAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemset2DAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    size_t pitch;
    rpc_read(client, &pitch, sizeof(pitch));
    int value;
    rpc_read(client, &value, sizeof(value));
    size_t width;
    rpc_read(client, &width, sizeof(width));
    size_t height;
    rpc_read(client, &height, sizeof(height));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemset2DAsync(devPtr, pitch, value, width, height, stream);
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

int handle_cudaMemset3DAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemset3DAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaPitchedPtr pitchedDevPtr;
    rpc_read(client, &pitchedDevPtr, sizeof(pitchedDevPtr));
    int value;
    rpc_read(client, &value, sizeof(value));
    struct cudaExtent extent;
    rpc_read(client, &extent, sizeof(extent));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemset3DAsync(pitchedDevPtr, value, extent, stream);
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

int handle_cudaGetSymbolSize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetSymbolSize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t *size;
    rpc_read(client, &size, sizeof(size));
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetSymbolSize(size, symbol);
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

int handle_cudaMemPrefetchAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPrefetchAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    int dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);
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

int handle_cudaMemAdvise(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemAdvise called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    enum cudaMemoryAdvise advice;
    rpc_read(client, &advice, sizeof(advice));
    int device;
    rpc_read(client, &device, sizeof(device));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemAdvise(devPtr, count, advice, device);
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

int handle_cudaMemRangeGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemRangeGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *data;
    rpc_read(client, &data, sizeof(data));
    size_t dataSize;
    rpc_read(client, &dataSize, sizeof(dataSize));
    enum cudaMemRangeAttribute attribute;
    rpc_read(client, &attribute, sizeof(attribute));
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
    _result = cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);
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

int handle_cudaMemcpyToArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyToArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaArray_t dst;
    rpc_read(client, &dst, sizeof(dst));
    size_t wOffset;
    rpc_read(client, &wOffset, sizeof(wOffset));
    size_t hOffset;
    rpc_read(client, &hOffset, sizeof(hOffset));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
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

int handle_cudaMemcpyFromArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyFromArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    cudaArray_const_t src;
    rpc_read(client, &src, sizeof(src));
    size_t wOffset;
    rpc_read(client, &wOffset, sizeof(wOffset));
    size_t hOffset;
    rpc_read(client, &hOffset, sizeof(hOffset));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
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

int handle_cudaMemcpyArrayToArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyArrayToArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaArray_t dst;
    rpc_read(client, &dst, sizeof(dst));
    size_t wOffsetDst;
    rpc_read(client, &wOffsetDst, sizeof(wOffsetDst));
    size_t hOffsetDst;
    rpc_read(client, &hOffsetDst, sizeof(hOffsetDst));
    cudaArray_const_t src;
    rpc_read(client, &src, sizeof(src));
    size_t wOffsetSrc;
    rpc_read(client, &wOffsetSrc, sizeof(wOffsetSrc));
    size_t hOffsetSrc;
    rpc_read(client, &hOffsetSrc, sizeof(hOffsetSrc));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);
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

int handle_cudaMemcpyToArrayAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyToArrayAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaArray_t dst;
    rpc_read(client, &dst, sizeof(dst));
    size_t wOffset;
    rpc_read(client, &wOffset, sizeof(wOffset));
    size_t hOffset;
    rpc_read(client, &hOffset, sizeof(hOffset));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream);
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

int handle_cudaMemcpyFromArrayAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyFromArrayAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    cudaArray_const_t src;
    rpc_read(client, &src, sizeof(src));
    size_t wOffset;
    rpc_read(client, &wOffset, sizeof(wOffset));
    size_t hOffset;
    rpc_read(client, &hOffset, sizeof(hOffset));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream);
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

int handle_cudaMallocAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMallocAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *devPtr;
    size_t size;
    rpc_read(client, &size, sizeof(size));
    cudaStream_t hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMallocAsync(&devPtr, size, hStream);
    rpc_write(client, &devPtr, sizeof(devPtr));
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

int handle_cudaFreeAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFreeAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    cudaStream_t hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFreeAsync(devPtr, hStream);
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

int handle_cudaMemPoolTrimTo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolTrimTo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaMemPool_t memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    size_t minBytesToKeep;
    rpc_read(client, &minBytesToKeep, sizeof(minBytesToKeep));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolTrimTo(memPool, minBytesToKeep);
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

int handle_cudaMemPoolSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaMemPool_t memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    enum cudaMemPoolAttr attr;
    rpc_read(client, &attr, sizeof(attr));
    void *value;
    rpc_read(client, &value, sizeof(value));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolSetAttribute(memPool, attr, value);
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

int handle_cudaMemPoolGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaMemPool_t memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    enum cudaMemPoolAttr attr;
    rpc_read(client, &attr, sizeof(attr));
    void *value;
    rpc_read(client, &value, sizeof(value));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolGetAttribute(memPool, attr, value);
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

int handle_cudaMemPoolSetAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolSetAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaMemPool_t memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    struct cudaMemAccessDesc *descList;
    rpc_read(client, &descList, sizeof(descList));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolSetAccess(memPool, descList, count);
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

int handle_cudaMemPoolGetAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolGetAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    enum cudaMemAccessFlags *flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaMemPool_t memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    struct cudaMemLocation *location;
    rpc_read(client, &location, sizeof(location));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolGetAccess(flags, memPool, location);
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

int handle_cudaMemPoolCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaMemPool_t *memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    struct cudaMemPoolProps *poolProps;
    rpc_read(client, &poolProps, sizeof(poolProps));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolCreate(memPool, poolProps);
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

int handle_cudaMemPoolDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaMemPool_t memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolDestroy(memPool);
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

int handle_cudaMallocFromPoolAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMallocFromPoolAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *ptr;
    size_t size;
    rpc_read(client, &size, sizeof(size));
    cudaMemPool_t memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMallocFromPoolAsync(&ptr, size, memPool, stream);
    rpc_write(client, &ptr, sizeof(ptr));
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

int handle_cudaMemPoolExportToShareableHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolExportToShareableHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *shareableHandle;
    rpc_read(client, &shareableHandle, sizeof(shareableHandle));
    cudaMemPool_t memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    enum cudaMemAllocationHandleType handleType;
    rpc_read(client, &handleType, sizeof(handleType));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolExportToShareableHandle(shareableHandle, memPool, handleType, flags);
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

int handle_cudaMemPoolImportFromShareableHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolImportFromShareableHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaMemPool_t *memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    void *shareableHandle;
    rpc_read(client, &shareableHandle, sizeof(shareableHandle));
    enum cudaMemAllocationHandleType handleType;
    rpc_read(client, &handleType, sizeof(handleType));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolImportFromShareableHandle(memPool, shareableHandle, handleType, flags);
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

int handle_cudaMemPoolExportPointer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolExportPointer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaMemPoolPtrExportData *exportData;
    rpc_read(client, &exportData, sizeof(exportData));
    void *ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolExportPointer(exportData, ptr);
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

int handle_cudaMemPoolImportPointer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolImportPointer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *ptr;
    cudaMemPool_t memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    struct cudaMemPoolPtrExportData *exportData;
    rpc_read(client, &exportData, sizeof(exportData));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolImportPointer(&ptr, memPool, exportData);
    rpc_write(client, &ptr, sizeof(ptr));
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

int handle_cudaPointerGetAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaPointerGetAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaPointerAttributes *attributes;
    rpc_read(client, &attributes, sizeof(attributes));
    void *ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaPointerGetAttributes(attributes, ptr);
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

int handle_cudaDeviceCanAccessPeer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceCanAccessPeer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int *canAccessPeer;
    rpc_read(client, &canAccessPeer, sizeof(canAccessPeer));
    int device;
    rpc_read(client, &device, sizeof(device));
    int peerDevice;
    rpc_read(client, &peerDevice, sizeof(peerDevice));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
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

int handle_cudaDeviceEnablePeerAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceEnablePeerAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int peerDevice;
    rpc_read(client, &peerDevice, sizeof(peerDevice));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceEnablePeerAccess(peerDevice, flags);
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

int handle_cudaDeviceDisablePeerAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceDisablePeerAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int peerDevice;
    rpc_read(client, &peerDevice, sizeof(peerDevice));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceDisablePeerAccess(peerDevice);
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

int handle_cudaGraphicsUnregisterResource(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsUnregisterResource called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphicsResource_t resource;
    rpc_read(client, &resource, sizeof(resource));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsUnregisterResource(resource);
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

int handle_cudaGraphicsResourceSetMapFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsResourceSetMapFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphicsResource_t resource;
    rpc_read(client, &resource, sizeof(resource));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsResourceSetMapFlags(resource, flags);
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

int handle_cudaGraphicsMapResources(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsMapResources called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int count;
    rpc_read(client, &count, sizeof(count));
    cudaGraphicsResource_t *resources;
    rpc_read(client, &resources, sizeof(resources));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsMapResources(count, resources, stream);
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

int handle_cudaGraphicsUnmapResources(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsUnmapResources called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int count;
    rpc_read(client, &count, sizeof(count));
    cudaGraphicsResource_t *resources;
    rpc_read(client, &resources, sizeof(resources));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsUnmapResources(count, resources, stream);
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

int handle_cudaGraphicsResourceGetMappedPointer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsResourceGetMappedPointer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *devPtr;
    size_t *size;
    rpc_read(client, &size, sizeof(size));
    cudaGraphicsResource_t resource;
    rpc_read(client, &resource, sizeof(resource));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsResourceGetMappedPointer(&devPtr, size, resource);
    rpc_write(client, &devPtr, sizeof(devPtr));
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

int handle_cudaGraphicsSubResourceGetMappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsSubResourceGetMappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaArray_t *array;
    rpc_read(client, &array, sizeof(array));
    cudaGraphicsResource_t resource;
    rpc_read(client, &resource, sizeof(resource));
    unsigned int arrayIndex;
    rpc_read(client, &arrayIndex, sizeof(arrayIndex));
    unsigned int mipLevel;
    rpc_read(client, &mipLevel, sizeof(mipLevel));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel);
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

int handle_cudaGraphicsResourceGetMappedMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsResourceGetMappedMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaMipmappedArray_t *mipmappedArray;
    rpc_read(client, &mipmappedArray, sizeof(mipmappedArray));
    cudaGraphicsResource_t resource;
    rpc_read(client, &resource, sizeof(resource));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource);
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

int handle_cudaBindTexture(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaBindTexture called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t *offset;
    rpc_read(client, &offset, sizeof(offset));
    struct textureReference *texref;
    rpc_read(client, &texref, sizeof(texref));
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    struct cudaChannelFormatDesc *desc;
    rpc_read(client, &desc, sizeof(desc));
    size_t size;
    rpc_read(client, &size, sizeof(size));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaBindTexture(offset, texref, devPtr, desc, size);
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

int handle_cudaBindTexture2D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaBindTexture2D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t *offset;
    rpc_read(client, &offset, sizeof(offset));
    struct textureReference *texref;
    rpc_read(client, &texref, sizeof(texref));
    void *devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    struct cudaChannelFormatDesc *desc;
    rpc_read(client, &desc, sizeof(desc));
    size_t width;
    rpc_read(client, &width, sizeof(width));
    size_t height;
    rpc_read(client, &height, sizeof(height));
    size_t pitch;
    rpc_read(client, &pitch, sizeof(pitch));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaBindTexture2D(offset, texref, devPtr, desc, width, height, pitch);
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

int handle_cudaBindTextureToArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaBindTextureToArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct textureReference *texref;
    rpc_read(client, &texref, sizeof(texref));
    cudaArray_const_t array;
    rpc_read(client, &array, sizeof(array));
    struct cudaChannelFormatDesc *desc;
    rpc_read(client, &desc, sizeof(desc));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaBindTextureToArray(texref, array, desc);
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

int handle_cudaBindTextureToMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaBindTextureToMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct textureReference *texref;
    rpc_read(client, &texref, sizeof(texref));
    cudaMipmappedArray_const_t mipmappedArray;
    rpc_read(client, &mipmappedArray, sizeof(mipmappedArray));
    struct cudaChannelFormatDesc *desc;
    rpc_read(client, &desc, sizeof(desc));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaBindTextureToMipmappedArray(texref, mipmappedArray, desc);
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

int handle_cudaUnbindTexture(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaUnbindTexture called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct textureReference *texref;
    rpc_read(client, &texref, sizeof(texref));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaUnbindTexture(texref);
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

int handle_cudaGetTextureAlignmentOffset(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetTextureAlignmentOffset called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t *offset;
    rpc_read(client, &offset, sizeof(offset));
    struct textureReference *texref;
    rpc_read(client, &texref, sizeof(texref));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetTextureAlignmentOffset(offset, texref);
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

int handle_cudaGetTextureReference(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetTextureReference called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    const struct textureReference *texref;
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetTextureReference(&texref, symbol);
    rpc_write(client, texref, sizeof(struct textureReference));
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

int handle_cudaBindSurfaceToArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaBindSurfaceToArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct surfaceReference *surfref;
    rpc_read(client, &surfref, sizeof(surfref));
    cudaArray_const_t array;
    rpc_read(client, &array, sizeof(array));
    struct cudaChannelFormatDesc *desc;
    rpc_read(client, &desc, sizeof(desc));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaBindSurfaceToArray(surfref, array, desc);
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

int handle_cudaGetSurfaceReference(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetSurfaceReference called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    const struct surfaceReference *surfref;
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetSurfaceReference(&surfref, symbol);
    rpc_write(client, surfref, sizeof(struct surfaceReference));
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

int handle_cudaGetChannelDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetChannelDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaChannelFormatDesc *desc;
    rpc_read(client, &desc, sizeof(desc));
    cudaArray_const_t array;
    rpc_read(client, &array, sizeof(array));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetChannelDesc(desc, array);
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

int handle_cudaCreateChannelDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaCreateChannelDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int x;
    rpc_read(client, &x, sizeof(x));
    int y;
    rpc_read(client, &y, sizeof(y));
    int z;
    rpc_read(client, &z, sizeof(z));
    int w;
    rpc_read(client, &w, sizeof(w));
    enum cudaChannelFormatKind f;
    rpc_read(client, &f, sizeof(f));
    struct cudaChannelFormatDesc _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaCreateChannelDesc(x, y, z, w, f);
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

int handle_cudaCreateTextureObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaCreateTextureObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaTextureObject_t *pTexObject;
    rpc_read(client, &pTexObject, sizeof(pTexObject));
    struct cudaResourceDesc *pResDesc;
    rpc_read(client, &pResDesc, sizeof(pResDesc));
    struct cudaTextureDesc *pTexDesc;
    rpc_read(client, &pTexDesc, sizeof(pTexDesc));
    struct cudaResourceViewDesc *pResViewDesc;
    rpc_read(client, &pResViewDesc, sizeof(pResViewDesc));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
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

int handle_cudaDestroyTextureObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDestroyTextureObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaTextureObject_t texObject;
    rpc_read(client, &texObject, sizeof(texObject));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDestroyTextureObject(texObject);
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

int handle_cudaGetTextureObjectResourceDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetTextureObjectResourceDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaResourceDesc *pResDesc;
    rpc_read(client, &pResDesc, sizeof(pResDesc));
    cudaTextureObject_t texObject;
    rpc_read(client, &texObject, sizeof(texObject));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetTextureObjectResourceDesc(pResDesc, texObject);
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

int handle_cudaGetTextureObjectTextureDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetTextureObjectTextureDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaTextureDesc *pTexDesc;
    rpc_read(client, &pTexDesc, sizeof(pTexDesc));
    cudaTextureObject_t texObject;
    rpc_read(client, &texObject, sizeof(texObject));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetTextureObjectTextureDesc(pTexDesc, texObject);
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

int handle_cudaGetTextureObjectResourceViewDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetTextureObjectResourceViewDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaResourceViewDesc *pResViewDesc;
    rpc_read(client, &pResViewDesc, sizeof(pResViewDesc));
    cudaTextureObject_t texObject;
    rpc_read(client, &texObject, sizeof(texObject));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject);
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

int handle_cudaCreateSurfaceObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaCreateSurfaceObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaSurfaceObject_t *pSurfObject;
    rpc_read(client, &pSurfObject, sizeof(pSurfObject));
    struct cudaResourceDesc *pResDesc;
    rpc_read(client, &pResDesc, sizeof(pResDesc));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaCreateSurfaceObject(pSurfObject, pResDesc);
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

int handle_cudaDestroySurfaceObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDestroySurfaceObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaSurfaceObject_t surfObject;
    rpc_read(client, &surfObject, sizeof(surfObject));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDestroySurfaceObject(surfObject);
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

int handle_cudaGetSurfaceObjectResourceDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetSurfaceObjectResourceDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    struct cudaResourceDesc *pResDesc;
    rpc_read(client, &pResDesc, sizeof(pResDesc));
    cudaSurfaceObject_t surfObject;
    rpc_read(client, &surfObject, sizeof(surfObject));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject);
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

int handle_cudaDriverGetVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDriverGetVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int *driverVersion;
    rpc_read(client, &driverVersion, sizeof(driverVersion));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDriverGetVersion(driverVersion);
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

int handle_cudaRuntimeGetVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaRuntimeGetVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int *runtimeVersion;
    rpc_read(client, &runtimeVersion, sizeof(runtimeVersion));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaRuntimeGetVersion(runtimeVersion);
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

int handle_cudaGraphCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraph_t *pGraph;
    rpc_read(client, &pGraph, sizeof(pGraph));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphCreate(pGraph, flags);
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

int handle_cudaGraphAddKernelNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddKernelNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    struct cudaKernelNodeParams *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
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

int handle_cudaGraphKernelNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphKernelNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    struct cudaKernelNodeParams *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphKernelNodeGetParams(node, pNodeParams);
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

int handle_cudaGraphKernelNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphKernelNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    struct cudaKernelNodeParams *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphKernelNodeSetParams(node, pNodeParams);
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

int handle_cudaGraphKernelNodeCopyAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphKernelNodeCopyAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t hSrc;
    rpc_read(client, &hSrc, sizeof(hSrc));
    cudaGraphNode_t hDst;
    rpc_read(client, &hDst, sizeof(hDst));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphKernelNodeCopyAttributes(hSrc, hDst);
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

int handle_cudaGraphKernelNodeGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphKernelNodeGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    enum cudaKernelNodeAttrID attr;
    rpc_read(client, &attr, sizeof(attr));
    union cudaKernelNodeAttrValue *value_out;
    rpc_read(client, &value_out, sizeof(value_out));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphKernelNodeGetAttribute(hNode, attr, value_out);
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

int handle_cudaGraphKernelNodeSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphKernelNodeSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    enum cudaKernelNodeAttrID attr;
    rpc_read(client, &attr, sizeof(attr));
    union cudaKernelNodeAttrValue *value;
    rpc_read(client, &value, sizeof(value));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphKernelNodeSetAttribute(hNode, attr, value);
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

int handle_cudaGraphAddMemcpyNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemcpyNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    struct cudaMemcpy3DParms *pCopyParams;
    rpc_read(client, &pCopyParams, sizeof(pCopyParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams);
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

int handle_cudaGraphAddMemcpyNodeToSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemcpyNodeToSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    size_t offset;
    rpc_read(client, &offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind);
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

int handle_cudaGraphAddMemcpyNodeFromSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemcpyNodeFromSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    size_t offset;
    rpc_read(client, &offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind);
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

int handle_cudaGraphAddMemcpyNode1D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemcpyNode1D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind);
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

int handle_cudaGraphMemcpyNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemcpyNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    struct cudaMemcpy3DParms *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemcpyNodeGetParams(node, pNodeParams);
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

int handle_cudaGraphMemcpyNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemcpyNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    struct cudaMemcpy3DParms *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemcpyNodeSetParams(node, pNodeParams);
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

int handle_cudaGraphMemcpyNodeSetParamsToSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemcpyNodeSetParamsToSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    size_t offset;
    rpc_read(client, &offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemcpyNodeSetParamsToSymbol(node, symbol, src, count, offset, kind);
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

int handle_cudaGraphMemcpyNodeSetParamsFromSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemcpyNodeSetParamsFromSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    size_t offset;
    rpc_read(client, &offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, symbol, count, offset, kind);
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

int handle_cudaGraphMemcpyNodeSetParams1D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemcpyNodeSetParams1D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind);
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

int handle_cudaGraphAddMemsetNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemsetNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    struct cudaMemsetParams *pMemsetParams;
    rpc_read(client, &pMemsetParams, sizeof(pMemsetParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams);
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

int handle_cudaGraphMemsetNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemsetNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    struct cudaMemsetParams *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemsetNodeGetParams(node, pNodeParams);
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

int handle_cudaGraphMemsetNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemsetNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    struct cudaMemsetParams *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemsetNodeSetParams(node, pNodeParams);
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

int handle_cudaGraphAddHostNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddHostNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    struct cudaHostNodeParams *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
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

int handle_cudaGraphHostNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphHostNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    struct cudaHostNodeParams *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphHostNodeGetParams(node, pNodeParams);
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

int handle_cudaGraphHostNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphHostNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    struct cudaHostNodeParams *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphHostNodeSetParams(node, pNodeParams);
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

int handle_cudaGraphAddChildGraphNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddChildGraphNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    cudaGraph_t childGraph;
    rpc_read(client, &childGraph, sizeof(childGraph));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph);
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

int handle_cudaGraphChildGraphNodeGetGraph(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphChildGraphNodeGetGraph called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    cudaGraph_t *pGraph;
    rpc_read(client, &pGraph, sizeof(pGraph));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphChildGraphNodeGetGraph(node, pGraph);
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

int handle_cudaGraphAddEmptyNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddEmptyNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies);
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

int handle_cudaGraphAddEventRecordNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddEventRecordNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    cudaEvent_t event;
    rpc_read(client, &event, sizeof(event));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event);
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

int handle_cudaGraphEventRecordNodeGetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphEventRecordNodeGetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    cudaEvent_t *event_out;
    rpc_read(client, &event_out, sizeof(event_out));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphEventRecordNodeGetEvent(node, event_out);
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

int handle_cudaGraphEventRecordNodeSetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphEventRecordNodeSetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    cudaEvent_t event;
    rpc_read(client, &event, sizeof(event));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphEventRecordNodeSetEvent(node, event);
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

int handle_cudaGraphAddEventWaitNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddEventWaitNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    cudaEvent_t event;
    rpc_read(client, &event, sizeof(event));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event);
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

int handle_cudaGraphEventWaitNodeGetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphEventWaitNodeGetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    cudaEvent_t *event_out;
    rpc_read(client, &event_out, sizeof(event_out));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphEventWaitNodeGetEvent(node, event_out);
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

int handle_cudaGraphEventWaitNodeSetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphEventWaitNodeSetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    cudaEvent_t event;
    rpc_read(client, &event, sizeof(event));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphEventWaitNodeSetEvent(node, event);
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

int handle_cudaGraphAddExternalSemaphoresSignalNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddExternalSemaphoresSignalNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    struct cudaExternalSemaphoreSignalNodeParams *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
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

int handle_cudaGraphExternalSemaphoresSignalNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExternalSemaphoresSignalNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    struct cudaExternalSemaphoreSignalNodeParams *params_out;
    rpc_read(client, &params_out, sizeof(params_out));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out);
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

int handle_cudaGraphExternalSemaphoresSignalNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExternalSemaphoresSignalNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    struct cudaExternalSemaphoreSignalNodeParams *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams);
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

int handle_cudaGraphAddExternalSemaphoresWaitNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddExternalSemaphoresWaitNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    struct cudaExternalSemaphoreWaitNodeParams *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
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

int handle_cudaGraphExternalSemaphoresWaitNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExternalSemaphoresWaitNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    struct cudaExternalSemaphoreWaitNodeParams *params_out;
    rpc_read(client, &params_out, sizeof(params_out));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out);
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

int handle_cudaGraphExternalSemaphoresWaitNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExternalSemaphoresWaitNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    struct cudaExternalSemaphoreWaitNodeParams *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams);
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

int handle_cudaGraphAddMemAllocNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemAllocNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    struct cudaMemAllocNodeParams *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
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

int handle_cudaGraphMemAllocNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemAllocNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    struct cudaMemAllocNodeParams *params_out;
    rpc_read(client, &params_out, sizeof(params_out));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemAllocNodeGetParams(node, params_out);
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

int handle_cudaGraphAddMemFreeNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemFreeNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pGraphNode;
    rpc_read(client, &pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    void *dptr;
    rpc_read(client, &dptr, sizeof(dptr));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dptr);
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

int handle_cudaGraphMemFreeNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemFreeNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    void *dptr_out;
    rpc_read(client, &dptr_out, sizeof(dptr_out));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemFreeNodeGetParams(node, dptr_out);
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

int handle_cudaDeviceGraphMemTrim(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGraphMemTrim called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int device;
    rpc_read(client, &device, sizeof(device));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGraphMemTrim(device);
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

int handle_cudaDeviceGetGraphMemAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetGraphMemAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int device;
    rpc_read(client, &device, sizeof(device));
    enum cudaGraphMemAttributeType attr;
    rpc_read(client, &attr, sizeof(attr));
    void *value;
    rpc_read(client, &value, sizeof(value));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetGraphMemAttribute(device, attr, value);
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

int handle_cudaDeviceSetGraphMemAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceSetGraphMemAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int device;
    rpc_read(client, &device, sizeof(device));
    enum cudaGraphMemAttributeType attr;
    rpc_read(client, &attr, sizeof(attr));
    void *value;
    rpc_read(client, &value, sizeof(value));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceSetGraphMemAttribute(device, attr, value);
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

int handle_cudaGraphClone(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphClone called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraph_t *pGraphClone;
    rpc_read(client, &pGraphClone, sizeof(pGraphClone));
    cudaGraph_t originalGraph;
    rpc_read(client, &originalGraph, sizeof(originalGraph));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphClone(pGraphClone, originalGraph);
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

int handle_cudaGraphNodeFindInClone(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphNodeFindInClone called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t *pNode;
    rpc_read(client, &pNode, sizeof(pNode));
    cudaGraphNode_t originalNode;
    rpc_read(client, &originalNode, sizeof(originalNode));
    cudaGraph_t clonedGraph;
    rpc_read(client, &clonedGraph, sizeof(clonedGraph));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphNodeFindInClone(pNode, originalNode, clonedGraph);
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

int handle_cudaGraphNodeGetType(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphNodeGetType called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    enum cudaGraphNodeType *pType;
    rpc_read(client, &pType, sizeof(pType));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphNodeGetType(node, pType);
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

int handle_cudaGraphGetNodes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphGetNodes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *nodes;
    rpc_read(client, &nodes, sizeof(nodes));
    size_t *numNodes;
    rpc_read(client, &numNodes, sizeof(numNodes));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphGetNodes(graph, nodes, numNodes);
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

int handle_cudaGraphGetRootNodes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphGetRootNodes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pRootNodes;
    rpc_read(client, &pRootNodes, sizeof(pRootNodes));
    size_t *pNumRootNodes;
    rpc_read(client, &pNumRootNodes, sizeof(pNumRootNodes));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes);
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

int handle_cudaGraphGetEdges(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphGetEdges called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *from;
    rpc_read(client, &from, sizeof(from));
    cudaGraphNode_t *to;
    rpc_read(client, &to, sizeof(to));
    size_t *numEdges;
    rpc_read(client, &numEdges, sizeof(numEdges));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphGetEdges(graph, from, to, numEdges);
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

int handle_cudaGraphNodeGetDependencies(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphNodeGetDependencies called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    cudaGraphNode_t *pDependencies;
    rpc_read(client, &pDependencies, sizeof(pDependencies));
    size_t *pNumDependencies;
    rpc_read(client, &pNumDependencies, sizeof(pNumDependencies));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphNodeGetDependencies(node, pDependencies, pNumDependencies);
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

int handle_cudaGraphNodeGetDependentNodes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphNodeGetDependentNodes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    cudaGraphNode_t *pDependentNodes;
    rpc_read(client, &pDependentNodes, sizeof(pDependentNodes));
    size_t *pNumDependentNodes;
    rpc_read(client, &pNumDependentNodes, sizeof(pNumDependentNodes));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphNodeGetDependentNodes(node, pDependentNodes, pNumDependentNodes);
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

int handle_cudaGraphAddDependencies(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddDependencies called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *from;
    rpc_read(client, &from, sizeof(from));
    cudaGraphNode_t *to;
    rpc_read(client, &to, sizeof(to));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddDependencies(graph, from, to, numDependencies);
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

int handle_cudaGraphRemoveDependencies(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphRemoveDependencies called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *from;
    rpc_read(client, &from, sizeof(from));
    cudaGraphNode_t *to;
    rpc_read(client, &to, sizeof(to));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphRemoveDependencies(graph, from, to, numDependencies);
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

int handle_cudaGraphDestroyNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphDestroyNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphDestroyNode(node);
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

int handle_cudaGraphInstantiate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphInstantiate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t *pGraphExec;
    rpc_read(client, &pGraphExec, sizeof(pGraphExec));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaGraphNode_t *pErrorNode;
    rpc_read(client, &pErrorNode, sizeof(pErrorNode));
    char pLogBuffer[1024];
    size_t bufferSize;
    rpc_read(client, &bufferSize, sizeof(bufferSize));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize);
    rpc_write(client, pLogBuffer, strlen(pLogBuffer) + 1, true);
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

int handle_cudaGraphInstantiateWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphInstantiateWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t *pGraphExec;
    rpc_read(client, &pGraphExec, sizeof(pGraphExec));
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    unsigned long long flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphInstantiateWithFlags(pGraphExec, graph, flags);
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

int handle_cudaGraphExecKernelNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecKernelNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    struct cudaKernelNodeParams *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams);
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

int handle_cudaGraphExecMemcpyNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecMemcpyNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    struct cudaMemcpy3DParms *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams);
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

int handle_cudaGraphExecMemcpyNodeSetParamsToSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecMemcpyNodeSetParamsToSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    size_t offset;
    rpc_read(client, &offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, symbol, src, count, offset, kind);
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

int handle_cudaGraphExecMemcpyNodeSetParamsFromSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecMemcpyNodeSetParamsFromSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    void *symbol;
    rpc_read(client, &symbol, sizeof(symbol));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    size_t offset;
    rpc_read(client, &offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, symbol, count, offset, kind);
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

int handle_cudaGraphExecMemcpyNodeSetParams1D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecMemcpyNodeSetParams1D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    void *dst;
    rpc_read(client, &dst, sizeof(dst));
    void *src;
    rpc_read(client, &src, sizeof(src));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    enum cudaMemcpyKind kind;
    rpc_read(client, &kind, sizeof(kind));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind);
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

int handle_cudaGraphExecMemsetNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecMemsetNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    struct cudaMemsetParams *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams);
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

int handle_cudaGraphExecHostNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecHostNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    struct cudaHostNodeParams *pNodeParams;
    rpc_read(client, &pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams);
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

int handle_cudaGraphExecChildGraphNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecChildGraphNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    rpc_read(client, &node, sizeof(node));
    cudaGraph_t childGraph;
    rpc_read(client, &childGraph, sizeof(childGraph));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph);
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

int handle_cudaGraphExecEventRecordNodeSetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecEventRecordNodeSetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    cudaEvent_t event;
    rpc_read(client, &event, sizeof(event));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);
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

int handle_cudaGraphExecEventWaitNodeSetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecEventWaitNodeSetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    cudaEvent_t event;
    rpc_read(client, &event, sizeof(event));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);
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

int handle_cudaGraphExecExternalSemaphoresSignalNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecExternalSemaphoresSignalNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    struct cudaExternalSemaphoreSignalNodeParams *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams);
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

int handle_cudaGraphExecExternalSemaphoresWaitNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecExternalSemaphoresWaitNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    struct cudaExternalSemaphoreWaitNodeParams *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams);
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

int handle_cudaGraphExecUpdate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecUpdate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cudaGraph_t hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    cudaGraphNode_t *hErrorNode_out;
    rpc_read(client, &hErrorNode_out, sizeof(hErrorNode_out));
    enum cudaGraphExecUpdateResult *updateResult_out;
    rpc_read(client, &updateResult_out, sizeof(updateResult_out));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out);
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

int handle_cudaGraphUpload(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphUpload called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t graphExec;
    rpc_read(client, &graphExec, sizeof(graphExec));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphUpload(graphExec, stream);
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

int handle_cudaGraphLaunch(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphLaunch called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t graphExec;
    rpc_read(client, &graphExec, sizeof(graphExec));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphLaunch(graphExec, stream);
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

int handle_cudaGraphExecDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraphExec_t graphExec;
    rpc_read(client, &graphExec, sizeof(graphExec));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecDestroy(graphExec);
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

int handle_cudaGraphDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphDestroy(graph);
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

int handle_cudaGraphDebugDotPrint(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphDebugDotPrint called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    char *path = nullptr;
    rpc_read(client, &path, 0, true);
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(path);
    _result = cudaGraphDebugDotPrint(graph, path, flags);
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

int handle_cudaUserObjectCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaUserObjectCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaUserObject_t *object_out;
    rpc_read(client, &object_out, sizeof(object_out));
    void *ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    cudaHostFn_t destroy;
    rpc_read(client, &destroy, sizeof(destroy));
    unsigned int initialRefcount;
    rpc_read(client, &initialRefcount, sizeof(initialRefcount));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags);
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

int handle_cudaUserObjectRetain(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaUserObjectRetain called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaUserObject_t object;
    rpc_read(client, &object, sizeof(object));
    unsigned int count;
    rpc_read(client, &count, sizeof(count));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaUserObjectRetain(object, count);
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

int handle_cudaUserObjectRelease(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaUserObjectRelease called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaUserObject_t object;
    rpc_read(client, &object, sizeof(object));
    unsigned int count;
    rpc_read(client, &count, sizeof(count));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaUserObjectRelease(object, count);
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

int handle_cudaGraphRetainUserObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphRetainUserObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaUserObject_t object;
    rpc_read(client, &object, sizeof(object));
    unsigned int count;
    rpc_read(client, &count, sizeof(count));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphRetainUserObject(graph, object, count, flags);
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

int handle_cudaGraphReleaseUserObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphReleaseUserObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaGraph_t graph;
    rpc_read(client, &graph, sizeof(graph));
    cudaUserObject_t object;
    rpc_read(client, &object, sizeof(object));
    unsigned int count;
    rpc_read(client, &count, sizeof(count));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphReleaseUserObject(graph, object, count);
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

int handle_cudaGetDriverEntryPoint(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetDriverEntryPoint called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    char *symbol = nullptr;
    rpc_read(client, &symbol, 0, true);
    void *funcPtr;
    unsigned long long flags;
    rpc_read(client, &flags, sizeof(flags));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(symbol);
    _result = cudaGetDriverEntryPoint(symbol, &funcPtr, flags);
    rpc_write(client, &funcPtr, sizeof(funcPtr));
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

int handle_cudaGetExportTable(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetExportTable called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    const void *ppExportTable;
    cudaUUID_t *pExportTableId;
    rpc_read(client, &pExportTableId, sizeof(pExportTableId));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetExportTable(&ppExportTable, pExportTableId);
    rpc_write(client, &ppExportTable, sizeof(ppExportTable));
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

int handle_cudaGetFuncBySymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetFuncBySymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cudaFunction_t *functionPtr;
    rpc_read(client, &functionPtr, sizeof(functionPtr));
    void *symbolPtr;
    rpc_read(client, &symbolPtr, sizeof(symbolPtr));
    cudaError_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetFuncBySymbol(functionPtr, symbolPtr);
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
