#include <iostream>
#include <map>
#include <string.h>
#include "hook_api.h"
#include "handle_server.h"
#include "rpc/rpc_core.h"
#include "cuda_runtime_api.h"

using namespace rpc;
int handle_cudaDeviceReset(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceReset called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceReset();
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

int handle_cudaDeviceSynchronize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceSynchronize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceSynchronize();
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

int handle_cudaDeviceSetLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceSetLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    enum cudaLimit limit;
    conn->read(&limit, sizeof(limit));
    size_t value;
    conn->read(&value, sizeof(value));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceSetLimit(limit, value);
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

int handle_cudaDeviceGetLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    size_t *pValue;
    conn->read(&pValue, sizeof(pValue));
    enum cudaLimit limit;
    conn->read(&limit, sizeof(limit));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetLimit(pValue, limit);
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

int handle_cudaDeviceGetTexture1DLinearMaxWidth(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetTexture1DLinearMaxWidth called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    size_t *maxWidthInElements;
    conn->read(&maxWidthInElements, sizeof(maxWidthInElements));
    struct cudaChannelFormatDesc *fmtDesc = nullptr;
    conn->read(&fmtDesc, sizeof(fmtDesc));
    int device;
    conn->read(&device, sizeof(device));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device);
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

int handle_cudaDeviceGetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    enum cudaFuncCache *pCacheConfig;
    conn->read(&pCacheConfig, sizeof(pCacheConfig));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetCacheConfig(pCacheConfig);
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

int handle_cudaDeviceGetStreamPriorityRange(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetStreamPriorityRange called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *leastPriority;
    conn->read(&leastPriority, sizeof(leastPriority));
    int *greatestPriority;
    conn->read(&greatestPriority, sizeof(greatestPriority));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
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

int handle_cudaDeviceSetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceSetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    enum cudaFuncCache cacheConfig;
    conn->read(&cacheConfig, sizeof(cacheConfig));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceSetCacheConfig(cacheConfig);
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

int handle_cudaDeviceGetSharedMemConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetSharedMemConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    enum cudaSharedMemConfig *pConfig;
    conn->read(&pConfig, sizeof(pConfig));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetSharedMemConfig(pConfig);
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

int handle_cudaDeviceSetSharedMemConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceSetSharedMemConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    enum cudaSharedMemConfig config;
    conn->read(&config, sizeof(config));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceSetSharedMemConfig(config);
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

int handle_cudaDeviceGetByPCIBusId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetByPCIBusId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *device;
    conn->read(&device, sizeof(device));
    char *pciBusId = nullptr;
    conn->read(&pciBusId, 0, true);
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(pciBusId);
    _result = cudaDeviceGetByPCIBusId(device, pciBusId);
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

int handle_cudaDeviceGetPCIBusId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetPCIBusId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    char pciBusId[1024];
    int len;
    conn->read(&len, sizeof(len));
    int device;
    conn->read(&device, sizeof(device));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetPCIBusId(pciBusId, len, device);
    if(len > 0) {
        conn->write(pciBusId, strlen(pciBusId) + 1, true);
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

int handle_cudaIpcGetEventHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaIpcGetEventHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaIpcEventHandle_t *handle;
    conn->read(&handle, sizeof(handle));
    cudaEvent_t event;
    conn->read(&event, sizeof(event));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaIpcGetEventHandle(handle, event);
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

int handle_cudaIpcOpenEventHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaIpcOpenEventHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaEvent_t *event;
    conn->read(&event, sizeof(event));
    cudaIpcEventHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaIpcOpenEventHandle(event, handle);
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

int handle_cudaIpcGetMemHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaIpcGetMemHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaIpcMemHandle_t *handle;
    conn->read(&handle, sizeof(handle));
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaIpcGetMemHandle(handle, devPtr);
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

int handle_cudaIpcOpenMemHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaIpcOpenMemHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *devPtr;
    cudaIpcMemHandle_t handle;
    conn->read(&handle, sizeof(handle));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaIpcOpenMemHandle(&devPtr, handle, flags);
    conn->write(&devPtr, sizeof(devPtr));
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

int handle_cudaIpcCloseMemHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaIpcCloseMemHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaIpcCloseMemHandle(devPtr);
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

int handle_cudaDeviceFlushGPUDirectRDMAWrites(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceFlushGPUDirectRDMAWrites called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    enum cudaFlushGPUDirectRDMAWritesTarget target;
    conn->read(&target, sizeof(target));
    enum cudaFlushGPUDirectRDMAWritesScope scope;
    conn->read(&scope, sizeof(scope));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceFlushGPUDirectRDMAWrites(target, scope);
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

int handle_cudaThreadExit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadExit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadExit();
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

int handle_cudaThreadSynchronize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadSynchronize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadSynchronize();
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

int handle_cudaThreadSetLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadSetLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    enum cudaLimit limit;
    conn->read(&limit, sizeof(limit));
    size_t value;
    conn->read(&value, sizeof(value));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadSetLimit(limit, value);
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

int handle_cudaThreadGetLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadGetLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    size_t *pValue;
    conn->read(&pValue, sizeof(pValue));
    enum cudaLimit limit;
    conn->read(&limit, sizeof(limit));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadGetLimit(pValue, limit);
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

int handle_cudaThreadGetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadGetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    enum cudaFuncCache *pCacheConfig;
    conn->read(&pCacheConfig, sizeof(pCacheConfig));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadGetCacheConfig(pCacheConfig);
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

int handle_cudaThreadSetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadSetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    enum cudaFuncCache cacheConfig;
    conn->read(&cacheConfig, sizeof(cacheConfig));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadSetCacheConfig(cacheConfig);
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

int handle_cudaGetLastError(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetLastError called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetLastError();
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

int handle_cudaPeekAtLastError(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaPeekAtLastError called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaPeekAtLastError();
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

int handle_cudaGetDeviceCount(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetDeviceCount called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *count;
    conn->read(&count, sizeof(count));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetDeviceCount(count);
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

int handle_cudaGetDeviceProperties(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetDeviceProperties called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaDeviceProp *prop;
    conn->read(&prop, sizeof(prop));
    int device;
    conn->read(&device, sizeof(device));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetDeviceProperties(prop, device);
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

int handle_cudaDeviceGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *value;
    conn->read(&value, sizeof(value));
    enum cudaDeviceAttr attr;
    conn->read(&attr, sizeof(attr));
    int device;
    conn->read(&device, sizeof(device));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetAttribute(value, attr, device);
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

int handle_cudaDeviceGetDefaultMemPool(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetDefaultMemPool called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaMemPool_t *memPool;
    conn->read(&memPool, sizeof(memPool));
    int device;
    conn->read(&device, sizeof(device));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetDefaultMemPool(memPool, device);
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

int handle_cudaDeviceSetMemPool(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceSetMemPool called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int device;
    conn->read(&device, sizeof(device));
    cudaMemPool_t memPool;
    conn->read(&memPool, sizeof(memPool));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceSetMemPool(device, memPool);
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

int handle_cudaDeviceGetMemPool(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetMemPool called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaMemPool_t *memPool;
    conn->read(&memPool, sizeof(memPool));
    int device;
    conn->read(&device, sizeof(device));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetMemPool(memPool, device);
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

int handle_cudaDeviceGetNvSciSyncAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetNvSciSyncAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *nvSciSyncAttrList;
    conn->read(&nvSciSyncAttrList, sizeof(nvSciSyncAttrList));
    int device;
    conn->read(&device, sizeof(device));
    int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, device, flags);
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

int handle_cudaDeviceGetP2PAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetP2PAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *value;
    conn->read(&value, sizeof(value));
    enum cudaDeviceP2PAttr attr;
    conn->read(&attr, sizeof(attr));
    int srcDevice;
    conn->read(&srcDevice, sizeof(srcDevice));
    int dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice);
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

int handle_cudaChooseDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaChooseDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *device;
    conn->read(&device, sizeof(device));
    struct cudaDeviceProp *prop = nullptr;
    conn->read(&prop, sizeof(prop));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaChooseDevice(device, prop);
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

int handle_cudaSetDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaSetDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int device;
    conn->read(&device, sizeof(device));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaSetDevice(device);
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

int handle_cudaGetDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *device;
    conn->read(&device, sizeof(device));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetDevice(device);
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

int handle_cudaSetValidDevices(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaSetValidDevices called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *device_arr;
    conn->read(&device_arr, sizeof(device_arr));
    int len;
    conn->read(&len, sizeof(len));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaSetValidDevices(device_arr, len);
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

int handle_cudaSetDeviceFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaSetDeviceFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaSetDeviceFlags(flags);
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

int handle_cudaGetDeviceFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetDeviceFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    unsigned int *flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetDeviceFlags(flags);
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

int handle_cudaStreamCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t *pStream;
    conn->read(&pStream, sizeof(pStream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamCreate(pStream);
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

int handle_cudaStreamCreateWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamCreateWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t *pStream;
    conn->read(&pStream, sizeof(pStream));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamCreateWithFlags(pStream, flags);
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

int handle_cudaStreamCreateWithPriority(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamCreateWithPriority called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t *pStream;
    conn->read(&pStream, sizeof(pStream));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    int priority;
    conn->read(&priority, sizeof(priority));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamCreateWithPriority(pStream, flags, priority);
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

int handle_cudaStreamGetPriority(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamGetPriority called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t hStream;
    conn->read(&hStream, sizeof(hStream));
    int *priority;
    conn->read(&priority, sizeof(priority));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamGetPriority(hStream, priority);
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

int handle_cudaStreamGetFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamGetFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t hStream;
    conn->read(&hStream, sizeof(hStream));
    unsigned int *flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamGetFlags(hStream, flags);
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

int handle_cudaCtxResetPersistingL2Cache(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaCtxResetPersistingL2Cache called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaCtxResetPersistingL2Cache();
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

int handle_cudaStreamCopyAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamCopyAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t dst;
    conn->read(&dst, sizeof(dst));
    cudaStream_t src;
    conn->read(&src, sizeof(src));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamCopyAttributes(dst, src);
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

int handle_cudaStreamGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t hStream;
    conn->read(&hStream, sizeof(hStream));
    enum cudaStreamAttrID attr;
    conn->read(&attr, sizeof(attr));
    union cudaStreamAttrValue *value_out;
    conn->read(&value_out, sizeof(value_out));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamGetAttribute(hStream, attr, value_out);
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

int handle_cudaStreamSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t hStream;
    conn->read(&hStream, sizeof(hStream));
    enum cudaStreamAttrID attr;
    conn->read(&attr, sizeof(attr));
    union cudaStreamAttrValue *value = nullptr;
    conn->read(&value, sizeof(value));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamSetAttribute(hStream, attr, value);
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

int handle_cudaStreamDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamDestroy(stream);
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

int handle_cudaStreamWaitEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamWaitEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaEvent_t event;
    conn->read(&event, sizeof(event));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamWaitEvent(stream, event, flags);
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

int handle_cudaStreamAddCallback(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamAddCallback called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaStreamCallback_t callback;
    conn->read(&callback, sizeof(callback));
    void *userData;
    conn->read(&userData, sizeof(userData));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamAddCallback(stream, callback, userData, flags);
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

int handle_cudaStreamSynchronize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamSynchronize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamSynchronize(stream);
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

int handle_cudaStreamQuery(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamQuery called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamQuery(stream);
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

int handle_cudaStreamAttachMemAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamAttachMemAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    size_t length;
    conn->read(&length, sizeof(length));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamAttachMemAsync(stream, devPtr, length, flags);
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

int handle_cudaStreamBeginCapture(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamBeginCapture called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    enum cudaStreamCaptureMode mode;
    conn->read(&mode, sizeof(mode));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamBeginCapture(stream, mode);
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

int handle_cudaThreadExchangeStreamCaptureMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaThreadExchangeStreamCaptureMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    enum cudaStreamCaptureMode *mode;
    conn->read(&mode, sizeof(mode));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaThreadExchangeStreamCaptureMode(mode);
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

int handle_cudaStreamEndCapture(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamEndCapture called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaGraph_t *pGraph;
    conn->read(&pGraph, sizeof(pGraph));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamEndCapture(stream, pGraph);
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

int handle_cudaStreamIsCapturing(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamIsCapturing called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    enum cudaStreamCaptureStatus *pCaptureStatus;
    conn->read(&pCaptureStatus, sizeof(pCaptureStatus));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamIsCapturing(stream, pCaptureStatus);
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

int handle_cudaStreamGetCaptureInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamGetCaptureInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    enum cudaStreamCaptureStatus *pCaptureStatus;
    conn->read(&pCaptureStatus, sizeof(pCaptureStatus));
    unsigned long long *pId;
    conn->read(&pId, sizeof(pId));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamGetCaptureInfo(stream, pCaptureStatus, pId);
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

int handle_cudaStreamGetCaptureInfo_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamGetCaptureInfo_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    enum cudaStreamCaptureStatus *captureStatus_out;
    conn->read(&captureStatus_out, sizeof(captureStatus_out));
    unsigned long long *id_out;
    conn->read(&id_out, sizeof(id_out));
    cudaGraph_t *graph_out;
    conn->read(&graph_out, sizeof(graph_out));
    const cudaGraphNode_t *dependencies_out;
    size_t *numDependencies_out;
    conn->read(&numDependencies_out, sizeof(numDependencies_out));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, &dependencies_out, numDependencies_out);
    conn->write(dependencies_out, sizeof(cudaGraphNode_t));
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

int handle_cudaStreamUpdateCaptureDependencies(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaStreamUpdateCaptureDependencies called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaGraphNode_t *dependencies;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags);
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

int handle_cudaEventCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaEvent_t *event;
    conn->read(&event, sizeof(event));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventCreate(event);
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

int handle_cudaEventCreateWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventCreateWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaEvent_t *event;
    conn->read(&event, sizeof(event));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventCreateWithFlags(event, flags);
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

int handle_cudaEventRecord(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventRecord called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaEvent_t event;
    conn->read(&event, sizeof(event));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventRecord(event, stream);
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

int handle_cudaEventRecordWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventRecordWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaEvent_t event;
    conn->read(&event, sizeof(event));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventRecordWithFlags(event, stream, flags);
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

int handle_cudaEventQuery(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventQuery called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaEvent_t event;
    conn->read(&event, sizeof(event));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventQuery(event);
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

int handle_cudaEventSynchronize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventSynchronize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaEvent_t event;
    conn->read(&event, sizeof(event));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventSynchronize(event);
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

int handle_cudaEventDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaEvent_t event;
    conn->read(&event, sizeof(event));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventDestroy(event);
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

int handle_cudaEventElapsedTime(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaEventElapsedTime called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    float *ms;
    conn->read(&ms, sizeof(ms));
    cudaEvent_t start;
    conn->read(&start, sizeof(start));
    cudaEvent_t end;
    conn->read(&end, sizeof(end));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaEventElapsedTime(ms, start, end);
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

int handle_cudaImportExternalMemory(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaImportExternalMemory called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaExternalMemory_t *extMem_out;
    conn->read(&extMem_out, sizeof(extMem_out));
    struct cudaExternalMemoryHandleDesc *memHandleDesc = nullptr;
    conn->read(&memHandleDesc, sizeof(memHandleDesc));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaImportExternalMemory(extMem_out, memHandleDesc);
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

int handle_cudaExternalMemoryGetMappedBuffer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaExternalMemoryGetMappedBuffer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *devPtr;
    cudaExternalMemory_t extMem;
    conn->read(&extMem, sizeof(extMem));
    struct cudaExternalMemoryBufferDesc *bufferDesc = nullptr;
    conn->read(&bufferDesc, sizeof(bufferDesc));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, bufferDesc);
    conn->write(&devPtr, sizeof(devPtr));
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

int handle_cudaExternalMemoryGetMappedMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaExternalMemoryGetMappedMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaMipmappedArray_t *mipmap;
    conn->read(&mipmap, sizeof(mipmap));
    cudaExternalMemory_t extMem;
    conn->read(&extMem, sizeof(extMem));
    struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc = nullptr;
    conn->read(&mipmapDesc, sizeof(mipmapDesc));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc);
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

int handle_cudaDestroyExternalMemory(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDestroyExternalMemory called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaExternalMemory_t extMem;
    conn->read(&extMem, sizeof(extMem));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDestroyExternalMemory(extMem);
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

int handle_cudaImportExternalSemaphore(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaImportExternalSemaphore called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaExternalSemaphore_t *extSem_out;
    conn->read(&extSem_out, sizeof(extSem_out));
    struct cudaExternalSemaphoreHandleDesc *semHandleDesc = nullptr;
    conn->read(&semHandleDesc, sizeof(semHandleDesc));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaImportExternalSemaphore(extSem_out, semHandleDesc);
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

int handle_cudaSignalExternalSemaphoresAsync_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaSignalExternalSemaphoresAsync_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaExternalSemaphore_t *extSemArray = nullptr;
    conn->read(&extSemArray, sizeof(extSemArray));
    struct cudaExternalSemaphoreSignalParams *paramsArray = nullptr;
    conn->read(&paramsArray, sizeof(paramsArray));
    unsigned int numExtSems;
    conn->read(&numExtSems, sizeof(numExtSems));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaSignalExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream);
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

int handle_cudaWaitExternalSemaphoresAsync_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaWaitExternalSemaphoresAsync_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaExternalSemaphore_t *extSemArray = nullptr;
    conn->read(&extSemArray, sizeof(extSemArray));
    struct cudaExternalSemaphoreWaitParams *paramsArray = nullptr;
    conn->read(&paramsArray, sizeof(paramsArray));
    unsigned int numExtSems;
    conn->read(&numExtSems, sizeof(numExtSems));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaWaitExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream);
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

int handle_cudaDestroyExternalSemaphore(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDestroyExternalSemaphore called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaExternalSemaphore_t extSem;
    conn->read(&extSem, sizeof(extSem));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDestroyExternalSemaphore(extSem);
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

int handle_cudaLaunchCooperativeKernelMultiDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaLaunchCooperativeKernelMultiDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaLaunchParams *launchParamsList;
    conn->read(&launchParamsList, sizeof(launchParamsList));
    unsigned int numDevices;
    conn->read(&numDevices, sizeof(numDevices));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
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

int handle_cudaFuncSetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFuncSetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *func;
    conn->read(&func, sizeof(func));
    enum cudaFuncCache cacheConfig;
    conn->read(&cacheConfig, sizeof(cacheConfig));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFuncSetCacheConfig(func, cacheConfig);
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

int handle_cudaFuncSetSharedMemConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFuncSetSharedMemConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *func;
    conn->read(&func, sizeof(func));
    enum cudaSharedMemConfig config;
    conn->read(&config, sizeof(config));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFuncSetSharedMemConfig(func, config);
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

int handle_cudaFuncGetAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFuncGetAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaFuncAttributes *attr;
    conn->read(&attr, sizeof(attr));
    void *func;
    conn->read(&func, sizeof(func));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFuncGetAttributes(attr, func);
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

int handle_cudaFuncSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFuncSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *func;
    conn->read(&func, sizeof(func));
    enum cudaFuncAttribute attr;
    conn->read(&attr, sizeof(attr));
    int value;
    conn->read(&value, sizeof(value));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFuncSetAttribute(func, attr, value);
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

int handle_cudaSetDoubleForDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaSetDoubleForDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    double *d;
    conn->read(&d, sizeof(d));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaSetDoubleForDevice(d);
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

int handle_cudaSetDoubleForHost(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaSetDoubleForHost called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    double *d;
    conn->read(&d, sizeof(d));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaSetDoubleForHost(d);
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

int handle_cudaLaunchHostFunc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaLaunchHostFunc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaHostFn_t fn;
    conn->read(&fn, sizeof(fn));
    void *userData;
    conn->read(&userData, sizeof(userData));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaLaunchHostFunc(stream, fn, userData);
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

int handle_cudaOccupancyMaxActiveBlocksPerMultiprocessor(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaOccupancyMaxActiveBlocksPerMultiprocessor called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *numBlocks;
    conn->read(&numBlocks, sizeof(numBlocks));
    void *func;
    conn->read(&func, sizeof(func));
    int blockSize;
    conn->read(&blockSize, sizeof(blockSize));
    size_t dynamicSMemSize;
    conn->read(&dynamicSMemSize, sizeof(dynamicSMemSize));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
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

int handle_cudaOccupancyAvailableDynamicSMemPerBlock(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaOccupancyAvailableDynamicSMemPerBlock called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    size_t *dynamicSmemSize;
    conn->read(&dynamicSmemSize, sizeof(dynamicSmemSize));
    void *func;
    conn->read(&func, sizeof(func));
    int numBlocks;
    conn->read(&numBlocks, sizeof(numBlocks));
    int blockSize;
    conn->read(&blockSize, sizeof(blockSize));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize);
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

int handle_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *numBlocks;
    conn->read(&numBlocks, sizeof(numBlocks));
    void *func;
    conn->read(&func, sizeof(func));
    int blockSize;
    conn->read(&blockSize, sizeof(blockSize));
    size_t dynamicSMemSize;
    conn->read(&dynamicSMemSize, sizeof(dynamicSMemSize));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
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

int handle_cudaMallocArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMallocArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaArray_t *array;
    conn->read(&array, sizeof(array));
    struct cudaChannelFormatDesc *desc = nullptr;
    conn->read(&desc, sizeof(desc));
    size_t width;
    conn->read(&width, sizeof(width));
    size_t height;
    conn->read(&height, sizeof(height));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMallocArray(array, desc, width, height, flags);
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

int handle_cudaFreeArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFreeArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaArray_t array;
    conn->read(&array, sizeof(array));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFreeArray(array);
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

int handle_cudaFreeMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFreeMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaMipmappedArray_t mipmappedArray;
    conn->read(&mipmappedArray, sizeof(mipmappedArray));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFreeMipmappedArray(mipmappedArray);
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

int handle_cudaHostGetDevicePointer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaHostGetDevicePointer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *pDevice;
    void *pHost;
    conn->read(&pHost, sizeof(pHost));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaHostGetDevicePointer(&pDevice, pHost, flags);
    conn->write(&pDevice, sizeof(pDevice));
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

int handle_cudaHostGetFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaHostGetFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    unsigned int *pFlags;
    conn->read(&pFlags, sizeof(pFlags));
    void *pHost;
    conn->read(&pHost, sizeof(pHost));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaHostGetFlags(pFlags, pHost);
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

int handle_cudaMalloc3DArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMalloc3DArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaArray_t *array;
    conn->read(&array, sizeof(array));
    struct cudaChannelFormatDesc *desc = nullptr;
    conn->read(&desc, sizeof(desc));
    struct cudaExtent extent;
    conn->read(&extent, sizeof(extent));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMalloc3DArray(array, desc, extent, flags);
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

int handle_cudaMallocMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMallocMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaMipmappedArray_t *mipmappedArray;
    conn->read(&mipmappedArray, sizeof(mipmappedArray));
    struct cudaChannelFormatDesc *desc = nullptr;
    conn->read(&desc, sizeof(desc));
    struct cudaExtent extent;
    conn->read(&extent, sizeof(extent));
    unsigned int numLevels;
    conn->read(&numLevels, sizeof(numLevels));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags);
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

int handle_cudaGetMipmappedArrayLevel(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetMipmappedArrayLevel called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaArray_t *levelArray;
    conn->read(&levelArray, sizeof(levelArray));
    cudaMipmappedArray_const_t mipmappedArray;
    conn->read(&mipmappedArray, sizeof(mipmappedArray));
    unsigned int level;
    conn->read(&level, sizeof(level));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level);
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

int handle_cudaMemcpy3D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy3D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaMemcpy3DParms *p = nullptr;
    conn->read(&p, sizeof(p));
    void *_0sptr;
    conn->read(&_0sptr, sizeof(_0sptr));
    void *_0dptr;
    conn->read(&_0dptr, sizeof(_0dptr));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    if(p != nullptr) {
        p->srcPtr.ptr = _0sptr;
        p->dstPtr.ptr = _0dptr;
    }
    _result = cudaMemcpy3D(p);
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

int handle_cudaMemcpy3DPeer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy3DPeer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaMemcpy3DPeerParms *p = nullptr;
    conn->read(&p, sizeof(p));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy3DPeer(p);
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

int handle_cudaMemcpy3DAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy3DAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaMemcpy3DParms *p = nullptr;
    conn->read(&p, sizeof(p));
    void *_0sptr;
    conn->read(&_0sptr, sizeof(_0sptr));
    void *_0dptr;
    conn->read(&_0dptr, sizeof(_0dptr));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    if(p != nullptr) {
        p->srcPtr.ptr = _0sptr;
        p->dstPtr.ptr = _0dptr;
    }
    _result = cudaMemcpy3DAsync(p, stream);
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

int handle_cudaMemcpy3DPeerAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy3DPeerAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaMemcpy3DPeerParms *p = nullptr;
    conn->read(&p, sizeof(p));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy3DPeerAsync(p, stream);
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

int handle_cudaMemGetInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemGetInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    size_t *free;
    conn->read(&free, sizeof(free));
    size_t *total;
    conn->read(&total, sizeof(total));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemGetInfo(free, total);
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

int handle_cudaArrayGetInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaArrayGetInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaChannelFormatDesc *desc;
    conn->read(&desc, sizeof(desc));
    struct cudaExtent *extent;
    conn->read(&extent, sizeof(extent));
    unsigned int *flags;
    conn->read(&flags, sizeof(flags));
    cudaArray_t array;
    conn->read(&array, sizeof(array));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaArrayGetInfo(desc, extent, flags, array);
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

int handle_cudaArrayGetPlane(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaArrayGetPlane called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaArray_t *pPlaneArray;
    conn->read(&pPlaneArray, sizeof(pPlaneArray));
    cudaArray_t hArray;
    conn->read(&hArray, sizeof(hArray));
    unsigned int planeIdx;
    conn->read(&planeIdx, sizeof(planeIdx));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaArrayGetPlane(pPlaneArray, hArray, planeIdx);
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

int handle_cudaArrayGetSparseProperties(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaArrayGetSparseProperties called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaArraySparseProperties *sparseProperties;
    conn->read(&sparseProperties, sizeof(sparseProperties));
    cudaArray_t array;
    conn->read(&array, sizeof(array));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaArrayGetSparseProperties(sparseProperties, array);
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

int handle_cudaMipmappedArrayGetSparseProperties(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMipmappedArrayGetSparseProperties called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaArraySparseProperties *sparseProperties;
    conn->read(&sparseProperties, sizeof(sparseProperties));
    cudaMipmappedArray_t mipmap;
    conn->read(&mipmap, sizeof(mipmap));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMipmappedArrayGetSparseProperties(sparseProperties, mipmap);
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

int handle_cudaMemcpy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *dst;
    conn->read(&dst, sizeof(dst));
    void *src;
    conn->read(&src, sizeof(src));
    size_t count;
    conn->read(&count, sizeof(count));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy(dst, src, count, kind);
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

int handle_cudaMemcpyPeer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyPeer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *dst;
    conn->read(&dst, sizeof(dst));
    int dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    void *src;
    conn->read(&src, sizeof(src));
    int srcDevice;
    conn->read(&srcDevice, sizeof(srcDevice));
    size_t count;
    conn->read(&count, sizeof(count));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
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

int handle_cudaMemcpy2D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *dst;
    conn->read(&dst, sizeof(dst));
    size_t dpitch;
    conn->read(&dpitch, sizeof(dpitch));
    void *src;
    conn->read(&src, sizeof(src));
    size_t spitch;
    conn->read(&spitch, sizeof(spitch));
    size_t width;
    conn->read(&width, sizeof(width));
    size_t height;
    conn->read(&height, sizeof(height));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
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

int handle_cudaMemcpy2DToArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2DToArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaArray_t dst;
    conn->read(&dst, sizeof(dst));
    size_t wOffset;
    conn->read(&wOffset, sizeof(wOffset));
    size_t hOffset;
    conn->read(&hOffset, sizeof(hOffset));
    void *src;
    conn->read(&src, sizeof(src));
    size_t spitch;
    conn->read(&spitch, sizeof(spitch));
    size_t width;
    conn->read(&width, sizeof(width));
    size_t height;
    conn->read(&height, sizeof(height));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
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

int handle_cudaMemcpy2DFromArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2DFromArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *dst;
    conn->read(&dst, sizeof(dst));
    size_t dpitch;
    conn->read(&dpitch, sizeof(dpitch));
    cudaArray_const_t src;
    conn->read(&src, sizeof(src));
    size_t wOffset;
    conn->read(&wOffset, sizeof(wOffset));
    size_t hOffset;
    conn->read(&hOffset, sizeof(hOffset));
    size_t width;
    conn->read(&width, sizeof(width));
    size_t height;
    conn->read(&height, sizeof(height));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
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

int handle_cudaMemcpy2DArrayToArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2DArrayToArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaArray_t dst;
    conn->read(&dst, sizeof(dst));
    size_t wOffsetDst;
    conn->read(&wOffsetDst, sizeof(wOffsetDst));
    size_t hOffsetDst;
    conn->read(&hOffsetDst, sizeof(hOffsetDst));
    cudaArray_const_t src;
    conn->read(&src, sizeof(src));
    size_t wOffsetSrc;
    conn->read(&wOffsetSrc, sizeof(wOffsetSrc));
    size_t hOffsetSrc;
    conn->read(&hOffsetSrc, sizeof(hOffsetSrc));
    size_t width;
    conn->read(&width, sizeof(width));
    size_t height;
    conn->read(&height, sizeof(height));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
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

int handle_cudaMemcpyToSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyToSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    void *src;
    conn->read(&src, sizeof(src));
    size_t count;
    conn->read(&count, sizeof(count));
    size_t offset;
    conn->read(&offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyToSymbol(symbol, src, count, offset, kind);
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

int handle_cudaMemcpyFromSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyFromSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *dst;
    conn->read(&dst, sizeof(dst));
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    size_t count;
    conn->read(&count, sizeof(count));
    size_t offset;
    conn->read(&offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
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

int handle_cudaMemcpyAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *dst;
    conn->read(&dst, sizeof(dst));
    void *src;
    conn->read(&src, sizeof(src));
    size_t count;
    conn->read(&count, sizeof(count));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyAsync(dst, src, count, kind, stream);
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

int handle_cudaMemcpyPeerAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyPeerAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *dst;
    conn->read(&dst, sizeof(dst));
    int dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    void *src;
    conn->read(&src, sizeof(src));
    int srcDevice;
    conn->read(&srcDevice, sizeof(srcDevice));
    size_t count;
    conn->read(&count, sizeof(count));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
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

int handle_cudaMemcpy2DAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2DAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *dst;
    conn->read(&dst, sizeof(dst));
    size_t dpitch;
    conn->read(&dpitch, sizeof(dpitch));
    void *src;
    conn->read(&src, sizeof(src));
    size_t spitch;
    conn->read(&spitch, sizeof(spitch));
    size_t width;
    conn->read(&width, sizeof(width));
    size_t height;
    conn->read(&height, sizeof(height));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
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

int handle_cudaMemcpy2DToArrayAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2DToArrayAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaArray_t dst;
    conn->read(&dst, sizeof(dst));
    size_t wOffset;
    conn->read(&wOffset, sizeof(wOffset));
    size_t hOffset;
    conn->read(&hOffset, sizeof(hOffset));
    void *src;
    conn->read(&src, sizeof(src));
    size_t spitch;
    conn->read(&spitch, sizeof(spitch));
    size_t width;
    conn->read(&width, sizeof(width));
    size_t height;
    conn->read(&height, sizeof(height));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
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

int handle_cudaMemcpy2DFromArrayAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpy2DFromArrayAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *dst;
    conn->read(&dst, sizeof(dst));
    size_t dpitch;
    conn->read(&dpitch, sizeof(dpitch));
    cudaArray_const_t src;
    conn->read(&src, sizeof(src));
    size_t wOffset;
    conn->read(&wOffset, sizeof(wOffset));
    size_t hOffset;
    conn->read(&hOffset, sizeof(hOffset));
    size_t width;
    conn->read(&width, sizeof(width));
    size_t height;
    conn->read(&height, sizeof(height));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
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

int handle_cudaMemcpyToSymbolAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyToSymbolAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    void *src;
    conn->read(&src, sizeof(src));
    size_t count;
    conn->read(&count, sizeof(count));
    size_t offset;
    conn->read(&offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);
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

int handle_cudaMemcpyFromSymbolAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyFromSymbolAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *dst;
    conn->read(&dst, sizeof(dst));
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    size_t count;
    conn->read(&count, sizeof(count));
    size_t offset;
    conn->read(&offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, stream);
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

int handle_cudaMemset(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemset called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    int value;
    conn->read(&value, sizeof(value));
    size_t count;
    conn->read(&count, sizeof(count));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemset(devPtr, value, count);
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

int handle_cudaMemset2D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemset2D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    size_t pitch;
    conn->read(&pitch, sizeof(pitch));
    int value;
    conn->read(&value, sizeof(value));
    size_t width;
    conn->read(&width, sizeof(width));
    size_t height;
    conn->read(&height, sizeof(height));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemset2D(devPtr, pitch, value, width, height);
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

int handle_cudaMemset3D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemset3D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaPitchedPtr pitchedDevPtr;
    conn->read(&pitchedDevPtr, sizeof(pitchedDevPtr));
    int value;
    conn->read(&value, sizeof(value));
    struct cudaExtent extent;
    conn->read(&extent, sizeof(extent));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemset3D(pitchedDevPtr, value, extent);
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

int handle_cudaMemsetAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemsetAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    int value;
    conn->read(&value, sizeof(value));
    size_t count;
    conn->read(&count, sizeof(count));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemsetAsync(devPtr, value, count, stream);
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

int handle_cudaMemset2DAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemset2DAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    size_t pitch;
    conn->read(&pitch, sizeof(pitch));
    int value;
    conn->read(&value, sizeof(value));
    size_t width;
    conn->read(&width, sizeof(width));
    size_t height;
    conn->read(&height, sizeof(height));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemset2DAsync(devPtr, pitch, value, width, height, stream);
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

int handle_cudaMemset3DAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemset3DAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaPitchedPtr pitchedDevPtr;
    conn->read(&pitchedDevPtr, sizeof(pitchedDevPtr));
    int value;
    conn->read(&value, sizeof(value));
    struct cudaExtent extent;
    conn->read(&extent, sizeof(extent));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemset3DAsync(pitchedDevPtr, value, extent, stream);
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

int handle_cudaGetSymbolSize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetSymbolSize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    size_t *size;
    conn->read(&size, sizeof(size));
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetSymbolSize(size, symbol);
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

int handle_cudaMemPrefetchAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPrefetchAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    size_t count;
    conn->read(&count, sizeof(count));
    int dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);
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

int handle_cudaMemAdvise(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemAdvise called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    size_t count;
    conn->read(&count, sizeof(count));
    enum cudaMemoryAdvise advice;
    conn->read(&advice, sizeof(advice));
    int device;
    conn->read(&device, sizeof(device));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemAdvise(devPtr, count, advice, device);
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

int handle_cudaMemRangeGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemRangeGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *data;
    conn->read(&data, sizeof(data));
    size_t dataSize;
    conn->read(&dataSize, sizeof(dataSize));
    enum cudaMemRangeAttribute attribute;
    conn->read(&attribute, sizeof(attribute));
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
    _result = cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);
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

int handle_cudaMemcpyToArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyToArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaArray_t dst;
    conn->read(&dst, sizeof(dst));
    size_t wOffset;
    conn->read(&wOffset, sizeof(wOffset));
    size_t hOffset;
    conn->read(&hOffset, sizeof(hOffset));
    void *src;
    conn->read(&src, sizeof(src));
    size_t count;
    conn->read(&count, sizeof(count));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
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

int handle_cudaMemcpyFromArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyFromArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *dst;
    conn->read(&dst, sizeof(dst));
    cudaArray_const_t src;
    conn->read(&src, sizeof(src));
    size_t wOffset;
    conn->read(&wOffset, sizeof(wOffset));
    size_t hOffset;
    conn->read(&hOffset, sizeof(hOffset));
    size_t count;
    conn->read(&count, sizeof(count));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
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

int handle_cudaMemcpyArrayToArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyArrayToArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaArray_t dst;
    conn->read(&dst, sizeof(dst));
    size_t wOffsetDst;
    conn->read(&wOffsetDst, sizeof(wOffsetDst));
    size_t hOffsetDst;
    conn->read(&hOffsetDst, sizeof(hOffsetDst));
    cudaArray_const_t src;
    conn->read(&src, sizeof(src));
    size_t wOffsetSrc;
    conn->read(&wOffsetSrc, sizeof(wOffsetSrc));
    size_t hOffsetSrc;
    conn->read(&hOffsetSrc, sizeof(hOffsetSrc));
    size_t count;
    conn->read(&count, sizeof(count));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);
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

int handle_cudaMemcpyToArrayAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyToArrayAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaArray_t dst;
    conn->read(&dst, sizeof(dst));
    size_t wOffset;
    conn->read(&wOffset, sizeof(wOffset));
    size_t hOffset;
    conn->read(&hOffset, sizeof(hOffset));
    void *src;
    conn->read(&src, sizeof(src));
    size_t count;
    conn->read(&count, sizeof(count));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream);
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

int handle_cudaMemcpyFromArrayAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemcpyFromArrayAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *dst;
    conn->read(&dst, sizeof(dst));
    cudaArray_const_t src;
    conn->read(&src, sizeof(src));
    size_t wOffset;
    conn->read(&wOffset, sizeof(wOffset));
    size_t hOffset;
    conn->read(&hOffset, sizeof(hOffset));
    size_t count;
    conn->read(&count, sizeof(count));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream);
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

int handle_cudaMallocAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMallocAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *devPtr;
    size_t size;
    conn->read(&size, sizeof(size));
    cudaStream_t hStream;
    conn->read(&hStream, sizeof(hStream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMallocAsync(&devPtr, size, hStream);
    conn->write(&devPtr, sizeof(devPtr));
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

int handle_cudaFreeAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaFreeAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    cudaStream_t hStream;
    conn->read(&hStream, sizeof(hStream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaFreeAsync(devPtr, hStream);
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

int handle_cudaMemPoolTrimTo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolTrimTo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaMemPool_t memPool;
    conn->read(&memPool, sizeof(memPool));
    size_t minBytesToKeep;
    conn->read(&minBytesToKeep, sizeof(minBytesToKeep));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolTrimTo(memPool, minBytesToKeep);
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

int handle_cudaMemPoolSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaMemPool_t memPool;
    conn->read(&memPool, sizeof(memPool));
    enum cudaMemPoolAttr attr;
    conn->read(&attr, sizeof(attr));
    void *value;
    conn->read(&value, sizeof(value));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolSetAttribute(memPool, attr, value);
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

int handle_cudaMemPoolGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaMemPool_t memPool;
    conn->read(&memPool, sizeof(memPool));
    enum cudaMemPoolAttr attr;
    conn->read(&attr, sizeof(attr));
    void *value;
    conn->read(&value, sizeof(value));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolGetAttribute(memPool, attr, value);
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

int handle_cudaMemPoolSetAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolSetAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaMemPool_t memPool;
    conn->read(&memPool, sizeof(memPool));
    struct cudaMemAccessDesc *descList = nullptr;
    conn->read(&descList, sizeof(descList));
    size_t count;
    conn->read(&count, sizeof(count));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolSetAccess(memPool, descList, count);
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

int handle_cudaMemPoolGetAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolGetAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    enum cudaMemAccessFlags *flags;
    conn->read(&flags, sizeof(flags));
    cudaMemPool_t memPool;
    conn->read(&memPool, sizeof(memPool));
    struct cudaMemLocation *location;
    conn->read(&location, sizeof(location));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolGetAccess(flags, memPool, location);
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

int handle_cudaMemPoolCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaMemPool_t *memPool;
    conn->read(&memPool, sizeof(memPool));
    struct cudaMemPoolProps *poolProps = nullptr;
    conn->read(&poolProps, sizeof(poolProps));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolCreate(memPool, poolProps);
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

int handle_cudaMemPoolDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaMemPool_t memPool;
    conn->read(&memPool, sizeof(memPool));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolDestroy(memPool);
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

int handle_cudaMallocFromPoolAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMallocFromPoolAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *ptr;
    size_t size;
    conn->read(&size, sizeof(size));
    cudaMemPool_t memPool;
    conn->read(&memPool, sizeof(memPool));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMallocFromPoolAsync(&ptr, size, memPool, stream);
    conn->write(&ptr, sizeof(ptr));
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

int handle_cudaMemPoolExportToShareableHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolExportToShareableHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *shareableHandle;
    conn->read(&shareableHandle, sizeof(shareableHandle));
    cudaMemPool_t memPool;
    conn->read(&memPool, sizeof(memPool));
    enum cudaMemAllocationHandleType handleType;
    conn->read(&handleType, sizeof(handleType));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolExportToShareableHandle(shareableHandle, memPool, handleType, flags);
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

int handle_cudaMemPoolImportFromShareableHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolImportFromShareableHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaMemPool_t *memPool;
    conn->read(&memPool, sizeof(memPool));
    void *shareableHandle;
    conn->read(&shareableHandle, sizeof(shareableHandle));
    enum cudaMemAllocationHandleType handleType;
    conn->read(&handleType, sizeof(handleType));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolImportFromShareableHandle(memPool, shareableHandle, handleType, flags);
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

int handle_cudaMemPoolExportPointer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolExportPointer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaMemPoolPtrExportData *exportData;
    conn->read(&exportData, sizeof(exportData));
    void *ptr;
    conn->read(&ptr, sizeof(ptr));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolExportPointer(exportData, ptr);
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

int handle_cudaMemPoolImportPointer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaMemPoolImportPointer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *ptr;
    cudaMemPool_t memPool;
    conn->read(&memPool, sizeof(memPool));
    struct cudaMemPoolPtrExportData *exportData;
    conn->read(&exportData, sizeof(exportData));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaMemPoolImportPointer(&ptr, memPool, exportData);
    conn->write(&ptr, sizeof(ptr));
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

int handle_cudaPointerGetAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaPointerGetAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaPointerAttributes *attributes;
    conn->read(&attributes, sizeof(attributes));
    void *ptr;
    conn->read(&ptr, sizeof(ptr));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaPointerGetAttributes(attributes, ptr);
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

int handle_cudaDeviceCanAccessPeer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceCanAccessPeer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *canAccessPeer;
    conn->read(&canAccessPeer, sizeof(canAccessPeer));
    int device;
    conn->read(&device, sizeof(device));
    int peerDevice;
    conn->read(&peerDevice, sizeof(peerDevice));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
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

int handle_cudaDeviceEnablePeerAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceEnablePeerAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int peerDevice;
    conn->read(&peerDevice, sizeof(peerDevice));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceEnablePeerAccess(peerDevice, flags);
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

int handle_cudaDeviceDisablePeerAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceDisablePeerAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int peerDevice;
    conn->read(&peerDevice, sizeof(peerDevice));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceDisablePeerAccess(peerDevice);
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

int handle_cudaGraphicsUnregisterResource(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsUnregisterResource called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphicsResource_t resource;
    conn->read(&resource, sizeof(resource));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsUnregisterResource(resource);
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

int handle_cudaGraphicsResourceSetMapFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsResourceSetMapFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphicsResource_t resource;
    conn->read(&resource, sizeof(resource));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsResourceSetMapFlags(resource, flags);
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

int handle_cudaGraphicsMapResources(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsMapResources called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int count;
    conn->read(&count, sizeof(count));
    cudaGraphicsResource_t *resources;
    conn->read(&resources, sizeof(resources));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsMapResources(count, resources, stream);
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

int handle_cudaGraphicsUnmapResources(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsUnmapResources called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int count;
    conn->read(&count, sizeof(count));
    cudaGraphicsResource_t *resources;
    conn->read(&resources, sizeof(resources));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsUnmapResources(count, resources, stream);
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

int handle_cudaGraphicsResourceGetMappedPointer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsResourceGetMappedPointer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *devPtr;
    size_t *size;
    conn->read(&size, sizeof(size));
    cudaGraphicsResource_t resource;
    conn->read(&resource, sizeof(resource));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsResourceGetMappedPointer(&devPtr, size, resource);
    conn->write(&devPtr, sizeof(devPtr));
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

int handle_cudaGraphicsSubResourceGetMappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsSubResourceGetMappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaArray_t *array;
    conn->read(&array, sizeof(array));
    cudaGraphicsResource_t resource;
    conn->read(&resource, sizeof(resource));
    unsigned int arrayIndex;
    conn->read(&arrayIndex, sizeof(arrayIndex));
    unsigned int mipLevel;
    conn->read(&mipLevel, sizeof(mipLevel));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel);
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

int handle_cudaGraphicsResourceGetMappedMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphicsResourceGetMappedMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaMipmappedArray_t *mipmappedArray;
    conn->read(&mipmappedArray, sizeof(mipmappedArray));
    cudaGraphicsResource_t resource;
    conn->read(&resource, sizeof(resource));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource);
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

int handle_cudaBindTexture(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaBindTexture called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    size_t *offset;
    conn->read(&offset, sizeof(offset));
    struct textureReference *texref = nullptr;
    conn->read(&texref, sizeof(texref));
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    struct cudaChannelFormatDesc *desc = nullptr;
    conn->read(&desc, sizeof(desc));
    size_t size;
    conn->read(&size, sizeof(size));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaBindTexture(offset, texref, devPtr, desc, size);
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

int handle_cudaBindTexture2D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaBindTexture2D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    size_t *offset;
    conn->read(&offset, sizeof(offset));
    struct textureReference *texref = nullptr;
    conn->read(&texref, sizeof(texref));
    void *devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    struct cudaChannelFormatDesc *desc = nullptr;
    conn->read(&desc, sizeof(desc));
    size_t width;
    conn->read(&width, sizeof(width));
    size_t height;
    conn->read(&height, sizeof(height));
    size_t pitch;
    conn->read(&pitch, sizeof(pitch));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaBindTexture2D(offset, texref, devPtr, desc, width, height, pitch);
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

int handle_cudaBindTextureToArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaBindTextureToArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct textureReference *texref = nullptr;
    conn->read(&texref, sizeof(texref));
    cudaArray_const_t array;
    conn->read(&array, sizeof(array));
    struct cudaChannelFormatDesc *desc = nullptr;
    conn->read(&desc, sizeof(desc));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaBindTextureToArray(texref, array, desc);
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

int handle_cudaBindTextureToMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaBindTextureToMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct textureReference *texref = nullptr;
    conn->read(&texref, sizeof(texref));
    cudaMipmappedArray_const_t mipmappedArray;
    conn->read(&mipmappedArray, sizeof(mipmappedArray));
    struct cudaChannelFormatDesc *desc = nullptr;
    conn->read(&desc, sizeof(desc));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaBindTextureToMipmappedArray(texref, mipmappedArray, desc);
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

int handle_cudaUnbindTexture(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaUnbindTexture called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct textureReference *texref = nullptr;
    conn->read(&texref, sizeof(texref));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaUnbindTexture(texref);
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

int handle_cudaGetTextureAlignmentOffset(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetTextureAlignmentOffset called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    size_t *offset;
    conn->read(&offset, sizeof(offset));
    struct textureReference *texref = nullptr;
    conn->read(&texref, sizeof(texref));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetTextureAlignmentOffset(offset, texref);
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

int handle_cudaGetTextureReference(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetTextureReference called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    const struct textureReference *texref;
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetTextureReference(&texref, symbol);
    conn->write(texref, sizeof(struct textureReference));
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

int handle_cudaBindSurfaceToArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaBindSurfaceToArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct surfaceReference *surfref = nullptr;
    conn->read(&surfref, sizeof(surfref));
    cudaArray_const_t array;
    conn->read(&array, sizeof(array));
    struct cudaChannelFormatDesc *desc = nullptr;
    conn->read(&desc, sizeof(desc));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaBindSurfaceToArray(surfref, array, desc);
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

int handle_cudaGetSurfaceReference(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetSurfaceReference called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    const struct surfaceReference *surfref;
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetSurfaceReference(&surfref, symbol);
    conn->write(surfref, sizeof(struct surfaceReference));
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

int handle_cudaGetChannelDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetChannelDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaChannelFormatDesc *desc;
    conn->read(&desc, sizeof(desc));
    cudaArray_const_t array;
    conn->read(&array, sizeof(array));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetChannelDesc(desc, array);
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

int handle_cudaCreateChannelDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaCreateChannelDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int x;
    conn->read(&x, sizeof(x));
    int y;
    conn->read(&y, sizeof(y));
    int z;
    conn->read(&z, sizeof(z));
    int w;
    conn->read(&w, sizeof(w));
    enum cudaChannelFormatKind f;
    conn->read(&f, sizeof(f));
    struct cudaChannelFormatDesc _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaCreateChannelDesc(x, y, z, w, f);
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

int handle_cudaCreateTextureObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaCreateTextureObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaTextureObject_t *pTexObject;
    conn->read(&pTexObject, sizeof(pTexObject));
    struct cudaResourceDesc *pResDesc = nullptr;
    conn->read(&pResDesc, sizeof(pResDesc));
    struct cudaTextureDesc *pTexDesc = nullptr;
    conn->read(&pTexDesc, sizeof(pTexDesc));
    struct cudaResourceViewDesc *pResViewDesc = nullptr;
    conn->read(&pResViewDesc, sizeof(pResViewDesc));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
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

int handle_cudaDestroyTextureObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDestroyTextureObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaTextureObject_t texObject;
    conn->read(&texObject, sizeof(texObject));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDestroyTextureObject(texObject);
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

int handle_cudaGetTextureObjectResourceDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetTextureObjectResourceDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaResourceDesc *pResDesc;
    conn->read(&pResDesc, sizeof(pResDesc));
    cudaTextureObject_t texObject;
    conn->read(&texObject, sizeof(texObject));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetTextureObjectResourceDesc(pResDesc, texObject);
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

int handle_cudaGetTextureObjectTextureDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetTextureObjectTextureDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaTextureDesc *pTexDesc;
    conn->read(&pTexDesc, sizeof(pTexDesc));
    cudaTextureObject_t texObject;
    conn->read(&texObject, sizeof(texObject));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetTextureObjectTextureDesc(pTexDesc, texObject);
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

int handle_cudaGetTextureObjectResourceViewDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetTextureObjectResourceViewDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaResourceViewDesc *pResViewDesc;
    conn->read(&pResViewDesc, sizeof(pResViewDesc));
    cudaTextureObject_t texObject;
    conn->read(&texObject, sizeof(texObject));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject);
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

int handle_cudaCreateSurfaceObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaCreateSurfaceObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaSurfaceObject_t *pSurfObject;
    conn->read(&pSurfObject, sizeof(pSurfObject));
    struct cudaResourceDesc *pResDesc = nullptr;
    conn->read(&pResDesc, sizeof(pResDesc));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaCreateSurfaceObject(pSurfObject, pResDesc);
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

int handle_cudaDestroySurfaceObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDestroySurfaceObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaSurfaceObject_t surfObject;
    conn->read(&surfObject, sizeof(surfObject));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDestroySurfaceObject(surfObject);
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

int handle_cudaGetSurfaceObjectResourceDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetSurfaceObjectResourceDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    struct cudaResourceDesc *pResDesc;
    conn->read(&pResDesc, sizeof(pResDesc));
    cudaSurfaceObject_t surfObject;
    conn->read(&surfObject, sizeof(surfObject));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject);
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

int handle_cudaDriverGetVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDriverGetVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *driverVersion;
    conn->read(&driverVersion, sizeof(driverVersion));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDriverGetVersion(driverVersion);
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

int handle_cudaRuntimeGetVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaRuntimeGetVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *runtimeVersion;
    conn->read(&runtimeVersion, sizeof(runtimeVersion));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaRuntimeGetVersion(runtimeVersion);
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

int handle_cudaGraphCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraph_t *pGraph;
    conn->read(&pGraph, sizeof(pGraph));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphCreate(pGraph, flags);
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

int handle_cudaGraphAddKernelNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddKernelNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    struct cudaKernelNodeParams *pNodeParams = nullptr;
    conn->read(&pNodeParams, sizeof(pNodeParams));
    void **args = nullptr;
    int arg_count;
    conn->read(&arg_count, sizeof(arg_count));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    args = (void **)conn->get_host_buffer(sizeof(void *) * arg_count);
    if(args == nullptr) {
        std::cerr << "Failed to allocate args" << std::endl;
        return 1;
    }
    if(conn->read_all_now(args, nullptr, arg_count) != RpcError::OK) {
        std::cerr << "Failed to read args" << std::endl;
        conn->free_host_buffer(args);
        return 1;
    }
    pNodeParams->kernelParams = args;
    _result = cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
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

int handle_cudaGraphKernelNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphKernelNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    struct cudaKernelNodeParams *pNodeParams;
    conn->read(&pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphKernelNodeGetParams(node, pNodeParams);
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

int handle_cudaGraphKernelNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphKernelNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    struct cudaKernelNodeParams *pNodeParams = nullptr;
    conn->read(&pNodeParams, sizeof(pNodeParams));
    void **args = nullptr;
    int arg_count;
    conn->read(&arg_count, sizeof(arg_count));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    args = (void **)conn->get_host_buffer(sizeof(void *) * arg_count);
    if(args == nullptr) {
        std::cerr << "Failed to allocate args" << std::endl;
        return 1;
    }
    if(conn->read_all_now(args, nullptr, arg_count) != RpcError::OK) {
        std::cerr << "Failed to read args" << std::endl;
        conn->free_host_buffer(args);
        return 1;
    }
    pNodeParams->kernelParams = args;
    _result = cudaGraphKernelNodeSetParams(node, pNodeParams);
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

int handle_cudaGraphKernelNodeCopyAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphKernelNodeCopyAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t hSrc;
    conn->read(&hSrc, sizeof(hSrc));
    cudaGraphNode_t hDst;
    conn->read(&hDst, sizeof(hDst));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphKernelNodeCopyAttributes(hSrc, hDst);
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

int handle_cudaGraphKernelNodeGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphKernelNodeGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t hNode;
    conn->read(&hNode, sizeof(hNode));
    enum cudaKernelNodeAttrID attr;
    conn->read(&attr, sizeof(attr));
    union cudaKernelNodeAttrValue *value_out;
    conn->read(&value_out, sizeof(value_out));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphKernelNodeGetAttribute(hNode, attr, value_out);
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

int handle_cudaGraphKernelNodeSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphKernelNodeSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t hNode;
    conn->read(&hNode, sizeof(hNode));
    enum cudaKernelNodeAttrID attr;
    conn->read(&attr, sizeof(attr));
    union cudaKernelNodeAttrValue *value = nullptr;
    conn->read(&value, sizeof(value));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphKernelNodeSetAttribute(hNode, attr, value);
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

int handle_cudaGraphAddMemcpyNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemcpyNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    struct cudaMemcpy3DParms *pCopyParams = nullptr;
    conn->read(&pCopyParams, sizeof(pCopyParams));
    void *_0sptr;
    conn->read(&_0sptr, sizeof(_0sptr));
    void *_0dptr;
    conn->read(&_0dptr, sizeof(_0dptr));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    if(pCopyParams != nullptr) {
        pCopyParams->srcPtr.ptr = _0sptr;
        pCopyParams->dstPtr.ptr = _0dptr;
    }
    _result = cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams);
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

int handle_cudaGraphAddMemcpyNodeToSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemcpyNodeToSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    void *src;
    conn->read(&src, sizeof(src));
    size_t count;
    conn->read(&count, sizeof(count));
    size_t offset;
    conn->read(&offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind);
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

int handle_cudaGraphAddMemcpyNodeFromSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemcpyNodeFromSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    void *dst;
    conn->read(&dst, sizeof(dst));
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    size_t count;
    conn->read(&count, sizeof(count));
    size_t offset;
    conn->read(&offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind);
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

int handle_cudaGraphAddMemcpyNode1D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemcpyNode1D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    void *dst;
    conn->read(&dst, sizeof(dst));
    void *src;
    conn->read(&src, sizeof(src));
    size_t count;
    conn->read(&count, sizeof(count));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind);
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

int handle_cudaGraphMemcpyNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemcpyNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    struct cudaMemcpy3DParms *pNodeParams = nullptr;
    conn->read(&pNodeParams, sizeof(pNodeParams));
    void *_0sptr;
    conn->read(&_0sptr, sizeof(_0sptr));
    void *_0dptr;
    conn->read(&_0dptr, sizeof(_0dptr));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    if(pNodeParams != nullptr) {
        pNodeParams->srcPtr.ptr = _0sptr;
        pNodeParams->dstPtr.ptr = _0dptr;
    }
    _result = cudaGraphMemcpyNodeSetParams(node, pNodeParams);
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

int handle_cudaGraphMemcpyNodeSetParamsToSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemcpyNodeSetParamsToSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    void *src;
    conn->read(&src, sizeof(src));
    size_t count;
    conn->read(&count, sizeof(count));
    size_t offset;
    conn->read(&offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemcpyNodeSetParamsToSymbol(node, symbol, src, count, offset, kind);
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

int handle_cudaGraphMemcpyNodeSetParamsFromSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemcpyNodeSetParamsFromSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    void *dst;
    conn->read(&dst, sizeof(dst));
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    size_t count;
    conn->read(&count, sizeof(count));
    size_t offset;
    conn->read(&offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, symbol, count, offset, kind);
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

int handle_cudaGraphMemcpyNodeSetParams1D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemcpyNodeSetParams1D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    void *dst;
    conn->read(&dst, sizeof(dst));
    void *src;
    conn->read(&src, sizeof(src));
    size_t count;
    conn->read(&count, sizeof(count));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind);
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

int handle_cudaGraphAddMemsetNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemsetNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    struct cudaMemsetParams *pMemsetParams = nullptr;
    conn->read(&pMemsetParams, sizeof(pMemsetParams));
    void *_0dst;
    conn->read(&_0dst, sizeof(_0dst));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    if(pMemsetParams != nullptr) {
        pMemsetParams->dst = _0dst;
    }
    _result = cudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams);
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

int handle_cudaGraphMemsetNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemsetNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    struct cudaMemsetParams *pNodeParams;
    conn->read(&pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemsetNodeGetParams(node, pNodeParams);
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

int handle_cudaGraphMemsetNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemsetNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    struct cudaMemsetParams *pNodeParams = nullptr;
    conn->read(&pNodeParams, sizeof(pNodeParams));
    void *_0dst;
    conn->read(&_0dst, sizeof(_0dst));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    if(pNodeParams != nullptr) {
        pNodeParams->dst = _0dst;
    }
    _result = cudaGraphMemsetNodeSetParams(node, pNodeParams);
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

int handle_cudaGraphAddHostNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddHostNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    struct cudaHostNodeParams *pNodeParams = nullptr;
    conn->read(&pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
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

int handle_cudaGraphHostNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphHostNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    struct cudaHostNodeParams *pNodeParams;
    conn->read(&pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphHostNodeGetParams(node, pNodeParams);
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

int handle_cudaGraphHostNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphHostNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    struct cudaHostNodeParams *pNodeParams = nullptr;
    conn->read(&pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphHostNodeSetParams(node, pNodeParams);
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

int handle_cudaGraphAddChildGraphNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddChildGraphNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    cudaGraph_t childGraph;
    conn->read(&childGraph, sizeof(childGraph));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph);
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

int handle_cudaGraphChildGraphNodeGetGraph(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphChildGraphNodeGetGraph called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    cudaGraph_t *pGraph;
    conn->read(&pGraph, sizeof(pGraph));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphChildGraphNodeGetGraph(node, pGraph);
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

int handle_cudaGraphAddEmptyNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddEmptyNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies);
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

int handle_cudaGraphAddEventRecordNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddEventRecordNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    cudaEvent_t event;
    conn->read(&event, sizeof(event));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event);
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

int handle_cudaGraphEventRecordNodeGetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphEventRecordNodeGetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    cudaEvent_t *event_out;
    conn->read(&event_out, sizeof(event_out));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphEventRecordNodeGetEvent(node, event_out);
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

int handle_cudaGraphEventRecordNodeSetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphEventRecordNodeSetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    cudaEvent_t event;
    conn->read(&event, sizeof(event));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphEventRecordNodeSetEvent(node, event);
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

int handle_cudaGraphAddEventWaitNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddEventWaitNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    cudaEvent_t event;
    conn->read(&event, sizeof(event));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event);
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

int handle_cudaGraphEventWaitNodeGetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphEventWaitNodeGetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    cudaEvent_t *event_out;
    conn->read(&event_out, sizeof(event_out));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphEventWaitNodeGetEvent(node, event_out);
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

int handle_cudaGraphEventWaitNodeSetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphEventWaitNodeSetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    cudaEvent_t event;
    conn->read(&event, sizeof(event));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphEventWaitNodeSetEvent(node, event);
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

int handle_cudaGraphAddExternalSemaphoresSignalNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddExternalSemaphoresSignalNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    struct cudaExternalSemaphoreSignalNodeParams *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
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

int handle_cudaGraphExternalSemaphoresSignalNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExternalSemaphoresSignalNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t hNode;
    conn->read(&hNode, sizeof(hNode));
    struct cudaExternalSemaphoreSignalNodeParams *params_out;
    conn->read(&params_out, sizeof(params_out));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out);
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

int handle_cudaGraphExternalSemaphoresSignalNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExternalSemaphoresSignalNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t hNode;
    conn->read(&hNode, sizeof(hNode));
    struct cudaExternalSemaphoreSignalNodeParams *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams);
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

int handle_cudaGraphAddExternalSemaphoresWaitNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddExternalSemaphoresWaitNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    struct cudaExternalSemaphoreWaitNodeParams *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
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

int handle_cudaGraphExternalSemaphoresWaitNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExternalSemaphoresWaitNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t hNode;
    conn->read(&hNode, sizeof(hNode));
    struct cudaExternalSemaphoreWaitNodeParams *params_out;
    conn->read(&params_out, sizeof(params_out));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out);
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

int handle_cudaGraphExternalSemaphoresWaitNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExternalSemaphoresWaitNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t hNode;
    conn->read(&hNode, sizeof(hNode));
    struct cudaExternalSemaphoreWaitNodeParams *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams);
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

int handle_cudaGraphAddMemAllocNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemAllocNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    struct cudaMemAllocNodeParams *nodeParams;
    conn->read(&nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
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

int handle_cudaGraphMemAllocNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemAllocNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    struct cudaMemAllocNodeParams *params_out;
    conn->read(&params_out, sizeof(params_out));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemAllocNodeGetParams(node, params_out);
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

int handle_cudaGraphAddMemFreeNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddMemFreeNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pGraphNode;
    conn->read(&pGraphNode, sizeof(pGraphNode));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pDependencies = nullptr;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    void *dptr;
    conn->read(&dptr, sizeof(dptr));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dptr);
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

int handle_cudaGraphMemFreeNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphMemFreeNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    void *dptr_out;
    conn->read(&dptr_out, sizeof(dptr_out));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphMemFreeNodeGetParams(node, dptr_out);
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

int handle_cudaDeviceGraphMemTrim(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGraphMemTrim called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int device;
    conn->read(&device, sizeof(device));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGraphMemTrim(device);
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

int handle_cudaDeviceGetGraphMemAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceGetGraphMemAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int device;
    conn->read(&device, sizeof(device));
    enum cudaGraphMemAttributeType attr;
    conn->read(&attr, sizeof(attr));
    void *value;
    conn->read(&value, sizeof(value));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceGetGraphMemAttribute(device, attr, value);
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

int handle_cudaDeviceSetGraphMemAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaDeviceSetGraphMemAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int device;
    conn->read(&device, sizeof(device));
    enum cudaGraphMemAttributeType attr;
    conn->read(&attr, sizeof(attr));
    void *value;
    conn->read(&value, sizeof(value));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaDeviceSetGraphMemAttribute(device, attr, value);
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

int handle_cudaGraphClone(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphClone called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraph_t *pGraphClone;
    conn->read(&pGraphClone, sizeof(pGraphClone));
    cudaGraph_t originalGraph;
    conn->read(&originalGraph, sizeof(originalGraph));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphClone(pGraphClone, originalGraph);
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

int handle_cudaGraphNodeFindInClone(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphNodeFindInClone called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t *pNode;
    conn->read(&pNode, sizeof(pNode));
    cudaGraphNode_t originalNode;
    conn->read(&originalNode, sizeof(originalNode));
    cudaGraph_t clonedGraph;
    conn->read(&clonedGraph, sizeof(clonedGraph));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphNodeFindInClone(pNode, originalNode, clonedGraph);
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

int handle_cudaGraphNodeGetType(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphNodeGetType called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    enum cudaGraphNodeType *pType;
    conn->read(&pType, sizeof(pType));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphNodeGetType(node, pType);
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

int handle_cudaGraphGetNodes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphGetNodes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *nodes;
    conn->read(&nodes, sizeof(nodes));
    size_t *numNodes;
    conn->read(&numNodes, sizeof(numNodes));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphGetNodes(graph, nodes, numNodes);
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

int handle_cudaGraphGetRootNodes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphGetRootNodes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pRootNodes;
    conn->read(&pRootNodes, sizeof(pRootNodes));
    size_t *pNumRootNodes;
    conn->read(&pNumRootNodes, sizeof(pNumRootNodes));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes);
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

int handle_cudaGraphGetEdges(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphGetEdges called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *from;
    conn->read(&from, sizeof(from));
    cudaGraphNode_t *to;
    conn->read(&to, sizeof(to));
    size_t *numEdges;
    conn->read(&numEdges, sizeof(numEdges));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphGetEdges(graph, from, to, numEdges);
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

int handle_cudaGraphNodeGetDependencies(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphNodeGetDependencies called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    cudaGraphNode_t *pDependencies;
    conn->read(&pDependencies, sizeof(pDependencies));
    size_t *pNumDependencies;
    conn->read(&pNumDependencies, sizeof(pNumDependencies));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphNodeGetDependencies(node, pDependencies, pNumDependencies);
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

int handle_cudaGraphNodeGetDependentNodes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphNodeGetDependentNodes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    cudaGraphNode_t *pDependentNodes;
    conn->read(&pDependentNodes, sizeof(pDependentNodes));
    size_t *pNumDependentNodes;
    conn->read(&pNumDependentNodes, sizeof(pNumDependentNodes));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphNodeGetDependentNodes(node, pDependentNodes, pNumDependentNodes);
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

int handle_cudaGraphAddDependencies(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphAddDependencies called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *from = nullptr;
    conn->read(&from, sizeof(from));
    cudaGraphNode_t *to = nullptr;
    conn->read(&to, sizeof(to));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphAddDependencies(graph, from, to, numDependencies);
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

int handle_cudaGraphRemoveDependencies(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphRemoveDependencies called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *from = nullptr;
    conn->read(&from, sizeof(from));
    cudaGraphNode_t *to = nullptr;
    conn->read(&to, sizeof(to));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphRemoveDependencies(graph, from, to, numDependencies);
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

int handle_cudaGraphDestroyNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphDestroyNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphDestroyNode(node);
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

int handle_cudaGraphInstantiate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphInstantiate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t *pGraphExec;
    conn->read(&pGraphExec, sizeof(pGraphExec));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaGraphNode_t *pErrorNode;
    conn->read(&pErrorNode, sizeof(pErrorNode));
    char pLogBuffer[1024];
    size_t bufferSize;
    conn->read(&bufferSize, sizeof(bufferSize));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize);
    if(bufferSize > 0) {
        conn->write(pLogBuffer, strlen(pLogBuffer) + 1, true);
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

int handle_cudaGraphInstantiateWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphInstantiateWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t *pGraphExec;
    conn->read(&pGraphExec, sizeof(pGraphExec));
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    unsigned long long flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphInstantiateWithFlags(pGraphExec, graph, flags);
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

int handle_cudaGraphExecKernelNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecKernelNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    struct cudaKernelNodeParams *pNodeParams = nullptr;
    conn->read(&pNodeParams, sizeof(pNodeParams));
    void **args = nullptr;
    int arg_count;
    conn->read(&arg_count, sizeof(arg_count));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    args = (void **)conn->get_host_buffer(sizeof(void *) * arg_count);
    if(args == nullptr) {
        std::cerr << "Failed to allocate args" << std::endl;
        return 1;
    }
    if(conn->read_all_now(args, nullptr, arg_count) != RpcError::OK) {
        std::cerr << "Failed to read args" << std::endl;
        conn->free_host_buffer(args);
        return 1;
    }
    pNodeParams->kernelParams = args;
    _result = cudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams);
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

int handle_cudaGraphExecMemcpyNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecMemcpyNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    struct cudaMemcpy3DParms *pNodeParams = nullptr;
    conn->read(&pNodeParams, sizeof(pNodeParams));
    void *_0sptr;
    conn->read(&_0sptr, sizeof(_0sptr));
    void *_0dptr;
    conn->read(&_0dptr, sizeof(_0dptr));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    if(pNodeParams != nullptr) {
        pNodeParams->srcPtr.ptr = _0sptr;
        pNodeParams->dstPtr.ptr = _0dptr;
    }
    _result = cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams);
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

int handle_cudaGraphExecMemcpyNodeSetParamsToSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecMemcpyNodeSetParamsToSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    void *src;
    conn->read(&src, sizeof(src));
    size_t count;
    conn->read(&count, sizeof(count));
    size_t offset;
    conn->read(&offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, symbol, src, count, offset, kind);
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

int handle_cudaGraphExecMemcpyNodeSetParamsFromSymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecMemcpyNodeSetParamsFromSymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    void *dst;
    conn->read(&dst, sizeof(dst));
    void *symbol;
    conn->read(&symbol, sizeof(symbol));
    size_t count;
    conn->read(&count, sizeof(count));
    size_t offset;
    conn->read(&offset, sizeof(offset));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, symbol, count, offset, kind);
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

int handle_cudaGraphExecMemcpyNodeSetParams1D(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecMemcpyNodeSetParams1D called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    void *dst;
    conn->read(&dst, sizeof(dst));
    void *src;
    conn->read(&src, sizeof(src));
    size_t count;
    conn->read(&count, sizeof(count));
    enum cudaMemcpyKind kind;
    conn->read(&kind, sizeof(kind));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind);
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

int handle_cudaGraphExecMemsetNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecMemsetNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    struct cudaMemsetParams *pNodeParams = nullptr;
    conn->read(&pNodeParams, sizeof(pNodeParams));
    void *_0dst;
    conn->read(&_0dst, sizeof(_0dst));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    if(pNodeParams != nullptr) {
        pNodeParams->dst = _0dst;
    }
    _result = cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams);
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

int handle_cudaGraphExecHostNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecHostNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    struct cudaHostNodeParams *pNodeParams = nullptr;
    conn->read(&pNodeParams, sizeof(pNodeParams));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams);
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

int handle_cudaGraphExecChildGraphNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecChildGraphNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t node;
    conn->read(&node, sizeof(node));
    cudaGraph_t childGraph;
    conn->read(&childGraph, sizeof(childGraph));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph);
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

int handle_cudaGraphExecEventRecordNodeSetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecEventRecordNodeSetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t hNode;
    conn->read(&hNode, sizeof(hNode));
    cudaEvent_t event;
    conn->read(&event, sizeof(event));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);
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

int handle_cudaGraphExecEventWaitNodeSetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecEventWaitNodeSetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t hNode;
    conn->read(&hNode, sizeof(hNode));
    cudaEvent_t event;
    conn->read(&event, sizeof(event));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);
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

int handle_cudaGraphExecExternalSemaphoresSignalNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecExternalSemaphoresSignalNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t hNode;
    conn->read(&hNode, sizeof(hNode));
    struct cudaExternalSemaphoreSignalNodeParams *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams);
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

int handle_cudaGraphExecExternalSemaphoresWaitNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecExternalSemaphoresWaitNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    cudaGraphNode_t hNode;
    conn->read(&hNode, sizeof(hNode));
    struct cudaExternalSemaphoreWaitNodeParams *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams);
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

int handle_cudaGraphExecUpdate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecUpdate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    cudaGraph_t hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    cudaGraphNode_t *hErrorNode_out;
    conn->read(&hErrorNode_out, sizeof(hErrorNode_out));
    enum cudaGraphExecUpdateResult *updateResult_out;
    conn->read(&updateResult_out, sizeof(updateResult_out));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out);
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

int handle_cudaGraphUpload(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphUpload called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t graphExec;
    conn->read(&graphExec, sizeof(graphExec));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphUpload(graphExec, stream);
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

int handle_cudaGraphLaunch(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphLaunch called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t graphExec;
    conn->read(&graphExec, sizeof(graphExec));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphLaunch(graphExec, stream);
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

int handle_cudaGraphExecDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphExecDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraphExec_t graphExec;
    conn->read(&graphExec, sizeof(graphExec));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphExecDestroy(graphExec);
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

int handle_cudaGraphDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphDestroy(graph);
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

int handle_cudaGraphDebugDotPrint(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphDebugDotPrint called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    char *path = nullptr;
    conn->read(&path, 0, true);
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(path);
    _result = cudaGraphDebugDotPrint(graph, path, flags);
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

int handle_cudaUserObjectCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaUserObjectCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaUserObject_t *object_out;
    conn->read(&object_out, sizeof(object_out));
    void *ptr;
    conn->read(&ptr, sizeof(ptr));
    cudaHostFn_t destroy;
    conn->read(&destroy, sizeof(destroy));
    unsigned int initialRefcount;
    conn->read(&initialRefcount, sizeof(initialRefcount));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags);
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

int handle_cudaUserObjectRetain(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaUserObjectRetain called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaUserObject_t object;
    conn->read(&object, sizeof(object));
    unsigned int count;
    conn->read(&count, sizeof(count));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaUserObjectRetain(object, count);
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

int handle_cudaUserObjectRelease(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaUserObjectRelease called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaUserObject_t object;
    conn->read(&object, sizeof(object));
    unsigned int count;
    conn->read(&count, sizeof(count));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaUserObjectRelease(object, count);
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

int handle_cudaGraphRetainUserObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphRetainUserObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaUserObject_t object;
    conn->read(&object, sizeof(object));
    unsigned int count;
    conn->read(&count, sizeof(count));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphRetainUserObject(graph, object, count, flags);
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

int handle_cudaGraphReleaseUserObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGraphReleaseUserObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaGraph_t graph;
    conn->read(&graph, sizeof(graph));
    cudaUserObject_t object;
    conn->read(&object, sizeof(object));
    unsigned int count;
    conn->read(&count, sizeof(count));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGraphReleaseUserObject(graph, object, count);
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

int handle_cudaGetDriverEntryPoint(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetDriverEntryPoint called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    char *symbol = nullptr;
    conn->read(&symbol, 0, true);
    void *funcPtr;
    unsigned long long flags;
    conn->read(&flags, sizeof(flags));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(symbol);
    _result = cudaGetDriverEntryPoint(symbol, &funcPtr, flags);
    conn->write(&funcPtr, sizeof(funcPtr));
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

int handle_cudaGetExportTable(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetExportTable called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    const void *ppExportTable;
    cudaUUID_t *pExportTableId = nullptr;
    conn->read(&pExportTableId, sizeof(pExportTableId));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetExportTable(&ppExportTable, pExportTableId);
    conn->write(&ppExportTable, sizeof(ppExportTable));
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

int handle_cudaGetFuncBySymbol(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cudaGetFuncBySymbol called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cudaFunction_t *functionPtr;
    conn->read(&functionPtr, sizeof(functionPtr));
    void *symbolPtr;
    conn->read(&symbolPtr, sizeof(symbolPtr));
    cudaError_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cudaGetFuncBySymbol(functionPtr, symbolPtr);
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
