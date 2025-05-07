#include <iostream>
#include <map>
#include "cuda_runtime_api.h"

#include "hook_api.h"
#include "client.h"
extern "C" cudaError_t cudaDeviceReset() {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceReset called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceReset);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceSynchronize() {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceSynchronize called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceSynchronize);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceSetLimit called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceSetLimit);
    conn->write(&limit, sizeof(limit));
    conn->write(&value, sizeof(value));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetLimit called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pValue;
    mem2server(conn, &_0pValue, (void *)pValue, sizeof(*pValue));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGetLimit);
    conn->write(&_0pValue, sizeof(_0pValue));
    updateTmpPtr((void *)pValue, _0pValue);
    conn->write(&limit, sizeof(limit));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pValue, sizeof(*pValue), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, const struct cudaChannelFormatDesc *fmtDesc, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetTexture1DLinearMaxWidth called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0maxWidthInElements;
    mem2server(conn, &_0maxWidthInElements, (void *)maxWidthInElements, sizeof(*maxWidthInElements));
    void *_0fmtDesc;
    mem2server(conn, &_0fmtDesc, (void *)fmtDesc, sizeof(*fmtDesc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGetTexture1DLinearMaxWidth);
    conn->write(&_0maxWidthInElements, sizeof(_0maxWidthInElements));
    updateTmpPtr((void *)maxWidthInElements, _0maxWidthInElements);
    conn->write(&_0fmtDesc, sizeof(_0fmtDesc));
    updateTmpPtr((void *)fmtDesc, _0fmtDesc);
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)maxWidthInElements, sizeof(*maxWidthInElements), true);
    mem2client(conn, (void *)fmtDesc, sizeof(*fmtDesc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetCacheConfig called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pCacheConfig;
    mem2server(conn, &_0pCacheConfig, (void *)pCacheConfig, sizeof(*pCacheConfig));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGetCacheConfig);
    conn->write(&_0pCacheConfig, sizeof(_0pCacheConfig));
    updateTmpPtr((void *)pCacheConfig, _0pCacheConfig);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pCacheConfig, sizeof(*pCacheConfig), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetStreamPriorityRange called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0leastPriority;
    mem2server(conn, &_0leastPriority, (void *)leastPriority, sizeof(*leastPriority));
    void *_0greatestPriority;
    mem2server(conn, &_0greatestPriority, (void *)greatestPriority, sizeof(*greatestPriority));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGetStreamPriorityRange);
    conn->write(&_0leastPriority, sizeof(_0leastPriority));
    updateTmpPtr((void *)leastPriority, _0leastPriority);
    conn->write(&_0greatestPriority, sizeof(_0greatestPriority));
    updateTmpPtr((void *)greatestPriority, _0greatestPriority);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)leastPriority, sizeof(*leastPriority), true);
    mem2client(conn, (void *)greatestPriority, sizeof(*greatestPriority), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceSetCacheConfig called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceSetCacheConfig);
    conn->write(&cacheConfig, sizeof(cacheConfig));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetSharedMemConfig called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pConfig;
    mem2server(conn, &_0pConfig, (void *)pConfig, sizeof(*pConfig));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGetSharedMemConfig);
    conn->write(&_0pConfig, sizeof(_0pConfig));
    updateTmpPtr((void *)pConfig, _0pConfig);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pConfig, sizeof(*pConfig), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceSetSharedMemConfig called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceSetSharedMemConfig);
    conn->write(&config, sizeof(config));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetByPCIBusId called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0device;
    mem2server(conn, &_0device, (void *)device, sizeof(*device));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGetByPCIBusId);
    conn->write(&_0device, sizeof(_0device));
    updateTmpPtr((void *)device, _0device);
    conn->write(pciBusId, strlen(pciBusId) + 1, true);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)device, sizeof(*device), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetPCIBusId called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGetPCIBusId);
    if(len > 0) {
        conn->read(pciBusId, len, true);
    }
    conn->write(&len, sizeof(len));
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaIpcGetEventHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0handle;
    mem2server(conn, &_0handle, (void *)handle, sizeof(*handle));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaIpcGetEventHandle);
    conn->write(&_0handle, sizeof(_0handle));
    updateTmpPtr((void *)handle, _0handle);
    conn->write(&event, sizeof(event));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)handle, sizeof(*handle), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle) {
#ifdef DEBUG
    std::cout << "Hook: cudaIpcOpenEventHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0event;
    mem2server(conn, &_0event, (void *)event, sizeof(*event));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaIpcOpenEventHandle);
    conn->write(&_0event, sizeof(_0event));
    updateTmpPtr((void *)event, _0event);
    conn->write(&handle, sizeof(handle));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)event, sizeof(*event), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr) {
#ifdef DEBUG
    std::cout << "Hook: cudaIpcGetMemHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0handle;
    mem2server(conn, &_0handle, (void *)handle, sizeof(*handle));
    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaIpcGetMemHandle);
    conn->write(&_0handle, sizeof(_0handle));
    updateTmpPtr((void *)handle, _0handle);
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)handle, sizeof(*handle), true);
    mem2client(conn, (void *)devPtr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaIpcOpenMemHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaIpcOpenMemHandle);
    conn->read(devPtr, sizeof(void *));
    conn->write(&handle, sizeof(handle));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaIpcCloseMemHandle(void *devPtr) {
#ifdef DEBUG
    std::cout << "Hook: cudaIpcCloseMemHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaIpcCloseMemHandle);
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devPtr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(enum cudaFlushGPUDirectRDMAWritesTarget target, enum cudaFlushGPUDirectRDMAWritesScope scope) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceFlushGPUDirectRDMAWrites called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceFlushGPUDirectRDMAWrites);
    conn->write(&target, sizeof(target));
    conn->write(&scope, sizeof(scope));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaThreadExit() {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadExit called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaThreadExit);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaThreadSynchronize() {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadSynchronize called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaThreadSynchronize);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value) {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadSetLimit called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaThreadSetLimit);
    conn->write(&limit, sizeof(limit));
    conn->write(&value, sizeof(value));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit) {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadGetLimit called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pValue;
    mem2server(conn, &_0pValue, (void *)pValue, sizeof(*pValue));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaThreadGetLimit);
    conn->write(&_0pValue, sizeof(_0pValue));
    updateTmpPtr((void *)pValue, _0pValue);
    conn->write(&limit, sizeof(limit));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pValue, sizeof(*pValue), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadGetCacheConfig called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pCacheConfig;
    mem2server(conn, &_0pCacheConfig, (void *)pCacheConfig, sizeof(*pCacheConfig));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaThreadGetCacheConfig);
    conn->write(&_0pCacheConfig, sizeof(_0pCacheConfig));
    updateTmpPtr((void *)pCacheConfig, _0pCacheConfig);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pCacheConfig, sizeof(*pCacheConfig), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig) {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadSetCacheConfig called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaThreadSetCacheConfig);
    conn->write(&cacheConfig, sizeof(cacheConfig));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetLastError() {
#ifdef DEBUG
    std::cout << "Hook: cudaGetLastError called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetLastError);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaPeekAtLastError() {
#ifdef DEBUG
    std::cout << "Hook: cudaPeekAtLastError called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaPeekAtLastError);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetDeviceCount(int *count) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetDeviceCount called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0count;
    mem2server(conn, &_0count, (void *)count, sizeof(*count));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetDeviceCount);
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)count, sizeof(*count), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetDeviceProperties called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0prop;
    mem2server(conn, &_0prop, (void *)prop, sizeof(*prop));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetDeviceProperties);
    conn->write(&_0prop, sizeof(_0prop));
    updateTmpPtr((void *)prop, _0prop);
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)prop, sizeof(*prop), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0value;
    mem2server(conn, &_0value, (void *)value, sizeof(*value));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGetAttribute);
    conn->write(&_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    conn->write(&attr, sizeof(attr));
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)value, sizeof(*value), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetDefaultMemPool called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0memPool;
    mem2server(conn, &_0memPool, (void *)memPool, sizeof(*memPool));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGetDefaultMemPool);
    conn->write(&_0memPool, sizeof(_0memPool));
    updateTmpPtr((void *)memPool, _0memPool);
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)memPool, sizeof(*memPool), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceSetMemPool called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceSetMemPool);
    conn->write(&device, sizeof(device));
    conn->write(&memPool, sizeof(memPool));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetMemPool called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0memPool;
    mem2server(conn, &_0memPool, (void *)memPool, sizeof(*memPool));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGetMemPool);
    conn->write(&_0memPool, sizeof(_0memPool));
    updateTmpPtr((void *)memPool, _0memPool);
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)memPool, sizeof(*memPool), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, int device, int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetNvSciSyncAttributes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0nvSciSyncAttrList;
    mem2server(conn, &_0nvSciSyncAttrList, (void *)nvSciSyncAttrList, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGetNvSciSyncAttributes);
    conn->write(&_0nvSciSyncAttrList, sizeof(_0nvSciSyncAttrList));
    updateTmpPtr((void *)nvSciSyncAttrList, _0nvSciSyncAttrList);
    conn->write(&device, sizeof(device));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)nvSciSyncAttrList, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetP2PAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0value;
    mem2server(conn, &_0value, (void *)value, sizeof(*value));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGetP2PAttribute);
    conn->write(&_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    conn->write(&attr, sizeof(attr));
    conn->write(&srcDevice, sizeof(srcDevice));
    conn->write(&dstDevice, sizeof(dstDevice));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)value, sizeof(*value), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
#ifdef DEBUG
    std::cout << "Hook: cudaChooseDevice called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0device;
    mem2server(conn, &_0device, (void *)device, sizeof(*device));
    void *_0prop;
    mem2server(conn, &_0prop, (void *)prop, sizeof(*prop));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaChooseDevice);
    conn->write(&_0device, sizeof(_0device));
    updateTmpPtr((void *)device, _0device);
    conn->write(&_0prop, sizeof(_0prop));
    updateTmpPtr((void *)prop, _0prop);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)device, sizeof(*device), true);
    mem2client(conn, (void *)prop, sizeof(*prop), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaSetDevice(int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaSetDevice called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaSetDevice);
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetDevice(int *device) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetDevice called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0device;
    mem2server(conn, &_0device, (void *)device, sizeof(*device));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetDevice);
    conn->write(&_0device, sizeof(_0device));
    updateTmpPtr((void *)device, _0device);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)device, sizeof(*device), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaSetValidDevices(int *device_arr, int len) {
#ifdef DEBUG
    std::cout << "Hook: cudaSetValidDevices called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0device_arr;
    mem2server(conn, &_0device_arr, (void *)device_arr, sizeof(*device_arr));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaSetValidDevices);
    conn->write(&_0device_arr, sizeof(_0device_arr));
    updateTmpPtr((void *)device_arr, _0device_arr);
    conn->write(&len, sizeof(len));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)device_arr, sizeof(*device_arr), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaSetDeviceFlags(unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaSetDeviceFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaSetDeviceFlags);
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetDeviceFlags(unsigned int *flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetDeviceFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0flags;
    mem2server(conn, &_0flags, (void *)flags, sizeof(*flags));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetDeviceFlags);
    conn->write(&_0flags, sizeof(_0flags));
    updateTmpPtr((void *)flags, _0flags);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)flags, sizeof(*flags), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamCreate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pStream;
    mem2server(conn, &_0pStream, (void *)pStream, sizeof(*pStream));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamCreate);
    conn->write(&_0pStream, sizeof(_0pStream));
    updateTmpPtr((void *)pStream, _0pStream);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pStream, sizeof(*pStream), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamCreateWithFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pStream;
    mem2server(conn, &_0pStream, (void *)pStream, sizeof(*pStream));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamCreateWithFlags);
    conn->write(&_0pStream, sizeof(_0pStream));
    updateTmpPtr((void *)pStream, _0pStream);
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pStream, sizeof(*pStream), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamCreateWithPriority called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pStream;
    mem2server(conn, &_0pStream, (void *)pStream, sizeof(*pStream));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamCreateWithPriority);
    conn->write(&_0pStream, sizeof(_0pStream));
    updateTmpPtr((void *)pStream, _0pStream);
    conn->write(&flags, sizeof(flags));
    conn->write(&priority, sizeof(priority));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pStream, sizeof(*pStream), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamGetPriority called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0priority;
    mem2server(conn, &_0priority, (void *)priority, sizeof(*priority));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamGetPriority);
    conn->write(&hStream, sizeof(hStream));
    conn->write(&_0priority, sizeof(_0priority));
    updateTmpPtr((void *)priority, _0priority);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)priority, sizeof(*priority), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamGetFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0flags;
    mem2server(conn, &_0flags, (void *)flags, sizeof(*flags));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamGetFlags);
    conn->write(&hStream, sizeof(hStream));
    conn->write(&_0flags, sizeof(_0flags));
    updateTmpPtr((void *)flags, _0flags);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)flags, sizeof(*flags), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaCtxResetPersistingL2Cache() {
#ifdef DEBUG
    std::cout << "Hook: cudaCtxResetPersistingL2Cache called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaCtxResetPersistingL2Cache);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamCopyAttributes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamCopyAttributes);
    conn->write(&dst, sizeof(dst));
    conn->write(&src, sizeof(src));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, enum cudaStreamAttrID attr, union cudaStreamAttrValue *value_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamGetAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0value_out;
    mem2server(conn, &_0value_out, (void *)value_out, sizeof(*value_out));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamGetAttribute);
    conn->write(&hStream, sizeof(hStream));
    conn->write(&attr, sizeof(attr));
    conn->write(&_0value_out, sizeof(_0value_out));
    updateTmpPtr((void *)value_out, _0value_out);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)value_out, sizeof(*value_out), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, enum cudaStreamAttrID attr, const union cudaStreamAttrValue *value) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamSetAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0value;
    mem2server(conn, &_0value, (void *)value, sizeof(*value));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamSetAttribute);
    conn->write(&hStream, sizeof(hStream));
    conn->write(&attr, sizeof(attr));
    conn->write(&_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)value, sizeof(*value), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamDestroy called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamDestroy);
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamWaitEvent called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamWaitEvent);
    conn->write(&stream, sizeof(stream));
    conn->write(&event, sizeof(event));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamAddCallback called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0userData;
    mem2server(conn, &_0userData, (void *)userData, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamAddCallback);
    conn->write(&stream, sizeof(stream));
    conn->write(&callback, sizeof(callback));
    conn->write(&_0userData, sizeof(_0userData));
    updateTmpPtr((void *)userData, _0userData);
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)userData, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamSynchronize called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamSynchronize);
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamQuery called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamQuery);
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamAttachMemAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, length);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamAttachMemAsync);
    conn->write(&stream, sizeof(stream));
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->write(&length, sizeof(length));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devPtr, length, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamBeginCapture called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamBeginCapture);
    conn->write(&stream, sizeof(stream));
    conn->write(&mode, sizeof(mode));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode) {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadExchangeStreamCaptureMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0mode;
    mem2server(conn, &_0mode, (void *)mode, sizeof(*mode));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaThreadExchangeStreamCaptureMode);
    conn->write(&_0mode, sizeof(_0mode));
    updateTmpPtr((void *)mode, _0mode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)mode, sizeof(*mode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamEndCapture called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraph;
    mem2server(conn, &_0pGraph, (void *)pGraph, sizeof(*pGraph));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamEndCapture);
    conn->write(&stream, sizeof(stream));
    conn->write(&_0pGraph, sizeof(_0pGraph));
    updateTmpPtr((void *)pGraph, _0pGraph);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraph, sizeof(*pGraph), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamIsCapturing called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pCaptureStatus;
    mem2server(conn, &_0pCaptureStatus, (void *)pCaptureStatus, sizeof(*pCaptureStatus));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamIsCapturing);
    conn->write(&stream, sizeof(stream));
    conn->write(&_0pCaptureStatus, sizeof(_0pCaptureStatus));
    updateTmpPtr((void *)pCaptureStatus, _0pCaptureStatus);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pCaptureStatus, sizeof(*pCaptureStatus), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus, unsigned long long *pId) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamGetCaptureInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pCaptureStatus;
    mem2server(conn, &_0pCaptureStatus, (void *)pCaptureStatus, sizeof(*pCaptureStatus));
    void *_0pId;
    mem2server(conn, &_0pId, (void *)pId, sizeof(*pId));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamGetCaptureInfo);
    conn->write(&stream, sizeof(stream));
    conn->write(&_0pCaptureStatus, sizeof(_0pCaptureStatus));
    updateTmpPtr((void *)pCaptureStatus, _0pCaptureStatus);
    conn->write(&_0pId, sizeof(_0pId));
    updateTmpPtr((void *)pId, _0pId);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pCaptureStatus, sizeof(*pCaptureStatus), true);
    mem2client(conn, (void *)pId, sizeof(*pId), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out, unsigned long long *id_out, cudaGraph_t *graph_out, const cudaGraphNode_t **dependencies_out, size_t *numDependencies_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamGetCaptureInfo_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0captureStatus_out;
    mem2server(conn, &_0captureStatus_out, (void *)captureStatus_out, sizeof(*captureStatus_out));
    void *_0id_out;
    mem2server(conn, &_0id_out, (void *)id_out, sizeof(*id_out));
    void *_0graph_out;
    mem2server(conn, &_0graph_out, (void *)graph_out, sizeof(*graph_out));
    void *_0numDependencies_out;
    mem2server(conn, &_0numDependencies_out, (void *)numDependencies_out, sizeof(*numDependencies_out));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamGetCaptureInfo_v2);
    conn->write(&stream, sizeof(stream));
    conn->write(&_0captureStatus_out, sizeof(_0captureStatus_out));
    updateTmpPtr((void *)captureStatus_out, _0captureStatus_out);
    conn->write(&_0id_out, sizeof(_0id_out));
    updateTmpPtr((void *)id_out, _0id_out);
    conn->write(&_0graph_out, sizeof(_0graph_out));
    updateTmpPtr((void *)graph_out, _0graph_out);
    static cudaGraphNode_t _cudaStreamGetCaptureInfo_v2_dependencies_out;
    conn->read(&_cudaStreamGetCaptureInfo_v2_dependencies_out, sizeof(cudaGraphNode_t));
    conn->write(&_0numDependencies_out, sizeof(_0numDependencies_out));
    updateTmpPtr((void *)numDependencies_out, _0numDependencies_out);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    *dependencies_out = &_cudaStreamGetCaptureInfo_v2_dependencies_out;
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)captureStatus_out, sizeof(*captureStatus_out), true);
    mem2client(conn, (void *)id_out, sizeof(*id_out), true);
    mem2client(conn, (void *)graph_out, sizeof(*graph_out), true);
    mem2client(conn, (void *)numDependencies_out, sizeof(*numDependencies_out), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies, size_t numDependencies, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamUpdateCaptureDependencies called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dependencies;
    mem2server(conn, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaStreamUpdateCaptureDependencies);
    conn->write(&stream, sizeof(stream));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dependencies, sizeof(*dependencies), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaEventCreate(cudaEvent_t *event) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventCreate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0event;
    mem2server(conn, &_0event, (void *)event, sizeof(*event));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaEventCreate);
    conn->write(&_0event, sizeof(_0event));
    updateTmpPtr((void *)event, _0event);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)event, sizeof(*event), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventCreateWithFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0event;
    mem2server(conn, &_0event, (void *)event, sizeof(*event));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaEventCreateWithFlags);
    conn->write(&_0event, sizeof(_0event));
    updateTmpPtr((void *)event, _0event);
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)event, sizeof(*event), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventRecord called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaEventRecord);
    conn->write(&event, sizeof(event));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventRecordWithFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaEventRecordWithFlags);
    conn->write(&event, sizeof(event));
    conn->write(&stream, sizeof(stream));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventQuery called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaEventQuery);
    conn->write(&event, sizeof(event));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventSynchronize called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaEventSynchronize);
    conn->write(&event, sizeof(event));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventDestroy called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaEventDestroy);
    conn->write(&event, sizeof(event));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventElapsedTime called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0ms;
    mem2server(conn, &_0ms, (void *)ms, sizeof(*ms));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaEventElapsedTime);
    conn->write(&_0ms, sizeof(_0ms));
    updateTmpPtr((void *)ms, _0ms);
    conn->write(&start, sizeof(start));
    conn->write(&end, sizeof(end));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)ms, sizeof(*ms), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaImportExternalMemory(cudaExternalMemory_t *extMem_out, const struct cudaExternalMemoryHandleDesc *memHandleDesc) {
#ifdef DEBUG
    std::cout << "Hook: cudaImportExternalMemory called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0extMem_out;
    mem2server(conn, &_0extMem_out, (void *)extMem_out, sizeof(*extMem_out));
    void *_0memHandleDesc;
    mem2server(conn, &_0memHandleDesc, (void *)memHandleDesc, sizeof(*memHandleDesc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaImportExternalMemory);
    conn->write(&_0extMem_out, sizeof(_0extMem_out));
    updateTmpPtr((void *)extMem_out, _0extMem_out);
    conn->write(&_0memHandleDesc, sizeof(_0memHandleDesc));
    updateTmpPtr((void *)memHandleDesc, _0memHandleDesc);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)extMem_out, sizeof(*extMem_out), true);
    mem2client(conn, (void *)memHandleDesc, sizeof(*memHandleDesc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaExternalMemoryGetMappedBuffer(void **devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc *bufferDesc) {
#ifdef DEBUG
    std::cout << "Hook: cudaExternalMemoryGetMappedBuffer called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0bufferDesc;
    mem2server(conn, &_0bufferDesc, (void *)bufferDesc, sizeof(*bufferDesc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaExternalMemoryGetMappedBuffer);
    conn->read(devPtr, sizeof(void *));
    conn->write(&extMem, sizeof(extMem));
    conn->write(&_0bufferDesc, sizeof(_0bufferDesc));
    updateTmpPtr((void *)bufferDesc, _0bufferDesc);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)bufferDesc, sizeof(*bufferDesc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t *mipmap, cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc) {
#ifdef DEBUG
    std::cout << "Hook: cudaExternalMemoryGetMappedMipmappedArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0mipmap;
    mem2server(conn, &_0mipmap, (void *)mipmap, sizeof(*mipmap));
    void *_0mipmapDesc;
    mem2server(conn, &_0mipmapDesc, (void *)mipmapDesc, sizeof(*mipmapDesc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaExternalMemoryGetMappedMipmappedArray);
    conn->write(&_0mipmap, sizeof(_0mipmap));
    updateTmpPtr((void *)mipmap, _0mipmap);
    conn->write(&extMem, sizeof(extMem));
    conn->write(&_0mipmapDesc, sizeof(_0mipmapDesc));
    updateTmpPtr((void *)mipmapDesc, _0mipmapDesc);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)mipmap, sizeof(*mipmap), true);
    mem2client(conn, (void *)mipmapDesc, sizeof(*mipmapDesc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem) {
#ifdef DEBUG
    std::cout << "Hook: cudaDestroyExternalMemory called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDestroyExternalMemory);
    conn->write(&extMem, sizeof(extMem));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t *extSem_out, const struct cudaExternalSemaphoreHandleDesc *semHandleDesc) {
#ifdef DEBUG
    std::cout << "Hook: cudaImportExternalSemaphore called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0extSem_out;
    mem2server(conn, &_0extSem_out, (void *)extSem_out, sizeof(*extSem_out));
    void *_0semHandleDesc;
    mem2server(conn, &_0semHandleDesc, (void *)semHandleDesc, sizeof(*semHandleDesc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaImportExternalSemaphore);
    conn->write(&_0extSem_out, sizeof(_0extSem_out));
    updateTmpPtr((void *)extSem_out, _0extSem_out);
    conn->write(&_0semHandleDesc, sizeof(_0semHandleDesc));
    updateTmpPtr((void *)semHandleDesc, _0semHandleDesc);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)extSem_out, sizeof(*extSem_out), true);
    mem2client(conn, (void *)semHandleDesc, sizeof(*semHandleDesc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaSignalExternalSemaphoresAsync_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0extSemArray;
    mem2server(conn, &_0extSemArray, (void *)extSemArray, sizeof(*extSemArray));
    void *_0paramsArray;
    mem2server(conn, &_0paramsArray, (void *)paramsArray, sizeof(*paramsArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaSignalExternalSemaphoresAsync_v2);
    conn->write(&_0extSemArray, sizeof(_0extSemArray));
    updateTmpPtr((void *)extSemArray, _0extSemArray);
    conn->write(&_0paramsArray, sizeof(_0paramsArray));
    updateTmpPtr((void *)paramsArray, _0paramsArray);
    conn->write(&numExtSems, sizeof(numExtSems));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)extSemArray, sizeof(*extSemArray), true);
    mem2client(conn, (void *)paramsArray, sizeof(*paramsArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaWaitExternalSemaphoresAsync_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0extSemArray;
    mem2server(conn, &_0extSemArray, (void *)extSemArray, sizeof(*extSemArray));
    void *_0paramsArray;
    mem2server(conn, &_0paramsArray, (void *)paramsArray, sizeof(*paramsArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaWaitExternalSemaphoresAsync_v2);
    conn->write(&_0extSemArray, sizeof(_0extSemArray));
    updateTmpPtr((void *)extSemArray, _0extSemArray);
    conn->write(&_0paramsArray, sizeof(_0paramsArray));
    updateTmpPtr((void *)paramsArray, _0paramsArray);
    conn->write(&numExtSems, sizeof(numExtSems));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)extSemArray, sizeof(*extSemArray), true);
    mem2client(conn, (void *)paramsArray, sizeof(*paramsArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) {
#ifdef DEBUG
    std::cout << "Hook: cudaDestroyExternalSemaphore called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDestroyExternalSemaphore);
    conn->write(&extSem, sizeof(extSem));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaLaunchCooperativeKernelMultiDevice called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0launchParamsList;
    mem2server(conn, &_0launchParamsList, (void *)launchParamsList, sizeof(*launchParamsList));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaLaunchCooperativeKernelMultiDevice);
    conn->write(&_0launchParamsList, sizeof(_0launchParamsList));
    updateTmpPtr((void *)launchParamsList, _0launchParamsList);
    conn->write(&numDevices, sizeof(numDevices));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)launchParamsList, sizeof(*launchParamsList), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig) {
#ifdef DEBUG
    std::cout << "Hook: cudaFuncSetCacheConfig called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0func;
    mem2server(conn, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaFuncSetCacheConfig);
    conn->write(&_0func, sizeof(_0func));
    updateTmpPtr((void *)func, _0func);
    conn->write(&cacheConfig, sizeof(cacheConfig));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)func, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config) {
#ifdef DEBUG
    std::cout << "Hook: cudaFuncSetSharedMemConfig called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0func;
    mem2server(conn, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaFuncSetSharedMemConfig);
    conn->write(&_0func, sizeof(_0func));
    updateTmpPtr((void *)func, _0func);
    conn->write(&config, sizeof(config));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)func, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {
#ifdef DEBUG
    std::cout << "Hook: cudaFuncGetAttributes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0attr;
    mem2server(conn, &_0attr, (void *)attr, sizeof(*attr));
    void *_0func;
    mem2server(conn, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaFuncGetAttributes);
    conn->write(&_0attr, sizeof(_0attr));
    updateTmpPtr((void *)attr, _0attr);
    conn->write(&_0func, sizeof(_0func));
    updateTmpPtr((void *)func, _0func);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)attr, sizeof(*attr), true);
    mem2client(conn, (void *)func, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) {
#ifdef DEBUG
    std::cout << "Hook: cudaFuncSetAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0func;
    mem2server(conn, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaFuncSetAttribute);
    conn->write(&_0func, sizeof(_0func));
    updateTmpPtr((void *)func, _0func);
    conn->write(&attr, sizeof(attr));
    conn->write(&value, sizeof(value));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)func, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaSetDoubleForDevice(double *d) {
#ifdef DEBUG
    std::cout << "Hook: cudaSetDoubleForDevice called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0d;
    mem2server(conn, &_0d, (void *)d, sizeof(*d));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaSetDoubleForDevice);
    conn->write(&_0d, sizeof(_0d));
    updateTmpPtr((void *)d, _0d);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)d, sizeof(*d), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaSetDoubleForHost(double *d) {
#ifdef DEBUG
    std::cout << "Hook: cudaSetDoubleForHost called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0d;
    mem2server(conn, &_0d, (void *)d, sizeof(*d));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaSetDoubleForHost);
    conn->write(&_0d, sizeof(_0d));
    updateTmpPtr((void *)d, _0d);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)d, sizeof(*d), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData) {
#ifdef DEBUG
    std::cout << "Hook: cudaLaunchHostFunc called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0userData;
    mem2server(conn, &_0userData, (void *)userData, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaLaunchHostFunc);
    conn->write(&stream, sizeof(stream));
    conn->write(&fn, sizeof(fn));
    conn->write(&_0userData, sizeof(_0userData));
    updateTmpPtr((void *)userData, _0userData);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)userData, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) {
#ifdef DEBUG
    std::cout << "Hook: cudaOccupancyMaxActiveBlocksPerMultiprocessor called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0numBlocks;
    mem2server(conn, &_0numBlocks, (void *)numBlocks, sizeof(*numBlocks));
    void *_0func;
    mem2server(conn, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaOccupancyMaxActiveBlocksPerMultiprocessor);
    conn->write(&_0numBlocks, sizeof(_0numBlocks));
    updateTmpPtr((void *)numBlocks, _0numBlocks);
    conn->write(&_0func, sizeof(_0func));
    updateTmpPtr((void *)func, _0func);
    conn->write(&blockSize, sizeof(blockSize));
    conn->write(&dynamicSMemSize, sizeof(dynamicSMemSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)numBlocks, sizeof(*numBlocks), true);
    mem2client(conn, (void *)func, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, const void *func, int numBlocks, int blockSize) {
#ifdef DEBUG
    std::cout << "Hook: cudaOccupancyAvailableDynamicSMemPerBlock called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dynamicSmemSize;
    mem2server(conn, &_0dynamicSmemSize, (void *)dynamicSmemSize, sizeof(*dynamicSmemSize));
    void *_0func;
    mem2server(conn, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaOccupancyAvailableDynamicSMemPerBlock);
    conn->write(&_0dynamicSmemSize, sizeof(_0dynamicSmemSize));
    updateTmpPtr((void *)dynamicSmemSize, _0dynamicSmemSize);
    conn->write(&_0func, sizeof(_0func));
    updateTmpPtr((void *)func, _0func);
    conn->write(&numBlocks, sizeof(numBlocks));
    conn->write(&blockSize, sizeof(blockSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dynamicSmemSize, sizeof(*dynamicSmemSize), true);
    mem2client(conn, (void *)func, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0numBlocks;
    mem2server(conn, &_0numBlocks, (void *)numBlocks, sizeof(*numBlocks));
    void *_0func;
    mem2server(conn, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    conn->write(&_0numBlocks, sizeof(_0numBlocks));
    updateTmpPtr((void *)numBlocks, _0numBlocks);
    conn->write(&_0func, sizeof(_0func));
    updateTmpPtr((void *)func, _0func);
    conn->write(&blockSize, sizeof(blockSize));
    conn->write(&dynamicSMemSize, sizeof(dynamicSMemSize));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)numBlocks, sizeof(*numBlocks), true);
    mem2client(conn, (void *)func, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0array;
    mem2server(conn, &_0array, (void *)array, sizeof(*array));
    void *_0desc;
    mem2server(conn, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMallocArray);
    conn->write(&_0array, sizeof(_0array));
    updateTmpPtr((void *)array, _0array);
    conn->write(&_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    conn->write(&width, sizeof(width));
    conn->write(&height, sizeof(height));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)array, sizeof(*array), true);
    mem2client(conn, (void *)desc, sizeof(*desc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaFreeArray(cudaArray_t array) {
#ifdef DEBUG
    std::cout << "Hook: cudaFreeArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaFreeArray);
    conn->write(&array, sizeof(array));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) {
#ifdef DEBUG
    std::cout << "Hook: cudaFreeMipmappedArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaFreeMipmappedArray);
    conn->write(&mipmappedArray, sizeof(mipmappedArray));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaHostGetDevicePointer called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pHost;
    mem2server(conn, &_0pHost, (void *)pHost, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaHostGetDevicePointer);
    conn->read(pDevice, sizeof(void *));
    conn->write(&_0pHost, sizeof(_0pHost));
    updateTmpPtr((void *)pHost, _0pHost);
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pHost, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
#ifdef DEBUG
    std::cout << "Hook: cudaHostGetFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pFlags;
    mem2server(conn, &_0pFlags, (void *)pFlags, sizeof(*pFlags));
    void *_0pHost;
    mem2server(conn, &_0pHost, (void *)pHost, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaHostGetFlags);
    conn->write(&_0pFlags, sizeof(_0pFlags));
    updateTmpPtr((void *)pFlags, _0pFlags);
    conn->write(&_0pHost, sizeof(_0pHost));
    updateTmpPtr((void *)pHost, _0pHost);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pFlags, sizeof(*pFlags), true);
    mem2client(conn, (void *)pHost, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, struct cudaExtent extent, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaMalloc3DArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0array;
    mem2server(conn, &_0array, (void *)array, sizeof(*array));
    void *_0desc;
    mem2server(conn, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMalloc3DArray);
    conn->write(&_0array, sizeof(_0array));
    updateTmpPtr((void *)array, _0array);
    conn->write(&_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    conn->write(&extent, sizeof(extent));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)array, sizeof(*array), true);
    mem2client(conn, (void *)desc, sizeof(*desc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray, const struct cudaChannelFormatDesc *desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocMipmappedArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0mipmappedArray;
    mem2server(conn, &_0mipmappedArray, (void *)mipmappedArray, sizeof(*mipmappedArray));
    void *_0desc;
    mem2server(conn, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMallocMipmappedArray);
    conn->write(&_0mipmappedArray, sizeof(_0mipmappedArray));
    updateTmpPtr((void *)mipmappedArray, _0mipmappedArray);
    conn->write(&_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    conn->write(&extent, sizeof(extent));
    conn->write(&numLevels, sizeof(numLevels));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)mipmappedArray, sizeof(*mipmappedArray), true);
    mem2client(conn, (void *)desc, sizeof(*desc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetMipmappedArrayLevel called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0levelArray;
    mem2server(conn, &_0levelArray, (void *)levelArray, sizeof(*levelArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetMipmappedArrayLevel);
    conn->write(&_0levelArray, sizeof(_0levelArray));
    updateTmpPtr((void *)levelArray, _0levelArray);
    conn->write(&mipmappedArray, sizeof(mipmappedArray));
    conn->write(&level, sizeof(level));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)levelArray, sizeof(*levelArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy3D called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0p;
    mem2server(conn, &_0p, (void *)p, sizeof(*p));
    void *_0sptr = nullptr;
    void *_0dptr = nullptr;
    if(p != nullptr) {
        mem2server(conn, &_0sptr, (void *)p->srcPtr.ptr, sizeof(p->srcPtr.pitch * p->srcPtr.ysize));
        mem2server(conn, &_0dptr, (void *)p->dstPtr.ptr, sizeof(p->dstPtr.pitch * p->dstPtr.ysize));
    }
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpy3D);
    conn->write(&_0p, sizeof(_0p));
    updateTmpPtr((void *)p, _0p);
    conn->write(&_0sptr, sizeof(_0sptr));
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)p->srcPtr.ptr, _0sptr);
    updateTmpPtr((void *)p->dstPtr.ptr, _0dptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)p, sizeof(*p), true);
    if(p != nullptr) {
        _0sptr = (void *)p->srcPtr.ptr;
        _0dptr = (void *)p->dstPtr.ptr;
        mem2client(conn, (void *)p->srcPtr.ptr, sizeof(p->srcPtr.pitch * p->srcPtr.ysize), false);
        mem2client(conn, (void *)p->dstPtr.ptr, sizeof(p->dstPtr.pitch * p->dstPtr.ysize), false);
    }
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    if(p != nullptr) {
        const_cast<void *&>(p->srcPtr.ptr) = _0sptr;
        const_cast<void *&>(p->dstPtr.ptr) = _0dptr;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy3DPeer called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0p;
    mem2server(conn, &_0p, (void *)p, sizeof(*p));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpy3DPeer);
    conn->write(&_0p, sizeof(_0p));
    updateTmpPtr((void *)p, _0p);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)p, sizeof(*p), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy3DAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0p;
    mem2server(conn, &_0p, (void *)p, sizeof(*p));
    void *_0sptr = nullptr;
    void *_0dptr = nullptr;
    if(p != nullptr) {
        mem2server(conn, &_0sptr, (void *)p->srcPtr.ptr, sizeof(p->srcPtr.pitch * p->srcPtr.ysize));
        mem2server(conn, &_0dptr, (void *)p->dstPtr.ptr, sizeof(p->dstPtr.pitch * p->dstPtr.ysize));
    }
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpy3DAsync);
    conn->write(&_0p, sizeof(_0p));
    updateTmpPtr((void *)p, _0p);
    conn->write(&_0sptr, sizeof(_0sptr));
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)p->srcPtr.ptr, _0sptr);
    updateTmpPtr((void *)p->dstPtr.ptr, _0dptr);
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)p, sizeof(*p), true);
    if(p != nullptr) {
        _0sptr = (void *)p->srcPtr.ptr;
        _0dptr = (void *)p->dstPtr.ptr;
        mem2client(conn, (void *)p->srcPtr.ptr, sizeof(p->srcPtr.pitch * p->srcPtr.ysize), false);
        mem2client(conn, (void *)p->dstPtr.ptr, sizeof(p->dstPtr.pitch * p->dstPtr.ysize), false);
    }
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    if(p != nullptr) {
        const_cast<void *&>(p->srcPtr.ptr) = _0sptr;
        const_cast<void *&>(p->dstPtr.ptr) = _0dptr;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy3DPeerAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0p;
    mem2server(conn, &_0p, (void *)p, sizeof(*p));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpy3DPeerAsync);
    conn->write(&_0p, sizeof(_0p));
    updateTmpPtr((void *)p, _0p);
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)p, sizeof(*p), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemGetInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0free;
    mem2server(conn, &_0free, (void *)free, sizeof(*free));
    void *_0total;
    mem2server(conn, &_0total, (void *)total, sizeof(*total));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemGetInfo);
    conn->write(&_0free, sizeof(_0free));
    updateTmpPtr((void *)free, _0free);
    conn->write(&_0total, sizeof(_0total));
    updateTmpPtr((void *)total, _0total);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)free, sizeof(*free), true);
    mem2client(conn, (void *)total, sizeof(*total), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array) {
#ifdef DEBUG
    std::cout << "Hook: cudaArrayGetInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0desc;
    mem2server(conn, &_0desc, (void *)desc, sizeof(*desc));
    void *_0extent;
    mem2server(conn, &_0extent, (void *)extent, sizeof(*extent));
    void *_0flags;
    mem2server(conn, &_0flags, (void *)flags, sizeof(*flags));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaArrayGetInfo);
    conn->write(&_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    conn->write(&_0extent, sizeof(_0extent));
    updateTmpPtr((void *)extent, _0extent);
    conn->write(&_0flags, sizeof(_0flags));
    updateTmpPtr((void *)flags, _0flags);
    conn->write(&array, sizeof(array));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)desc, sizeof(*desc), true);
    mem2client(conn, (void *)extent, sizeof(*extent), true);
    mem2client(conn, (void *)flags, sizeof(*flags), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaArrayGetPlane(cudaArray_t *pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) {
#ifdef DEBUG
    std::cout << "Hook: cudaArrayGetPlane called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pPlaneArray;
    mem2server(conn, &_0pPlaneArray, (void *)pPlaneArray, sizeof(*pPlaneArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaArrayGetPlane);
    conn->write(&_0pPlaneArray, sizeof(_0pPlaneArray));
    updateTmpPtr((void *)pPlaneArray, _0pPlaneArray);
    conn->write(&hArray, sizeof(hArray));
    conn->write(&planeIdx, sizeof(planeIdx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pPlaneArray, sizeof(*pPlaneArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaArray_t array) {
#ifdef DEBUG
    std::cout << "Hook: cudaArrayGetSparseProperties called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0sparseProperties;
    mem2server(conn, &_0sparseProperties, (void *)sparseProperties, sizeof(*sparseProperties));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaArrayGetSparseProperties);
    conn->write(&_0sparseProperties, sizeof(_0sparseProperties));
    updateTmpPtr((void *)sparseProperties, _0sparseProperties);
    conn->write(&array, sizeof(array));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)sparseProperties, sizeof(*sparseProperties), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaMipmappedArray_t mipmap) {
#ifdef DEBUG
    std::cout << "Hook: cudaMipmappedArrayGetSparseProperties called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0sparseProperties;
    mem2server(conn, &_0sparseProperties, (void *)sparseProperties, sizeof(*sparseProperties));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMipmappedArrayGetSparseProperties);
    conn->write(&_0sparseProperties, sizeof(_0sparseProperties));
    updateTmpPtr((void *)sparseProperties, _0sparseProperties);
    conn->write(&mipmap, sizeof(mipmap));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)sparseProperties, sizeof(*sparseProperties), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpy);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&count, sizeof(count));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, count, true);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyPeer called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpyPeer);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&dstDevice, sizeof(dstDevice));
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&srcDevice, sizeof(srcDevice));
    conn->write(&count, sizeof(count));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, count, true);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2D called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, dpitch * height);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, spitch * height);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpy2D);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&dpitch, sizeof(dpitch));
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&spitch, sizeof(spitch));
    conn->write(&width, sizeof(width));
    conn->write(&height, sizeof(height));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, dpitch * height, true);
    mem2client(conn, (void *)src, spitch * height, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2DToArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, spitch * height);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpy2DToArray);
    conn->write(&dst, sizeof(dst));
    conn->write(&wOffset, sizeof(wOffset));
    conn->write(&hOffset, sizeof(hOffset));
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&spitch, sizeof(spitch));
    conn->write(&width, sizeof(width));
    conn->write(&height, sizeof(height));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)src, spitch * height, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2DFromArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, dpitch * height);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpy2DFromArray);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&dpitch, sizeof(dpitch));
    conn->write(&src, sizeof(src));
    conn->write(&wOffset, sizeof(wOffset));
    conn->write(&hOffset, sizeof(hOffset));
    conn->write(&width, sizeof(width));
    conn->write(&height, sizeof(height));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, dpitch * height, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2DArrayToArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpy2DArrayToArray);
    conn->write(&dst, sizeof(dst));
    conn->write(&wOffsetDst, sizeof(wOffsetDst));
    conn->write(&hOffsetDst, sizeof(hOffsetDst));
    conn->write(&src, sizeof(src));
    conn->write(&wOffsetSrc, sizeof(wOffsetSrc));
    conn->write(&hOffsetSrc, sizeof(hOffsetSrc));
    conn->write(&width, sizeof(width));
    conn->write(&height, sizeof(height));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyToSymbol called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0symbol;
    mem2server(conn, &_0symbol, (void *)symbol, -1);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpyToSymbol);
    conn->write(&_0symbol, sizeof(_0symbol));
    updateTmpPtr((void *)symbol, _0symbol);
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&count, sizeof(count));
    conn->write(&offset, sizeof(offset));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)symbol, -1, true);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyFromSymbol called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *_0symbol;
    mem2server(conn, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpyFromSymbol);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&_0symbol, sizeof(_0symbol));
    updateTmpPtr((void *)symbol, _0symbol);
    conn->write(&count, sizeof(count));
    conn->write(&offset, sizeof(offset));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, count, true);
    mem2client(conn, (void *)symbol, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpyAsync);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&count, sizeof(count));
    conn->write(&kind, sizeof(kind));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, count, true);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyPeerAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpyPeerAsync);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&dstDevice, sizeof(dstDevice));
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&srcDevice, sizeof(srcDevice));
    conn->write(&count, sizeof(count));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, count, true);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2DAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, dpitch * height);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, spitch * height);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpy2DAsync);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&dpitch, sizeof(dpitch));
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&spitch, sizeof(spitch));
    conn->write(&width, sizeof(width));
    conn->write(&height, sizeof(height));
    conn->write(&kind, sizeof(kind));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, dpitch * height, true);
    mem2client(conn, (void *)src, spitch * height, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2DToArrayAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, spitch * height);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpy2DToArrayAsync);
    conn->write(&dst, sizeof(dst));
    conn->write(&wOffset, sizeof(wOffset));
    conn->write(&hOffset, sizeof(hOffset));
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&spitch, sizeof(spitch));
    conn->write(&width, sizeof(width));
    conn->write(&height, sizeof(height));
    conn->write(&kind, sizeof(kind));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)src, spitch * height, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2DFromArrayAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, dpitch * height);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpy2DFromArrayAsync);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&dpitch, sizeof(dpitch));
    conn->write(&src, sizeof(src));
    conn->write(&wOffset, sizeof(wOffset));
    conn->write(&hOffset, sizeof(hOffset));
    conn->write(&width, sizeof(width));
    conn->write(&height, sizeof(height));
    conn->write(&kind, sizeof(kind));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, dpitch * height, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyToSymbolAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0symbol;
    mem2server(conn, &_0symbol, (void *)symbol, -1);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpyToSymbolAsync);
    conn->write(&_0symbol, sizeof(_0symbol));
    updateTmpPtr((void *)symbol, _0symbol);
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&count, sizeof(count));
    conn->write(&offset, sizeof(offset));
    conn->write(&kind, sizeof(kind));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)symbol, -1, true);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyFromSymbolAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *_0symbol;
    mem2server(conn, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpyFromSymbolAsync);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&_0symbol, sizeof(_0symbol));
    updateTmpPtr((void *)symbol, _0symbol);
    conn->write(&count, sizeof(count));
    conn->write(&offset, sizeof(offset));
    conn->write(&kind, sizeof(kind));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, count, true);
    mem2client(conn, (void *)symbol, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemset called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemset);
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->write(&value, sizeof(value));
    conn->write(&count, sizeof(count));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devPtr, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemset2D called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, pitch * height);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemset2D);
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->write(&pitch, sizeof(pitch));
    conn->write(&value, sizeof(value));
    conn->write(&width, sizeof(width));
    conn->write(&height, sizeof(height));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devPtr, pitch * height, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemset3D called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemset3D);
    conn->write(&pitchedDevPtr, sizeof(pitchedDevPtr));
    conn->write(&value, sizeof(value));
    conn->write(&extent, sizeof(extent));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemsetAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemsetAsync);
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->write(&value, sizeof(value));
    conn->write(&count, sizeof(count));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devPtr, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemset2DAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, pitch * height);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemset2DAsync);
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->write(&pitch, sizeof(pitch));
    conn->write(&value, sizeof(value));
    conn->write(&width, sizeof(width));
    conn->write(&height, sizeof(height));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devPtr, pitch * height, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemset3DAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemset3DAsync);
    conn->write(&pitchedDevPtr, sizeof(pitchedDevPtr));
    conn->write(&value, sizeof(value));
    conn->write(&extent, sizeof(extent));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetSymbolSize called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0size;
    mem2server(conn, &_0size, (void *)size, sizeof(*size));
    void *_0symbol;
    mem2server(conn, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetSymbolSize);
    conn->write(&_0size, sizeof(_0size));
    updateTmpPtr((void *)size, _0size);
    conn->write(&_0symbol, sizeof(_0symbol));
    updateTmpPtr((void *)symbol, _0symbol);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)size, sizeof(*size), true);
    mem2client(conn, (void *)symbol, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPrefetchAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemPrefetchAsync);
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->write(&count, sizeof(count));
    conn->write(&dstDevice, sizeof(dstDevice));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devPtr, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemAdvise called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemAdvise);
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->write(&count, sizeof(count));
    conn->write(&advice, sizeof(advice));
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devPtr, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemRangeGetAttribute(void *data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void *devPtr, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemRangeGetAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0data;
    mem2server(conn, &_0data, (void *)data, dataSize);
    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemRangeGetAttribute);
    conn->write(&_0data, sizeof(_0data));
    updateTmpPtr((void *)data, _0data);
    conn->write(&dataSize, sizeof(dataSize));
    conn->write(&attribute, sizeof(attribute));
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->write(&count, sizeof(count));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)data, dataSize, true);
    mem2client(conn, (void *)devPtr, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyToArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpyToArray);
    conn->write(&dst, sizeof(dst));
    conn->write(&wOffset, sizeof(wOffset));
    conn->write(&hOffset, sizeof(hOffset));
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&count, sizeof(count));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpyFromArray(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyFromArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpyFromArray);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&src, sizeof(src));
    conn->write(&wOffset, sizeof(wOffset));
    conn->write(&hOffset, sizeof(hOffset));
    conn->write(&count, sizeof(count));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyArrayToArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpyArrayToArray);
    conn->write(&dst, sizeof(dst));
    conn->write(&wOffsetDst, sizeof(wOffsetDst));
    conn->write(&hOffsetDst, sizeof(hOffsetDst));
    conn->write(&src, sizeof(src));
    conn->write(&wOffsetSrc, sizeof(wOffsetSrc));
    conn->write(&hOffsetSrc, sizeof(hOffsetSrc));
    conn->write(&count, sizeof(count));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyToArrayAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpyToArrayAsync);
    conn->write(&dst, sizeof(dst));
    conn->write(&wOffset, sizeof(wOffset));
    conn->write(&hOffset, sizeof(hOffset));
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&count, sizeof(count));
    conn->write(&kind, sizeof(kind));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyFromArrayAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemcpyFromArrayAsync);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&src, sizeof(src));
    conn->write(&wOffset, sizeof(wOffset));
    conn->write(&hOffset, sizeof(hOffset));
    conn->write(&count, sizeof(count));
    conn->write(&kind, sizeof(kind));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMallocAsync);
    conn->read(devPtr, sizeof(void *));
    conn->write(&size, sizeof(size));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t hStream) {
#ifdef DEBUG
    std::cout << "Hook: cudaFreeAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaFreeAsync);
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devPtr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolTrimTo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemPoolTrimTo);
    conn->write(&memPool, sizeof(memPool));
    conn->write(&minBytesToKeep, sizeof(minBytesToKeep));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolSetAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0value;
    mem2server(conn, &_0value, (void *)value, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemPoolSetAttribute);
    conn->write(&memPool, sizeof(memPool));
    conn->write(&attr, sizeof(attr));
    conn->write(&_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)value, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolGetAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0value;
    mem2server(conn, &_0value, (void *)value, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemPoolGetAttribute);
    conn->write(&memPool, sizeof(memPool));
    conn->write(&attr, sizeof(attr));
    conn->write(&_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)value, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const struct cudaMemAccessDesc *descList, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolSetAccess called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0descList;
    mem2server(conn, &_0descList, (void *)descList, sizeof(*descList));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemPoolSetAccess);
    conn->write(&memPool, sizeof(memPool));
    conn->write(&_0descList, sizeof(_0descList));
    updateTmpPtr((void *)descList, _0descList);
    conn->write(&count, sizeof(count));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)descList, sizeof(*descList), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemPoolGetAccess(enum cudaMemAccessFlags *flags, cudaMemPool_t memPool, struct cudaMemLocation *location) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolGetAccess called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0flags;
    mem2server(conn, &_0flags, (void *)flags, sizeof(*flags));
    void *_0location;
    mem2server(conn, &_0location, (void *)location, sizeof(*location));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemPoolGetAccess);
    conn->write(&_0flags, sizeof(_0flags));
    updateTmpPtr((void *)flags, _0flags);
    conn->write(&memPool, sizeof(memPool));
    conn->write(&_0location, sizeof(_0location));
    updateTmpPtr((void *)location, _0location);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)flags, sizeof(*flags), true);
    mem2client(conn, (void *)location, sizeof(*location), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemPoolCreate(cudaMemPool_t *memPool, const struct cudaMemPoolProps *poolProps) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolCreate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0memPool;
    mem2server(conn, &_0memPool, (void *)memPool, sizeof(*memPool));
    void *_0poolProps;
    mem2server(conn, &_0poolProps, (void *)poolProps, sizeof(*poolProps));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemPoolCreate);
    conn->write(&_0memPool, sizeof(_0memPool));
    updateTmpPtr((void *)memPool, _0memPool);
    conn->write(&_0poolProps, sizeof(_0poolProps));
    updateTmpPtr((void *)poolProps, _0poolProps);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)memPool, sizeof(*memPool), true);
    mem2client(conn, (void *)poolProps, sizeof(*poolProps), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolDestroy called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemPoolDestroy);
    conn->write(&memPool, sizeof(memPool));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocFromPoolAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMallocFromPoolAsync);
    conn->read(ptr, sizeof(void *));
    conn->write(&size, sizeof(size));
    conn->write(&memPool, sizeof(memPool));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemPoolExportToShareableHandle(void *shareableHandle, cudaMemPool_t memPool, enum cudaMemAllocationHandleType handleType, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolExportToShareableHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0shareableHandle;
    mem2server(conn, &_0shareableHandle, (void *)shareableHandle, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemPoolExportToShareableHandle);
    conn->write(&_0shareableHandle, sizeof(_0shareableHandle));
    updateTmpPtr((void *)shareableHandle, _0shareableHandle);
    conn->write(&memPool, sizeof(memPool));
    conn->write(&handleType, sizeof(handleType));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)shareableHandle, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t *memPool, void *shareableHandle, enum cudaMemAllocationHandleType handleType, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolImportFromShareableHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0memPool;
    mem2server(conn, &_0memPool, (void *)memPool, sizeof(*memPool));
    void *_0shareableHandle;
    mem2server(conn, &_0shareableHandle, (void *)shareableHandle, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemPoolImportFromShareableHandle);
    conn->write(&_0memPool, sizeof(_0memPool));
    updateTmpPtr((void *)memPool, _0memPool);
    conn->write(&_0shareableHandle, sizeof(_0shareableHandle));
    updateTmpPtr((void *)shareableHandle, _0shareableHandle);
    conn->write(&handleType, sizeof(handleType));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)memPool, sizeof(*memPool), true);
    mem2client(conn, (void *)shareableHandle, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData *exportData, void *ptr) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolExportPointer called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0exportData;
    mem2server(conn, &_0exportData, (void *)exportData, sizeof(*exportData));
    void *_0ptr;
    mem2server(conn, &_0ptr, (void *)ptr, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemPoolExportPointer);
    conn->write(&_0exportData, sizeof(_0exportData));
    updateTmpPtr((void *)exportData, _0exportData);
    conn->write(&_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)exportData, sizeof(*exportData), true);
    mem2client(conn, (void *)ptr, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemPoolImportPointer(void **ptr, cudaMemPool_t memPool, struct cudaMemPoolPtrExportData *exportData) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolImportPointer called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0exportData;
    mem2server(conn, &_0exportData, (void *)exportData, sizeof(*exportData));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemPoolImportPointer);
    conn->read(ptr, sizeof(void *));
    conn->write(&memPool, sizeof(memPool));
    conn->write(&_0exportData, sizeof(_0exportData));
    updateTmpPtr((void *)exportData, _0exportData);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)exportData, sizeof(*exportData), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr) {
#ifdef DEBUG
    std::cout << "Hook: cudaPointerGetAttributes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0attributes;
    mem2server(conn, &_0attributes, (void *)attributes, sizeof(*attributes));
    void *_0ptr;
    mem2server(conn, &_0ptr, (void *)ptr, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaPointerGetAttributes);
    conn->write(&_0attributes, sizeof(_0attributes));
    updateTmpPtr((void *)attributes, _0attributes);
    conn->write(&_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)attributes, sizeof(*attributes), true);
    mem2client(conn, (void *)ptr, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceCanAccessPeer called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0canAccessPeer;
    mem2server(conn, &_0canAccessPeer, (void *)canAccessPeer, sizeof(*canAccessPeer));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceCanAccessPeer);
    conn->write(&_0canAccessPeer, sizeof(_0canAccessPeer));
    updateTmpPtr((void *)canAccessPeer, _0canAccessPeer);
    conn->write(&device, sizeof(device));
    conn->write(&peerDevice, sizeof(peerDevice));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)canAccessPeer, sizeof(*canAccessPeer), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceEnablePeerAccess called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceEnablePeerAccess);
    conn->write(&peerDevice, sizeof(peerDevice));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceDisablePeerAccess called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceDisablePeerAccess);
    conn->write(&peerDevice, sizeof(peerDevice));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsUnregisterResource called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphicsUnregisterResource);
    conn->write(&resource, sizeof(resource));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsResourceSetMapFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphicsResourceSetMapFlags);
    conn->write(&resource, sizeof(resource));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsMapResources called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0resources;
    mem2server(conn, &_0resources, (void *)resources, sizeof(*resources));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphicsMapResources);
    conn->write(&count, sizeof(count));
    conn->write(&_0resources, sizeof(_0resources));
    updateTmpPtr((void *)resources, _0resources);
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)resources, sizeof(*resources), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsUnmapResources called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0resources;
    mem2server(conn, &_0resources, (void *)resources, sizeof(*resources));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphicsUnmapResources);
    conn->write(&count, sizeof(count));
    conn->write(&_0resources, sizeof(_0resources));
    updateTmpPtr((void *)resources, _0resources);
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)resources, sizeof(*resources), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsResourceGetMappedPointer called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0size;
    mem2server(conn, &_0size, (void *)size, sizeof(*size));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphicsResourceGetMappedPointer);
    conn->read(devPtr, sizeof(void *));
    conn->write(&_0size, sizeof(_0size));
    updateTmpPtr((void *)size, _0size);
    conn->write(&resource, sizeof(resource));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)size, sizeof(*size), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsSubResourceGetMappedArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0array;
    mem2server(conn, &_0array, (void *)array, sizeof(*array));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphicsSubResourceGetMappedArray);
    conn->write(&_0array, sizeof(_0array));
    updateTmpPtr((void *)array, _0array);
    conn->write(&resource, sizeof(resource));
    conn->write(&arrayIndex, sizeof(arrayIndex));
    conn->write(&mipLevel, sizeof(mipLevel));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)array, sizeof(*array), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsResourceGetMappedMipmappedArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0mipmappedArray;
    mem2server(conn, &_0mipmappedArray, (void *)mipmappedArray, sizeof(*mipmappedArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphicsResourceGetMappedMipmappedArray);
    conn->write(&_0mipmappedArray, sizeof(_0mipmappedArray));
    updateTmpPtr((void *)mipmappedArray, _0mipmappedArray);
    conn->write(&resource, sizeof(resource));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)mipmappedArray, sizeof(*mipmappedArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size) {
#ifdef DEBUG
    std::cout << "Hook: cudaBindTexture called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0offset;
    mem2server(conn, &_0offset, (void *)offset, sizeof(*offset));
    void *_0texref;
    mem2server(conn, &_0texref, (void *)texref, sizeof(*texref));
    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, size);
    void *_0desc;
    mem2server(conn, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaBindTexture);
    conn->write(&_0offset, sizeof(_0offset));
    updateTmpPtr((void *)offset, _0offset);
    conn->write(&_0texref, sizeof(_0texref));
    updateTmpPtr((void *)texref, _0texref);
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->write(&_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    conn->write(&size, sizeof(size));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)offset, sizeof(*offset), true);
    mem2client(conn, (void *)texref, sizeof(*texref), true);
    mem2client(conn, (void *)devPtr, size, true);
    mem2client(conn, (void *)desc, sizeof(*desc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaBindTexture2D(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, size_t pitch) {
#ifdef DEBUG
    std::cout << "Hook: cudaBindTexture2D called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0offset;
    mem2server(conn, &_0offset, (void *)offset, sizeof(*offset));
    void *_0texref;
    mem2server(conn, &_0texref, (void *)texref, sizeof(*texref));
    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, pitch * height);
    void *_0desc;
    mem2server(conn, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaBindTexture2D);
    conn->write(&_0offset, sizeof(_0offset));
    updateTmpPtr((void *)offset, _0offset);
    conn->write(&_0texref, sizeof(_0texref));
    updateTmpPtr((void *)texref, _0texref);
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->write(&_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    conn->write(&width, sizeof(width));
    conn->write(&height, sizeof(height));
    conn->write(&pitch, sizeof(pitch));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)offset, sizeof(*offset), true);
    mem2client(conn, (void *)texref, sizeof(*texref), true);
    mem2client(conn, (void *)devPtr, pitch * height, true);
    mem2client(conn, (void *)desc, sizeof(*desc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaBindTextureToArray(const struct textureReference *texref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc) {
#ifdef DEBUG
    std::cout << "Hook: cudaBindTextureToArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0texref;
    mem2server(conn, &_0texref, (void *)texref, sizeof(*texref));
    void *_0desc;
    mem2server(conn, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaBindTextureToArray);
    conn->write(&_0texref, sizeof(_0texref));
    updateTmpPtr((void *)texref, _0texref);
    conn->write(&array, sizeof(array));
    conn->write(&_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)texref, sizeof(*texref), true);
    mem2client(conn, (void *)desc, sizeof(*desc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaBindTextureToMipmappedArray(const struct textureReference *texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc *desc) {
#ifdef DEBUG
    std::cout << "Hook: cudaBindTextureToMipmappedArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0texref;
    mem2server(conn, &_0texref, (void *)texref, sizeof(*texref));
    void *_0desc;
    mem2server(conn, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaBindTextureToMipmappedArray);
    conn->write(&_0texref, sizeof(_0texref));
    updateTmpPtr((void *)texref, _0texref);
    conn->write(&mipmappedArray, sizeof(mipmappedArray));
    conn->write(&_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)texref, sizeof(*texref), true);
    mem2client(conn, (void *)desc, sizeof(*desc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaUnbindTexture(const struct textureReference *texref) {
#ifdef DEBUG
    std::cout << "Hook: cudaUnbindTexture called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0texref;
    mem2server(conn, &_0texref, (void *)texref, sizeof(*texref));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaUnbindTexture);
    conn->write(&_0texref, sizeof(_0texref));
    updateTmpPtr((void *)texref, _0texref);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)texref, sizeof(*texref), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetTextureAlignmentOffset called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0offset;
    mem2server(conn, &_0offset, (void *)offset, sizeof(*offset));
    void *_0texref;
    mem2server(conn, &_0texref, (void *)texref, sizeof(*texref));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetTextureAlignmentOffset);
    conn->write(&_0offset, sizeof(_0offset));
    updateTmpPtr((void *)offset, _0offset);
    conn->write(&_0texref, sizeof(_0texref));
    updateTmpPtr((void *)texref, _0texref);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)offset, sizeof(*offset), true);
    mem2client(conn, (void *)texref, sizeof(*texref), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetTextureReference(const struct textureReference **texref, const void *symbol) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetTextureReference called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0symbol;
    mem2server(conn, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetTextureReference);
    static struct textureReference _cudaGetTextureReference_texref;
    conn->read(&_cudaGetTextureReference_texref, sizeof(struct textureReference));
    conn->write(&_0symbol, sizeof(_0symbol));
    updateTmpPtr((void *)symbol, _0symbol);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    *texref = &_cudaGetTextureReference_texref;
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)symbol, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaBindSurfaceToArray(const struct surfaceReference *surfref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc) {
#ifdef DEBUG
    std::cout << "Hook: cudaBindSurfaceToArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0surfref;
    mem2server(conn, &_0surfref, (void *)surfref, sizeof(*surfref));
    void *_0desc;
    mem2server(conn, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaBindSurfaceToArray);
    conn->write(&_0surfref, sizeof(_0surfref));
    updateTmpPtr((void *)surfref, _0surfref);
    conn->write(&array, sizeof(array));
    conn->write(&_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)surfref, sizeof(*surfref), true);
    mem2client(conn, (void *)desc, sizeof(*desc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetSurfaceReference(const struct surfaceReference **surfref, const void *symbol) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetSurfaceReference called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0symbol;
    mem2server(conn, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetSurfaceReference);
    static struct surfaceReference _cudaGetSurfaceReference_surfref;
    conn->read(&_cudaGetSurfaceReference_surfref, sizeof(struct surfaceReference));
    conn->write(&_0symbol, sizeof(_0symbol));
    updateTmpPtr((void *)symbol, _0symbol);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    *surfref = &_cudaGetSurfaceReference_surfref;
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)symbol, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, cudaArray_const_t array) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetChannelDesc called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0desc;
    mem2server(conn, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetChannelDesc);
    conn->write(&_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    conn->write(&array, sizeof(array));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)desc, sizeof(*desc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f) {
#ifdef DEBUG
    std::cout << "Hook: cudaCreateChannelDesc called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    struct cudaChannelFormatDesc _result;
    conn->prepare_request(RPC_cudaCreateChannelDesc);
    conn->write(&x, sizeof(x));
    conn->write(&y, sizeof(y));
    conn->write(&z, sizeof(z));
    conn->write(&w, sizeof(w));
    conn->write(&f, sizeof(f));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject, const struct cudaResourceDesc *pResDesc, const struct cudaTextureDesc *pTexDesc, const struct cudaResourceViewDesc *pResViewDesc) {
#ifdef DEBUG
    std::cout << "Hook: cudaCreateTextureObject called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pTexObject;
    mem2server(conn, &_0pTexObject, (void *)pTexObject, sizeof(*pTexObject));
    void *_0pResDesc;
    mem2server(conn, &_0pResDesc, (void *)pResDesc, sizeof(*pResDesc));
    void *_0pTexDesc;
    mem2server(conn, &_0pTexDesc, (void *)pTexDesc, sizeof(*pTexDesc));
    void *_0pResViewDesc;
    mem2server(conn, &_0pResViewDesc, (void *)pResViewDesc, sizeof(*pResViewDesc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaCreateTextureObject);
    conn->write(&_0pTexObject, sizeof(_0pTexObject));
    updateTmpPtr((void *)pTexObject, _0pTexObject);
    conn->write(&_0pResDesc, sizeof(_0pResDesc));
    updateTmpPtr((void *)pResDesc, _0pResDesc);
    conn->write(&_0pTexDesc, sizeof(_0pTexDesc));
    updateTmpPtr((void *)pTexDesc, _0pTexDesc);
    conn->write(&_0pResViewDesc, sizeof(_0pResViewDesc));
    updateTmpPtr((void *)pResViewDesc, _0pResViewDesc);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pTexObject, sizeof(*pTexObject), true);
    mem2client(conn, (void *)pResDesc, sizeof(*pResDesc), true);
    mem2client(conn, (void *)pTexDesc, sizeof(*pTexDesc), true);
    mem2client(conn, (void *)pResViewDesc, sizeof(*pResViewDesc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
#ifdef DEBUG
    std::cout << "Hook: cudaDestroyTextureObject called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDestroyTextureObject);
    conn->write(&texObject, sizeof(texObject));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaTextureObject_t texObject) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetTextureObjectResourceDesc called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pResDesc;
    mem2server(conn, &_0pResDesc, (void *)pResDesc, sizeof(*pResDesc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetTextureObjectResourceDesc);
    conn->write(&_0pResDesc, sizeof(_0pResDesc));
    updateTmpPtr((void *)pResDesc, _0pResDesc);
    conn->write(&texObject, sizeof(texObject));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pResDesc, sizeof(*pResDesc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetTextureObjectTextureDesc called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pTexDesc;
    mem2server(conn, &_0pTexDesc, (void *)pTexDesc, sizeof(*pTexDesc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetTextureObjectTextureDesc);
    conn->write(&_0pTexDesc, sizeof(_0pTexDesc));
    updateTmpPtr((void *)pTexDesc, _0pTexDesc);
    conn->write(&texObject, sizeof(texObject));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pTexDesc, sizeof(*pTexDesc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetTextureObjectResourceViewDesc called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pResViewDesc;
    mem2server(conn, &_0pResViewDesc, (void *)pResViewDesc, sizeof(*pResViewDesc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetTextureObjectResourceViewDesc);
    conn->write(&_0pResViewDesc, sizeof(_0pResViewDesc));
    updateTmpPtr((void *)pResViewDesc, _0pResViewDesc);
    conn->write(&texObject, sizeof(texObject));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pResViewDesc, sizeof(*pResViewDesc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject, const struct cudaResourceDesc *pResDesc) {
#ifdef DEBUG
    std::cout << "Hook: cudaCreateSurfaceObject called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pSurfObject;
    mem2server(conn, &_0pSurfObject, (void *)pSurfObject, sizeof(*pSurfObject));
    void *_0pResDesc;
    mem2server(conn, &_0pResDesc, (void *)pResDesc, sizeof(*pResDesc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaCreateSurfaceObject);
    conn->write(&_0pSurfObject, sizeof(_0pSurfObject));
    updateTmpPtr((void *)pSurfObject, _0pSurfObject);
    conn->write(&_0pResDesc, sizeof(_0pResDesc));
    updateTmpPtr((void *)pResDesc, _0pResDesc);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pSurfObject, sizeof(*pSurfObject), true);
    mem2client(conn, (void *)pResDesc, sizeof(*pResDesc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
#ifdef DEBUG
    std::cout << "Hook: cudaDestroySurfaceObject called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDestroySurfaceObject);
    conn->write(&surfObject, sizeof(surfObject));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetSurfaceObjectResourceDesc called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pResDesc;
    mem2server(conn, &_0pResDesc, (void *)pResDesc, sizeof(*pResDesc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetSurfaceObjectResourceDesc);
    conn->write(&_0pResDesc, sizeof(_0pResDesc));
    updateTmpPtr((void *)pResDesc, _0pResDesc);
    conn->write(&surfObject, sizeof(surfObject));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pResDesc, sizeof(*pResDesc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDriverGetVersion(int *driverVersion) {
#ifdef DEBUG
    std::cout << "Hook: cudaDriverGetVersion called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0driverVersion;
    mem2server(conn, &_0driverVersion, (void *)driverVersion, sizeof(*driverVersion));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDriverGetVersion);
    conn->write(&_0driverVersion, sizeof(_0driverVersion));
    updateTmpPtr((void *)driverVersion, _0driverVersion);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)driverVersion, sizeof(*driverVersion), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
#ifdef DEBUG
    std::cout << "Hook: cudaRuntimeGetVersion called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0runtimeVersion;
    mem2server(conn, &_0runtimeVersion, (void *)runtimeVersion, sizeof(*runtimeVersion));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaRuntimeGetVersion);
    conn->write(&_0runtimeVersion, sizeof(_0runtimeVersion));
    updateTmpPtr((void *)runtimeVersion, _0runtimeVersion);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)runtimeVersion, sizeof(*runtimeVersion), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphCreate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraph;
    mem2server(conn, &_0pGraph, (void *)pGraph, sizeof(*pGraph));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphCreate);
    conn->write(&_0pGraph, sizeof(_0pGraph));
    updateTmpPtr((void *)pGraph, _0pGraph);
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraph, sizeof(*pGraph), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaKernelNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddKernelNode called" << std::endl;
#endif
    if(pNodeParams == nullptr) {
        return cudaErrorInvalidDeviceFunction;
    }
    FuncInfo *f = nullptr;
    for(auto &funcinfo : funcinfos) {
        if(funcinfo.fun_ptr == pNodeParams->func) {
            f = &funcinfo;
            break;
        }
    }
    if(f == nullptr) {
        return cudaErrorInvalidDeviceFunction;
    }

    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    for(int i = 0; i < f->param_count; i++) {
        mem2server(conn, &f->params[i].ptr, *((void **)pNodeParams->kernelParams[i]), -1);
    }
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0pNodeParams;
    mem2server(conn, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));

    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddKernelNode);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&_0pNodeParams, sizeof(_0pNodeParams));
    updateTmpPtr((void *)pNodeParams, _0pNodeParams);
    conn->write(&f->param_count, sizeof(f->param_count));
    for(int i = 0; i < f->param_count; i++) {
        conn->write(&f->params[i].ptr, f->params[i].size, true);
        updateTmpPtr(*((void **)pNodeParams->kernelParams[i]), f->params[i].ptr);
    }
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    mem2client(conn, (void *)pNodeParams, sizeof(*pNodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphKernelNodeGetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pNodeParams;
    mem2server(conn, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphKernelNodeGetParams);
    conn->write(&node, sizeof(node));
    conn->write(&_0pNodeParams, sizeof(_0pNodeParams));
    updateTmpPtr((void *)pNodeParams, _0pNodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pNodeParams, sizeof(*pNodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphKernelNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pNodeParams;
    mem2server(conn, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphKernelNodeSetParams);
    conn->write(&node, sizeof(node));
    conn->write(&_0pNodeParams, sizeof(_0pNodeParams));
    updateTmpPtr((void *)pNodeParams, _0pNodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pNodeParams, sizeof(*pNodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphKernelNodeCopyAttributes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphKernelNodeCopyAttributes);
    conn->write(&hSrc, sizeof(hSrc));
    conn->write(&hDst, sizeof(hDst));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, enum cudaKernelNodeAttrID attr, union cudaKernelNodeAttrValue *value_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphKernelNodeGetAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0value_out;
    mem2server(conn, &_0value_out, (void *)value_out, sizeof(*value_out));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphKernelNodeGetAttribute);
    conn->write(&hNode, sizeof(hNode));
    conn->write(&attr, sizeof(attr));
    conn->write(&_0value_out, sizeof(_0value_out));
    updateTmpPtr((void *)value_out, _0value_out);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)value_out, sizeof(*value_out), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, enum cudaKernelNodeAttrID attr, const union cudaKernelNodeAttrValue *value) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphKernelNodeSetAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0value;
    mem2server(conn, &_0value, (void *)value, sizeof(*value));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphKernelNodeSetAttribute);
    conn->write(&hNode, sizeof(hNode));
    conn->write(&attr, sizeof(attr));
    conn->write(&_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)value, sizeof(*value), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms *pCopyParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemcpyNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0pCopyParams;
    mem2server(conn, &_0pCopyParams, (void *)pCopyParams, sizeof(*pCopyParams));
    void *_0sptr = nullptr;
    void *_0dptr = nullptr;
    if(pCopyParams != nullptr) {
        mem2server(conn, &_0sptr, (void *)pCopyParams->srcPtr.ptr, pCopyParams->srcPtr.pitch * pCopyParams->srcPtr.ysize);
        mem2server(conn, &_0dptr, (void *)pCopyParams->dstPtr.ptr, pCopyParams->dstPtr.pitch * pCopyParams->dstPtr.ysize);
    }
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddMemcpyNode);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&_0pCopyParams, sizeof(_0pCopyParams));
    updateTmpPtr((void *)pCopyParams, _0pCopyParams);
    conn->write(&_0sptr, sizeof(_0sptr));
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)pCopyParams->srcPtr.ptr, _0sptr);
    updateTmpPtr((void *)pCopyParams->dstPtr.ptr, _0dptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    mem2client(conn, (void *)pCopyParams, sizeof(*pCopyParams), true);
    if(pCopyParams != nullptr) {
        _0sptr = (void *)pCopyParams->srcPtr.ptr;
        _0dptr = (void *)pCopyParams->dstPtr.ptr;
        mem2client(conn, (void *)pCopyParams->srcPtr.ptr, pCopyParams->srcPtr.pitch * pCopyParams->srcPtr.ysize, false);
        mem2client(conn, (void *)pCopyParams->dstPtr.ptr, pCopyParams->dstPtr.pitch * pCopyParams->dstPtr.ysize, false);
    }
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    if(pCopyParams != nullptr) {
        const_cast<void *&>(pCopyParams->srcPtr.ptr) = _0sptr;
        const_cast<void *&>(pCopyParams->dstPtr.ptr) = _0dptr;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemcpyNodeToSymbol called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0symbol;
    mem2server(conn, &_0symbol, (void *)symbol, -1);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddMemcpyNodeToSymbol);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&_0symbol, sizeof(_0symbol));
    updateTmpPtr((void *)symbol, _0symbol);
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&count, sizeof(count));
    conn->write(&offset, sizeof(offset));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    mem2client(conn, (void *)symbol, -1, true);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemcpyNodeFromSymbol called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *_0symbol;
    mem2server(conn, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddMemcpyNodeFromSymbol);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&_0symbol, sizeof(_0symbol));
    updateTmpPtr((void *)symbol, _0symbol);
    conn->write(&count, sizeof(count));
    conn->write(&offset, sizeof(offset));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    mem2client(conn, (void *)dst, count, true);
    mem2client(conn, (void *)symbol, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemcpyNode1D called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddMemcpyNode1D);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&count, sizeof(count));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    mem2client(conn, (void *)dst, count, true);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemcpyNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pNodeParams;
    mem2server(conn, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *_0sptr = nullptr;
    void *_0dptr = nullptr;
    if(pNodeParams != nullptr) {
        mem2server(conn, &_0sptr, (void *)pNodeParams->srcPtr.ptr, sizeof(pNodeParams->srcPtr.pitch * pNodeParams->srcPtr.ysize));
        mem2server(conn, &_0dptr, (void *)pNodeParams->dstPtr.ptr, sizeof(pNodeParams->dstPtr.pitch * pNodeParams->dstPtr.ysize));
    }
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphMemcpyNodeSetParams);
    conn->write(&node, sizeof(node));
    conn->write(&_0pNodeParams, sizeof(_0pNodeParams));
    updateTmpPtr((void *)pNodeParams, _0pNodeParams);
    conn->write(&_0sptr, sizeof(_0sptr));
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)pNodeParams->srcPtr.ptr, _0sptr);
    updateTmpPtr((void *)pNodeParams->dstPtr.ptr, _0dptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pNodeParams, sizeof(*pNodeParams), true);
    if(pNodeParams != nullptr) {
        _0sptr = (void *)pNodeParams->srcPtr.ptr;
        _0dptr = (void *)pNodeParams->dstPtr.ptr;
        mem2client(conn, (void *)pNodeParams->srcPtr.ptr, sizeof(pNodeParams->srcPtr.pitch * pNodeParams->srcPtr.ysize), false);
        mem2client(conn, (void *)pNodeParams->dstPtr.ptr, sizeof(pNodeParams->dstPtr.pitch * pNodeParams->dstPtr.ysize), false);
    }
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    if(pNodeParams != nullptr) {
        const_cast<void *&>(pNodeParams->srcPtr.ptr) = _0sptr;
        const_cast<void *&>(pNodeParams->dstPtr.ptr) = _0dptr;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemcpyNodeSetParamsToSymbol called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0symbol;
    mem2server(conn, &_0symbol, (void *)symbol, -1);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphMemcpyNodeSetParamsToSymbol);
    conn->write(&node, sizeof(node));
    conn->write(&_0symbol, sizeof(_0symbol));
    updateTmpPtr((void *)symbol, _0symbol);
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&count, sizeof(count));
    conn->write(&offset, sizeof(offset));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)symbol, -1, true);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemcpyNodeSetParamsFromSymbol called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *_0symbol;
    mem2server(conn, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphMemcpyNodeSetParamsFromSymbol);
    conn->write(&node, sizeof(node));
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&_0symbol, sizeof(_0symbol));
    updateTmpPtr((void *)symbol, _0symbol);
    conn->write(&count, sizeof(count));
    conn->write(&offset, sizeof(offset));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, count, true);
    mem2client(conn, (void *)symbol, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemcpyNodeSetParams1D called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphMemcpyNodeSetParams1D);
    conn->write(&node, sizeof(node));
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&count, sizeof(count));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, count, true);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemsetParams *pMemsetParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemsetNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0pMemsetParams;
    mem2server(conn, &_0pMemsetParams, (void *)pMemsetParams, sizeof(*pMemsetParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddMemsetNode);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&_0pMemsetParams, sizeof(_0pMemsetParams));
    updateTmpPtr((void *)pMemsetParams, _0pMemsetParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    mem2client(conn, (void *)pMemsetParams, sizeof(*pMemsetParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, struct cudaMemsetParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemsetNodeGetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pNodeParams;
    mem2server(conn, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphMemsetNodeGetParams);
    conn->write(&node, sizeof(node));
    conn->write(&_0pNodeParams, sizeof(_0pNodeParams));
    updateTmpPtr((void *)pNodeParams, _0pNodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pNodeParams, sizeof(*pNodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemsetNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pNodeParams;
    mem2server(conn, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphMemsetNodeSetParams);
    conn->write(&node, sizeof(node));
    conn->write(&_0pNodeParams, sizeof(_0pNodeParams));
    updateTmpPtr((void *)pNodeParams, _0pNodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pNodeParams, sizeof(*pNodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaHostNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddHostNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0pNodeParams;
    mem2server(conn, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddHostNode);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&_0pNodeParams, sizeof(_0pNodeParams));
    updateTmpPtr((void *)pNodeParams, _0pNodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    mem2client(conn, (void *)pNodeParams, sizeof(*pNodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, struct cudaHostNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphHostNodeGetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pNodeParams;
    mem2server(conn, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphHostNodeGetParams);
    conn->write(&node, sizeof(node));
    conn->write(&_0pNodeParams, sizeof(_0pNodeParams));
    updateTmpPtr((void *)pNodeParams, _0pNodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pNodeParams, sizeof(*pNodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphHostNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pNodeParams;
    mem2server(conn, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphHostNodeSetParams);
    conn->write(&node, sizeof(node));
    conn->write(&_0pNodeParams, sizeof(_0pNodeParams));
    updateTmpPtr((void *)pNodeParams, _0pNodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pNodeParams, sizeof(*pNodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaGraph_t childGraph) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddChildGraphNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddChildGraphNode);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&childGraph, sizeof(childGraph));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t *pGraph) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphChildGraphNodeGetGraph called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraph;
    mem2server(conn, &_0pGraph, (void *)pGraph, sizeof(*pGraph));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphChildGraphNodeGetGraph);
    conn->write(&node, sizeof(node));
    conn->write(&_0pGraph, sizeof(_0pGraph));
    updateTmpPtr((void *)pGraph, _0pGraph);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraph, sizeof(*pGraph), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddEmptyNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddEmptyNode);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddEventRecordNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddEventRecordNode);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&event, sizeof(event));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphEventRecordNodeGetEvent called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0event_out;
    mem2server(conn, &_0event_out, (void *)event_out, sizeof(*event_out));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphEventRecordNodeGetEvent);
    conn->write(&node, sizeof(node));
    conn->write(&_0event_out, sizeof(_0event_out));
    updateTmpPtr((void *)event_out, _0event_out);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)event_out, sizeof(*event_out), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphEventRecordNodeSetEvent called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphEventRecordNodeSetEvent);
    conn->write(&node, sizeof(node));
    conn->write(&event, sizeof(event));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddEventWaitNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddEventWaitNode);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&event, sizeof(event));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphEventWaitNodeGetEvent called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0event_out;
    mem2server(conn, &_0event_out, (void *)event_out, sizeof(*event_out));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphEventWaitNodeGetEvent);
    conn->write(&node, sizeof(node));
    conn->write(&_0event_out, sizeof(_0event_out));
    updateTmpPtr((void *)event_out, _0event_out);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)event_out, sizeof(*event_out), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphEventWaitNodeSetEvent called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphEventWaitNodeSetEvent);
    conn->write(&node, sizeof(node));
    conn->write(&event, sizeof(event));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddExternalSemaphoresSignalNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0nodeParams;
    mem2server(conn, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddExternalSemaphoresSignalNode);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    mem2client(conn, (void *)nodeParams, sizeof(*nodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreSignalNodeParams *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExternalSemaphoresSignalNodeGetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0params_out;
    mem2server(conn, &_0params_out, (void *)params_out, sizeof(*params_out));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExternalSemaphoresSignalNodeGetParams);
    conn->write(&hNode, sizeof(hNode));
    conn->write(&_0params_out, sizeof(_0params_out));
    updateTmpPtr((void *)params_out, _0params_out);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)params_out, sizeof(*params_out), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExternalSemaphoresSignalNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0nodeParams;
    mem2server(conn, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExternalSemaphoresSignalNodeSetParams);
    conn->write(&hNode, sizeof(hNode));
    conn->write(&_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)nodeParams, sizeof(*nodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddExternalSemaphoresWaitNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0nodeParams;
    mem2server(conn, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddExternalSemaphoresWaitNode);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    mem2client(conn, (void *)nodeParams, sizeof(*nodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreWaitNodeParams *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExternalSemaphoresWaitNodeGetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0params_out;
    mem2server(conn, &_0params_out, (void *)params_out, sizeof(*params_out));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExternalSemaphoresWaitNodeGetParams);
    conn->write(&hNode, sizeof(hNode));
    conn->write(&_0params_out, sizeof(_0params_out));
    updateTmpPtr((void *)params_out, _0params_out);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)params_out, sizeof(*params_out), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExternalSemaphoresWaitNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0nodeParams;
    mem2server(conn, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExternalSemaphoresWaitNodeSetParams);
    conn->write(&hNode, sizeof(hNode));
    conn->write(&_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)nodeParams, sizeof(*nodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, struct cudaMemAllocNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemAllocNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0nodeParams;
    mem2server(conn, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddMemAllocNode);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    mem2client(conn, (void *)nodeParams, sizeof(*nodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, struct cudaMemAllocNodeParams *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemAllocNodeGetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0params_out;
    mem2server(conn, &_0params_out, (void *)params_out, sizeof(*params_out));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphMemAllocNodeGetParams);
    conn->write(&node, sizeof(node));
    conn->write(&_0params_out, sizeof(_0params_out));
    updateTmpPtr((void *)params_out, _0params_out);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)params_out, sizeof(*params_out), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dptr) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemFreeNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphNode;
    mem2server(conn, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0dptr;
    mem2server(conn, &_0dptr, (void *)dptr, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddMemFreeNode);
    conn->write(&_0pGraphNode, sizeof(_0pGraphNode));
    updateTmpPtr((void *)pGraphNode, _0pGraphNode);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphNode, sizeof(*pGraphNode), true);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    mem2client(conn, (void *)dptr, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void *dptr_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemFreeNodeGetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dptr_out;
    mem2server(conn, &_0dptr_out, (void *)dptr_out, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphMemFreeNodeGetParams);
    conn->write(&node, sizeof(node));
    conn->write(&_0dptr_out, sizeof(_0dptr_out));
    updateTmpPtr((void *)dptr_out, _0dptr_out);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dptr_out, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGraphMemTrim(int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGraphMemTrim called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGraphMemTrim);
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetGraphMemAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0value;
    mem2server(conn, &_0value, (void *)value, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceGetGraphMemAttribute);
    conn->write(&device, sizeof(device));
    conn->write(&attr, sizeof(attr));
    conn->write(&_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)value, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaDeviceSetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceSetGraphMemAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0value;
    mem2server(conn, &_0value, (void *)value, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaDeviceSetGraphMemAttribute);
    conn->write(&device, sizeof(device));
    conn->write(&attr, sizeof(attr));
    conn->write(&_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)value, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphClone called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphClone;
    mem2server(conn, &_0pGraphClone, (void *)pGraphClone, sizeof(*pGraphClone));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphClone);
    conn->write(&_0pGraphClone, sizeof(_0pGraphClone));
    updateTmpPtr((void *)pGraphClone, _0pGraphClone);
    conn->write(&originalGraph, sizeof(originalGraph));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphClone, sizeof(*pGraphClone), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphNodeFindInClone called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pNode;
    mem2server(conn, &_0pNode, (void *)pNode, sizeof(*pNode));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphNodeFindInClone);
    conn->write(&_0pNode, sizeof(_0pNode));
    updateTmpPtr((void *)pNode, _0pNode);
    conn->write(&originalNode, sizeof(originalNode));
    conn->write(&clonedGraph, sizeof(clonedGraph));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pNode, sizeof(*pNode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType *pType) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphNodeGetType called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pType;
    mem2server(conn, &_0pType, (void *)pType, sizeof(*pType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphNodeGetType);
    conn->write(&node, sizeof(node));
    conn->write(&_0pType, sizeof(_0pType));
    updateTmpPtr((void *)pType, _0pType);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pType, sizeof(*pType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphGetNodes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0nodes;
    mem2server(conn, &_0nodes, (void *)nodes, sizeof(*nodes));
    void *_0numNodes;
    mem2server(conn, &_0numNodes, (void *)numNodes, sizeof(*numNodes));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphGetNodes);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0nodes, sizeof(_0nodes));
    updateTmpPtr((void *)nodes, _0nodes);
    conn->write(&_0numNodes, sizeof(_0numNodes));
    updateTmpPtr((void *)numNodes, _0numNodes);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)nodes, sizeof(*nodes), true);
    mem2client(conn, (void *)numNodes, sizeof(*numNodes), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphGetRootNodes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pRootNodes;
    mem2server(conn, &_0pRootNodes, (void *)pRootNodes, sizeof(*pRootNodes));
    void *_0pNumRootNodes;
    mem2server(conn, &_0pNumRootNodes, (void *)pNumRootNodes, sizeof(*pNumRootNodes));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphGetRootNodes);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pRootNodes, sizeof(_0pRootNodes));
    updateTmpPtr((void *)pRootNodes, _0pRootNodes);
    conn->write(&_0pNumRootNodes, sizeof(_0pNumRootNodes));
    updateTmpPtr((void *)pNumRootNodes, _0pNumRootNodes);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pRootNodes, sizeof(*pRootNodes), true);
    mem2client(conn, (void *)pNumRootNodes, sizeof(*pNumRootNodes), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphGetEdges called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0from;
    mem2server(conn, &_0from, (void *)from, sizeof(*from));
    void *_0to;
    mem2server(conn, &_0to, (void *)to, sizeof(*to));
    void *_0numEdges;
    mem2server(conn, &_0numEdges, (void *)numEdges, sizeof(*numEdges));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphGetEdges);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0from, sizeof(_0from));
    updateTmpPtr((void *)from, _0from);
    conn->write(&_0to, sizeof(_0to));
    updateTmpPtr((void *)to, _0to);
    conn->write(&_0numEdges, sizeof(_0numEdges));
    updateTmpPtr((void *)numEdges, _0numEdges);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)from, sizeof(*from), true);
    mem2client(conn, (void *)to, sizeof(*to), true);
    mem2client(conn, (void *)numEdges, sizeof(*numEdges), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies, size_t *pNumDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphNodeGetDependencies called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pDependencies;
    mem2server(conn, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0pNumDependencies;
    mem2server(conn, &_0pNumDependencies, (void *)pNumDependencies, sizeof(*pNumDependencies));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphNodeGetDependencies);
    conn->write(&node, sizeof(node));
    conn->write(&_0pDependencies, sizeof(_0pDependencies));
    updateTmpPtr((void *)pDependencies, _0pDependencies);
    conn->write(&_0pNumDependencies, sizeof(_0pNumDependencies));
    updateTmpPtr((void *)pNumDependencies, _0pNumDependencies);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pDependencies, sizeof(*pDependencies), true);
    mem2client(conn, (void *)pNumDependencies, sizeof(*pNumDependencies), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes, size_t *pNumDependentNodes) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphNodeGetDependentNodes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pDependentNodes;
    mem2server(conn, &_0pDependentNodes, (void *)pDependentNodes, sizeof(*pDependentNodes));
    void *_0pNumDependentNodes;
    mem2server(conn, &_0pNumDependentNodes, (void *)pNumDependentNodes, sizeof(*pNumDependentNodes));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphNodeGetDependentNodes);
    conn->write(&node, sizeof(node));
    conn->write(&_0pDependentNodes, sizeof(_0pDependentNodes));
    updateTmpPtr((void *)pDependentNodes, _0pDependentNodes);
    conn->write(&_0pNumDependentNodes, sizeof(_0pNumDependentNodes));
    updateTmpPtr((void *)pNumDependentNodes, _0pNumDependentNodes);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pDependentNodes, sizeof(*pDependentNodes), true);
    mem2client(conn, (void *)pNumDependentNodes, sizeof(*pNumDependentNodes), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddDependencies called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0from;
    mem2server(conn, &_0from, (void *)from, sizeof(*from));
    void *_0to;
    mem2server(conn, &_0to, (void *)to, sizeof(*to));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphAddDependencies);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0from, sizeof(_0from));
    updateTmpPtr((void *)from, _0from);
    conn->write(&_0to, sizeof(_0to));
    updateTmpPtr((void *)to, _0to);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)from, sizeof(*from), true);
    mem2client(conn, (void *)to, sizeof(*to), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphRemoveDependencies called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0from;
    mem2server(conn, &_0from, (void *)from, sizeof(*from));
    void *_0to;
    mem2server(conn, &_0to, (void *)to, sizeof(*to));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphRemoveDependencies);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0from, sizeof(_0from));
    updateTmpPtr((void *)from, _0from);
    conn->write(&_0to, sizeof(_0to));
    updateTmpPtr((void *)to, _0to);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)from, sizeof(*from), true);
    mem2client(conn, (void *)to, sizeof(*to), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphDestroyNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphDestroyNode);
    conn->write(&node, sizeof(node));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphInstantiate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphExec;
    mem2server(conn, &_0pGraphExec, (void *)pGraphExec, sizeof(*pGraphExec));
    void *_0pErrorNode;
    mem2server(conn, &_0pErrorNode, (void *)pErrorNode, sizeof(*pErrorNode));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphInstantiate);
    conn->write(&_0pGraphExec, sizeof(_0pGraphExec));
    updateTmpPtr((void *)pGraphExec, _0pGraphExec);
    conn->write(&graph, sizeof(graph));
    conn->write(&_0pErrorNode, sizeof(_0pErrorNode));
    updateTmpPtr((void *)pErrorNode, _0pErrorNode);
    if(bufferSize > 0) {
        conn->read(pLogBuffer, bufferSize, true);
    }
    conn->write(&bufferSize, sizeof(bufferSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphExec, sizeof(*pGraphExec), true);
    mem2client(conn, (void *)pErrorNode, sizeof(*pErrorNode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphInstantiateWithFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGraphExec;
    mem2server(conn, &_0pGraphExec, (void *)pGraphExec, sizeof(*pGraphExec));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphInstantiateWithFlags);
    conn->write(&_0pGraphExec, sizeof(_0pGraphExec));
    updateTmpPtr((void *)pGraphExec, _0pGraphExec);
    conn->write(&graph, sizeof(graph));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGraphExec, sizeof(*pGraphExec), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecKernelNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pNodeParams;
    mem2server(conn, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecKernelNodeSetParams);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&node, sizeof(node));
    conn->write(&_0pNodeParams, sizeof(_0pNodeParams));
    updateTmpPtr((void *)pNodeParams, _0pNodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pNodeParams, sizeof(*pNodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecMemcpyNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pNodeParams;
    mem2server(conn, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *_0sptr = nullptr;
    void *_0dptr = nullptr;
    if(pNodeParams != nullptr) {
        mem2server(conn, &_0sptr, (void *)pNodeParams->srcPtr.ptr, sizeof(pNodeParams->srcPtr.pitch * pNodeParams->srcPtr.ysize));
        mem2server(conn, &_0dptr, (void *)pNodeParams->dstPtr.ptr, sizeof(pNodeParams->dstPtr.pitch * pNodeParams->dstPtr.ysize));
    }
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecMemcpyNodeSetParams);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&node, sizeof(node));
    conn->write(&_0pNodeParams, sizeof(_0pNodeParams));
    updateTmpPtr((void *)pNodeParams, _0pNodeParams);
    conn->write(&_0sptr, sizeof(_0sptr));
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)pNodeParams->srcPtr.ptr, _0sptr);
    updateTmpPtr((void *)pNodeParams->dstPtr.ptr, _0dptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pNodeParams, sizeof(*pNodeParams), true);
    if(pNodeParams != nullptr) {
        _0sptr = (void *)pNodeParams->srcPtr.ptr;
        _0dptr = (void *)pNodeParams->dstPtr.ptr;
        mem2client(conn, (void *)pNodeParams->srcPtr.ptr, sizeof(pNodeParams->srcPtr.pitch * pNodeParams->srcPtr.ysize), false);
        mem2client(conn, (void *)pNodeParams->dstPtr.ptr, sizeof(pNodeParams->dstPtr.pitch * pNodeParams->dstPtr.ysize), false);
    }
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    if(pNodeParams != nullptr) {
        const_cast<void *&>(pNodeParams->srcPtr.ptr) = _0sptr;
        const_cast<void *&>(pNodeParams->dstPtr.ptr) = _0dptr;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecMemcpyNodeSetParamsToSymbol called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0symbol;
    mem2server(conn, &_0symbol, (void *)symbol, -1);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecMemcpyNodeSetParamsToSymbol);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&node, sizeof(node));
    conn->write(&_0symbol, sizeof(_0symbol));
    updateTmpPtr((void *)symbol, _0symbol);
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&count, sizeof(count));
    conn->write(&offset, sizeof(offset));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)symbol, -1, true);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecMemcpyNodeSetParamsFromSymbol called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *_0symbol;
    mem2server(conn, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecMemcpyNodeSetParamsFromSymbol);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&node, sizeof(node));
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&_0symbol, sizeof(_0symbol));
    updateTmpPtr((void *)symbol, _0symbol);
    conn->write(&count, sizeof(count));
    conn->write(&offset, sizeof(offset));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, count, true);
    mem2client(conn, (void *)symbol, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecMemcpyNodeSetParams1D called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecMemcpyNodeSetParams1D);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&node, sizeof(node));
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&count, sizeof(count));
    conn->write(&kind, sizeof(kind));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, count, true);
    mem2client(conn, (void *)src, count, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecMemsetNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pNodeParams;
    mem2server(conn, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecMemsetNodeSetParams);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&node, sizeof(node));
    conn->write(&_0pNodeParams, sizeof(_0pNodeParams));
    updateTmpPtr((void *)pNodeParams, _0pNodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pNodeParams, sizeof(*pNodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecHostNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pNodeParams;
    mem2server(conn, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecHostNodeSetParams);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&node, sizeof(node));
    conn->write(&_0pNodeParams, sizeof(_0pNodeParams));
    updateTmpPtr((void *)pNodeParams, _0pNodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pNodeParams, sizeof(*pNodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecChildGraphNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecChildGraphNodeSetParams);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&node, sizeof(node));
    conn->write(&childGraph, sizeof(childGraph));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecEventRecordNodeSetEvent called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecEventRecordNodeSetEvent);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&hNode, sizeof(hNode));
    conn->write(&event, sizeof(event));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecEventWaitNodeSetEvent called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecEventWaitNodeSetEvent);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&hNode, sizeof(hNode));
    conn->write(&event, sizeof(event));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecExternalSemaphoresSignalNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0nodeParams;
    mem2server(conn, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecExternalSemaphoresSignalNodeSetParams);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&hNode, sizeof(hNode));
    conn->write(&_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)nodeParams, sizeof(*nodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecExternalSemaphoresWaitNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0nodeParams;
    mem2server(conn, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecExternalSemaphoresWaitNodeSetParams);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&hNode, sizeof(hNode));
    conn->write(&_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)nodeParams, sizeof(*nodeParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t *hErrorNode_out, enum cudaGraphExecUpdateResult *updateResult_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecUpdate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0hErrorNode_out;
    mem2server(conn, &_0hErrorNode_out, (void *)hErrorNode_out, sizeof(*hErrorNode_out));
    void *_0updateResult_out;
    mem2server(conn, &_0updateResult_out, (void *)updateResult_out, sizeof(*updateResult_out));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecUpdate);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0hErrorNode_out, sizeof(_0hErrorNode_out));
    updateTmpPtr((void *)hErrorNode_out, _0hErrorNode_out);
    conn->write(&_0updateResult_out, sizeof(_0updateResult_out));
    updateTmpPtr((void *)updateResult_out, _0updateResult_out);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)hErrorNode_out, sizeof(*hErrorNode_out), true);
    mem2client(conn, (void *)updateResult_out, sizeof(*updateResult_out), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphUpload called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphUpload);
    conn->write(&graphExec, sizeof(graphExec));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphLaunch called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphLaunch);
    conn->write(&graphExec, sizeof(graphExec));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecDestroy called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphExecDestroy);
    conn->write(&graphExec, sizeof(graphExec));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphDestroy called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphDestroy);
    conn->write(&graph, sizeof(graph));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphDebugDotPrint called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphDebugDotPrint);
    conn->write(&graph, sizeof(graph));
    conn->write(path, strlen(path) + 1, true);
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaUserObjectCreate(cudaUserObject_t *object_out, void *ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaUserObjectCreate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0object_out;
    mem2server(conn, &_0object_out, (void *)object_out, sizeof(*object_out));
    void *_0ptr;
    mem2server(conn, &_0ptr, (void *)ptr, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaUserObjectCreate);
    conn->write(&_0object_out, sizeof(_0object_out));
    updateTmpPtr((void *)object_out, _0object_out);
    conn->write(&_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    conn->write(&destroy, sizeof(destroy));
    conn->write(&initialRefcount, sizeof(initialRefcount));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)object_out, sizeof(*object_out), true);
    mem2client(conn, (void *)ptr, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cudaUserObjectRetain called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaUserObjectRetain);
    conn->write(&object, sizeof(object));
    conn->write(&count, sizeof(count));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cudaUserObjectRelease called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaUserObjectRelease);
    conn->write(&object, sizeof(object));
    conn->write(&count, sizeof(count));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphRetainUserObject called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphRetainUserObject);
    conn->write(&graph, sizeof(graph));
    conn->write(&object, sizeof(object));
    conn->write(&count, sizeof(count));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphReleaseUserObject called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphReleaseUserObject);
    conn->write(&graph, sizeof(graph));
    conn->write(&object, sizeof(object));
    conn->write(&count, sizeof(count));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetDriverEntryPoint called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetDriverEntryPoint);
    conn->write(symbol, strlen(symbol) + 1, true);
    conn->read(funcPtr, sizeof(void *));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetExportTable called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pExportTableId;
    mem2server(conn, &_0pExportTableId, (void *)pExportTableId, sizeof(*pExportTableId));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetExportTable);
    conn->read(ppExportTable, sizeof(*ppExportTable));
    conn->write(&_0pExportTableId, sizeof(_0pExportTableId));
    updateTmpPtr((void *)pExportTableId, _0pExportTableId);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pExportTableId, sizeof(*pExportTableId), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGetFuncBySymbol(cudaFunction_t *functionPtr, const void *symbolPtr) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetFuncBySymbol called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0functionPtr;
    mem2server(conn, &_0functionPtr, (void *)functionPtr, sizeof(*functionPtr));
    void *_0symbolPtr;
    mem2server(conn, &_0symbolPtr, (void *)symbolPtr, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGetFuncBySymbol);
    conn->write(&_0functionPtr, sizeof(_0functionPtr));
    updateTmpPtr((void *)functionPtr, _0functionPtr);
    conn->write(&_0symbolPtr, sizeof(_0symbolPtr));
    updateTmpPtr((void *)symbolPtr, _0symbolPtr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)functionPtr, sizeof(*functionPtr), true);
    mem2client(conn, (void *)symbolPtr, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}
