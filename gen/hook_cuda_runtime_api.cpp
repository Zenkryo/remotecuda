#include <iostream>
#include <unordered_map>
#include "cuda_runtime_api.h"

#include "hook_api.h"
#include "../rpc.h"
extern void *(*real_dlsym)(void *, const char *);

extern "C" void mem2server(RpcClient *client, void **serverPtr, void *clientPtr, ssize_t size);
extern "C" void mem2client(RpcClient *client, void *clientPtr, ssize_t size);
void *get_so_handle(const std::string &so_file);
extern "C" cudaError_t cudaDeviceReset() {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceReset called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceReset);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceSynchronize() {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceSynchronize called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceSynchronize);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceSetLimit called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceSetLimit);
    rpc_write(client, &limit, sizeof(limit));
    rpc_write(client, &value, sizeof(value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetLimit called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pValue;
    mem2server(client, &_0pValue, (void *)pValue, sizeof(*pValue));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGetLimit);
    rpc_write(client, &_0pValue, sizeof(_0pValue));
    rpc_write(client, &limit, sizeof(limit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pValue, sizeof(*pValue));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, const struct cudaChannelFormatDesc *fmtDesc, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetTexture1DLinearMaxWidth called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0maxWidthInElements;
    mem2server(client, &_0maxWidthInElements, (void *)maxWidthInElements, sizeof(*maxWidthInElements));
    void *_0fmtDesc;
    mem2server(client, &_0fmtDesc, (void *)fmtDesc, sizeof(*fmtDesc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGetTexture1DLinearMaxWidth);
    rpc_write(client, &_0maxWidthInElements, sizeof(_0maxWidthInElements));
    rpc_write(client, &_0fmtDesc, sizeof(_0fmtDesc));
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)maxWidthInElements, sizeof(*maxWidthInElements));
    mem2client(client, (void *)fmtDesc, sizeof(*fmtDesc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetCacheConfig called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pCacheConfig;
    mem2server(client, &_0pCacheConfig, (void *)pCacheConfig, sizeof(*pCacheConfig));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGetCacheConfig);
    rpc_write(client, &_0pCacheConfig, sizeof(_0pCacheConfig));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pCacheConfig, sizeof(*pCacheConfig));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetStreamPriorityRange called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0leastPriority;
    mem2server(client, &_0leastPriority, (void *)leastPriority, sizeof(*leastPriority));
    void *_0greatestPriority;
    mem2server(client, &_0greatestPriority, (void *)greatestPriority, sizeof(*greatestPriority));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGetStreamPriorityRange);
    rpc_write(client, &_0leastPriority, sizeof(_0leastPriority));
    rpc_write(client, &_0greatestPriority, sizeof(_0greatestPriority));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)leastPriority, sizeof(*leastPriority));
    mem2client(client, (void *)greatestPriority, sizeof(*greatestPriority));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceSetCacheConfig called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceSetCacheConfig);
    rpc_write(client, &cacheConfig, sizeof(cacheConfig));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetSharedMemConfig called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pConfig;
    mem2server(client, &_0pConfig, (void *)pConfig, sizeof(*pConfig));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGetSharedMemConfig);
    rpc_write(client, &_0pConfig, sizeof(_0pConfig));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pConfig, sizeof(*pConfig));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceSetSharedMemConfig called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceSetSharedMemConfig);
    rpc_write(client, &config, sizeof(config));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetByPCIBusId called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0device;
    mem2server(client, &_0device, (void *)device, sizeof(*device));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGetByPCIBusId);
    rpc_write(client, &_0device, sizeof(_0device));
    rpc_write(client, pciBusId, strlen(pciBusId) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)device, sizeof(*device));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetPCIBusId called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGetPCIBusId);
    if(len > 0) {
        rpc_read(client, pciBusId, len, true);
    }
    rpc_write(client, &len, sizeof(len));
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaIpcGetEventHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0handle;
    mem2server(client, &_0handle, (void *)handle, sizeof(*handle));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaIpcGetEventHandle);
    rpc_write(client, &_0handle, sizeof(_0handle));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)handle, sizeof(*handle));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle) {
#ifdef DEBUG
    std::cout << "Hook: cudaIpcOpenEventHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0event;
    mem2server(client, &_0event, (void *)event, sizeof(*event));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaIpcOpenEventHandle);
    rpc_write(client, &_0event, sizeof(_0event));
    rpc_write(client, &handle, sizeof(handle));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)event, sizeof(*event));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr) {
#ifdef DEBUG
    std::cout << "Hook: cudaIpcGetMemHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0handle;
    mem2server(client, &_0handle, (void *)handle, sizeof(*handle));
    void *_0devPtr;
    mem2server(client, &_0devPtr, (void *)devPtr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaIpcGetMemHandle);
    rpc_write(client, &_0handle, sizeof(_0handle));
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)handle, sizeof(*handle));
    mem2client(client, (void *)devPtr, -1);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaIpcOpenMemHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaIpcOpenMemHandle);
    rpc_read(client, devPtr, sizeof(void *));
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaIpcCloseMemHandle(void *devPtr) {
#ifdef DEBUG
    std::cout << "Hook: cudaIpcCloseMemHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0devPtr;
    mem2server(client, &_0devPtr, (void *)devPtr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaIpcCloseMemHandle);
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)devPtr, -1);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(enum cudaFlushGPUDirectRDMAWritesTarget target, enum cudaFlushGPUDirectRDMAWritesScope scope) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceFlushGPUDirectRDMAWrites called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceFlushGPUDirectRDMAWrites);
    rpc_write(client, &target, sizeof(target));
    rpc_write(client, &scope, sizeof(scope));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaThreadExit() {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadExit called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaThreadExit);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaThreadSynchronize() {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadSynchronize called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaThreadSynchronize);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value) {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadSetLimit called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaThreadSetLimit);
    rpc_write(client, &limit, sizeof(limit));
    rpc_write(client, &value, sizeof(value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit) {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadGetLimit called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pValue;
    mem2server(client, &_0pValue, (void *)pValue, sizeof(*pValue));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaThreadGetLimit);
    rpc_write(client, &_0pValue, sizeof(_0pValue));
    rpc_write(client, &limit, sizeof(limit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pValue, sizeof(*pValue));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadGetCacheConfig called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pCacheConfig;
    mem2server(client, &_0pCacheConfig, (void *)pCacheConfig, sizeof(*pCacheConfig));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaThreadGetCacheConfig);
    rpc_write(client, &_0pCacheConfig, sizeof(_0pCacheConfig));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pCacheConfig, sizeof(*pCacheConfig));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig) {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadSetCacheConfig called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaThreadSetCacheConfig);
    rpc_write(client, &cacheConfig, sizeof(cacheConfig));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetLastError() {
#ifdef DEBUG
    std::cout << "Hook: cudaGetLastError called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetLastError);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaPeekAtLastError() {
#ifdef DEBUG
    std::cout << "Hook: cudaPeekAtLastError called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaPeekAtLastError);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetDeviceCount(int *count) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetDeviceCount called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0count;
    mem2server(client, &_0count, (void *)count, sizeof(*count));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetDeviceCount);
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)count, sizeof(*count));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetDeviceProperties called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0prop;
    mem2server(client, &_0prop, (void *)prop, sizeof(*prop));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetDeviceProperties);
    rpc_write(client, &_0prop, sizeof(_0prop));
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)prop, sizeof(*prop));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0value;
    mem2server(client, &_0value, (void *)value, sizeof(*value));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGetAttribute);
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, sizeof(*value));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetDefaultMemPool called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0memPool;
    mem2server(client, &_0memPool, (void *)memPool, sizeof(*memPool));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGetDefaultMemPool);
    rpc_write(client, &_0memPool, sizeof(_0memPool));
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)memPool, sizeof(*memPool));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceSetMemPool called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceSetMemPool);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &memPool, sizeof(memPool));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetMemPool called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0memPool;
    mem2server(client, &_0memPool, (void *)memPool, sizeof(*memPool));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGetMemPool);
    rpc_write(client, &_0memPool, sizeof(_0memPool));
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)memPool, sizeof(*memPool));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, int device, int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetNvSciSyncAttributes called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0nvSciSyncAttrList;
    mem2server(client, &_0nvSciSyncAttrList, (void *)nvSciSyncAttrList, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGetNvSciSyncAttributes);
    rpc_write(client, &_0nvSciSyncAttrList, sizeof(_0nvSciSyncAttrList));
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nvSciSyncAttrList, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetP2PAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0value;
    mem2server(client, &_0value, (void *)value, sizeof(*value));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGetP2PAttribute);
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &srcDevice, sizeof(srcDevice));
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, sizeof(*value));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
#ifdef DEBUG
    std::cout << "Hook: cudaChooseDevice called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0device;
    mem2server(client, &_0device, (void *)device, sizeof(*device));
    void *_0prop;
    mem2server(client, &_0prop, (void *)prop, sizeof(*prop));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaChooseDevice);
    rpc_write(client, &_0device, sizeof(_0device));
    rpc_write(client, &_0prop, sizeof(_0prop));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)device, sizeof(*device));
    mem2client(client, (void *)prop, sizeof(*prop));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaSetDevice(int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaSetDevice called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaSetDevice);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetDevice(int *device) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetDevice called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0device;
    mem2server(client, &_0device, (void *)device, sizeof(*device));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetDevice);
    rpc_write(client, &_0device, sizeof(_0device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)device, sizeof(*device));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaSetValidDevices(int *device_arr, int len) {
#ifdef DEBUG
    std::cout << "Hook: cudaSetValidDevices called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0device_arr;
    mem2server(client, &_0device_arr, (void *)device_arr, sizeof(*device_arr));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaSetValidDevices);
    rpc_write(client, &_0device_arr, sizeof(_0device_arr));
    rpc_write(client, &len, sizeof(len));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)device_arr, sizeof(*device_arr));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaSetDeviceFlags(unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaSetDeviceFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaSetDeviceFlags);
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetDeviceFlags(unsigned int *flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetDeviceFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0flags;
    mem2server(client, &_0flags, (void *)flags, sizeof(*flags));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetDeviceFlags);
    rpc_write(client, &_0flags, sizeof(_0flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)flags, sizeof(*flags));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamCreate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pStream;
    mem2server(client, &_0pStream, (void *)pStream, sizeof(*pStream));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamCreate);
    rpc_write(client, &_0pStream, sizeof(_0pStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pStream, sizeof(*pStream));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamCreateWithFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pStream;
    mem2server(client, &_0pStream, (void *)pStream, sizeof(*pStream));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamCreateWithFlags);
    rpc_write(client, &_0pStream, sizeof(_0pStream));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pStream, sizeof(*pStream));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamCreateWithPriority called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pStream;
    mem2server(client, &_0pStream, (void *)pStream, sizeof(*pStream));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamCreateWithPriority);
    rpc_write(client, &_0pStream, sizeof(_0pStream));
    rpc_write(client, &flags, sizeof(flags));
    rpc_write(client, &priority, sizeof(priority));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pStream, sizeof(*pStream));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamGetPriority called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0priority;
    mem2server(client, &_0priority, (void *)priority, sizeof(*priority));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamGetPriority);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0priority, sizeof(_0priority));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)priority, sizeof(*priority));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamGetFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0flags;
    mem2server(client, &_0flags, (void *)flags, sizeof(*flags));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamGetFlags);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0flags, sizeof(_0flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)flags, sizeof(*flags));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaCtxResetPersistingL2Cache() {
#ifdef DEBUG
    std::cout << "Hook: cudaCtxResetPersistingL2Cache called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaCtxResetPersistingL2Cache);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamCopyAttributes called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamCopyAttributes);
    rpc_write(client, &dst, sizeof(dst));
    rpc_write(client, &src, sizeof(src));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, enum cudaStreamAttrID attr, union cudaStreamAttrValue *value_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamGetAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0value_out;
    mem2server(client, &_0value_out, (void *)value_out, sizeof(*value_out));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamGetAttribute);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value_out, sizeof(_0value_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value_out, sizeof(*value_out));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, enum cudaStreamAttrID attr, const union cudaStreamAttrValue *value) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamSetAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0value;
    mem2server(client, &_0value, (void *)value, sizeof(*value));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamSetAttribute);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, sizeof(*value));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamDestroy called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamDestroy);
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamWaitEvent called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamWaitEvent);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &event, sizeof(event));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamAddCallback called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0userData;
    mem2server(client, &_0userData, (void *)userData, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamAddCallback);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &callback, sizeof(callback));
    rpc_write(client, &_0userData, sizeof(_0userData));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)userData, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamSynchronize called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamSynchronize);
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamQuery called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamQuery);
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamAttachMemAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0devPtr;
    mem2server(client, &_0devPtr, (void *)devPtr, length);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamAttachMemAsync);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    rpc_write(client, &length, sizeof(length));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)devPtr, length);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamBeginCapture called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamBeginCapture);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &mode, sizeof(mode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode) {
#ifdef DEBUG
    std::cout << "Hook: cudaThreadExchangeStreamCaptureMode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0mode;
    mem2server(client, &_0mode, (void *)mode, sizeof(*mode));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaThreadExchangeStreamCaptureMode);
    rpc_write(client, &_0mode, sizeof(_0mode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)mode, sizeof(*mode));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamEndCapture called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraph;
    mem2server(client, &_0pGraph, (void *)pGraph, sizeof(*pGraph));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamEndCapture);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &_0pGraph, sizeof(_0pGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraph, sizeof(*pGraph));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamIsCapturing called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pCaptureStatus;
    mem2server(client, &_0pCaptureStatus, (void *)pCaptureStatus, sizeof(*pCaptureStatus));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamIsCapturing);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &_0pCaptureStatus, sizeof(_0pCaptureStatus));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pCaptureStatus, sizeof(*pCaptureStatus));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus, unsigned long long *pId) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamGetCaptureInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pCaptureStatus;
    mem2server(client, &_0pCaptureStatus, (void *)pCaptureStatus, sizeof(*pCaptureStatus));
    void *_0pId;
    mem2server(client, &_0pId, (void *)pId, sizeof(*pId));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamGetCaptureInfo);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &_0pCaptureStatus, sizeof(_0pCaptureStatus));
    rpc_write(client, &_0pId, sizeof(_0pId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pCaptureStatus, sizeof(*pCaptureStatus));
    mem2client(client, (void *)pId, sizeof(*pId));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out, unsigned long long *id_out, cudaGraph_t *graph_out, const cudaGraphNode_t **dependencies_out, size_t *numDependencies_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamGetCaptureInfo_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0captureStatus_out;
    mem2server(client, &_0captureStatus_out, (void *)captureStatus_out, sizeof(*captureStatus_out));
    void *_0id_out;
    mem2server(client, &_0id_out, (void *)id_out, sizeof(*id_out));
    void *_0graph_out;
    mem2server(client, &_0graph_out, (void *)graph_out, sizeof(*graph_out));
    void *_0numDependencies_out;
    mem2server(client, &_0numDependencies_out, (void *)numDependencies_out, sizeof(*numDependencies_out));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamGetCaptureInfo_v2);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &_0captureStatus_out, sizeof(_0captureStatus_out));
    rpc_write(client, &_0id_out, sizeof(_0id_out));
    rpc_write(client, &_0graph_out, sizeof(_0graph_out));
    static cudaGraphNode_t _cudaStreamGetCaptureInfo_v2_dependencies_out;
    rpc_read(client, &_cudaStreamGetCaptureInfo_v2_dependencies_out, sizeof(cudaGraphNode_t));
    rpc_write(client, &_0numDependencies_out, sizeof(_0numDependencies_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    *dependencies_out = &_cudaStreamGetCaptureInfo_v2_dependencies_out;
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)captureStatus_out, sizeof(*captureStatus_out));
    mem2client(client, (void *)id_out, sizeof(*id_out));
    mem2client(client, (void *)graph_out, sizeof(*graph_out));
    mem2client(client, (void *)numDependencies_out, sizeof(*numDependencies_out));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies, size_t numDependencies, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaStreamUpdateCaptureDependencies called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dependencies;
    mem2server(client, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaStreamUpdateCaptureDependencies);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dependencies, sizeof(*dependencies));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaEventCreate(cudaEvent_t *event) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventCreate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0event;
    mem2server(client, &_0event, (void *)event, sizeof(*event));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaEventCreate);
    rpc_write(client, &_0event, sizeof(_0event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)event, sizeof(*event));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventCreateWithFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0event;
    mem2server(client, &_0event, (void *)event, sizeof(*event));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaEventCreateWithFlags);
    rpc_write(client, &_0event, sizeof(_0event));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)event, sizeof(*event));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventRecord called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaEventRecord);
    rpc_write(client, &event, sizeof(event));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventRecordWithFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaEventRecordWithFlags);
    rpc_write(client, &event, sizeof(event));
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventQuery called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaEventQuery);
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventSynchronize called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaEventSynchronize);
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventDestroy called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaEventDestroy);
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
#ifdef DEBUG
    std::cout << "Hook: cudaEventElapsedTime called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0ms;
    mem2server(client, &_0ms, (void *)ms, sizeof(*ms));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaEventElapsedTime);
    rpc_write(client, &_0ms, sizeof(_0ms));
    rpc_write(client, &start, sizeof(start));
    rpc_write(client, &end, sizeof(end));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)ms, sizeof(*ms));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaImportExternalMemory(cudaExternalMemory_t *extMem_out, const struct cudaExternalMemoryHandleDesc *memHandleDesc) {
#ifdef DEBUG
    std::cout << "Hook: cudaImportExternalMemory called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0extMem_out;
    mem2server(client, &_0extMem_out, (void *)extMem_out, sizeof(*extMem_out));
    void *_0memHandleDesc;
    mem2server(client, &_0memHandleDesc, (void *)memHandleDesc, sizeof(*memHandleDesc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaImportExternalMemory);
    rpc_write(client, &_0extMem_out, sizeof(_0extMem_out));
    rpc_write(client, &_0memHandleDesc, sizeof(_0memHandleDesc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)extMem_out, sizeof(*extMem_out));
    mem2client(client, (void *)memHandleDesc, sizeof(*memHandleDesc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaExternalMemoryGetMappedBuffer(void **devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc *bufferDesc) {
#ifdef DEBUG
    std::cout << "Hook: cudaExternalMemoryGetMappedBuffer called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0bufferDesc;
    mem2server(client, &_0bufferDesc, (void *)bufferDesc, sizeof(*bufferDesc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaExternalMemoryGetMappedBuffer);
    rpc_read(client, devPtr, sizeof(void *));
    rpc_write(client, &extMem, sizeof(extMem));
    rpc_write(client, &_0bufferDesc, sizeof(_0bufferDesc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)bufferDesc, sizeof(*bufferDesc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t *mipmap, cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc) {
#ifdef DEBUG
    std::cout << "Hook: cudaExternalMemoryGetMappedMipmappedArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0mipmap;
    mem2server(client, &_0mipmap, (void *)mipmap, sizeof(*mipmap));
    void *_0mipmapDesc;
    mem2server(client, &_0mipmapDesc, (void *)mipmapDesc, sizeof(*mipmapDesc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaExternalMemoryGetMappedMipmappedArray);
    rpc_write(client, &_0mipmap, sizeof(_0mipmap));
    rpc_write(client, &extMem, sizeof(extMem));
    rpc_write(client, &_0mipmapDesc, sizeof(_0mipmapDesc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)mipmap, sizeof(*mipmap));
    mem2client(client, (void *)mipmapDesc, sizeof(*mipmapDesc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem) {
#ifdef DEBUG
    std::cout << "Hook: cudaDestroyExternalMemory called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDestroyExternalMemory);
    rpc_write(client, &extMem, sizeof(extMem));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t *extSem_out, const struct cudaExternalSemaphoreHandleDesc *semHandleDesc) {
#ifdef DEBUG
    std::cout << "Hook: cudaImportExternalSemaphore called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0extSem_out;
    mem2server(client, &_0extSem_out, (void *)extSem_out, sizeof(*extSem_out));
    void *_0semHandleDesc;
    mem2server(client, &_0semHandleDesc, (void *)semHandleDesc, sizeof(*semHandleDesc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaImportExternalSemaphore);
    rpc_write(client, &_0extSem_out, sizeof(_0extSem_out));
    rpc_write(client, &_0semHandleDesc, sizeof(_0semHandleDesc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)extSem_out, sizeof(*extSem_out));
    mem2client(client, (void *)semHandleDesc, sizeof(*semHandleDesc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaSignalExternalSemaphoresAsync_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0extSemArray;
    mem2server(client, &_0extSemArray, (void *)extSemArray, sizeof(*extSemArray));
    void *_0paramsArray;
    mem2server(client, &_0paramsArray, (void *)paramsArray, sizeof(*paramsArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaSignalExternalSemaphoresAsync_v2);
    rpc_write(client, &_0extSemArray, sizeof(_0extSemArray));
    rpc_write(client, &_0paramsArray, sizeof(_0paramsArray));
    rpc_write(client, &numExtSems, sizeof(numExtSems));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)extSemArray, sizeof(*extSemArray));
    mem2client(client, (void *)paramsArray, sizeof(*paramsArray));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaWaitExternalSemaphoresAsync_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0extSemArray;
    mem2server(client, &_0extSemArray, (void *)extSemArray, sizeof(*extSemArray));
    void *_0paramsArray;
    mem2server(client, &_0paramsArray, (void *)paramsArray, sizeof(*paramsArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaWaitExternalSemaphoresAsync_v2);
    rpc_write(client, &_0extSemArray, sizeof(_0extSemArray));
    rpc_write(client, &_0paramsArray, sizeof(_0paramsArray));
    rpc_write(client, &numExtSems, sizeof(numExtSems));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)extSemArray, sizeof(*extSemArray));
    mem2client(client, (void *)paramsArray, sizeof(*paramsArray));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) {
#ifdef DEBUG
    std::cout << "Hook: cudaDestroyExternalSemaphore called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDestroyExternalSemaphore);
    rpc_write(client, &extSem, sizeof(extSem));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaLaunchCooperativeKernelMultiDevice called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0launchParamsList;
    mem2server(client, &_0launchParamsList, (void *)launchParamsList, sizeof(*launchParamsList));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaLaunchCooperativeKernelMultiDevice);
    rpc_write(client, &_0launchParamsList, sizeof(_0launchParamsList));
    rpc_write(client, &numDevices, sizeof(numDevices));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)launchParamsList, sizeof(*launchParamsList));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig) {
#ifdef DEBUG
    std::cout << "Hook: cudaFuncSetCacheConfig called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0func;
    mem2server(client, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaFuncSetCacheConfig);
    rpc_write(client, &_0func, sizeof(_0func));
    rpc_write(client, &cacheConfig, sizeof(cacheConfig));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)func, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config) {
#ifdef DEBUG
    std::cout << "Hook: cudaFuncSetSharedMemConfig called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0func;
    mem2server(client, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaFuncSetSharedMemConfig);
    rpc_write(client, &_0func, sizeof(_0func));
    rpc_write(client, &config, sizeof(config));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)func, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {
#ifdef DEBUG
    std::cout << "Hook: cudaFuncGetAttributes called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0attr;
    mem2server(client, &_0attr, (void *)attr, sizeof(*attr));
    void *_0func;
    mem2server(client, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaFuncGetAttributes);
    rpc_write(client, &_0attr, sizeof(_0attr));
    rpc_write(client, &_0func, sizeof(_0func));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)attr, sizeof(*attr));
    mem2client(client, (void *)func, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) {
#ifdef DEBUG
    std::cout << "Hook: cudaFuncSetAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0func;
    mem2server(client, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaFuncSetAttribute);
    rpc_write(client, &_0func, sizeof(_0func));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &value, sizeof(value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)func, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaSetDoubleForDevice(double *d) {
#ifdef DEBUG
    std::cout << "Hook: cudaSetDoubleForDevice called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0d;
    mem2server(client, &_0d, (void *)d, sizeof(*d));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaSetDoubleForDevice);
    rpc_write(client, &_0d, sizeof(_0d));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)d, sizeof(*d));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaSetDoubleForHost(double *d) {
#ifdef DEBUG
    std::cout << "Hook: cudaSetDoubleForHost called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0d;
    mem2server(client, &_0d, (void *)d, sizeof(*d));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaSetDoubleForHost);
    rpc_write(client, &_0d, sizeof(_0d));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)d, sizeof(*d));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData) {
#ifdef DEBUG
    std::cout << "Hook: cudaLaunchHostFunc called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0userData;
    mem2server(client, &_0userData, (void *)userData, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaLaunchHostFunc);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &fn, sizeof(fn));
    rpc_write(client, &_0userData, sizeof(_0userData));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)userData, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) {
#ifdef DEBUG
    std::cout << "Hook: cudaOccupancyMaxActiveBlocksPerMultiprocessor called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0numBlocks;
    mem2server(client, &_0numBlocks, (void *)numBlocks, sizeof(*numBlocks));
    void *_0func;
    mem2server(client, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaOccupancyMaxActiveBlocksPerMultiprocessor);
    rpc_write(client, &_0numBlocks, sizeof(_0numBlocks));
    rpc_write(client, &_0func, sizeof(_0func));
    rpc_write(client, &blockSize, sizeof(blockSize));
    rpc_write(client, &dynamicSMemSize, sizeof(dynamicSMemSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)numBlocks, sizeof(*numBlocks));
    mem2client(client, (void *)func, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, const void *func, int numBlocks, int blockSize) {
#ifdef DEBUG
    std::cout << "Hook: cudaOccupancyAvailableDynamicSMemPerBlock called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dynamicSmemSize;
    mem2server(client, &_0dynamicSmemSize, (void *)dynamicSmemSize, sizeof(*dynamicSmemSize));
    void *_0func;
    mem2server(client, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaOccupancyAvailableDynamicSMemPerBlock);
    rpc_write(client, &_0dynamicSmemSize, sizeof(_0dynamicSmemSize));
    rpc_write(client, &_0func, sizeof(_0func));
    rpc_write(client, &numBlocks, sizeof(numBlocks));
    rpc_write(client, &blockSize, sizeof(blockSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dynamicSmemSize, sizeof(*dynamicSmemSize));
    mem2client(client, (void *)func, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0numBlocks;
    mem2server(client, &_0numBlocks, (void *)numBlocks, sizeof(*numBlocks));
    void *_0func;
    mem2server(client, &_0func, (void *)func, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    rpc_write(client, &_0numBlocks, sizeof(_0numBlocks));
    rpc_write(client, &_0func, sizeof(_0func));
    rpc_write(client, &blockSize, sizeof(blockSize));
    rpc_write(client, &dynamicSMemSize, sizeof(dynamicSMemSize));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)numBlocks, sizeof(*numBlocks));
    mem2client(client, (void *)func, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0array;
    mem2server(client, &_0array, (void *)array, sizeof(*array));
    void *_0desc;
    mem2server(client, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMallocArray);
    rpc_write(client, &_0array, sizeof(_0array));
    rpc_write(client, &_0desc, sizeof(_0desc));
    rpc_write(client, &width, sizeof(width));
    rpc_write(client, &height, sizeof(height));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)array, sizeof(*array));
    mem2client(client, (void *)desc, sizeof(*desc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaFreeArray(cudaArray_t array) {
#ifdef DEBUG
    std::cout << "Hook: cudaFreeArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaFreeArray);
    rpc_write(client, &array, sizeof(array));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) {
#ifdef DEBUG
    std::cout << "Hook: cudaFreeMipmappedArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaFreeMipmappedArray);
    rpc_write(client, &mipmappedArray, sizeof(mipmappedArray));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaHostGetDevicePointer called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pHost;
    mem2server(client, &_0pHost, (void *)pHost, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaHostGetDevicePointer);
    rpc_read(client, pDevice, sizeof(void *));
    rpc_write(client, &_0pHost, sizeof(_0pHost));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pHost, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
#ifdef DEBUG
    std::cout << "Hook: cudaHostGetFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pFlags;
    mem2server(client, &_0pFlags, (void *)pFlags, sizeof(*pFlags));
    void *_0pHost;
    mem2server(client, &_0pHost, (void *)pHost, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaHostGetFlags);
    rpc_write(client, &_0pFlags, sizeof(_0pFlags));
    rpc_write(client, &_0pHost, sizeof(_0pHost));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pFlags, sizeof(*pFlags));
    mem2client(client, (void *)pHost, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, struct cudaExtent extent, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaMalloc3DArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0array;
    mem2server(client, &_0array, (void *)array, sizeof(*array));
    void *_0desc;
    mem2server(client, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMalloc3DArray);
    rpc_write(client, &_0array, sizeof(_0array));
    rpc_write(client, &_0desc, sizeof(_0desc));
    rpc_write(client, &extent, sizeof(extent));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)array, sizeof(*array));
    mem2client(client, (void *)desc, sizeof(*desc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray, const struct cudaChannelFormatDesc *desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocMipmappedArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0mipmappedArray;
    mem2server(client, &_0mipmappedArray, (void *)mipmappedArray, sizeof(*mipmappedArray));
    void *_0desc;
    mem2server(client, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMallocMipmappedArray);
    rpc_write(client, &_0mipmappedArray, sizeof(_0mipmappedArray));
    rpc_write(client, &_0desc, sizeof(_0desc));
    rpc_write(client, &extent, sizeof(extent));
    rpc_write(client, &numLevels, sizeof(numLevels));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)mipmappedArray, sizeof(*mipmappedArray));
    mem2client(client, (void *)desc, sizeof(*desc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetMipmappedArrayLevel called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0levelArray;
    mem2server(client, &_0levelArray, (void *)levelArray, sizeof(*levelArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetMipmappedArrayLevel);
    rpc_write(client, &_0levelArray, sizeof(_0levelArray));
    rpc_write(client, &mipmappedArray, sizeof(mipmappedArray));
    rpc_write(client, &level, sizeof(level));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)levelArray, sizeof(*levelArray));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy3D called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0p;
    mem2server(client, &_0p, (void *)p, sizeof(*p));
    void *_0sptr = nullptr;
    void *_0dptr = nullptr;
    if(p != nullptr) {
        mem2server(client, &_0sptr, (void *)p->srcPtr.ptr, sizeof(p->srcPtr.pitch * p->srcPtr.ysize));
        mem2server(client, &_0dptr, (void *)p->dstPtr.ptr, sizeof(p->dstPtr.pitch * p->dstPtr.ysize));
    }
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpy3D);
    rpc_write(client, &_0p, sizeof(_0p));
    rpc_write(client, &_0sptr, sizeof(_0sptr));
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)p, sizeof(*p));
    if(p != nullptr) {
        mem2client(client, (void *)p->srcPtr.ptr, sizeof(p->srcPtr.pitch * p->srcPtr.ysize));
        mem2client(client, (void *)p->dstPtr.ptr, sizeof(p->dstPtr.pitch * p->dstPtr.ysize));
    }
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy3DPeer called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0p;
    mem2server(client, &_0p, (void *)p, sizeof(*p));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpy3DPeer);
    rpc_write(client, &_0p, sizeof(_0p));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)p, sizeof(*p));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy3DAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0p;
    mem2server(client, &_0p, (void *)p, sizeof(*p));
    void *_0sptr = nullptr;
    void *_0dptr = nullptr;
    if(p != nullptr) {
        mem2server(client, &_0sptr, (void *)p->srcPtr.ptr, sizeof(p->srcPtr.pitch * p->srcPtr.ysize));
        mem2server(client, &_0dptr, (void *)p->dstPtr.ptr, sizeof(p->dstPtr.pitch * p->dstPtr.ysize));
    }
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpy3DAsync);
    rpc_write(client, &_0p, sizeof(_0p));
    rpc_write(client, &_0sptr, sizeof(_0sptr));
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)p, sizeof(*p));
    if(p != nullptr) {
        mem2client(client, (void *)p->srcPtr.ptr, sizeof(p->srcPtr.pitch * p->srcPtr.ysize));
        mem2client(client, (void *)p->dstPtr.ptr, sizeof(p->dstPtr.pitch * p->dstPtr.ysize));
    }
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy3DPeerAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0p;
    mem2server(client, &_0p, (void *)p, sizeof(*p));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpy3DPeerAsync);
    rpc_write(client, &_0p, sizeof(_0p));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)p, sizeof(*p));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemGetInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0free;
    mem2server(client, &_0free, (void *)free, sizeof(*free));
    void *_0total;
    mem2server(client, &_0total, (void *)total, sizeof(*total));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemGetInfo);
    rpc_write(client, &_0free, sizeof(_0free));
    rpc_write(client, &_0total, sizeof(_0total));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)free, sizeof(*free));
    mem2client(client, (void *)total, sizeof(*total));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array) {
#ifdef DEBUG
    std::cout << "Hook: cudaArrayGetInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0desc;
    mem2server(client, &_0desc, (void *)desc, sizeof(*desc));
    void *_0extent;
    mem2server(client, &_0extent, (void *)extent, sizeof(*extent));
    void *_0flags;
    mem2server(client, &_0flags, (void *)flags, sizeof(*flags));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaArrayGetInfo);
    rpc_write(client, &_0desc, sizeof(_0desc));
    rpc_write(client, &_0extent, sizeof(_0extent));
    rpc_write(client, &_0flags, sizeof(_0flags));
    rpc_write(client, &array, sizeof(array));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)desc, sizeof(*desc));
    mem2client(client, (void *)extent, sizeof(*extent));
    mem2client(client, (void *)flags, sizeof(*flags));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaArrayGetPlane(cudaArray_t *pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) {
#ifdef DEBUG
    std::cout << "Hook: cudaArrayGetPlane called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pPlaneArray;
    mem2server(client, &_0pPlaneArray, (void *)pPlaneArray, sizeof(*pPlaneArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaArrayGetPlane);
    rpc_write(client, &_0pPlaneArray, sizeof(_0pPlaneArray));
    rpc_write(client, &hArray, sizeof(hArray));
    rpc_write(client, &planeIdx, sizeof(planeIdx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pPlaneArray, sizeof(*pPlaneArray));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaArray_t array) {
#ifdef DEBUG
    std::cout << "Hook: cudaArrayGetSparseProperties called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0sparseProperties;
    mem2server(client, &_0sparseProperties, (void *)sparseProperties, sizeof(*sparseProperties));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaArrayGetSparseProperties);
    rpc_write(client, &_0sparseProperties, sizeof(_0sparseProperties));
    rpc_write(client, &array, sizeof(array));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)sparseProperties, sizeof(*sparseProperties));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaMipmappedArray_t mipmap) {
#ifdef DEBUG
    std::cout << "Hook: cudaMipmappedArrayGetSparseProperties called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0sparseProperties;
    mem2server(client, &_0sparseProperties, (void *)sparseProperties, sizeof(*sparseProperties));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMipmappedArrayGetSparseProperties);
    rpc_write(client, &_0sparseProperties, sizeof(_0sparseProperties));
    rpc_write(client, &mipmap, sizeof(mipmap));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)sparseProperties, sizeof(*sparseProperties));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpy);
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, count);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyPeer called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpyPeer);
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &srcDevice, sizeof(srcDevice));
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, count);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2D called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, dpitch * height);
    void *_0src;
    mem2server(client, &_0src, (void *)src, spitch * height);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpy2D);
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &dpitch, sizeof(dpitch));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &spitch, sizeof(spitch));
    rpc_write(client, &width, sizeof(width));
    rpc_write(client, &height, sizeof(height));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, dpitch * height);
    mem2client(client, (void *)src, spitch * height);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2DToArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0src;
    mem2server(client, &_0src, (void *)src, spitch * height);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpy2DToArray);
    rpc_write(client, &dst, sizeof(dst));
    rpc_write(client, &wOffset, sizeof(wOffset));
    rpc_write(client, &hOffset, sizeof(hOffset));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &spitch, sizeof(spitch));
    rpc_write(client, &width, sizeof(width));
    rpc_write(client, &height, sizeof(height));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)src, spitch * height);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2DFromArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, dpitch * height);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpy2DFromArray);
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &dpitch, sizeof(dpitch));
    rpc_write(client, &src, sizeof(src));
    rpc_write(client, &wOffset, sizeof(wOffset));
    rpc_write(client, &hOffset, sizeof(hOffset));
    rpc_write(client, &width, sizeof(width));
    rpc_write(client, &height, sizeof(height));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, dpitch * height);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2DArrayToArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpy2DArrayToArray);
    rpc_write(client, &dst, sizeof(dst));
    rpc_write(client, &wOffsetDst, sizeof(wOffsetDst));
    rpc_write(client, &hOffsetDst, sizeof(hOffsetDst));
    rpc_write(client, &src, sizeof(src));
    rpc_write(client, &wOffsetSrc, sizeof(wOffsetSrc));
    rpc_write(client, &hOffsetSrc, sizeof(hOffsetSrc));
    rpc_write(client, &width, sizeof(width));
    rpc_write(client, &height, sizeof(height));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyToSymbol called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0symbol;
    mem2server(client, &_0symbol, (void *)symbol, -1);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpyToSymbol);
    rpc_write(client, &_0symbol, sizeof(_0symbol));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)symbol, -1);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyFromSymbol called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *_0symbol;
    mem2server(client, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpyFromSymbol);
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &_0symbol, sizeof(_0symbol));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, count);
    mem2client(client, (void *)symbol, -1);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpyAsync);
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &kind, sizeof(kind));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, count);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyPeerAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpyPeerAsync);
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &srcDevice, sizeof(srcDevice));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, count);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2DAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, dpitch * height);
    void *_0src;
    mem2server(client, &_0src, (void *)src, spitch * height);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpy2DAsync);
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &dpitch, sizeof(dpitch));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &spitch, sizeof(spitch));
    rpc_write(client, &width, sizeof(width));
    rpc_write(client, &height, sizeof(height));
    rpc_write(client, &kind, sizeof(kind));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, dpitch * height);
    mem2client(client, (void *)src, spitch * height);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2DToArrayAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0src;
    mem2server(client, &_0src, (void *)src, spitch * height);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpy2DToArrayAsync);
    rpc_write(client, &dst, sizeof(dst));
    rpc_write(client, &wOffset, sizeof(wOffset));
    rpc_write(client, &hOffset, sizeof(hOffset));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &spitch, sizeof(spitch));
    rpc_write(client, &width, sizeof(width));
    rpc_write(client, &height, sizeof(height));
    rpc_write(client, &kind, sizeof(kind));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)src, spitch * height);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpy2DFromArrayAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, dpitch * height);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpy2DFromArrayAsync);
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &dpitch, sizeof(dpitch));
    rpc_write(client, &src, sizeof(src));
    rpc_write(client, &wOffset, sizeof(wOffset));
    rpc_write(client, &hOffset, sizeof(hOffset));
    rpc_write(client, &width, sizeof(width));
    rpc_write(client, &height, sizeof(height));
    rpc_write(client, &kind, sizeof(kind));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, dpitch * height);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyToSymbolAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0symbol;
    mem2server(client, &_0symbol, (void *)symbol, -1);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpyToSymbolAsync);
    rpc_write(client, &_0symbol, sizeof(_0symbol));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &kind, sizeof(kind));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)symbol, -1);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyFromSymbolAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *_0symbol;
    mem2server(client, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpyFromSymbolAsync);
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &_0symbol, sizeof(_0symbol));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &kind, sizeof(kind));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, count);
    mem2client(client, (void *)symbol, -1);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemset called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0devPtr;
    mem2server(client, &_0devPtr, (void *)devPtr, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemset);
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)devPtr, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemset2D called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0devPtr;
    mem2server(client, &_0devPtr, (void *)devPtr, pitch * height);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemset2D);
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    rpc_write(client, &pitch, sizeof(pitch));
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &width, sizeof(width));
    rpc_write(client, &height, sizeof(height));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)devPtr, pitch * height);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemset3D called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemset3D);
    rpc_write(client, &pitchedDevPtr, sizeof(pitchedDevPtr));
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &extent, sizeof(extent));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemsetAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0devPtr;
    mem2server(client, &_0devPtr, (void *)devPtr, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemsetAsync);
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)devPtr, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemset2DAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0devPtr;
    mem2server(client, &_0devPtr, (void *)devPtr, pitch * height);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemset2DAsync);
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    rpc_write(client, &pitch, sizeof(pitch));
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &width, sizeof(width));
    rpc_write(client, &height, sizeof(height));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)devPtr, pitch * height);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemset3DAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemset3DAsync);
    rpc_write(client, &pitchedDevPtr, sizeof(pitchedDevPtr));
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &extent, sizeof(extent));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetSymbolSize called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0size;
    mem2server(client, &_0size, (void *)size, sizeof(*size));
    void *_0symbol;
    mem2server(client, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetSymbolSize);
    rpc_write(client, &_0size, sizeof(_0size));
    rpc_write(client, &_0symbol, sizeof(_0symbol));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)size, sizeof(*size));
    mem2client(client, (void *)symbol, -1);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPrefetchAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0devPtr;
    mem2server(client, &_0devPtr, (void *)devPtr, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemPrefetchAsync);
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)devPtr, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemAdvise called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0devPtr;
    mem2server(client, &_0devPtr, (void *)devPtr, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemAdvise);
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &advice, sizeof(advice));
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)devPtr, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemRangeGetAttribute(void *data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void *devPtr, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemRangeGetAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0data;
    mem2server(client, &_0data, (void *)data, dataSize);
    void *_0devPtr;
    mem2server(client, &_0devPtr, (void *)devPtr, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemRangeGetAttribute);
    rpc_write(client, &_0data, sizeof(_0data));
    rpc_write(client, &dataSize, sizeof(dataSize));
    rpc_write(client, &attribute, sizeof(attribute));
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)data, dataSize);
    mem2client(client, (void *)devPtr, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyToArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpyToArray);
    rpc_write(client, &dst, sizeof(dst));
    rpc_write(client, &wOffset, sizeof(wOffset));
    rpc_write(client, &hOffset, sizeof(hOffset));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpyFromArray(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyFromArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpyFromArray);
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &src, sizeof(src));
    rpc_write(client, &wOffset, sizeof(wOffset));
    rpc_write(client, &hOffset, sizeof(hOffset));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyArrayToArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpyArrayToArray);
    rpc_write(client, &dst, sizeof(dst));
    rpc_write(client, &wOffsetDst, sizeof(wOffsetDst));
    rpc_write(client, &hOffsetDst, sizeof(hOffsetDst));
    rpc_write(client, &src, sizeof(src));
    rpc_write(client, &wOffsetSrc, sizeof(wOffsetSrc));
    rpc_write(client, &hOffsetSrc, sizeof(hOffsetSrc));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyToArrayAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpyToArrayAsync);
    rpc_write(client, &dst, sizeof(dst));
    rpc_write(client, &wOffset, sizeof(wOffset));
    rpc_write(client, &hOffset, sizeof(hOffset));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &kind, sizeof(kind));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemcpyFromArrayAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemcpyFromArrayAsync);
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &src, sizeof(src));
    rpc_write(client, &wOffset, sizeof(wOffset));
    rpc_write(client, &hOffset, sizeof(hOffset));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &kind, sizeof(kind));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMallocAsync);
    rpc_read(client, devPtr, sizeof(void *));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t hStream) {
#ifdef DEBUG
    std::cout << "Hook: cudaFreeAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0devPtr;
    mem2server(client, &_0devPtr, (void *)devPtr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaFreeAsync);
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)devPtr, -1);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolTrimTo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemPoolTrimTo);
    rpc_write(client, &memPool, sizeof(memPool));
    rpc_write(client, &minBytesToKeep, sizeof(minBytesToKeep));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolSetAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0value;
    mem2server(client, &_0value, (void *)value, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemPoolSetAttribute);
    rpc_write(client, &memPool, sizeof(memPool));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolGetAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0value;
    mem2server(client, &_0value, (void *)value, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemPoolGetAttribute);
    rpc_write(client, &memPool, sizeof(memPool));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const struct cudaMemAccessDesc *descList, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolSetAccess called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0descList;
    mem2server(client, &_0descList, (void *)descList, sizeof(*descList));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemPoolSetAccess);
    rpc_write(client, &memPool, sizeof(memPool));
    rpc_write(client, &_0descList, sizeof(_0descList));
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)descList, sizeof(*descList));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemPoolGetAccess(enum cudaMemAccessFlags *flags, cudaMemPool_t memPool, struct cudaMemLocation *location) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolGetAccess called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0flags;
    mem2server(client, &_0flags, (void *)flags, sizeof(*flags));
    void *_0location;
    mem2server(client, &_0location, (void *)location, sizeof(*location));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemPoolGetAccess);
    rpc_write(client, &_0flags, sizeof(_0flags));
    rpc_write(client, &memPool, sizeof(memPool));
    rpc_write(client, &_0location, sizeof(_0location));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)flags, sizeof(*flags));
    mem2client(client, (void *)location, sizeof(*location));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemPoolCreate(cudaMemPool_t *memPool, const struct cudaMemPoolProps *poolProps) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolCreate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0memPool;
    mem2server(client, &_0memPool, (void *)memPool, sizeof(*memPool));
    void *_0poolProps;
    mem2server(client, &_0poolProps, (void *)poolProps, sizeof(*poolProps));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemPoolCreate);
    rpc_write(client, &_0memPool, sizeof(_0memPool));
    rpc_write(client, &_0poolProps, sizeof(_0poolProps));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)memPool, sizeof(*memPool));
    mem2client(client, (void *)poolProps, sizeof(*poolProps));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolDestroy called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemPoolDestroy);
    rpc_write(client, &memPool, sizeof(memPool));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocFromPoolAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMallocFromPoolAsync);
    rpc_read(client, ptr, sizeof(void *));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &memPool, sizeof(memPool));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemPoolExportToShareableHandle(void *shareableHandle, cudaMemPool_t memPool, enum cudaMemAllocationHandleType handleType, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolExportToShareableHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0shareableHandle;
    mem2server(client, &_0shareableHandle, (void *)shareableHandle, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemPoolExportToShareableHandle);
    rpc_write(client, &_0shareableHandle, sizeof(_0shareableHandle));
    rpc_write(client, &memPool, sizeof(memPool));
    rpc_write(client, &handleType, sizeof(handleType));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)shareableHandle, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t *memPool, void *shareableHandle, enum cudaMemAllocationHandleType handleType, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolImportFromShareableHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0memPool;
    mem2server(client, &_0memPool, (void *)memPool, sizeof(*memPool));
    void *_0shareableHandle;
    mem2server(client, &_0shareableHandle, (void *)shareableHandle, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemPoolImportFromShareableHandle);
    rpc_write(client, &_0memPool, sizeof(_0memPool));
    rpc_write(client, &_0shareableHandle, sizeof(_0shareableHandle));
    rpc_write(client, &handleType, sizeof(handleType));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)memPool, sizeof(*memPool));
    mem2client(client, (void *)shareableHandle, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData *exportData, void *ptr) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolExportPointer called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0exportData;
    mem2server(client, &_0exportData, (void *)exportData, sizeof(*exportData));
    void *_0ptr;
    mem2server(client, &_0ptr, (void *)ptr, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemPoolExportPointer);
    rpc_write(client, &_0exportData, sizeof(_0exportData));
    rpc_write(client, &_0ptr, sizeof(_0ptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)exportData, sizeof(*exportData));
    mem2client(client, (void *)ptr, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemPoolImportPointer(void **ptr, cudaMemPool_t memPool, struct cudaMemPoolPtrExportData *exportData) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemPoolImportPointer called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0exportData;
    mem2server(client, &_0exportData, (void *)exportData, sizeof(*exportData));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaMemPoolImportPointer);
    rpc_read(client, ptr, sizeof(void *));
    rpc_write(client, &memPool, sizeof(memPool));
    rpc_write(client, &_0exportData, sizeof(_0exportData));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)exportData, sizeof(*exportData));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr) {
#ifdef DEBUG
    std::cout << "Hook: cudaPointerGetAttributes called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0attributes;
    mem2server(client, &_0attributes, (void *)attributes, sizeof(*attributes));
    void *_0ptr;
    mem2server(client, &_0ptr, (void *)ptr, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaPointerGetAttributes);
    rpc_write(client, &_0attributes, sizeof(_0attributes));
    rpc_write(client, &_0ptr, sizeof(_0ptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)attributes, sizeof(*attributes));
    mem2client(client, (void *)ptr, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceCanAccessPeer called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0canAccessPeer;
    mem2server(client, &_0canAccessPeer, (void *)canAccessPeer, sizeof(*canAccessPeer));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceCanAccessPeer);
    rpc_write(client, &_0canAccessPeer, sizeof(_0canAccessPeer));
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &peerDevice, sizeof(peerDevice));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)canAccessPeer, sizeof(*canAccessPeer));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceEnablePeerAccess called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceEnablePeerAccess);
    rpc_write(client, &peerDevice, sizeof(peerDevice));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceDisablePeerAccess called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceDisablePeerAccess);
    rpc_write(client, &peerDevice, sizeof(peerDevice));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsUnregisterResource called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphicsUnregisterResource);
    rpc_write(client, &resource, sizeof(resource));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsResourceSetMapFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphicsResourceSetMapFlags);
    rpc_write(client, &resource, sizeof(resource));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsMapResources called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0resources;
    mem2server(client, &_0resources, (void *)resources, sizeof(*resources));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphicsMapResources);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_0resources, sizeof(_0resources));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)resources, sizeof(*resources));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsUnmapResources called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0resources;
    mem2server(client, &_0resources, (void *)resources, sizeof(*resources));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphicsUnmapResources);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_0resources, sizeof(_0resources));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)resources, sizeof(*resources));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsResourceGetMappedPointer called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0size;
    mem2server(client, &_0size, (void *)size, sizeof(*size));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphicsResourceGetMappedPointer);
    rpc_read(client, devPtr, sizeof(void *));
    rpc_write(client, &_0size, sizeof(_0size));
    rpc_write(client, &resource, sizeof(resource));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)size, sizeof(*size));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsSubResourceGetMappedArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0array;
    mem2server(client, &_0array, (void *)array, sizeof(*array));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphicsSubResourceGetMappedArray);
    rpc_write(client, &_0array, sizeof(_0array));
    rpc_write(client, &resource, sizeof(resource));
    rpc_write(client, &arrayIndex, sizeof(arrayIndex));
    rpc_write(client, &mipLevel, sizeof(mipLevel));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)array, sizeof(*array));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphicsResourceGetMappedMipmappedArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0mipmappedArray;
    mem2server(client, &_0mipmappedArray, (void *)mipmappedArray, sizeof(*mipmappedArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphicsResourceGetMappedMipmappedArray);
    rpc_write(client, &_0mipmappedArray, sizeof(_0mipmappedArray));
    rpc_write(client, &resource, sizeof(resource));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)mipmappedArray, sizeof(*mipmappedArray));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size) {
#ifdef DEBUG
    std::cout << "Hook: cudaBindTexture called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0offset;
    mem2server(client, &_0offset, (void *)offset, sizeof(*offset));
    void *_0texref;
    mem2server(client, &_0texref, (void *)texref, sizeof(*texref));
    void *_0devPtr;
    mem2server(client, &_0devPtr, (void *)devPtr, size);
    void *_0desc;
    mem2server(client, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaBindTexture);
    rpc_write(client, &_0offset, sizeof(_0offset));
    rpc_write(client, &_0texref, sizeof(_0texref));
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    rpc_write(client, &_0desc, sizeof(_0desc));
    rpc_write(client, &size, sizeof(size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)offset, sizeof(*offset));
    mem2client(client, (void *)texref, sizeof(*texref));
    mem2client(client, (void *)devPtr, size);
    mem2client(client, (void *)desc, sizeof(*desc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaBindTexture2D(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, size_t pitch) {
#ifdef DEBUG
    std::cout << "Hook: cudaBindTexture2D called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0offset;
    mem2server(client, &_0offset, (void *)offset, sizeof(*offset));
    void *_0texref;
    mem2server(client, &_0texref, (void *)texref, sizeof(*texref));
    void *_0devPtr;
    mem2server(client, &_0devPtr, (void *)devPtr, pitch * height);
    void *_0desc;
    mem2server(client, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaBindTexture2D);
    rpc_write(client, &_0offset, sizeof(_0offset));
    rpc_write(client, &_0texref, sizeof(_0texref));
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    rpc_write(client, &_0desc, sizeof(_0desc));
    rpc_write(client, &width, sizeof(width));
    rpc_write(client, &height, sizeof(height));
    rpc_write(client, &pitch, sizeof(pitch));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)offset, sizeof(*offset));
    mem2client(client, (void *)texref, sizeof(*texref));
    mem2client(client, (void *)devPtr, pitch * height);
    mem2client(client, (void *)desc, sizeof(*desc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaBindTextureToArray(const struct textureReference *texref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc) {
#ifdef DEBUG
    std::cout << "Hook: cudaBindTextureToArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0texref;
    mem2server(client, &_0texref, (void *)texref, sizeof(*texref));
    void *_0desc;
    mem2server(client, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaBindTextureToArray);
    rpc_write(client, &_0texref, sizeof(_0texref));
    rpc_write(client, &array, sizeof(array));
    rpc_write(client, &_0desc, sizeof(_0desc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)texref, sizeof(*texref));
    mem2client(client, (void *)desc, sizeof(*desc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaBindTextureToMipmappedArray(const struct textureReference *texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc *desc) {
#ifdef DEBUG
    std::cout << "Hook: cudaBindTextureToMipmappedArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0texref;
    mem2server(client, &_0texref, (void *)texref, sizeof(*texref));
    void *_0desc;
    mem2server(client, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaBindTextureToMipmappedArray);
    rpc_write(client, &_0texref, sizeof(_0texref));
    rpc_write(client, &mipmappedArray, sizeof(mipmappedArray));
    rpc_write(client, &_0desc, sizeof(_0desc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)texref, sizeof(*texref));
    mem2client(client, (void *)desc, sizeof(*desc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaUnbindTexture(const struct textureReference *texref) {
#ifdef DEBUG
    std::cout << "Hook: cudaUnbindTexture called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0texref;
    mem2server(client, &_0texref, (void *)texref, sizeof(*texref));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaUnbindTexture);
    rpc_write(client, &_0texref, sizeof(_0texref));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)texref, sizeof(*texref));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetTextureAlignmentOffset called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0offset;
    mem2server(client, &_0offset, (void *)offset, sizeof(*offset));
    void *_0texref;
    mem2server(client, &_0texref, (void *)texref, sizeof(*texref));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetTextureAlignmentOffset);
    rpc_write(client, &_0offset, sizeof(_0offset));
    rpc_write(client, &_0texref, sizeof(_0texref));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)offset, sizeof(*offset));
    mem2client(client, (void *)texref, sizeof(*texref));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetTextureReference(const struct textureReference **texref, const void *symbol) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetTextureReference called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0symbol;
    mem2server(client, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetTextureReference);
    static struct textureReference _cudaGetTextureReference_texref;
    rpc_read(client, &_cudaGetTextureReference_texref, sizeof(struct textureReference));
    rpc_write(client, &_0symbol, sizeof(_0symbol));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    *texref = &_cudaGetTextureReference_texref;
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)symbol, -1);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaBindSurfaceToArray(const struct surfaceReference *surfref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc) {
#ifdef DEBUG
    std::cout << "Hook: cudaBindSurfaceToArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0surfref;
    mem2server(client, &_0surfref, (void *)surfref, sizeof(*surfref));
    void *_0desc;
    mem2server(client, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaBindSurfaceToArray);
    rpc_write(client, &_0surfref, sizeof(_0surfref));
    rpc_write(client, &array, sizeof(array));
    rpc_write(client, &_0desc, sizeof(_0desc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)surfref, sizeof(*surfref));
    mem2client(client, (void *)desc, sizeof(*desc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetSurfaceReference(const struct surfaceReference **surfref, const void *symbol) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetSurfaceReference called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0symbol;
    mem2server(client, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetSurfaceReference);
    static struct surfaceReference _cudaGetSurfaceReference_surfref;
    rpc_read(client, &_cudaGetSurfaceReference_surfref, sizeof(struct surfaceReference));
    rpc_write(client, &_0symbol, sizeof(_0symbol));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    *surfref = &_cudaGetSurfaceReference_surfref;
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)symbol, -1);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, cudaArray_const_t array) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetChannelDesc called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0desc;
    mem2server(client, &_0desc, (void *)desc, sizeof(*desc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetChannelDesc);
    rpc_write(client, &_0desc, sizeof(_0desc));
    rpc_write(client, &array, sizeof(array));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)desc, sizeof(*desc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f) {
#ifdef DEBUG
    std::cout << "Hook: cudaCreateChannelDesc called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    struct cudaChannelFormatDesc _result;
    rpc_prepare_request(client, RPC_cudaCreateChannelDesc);
    rpc_write(client, &x, sizeof(x));
    rpc_write(client, &y, sizeof(y));
    rpc_write(client, &z, sizeof(z));
    rpc_write(client, &w, sizeof(w));
    rpc_write(client, &f, sizeof(f));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject, const struct cudaResourceDesc *pResDesc, const struct cudaTextureDesc *pTexDesc, const struct cudaResourceViewDesc *pResViewDesc) {
#ifdef DEBUG
    std::cout << "Hook: cudaCreateTextureObject called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pTexObject;
    mem2server(client, &_0pTexObject, (void *)pTexObject, sizeof(*pTexObject));
    void *_0pResDesc;
    mem2server(client, &_0pResDesc, (void *)pResDesc, sizeof(*pResDesc));
    void *_0pTexDesc;
    mem2server(client, &_0pTexDesc, (void *)pTexDesc, sizeof(*pTexDesc));
    void *_0pResViewDesc;
    mem2server(client, &_0pResViewDesc, (void *)pResViewDesc, sizeof(*pResViewDesc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaCreateTextureObject);
    rpc_write(client, &_0pTexObject, sizeof(_0pTexObject));
    rpc_write(client, &_0pResDesc, sizeof(_0pResDesc));
    rpc_write(client, &_0pTexDesc, sizeof(_0pTexDesc));
    rpc_write(client, &_0pResViewDesc, sizeof(_0pResViewDesc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pTexObject, sizeof(*pTexObject));
    mem2client(client, (void *)pResDesc, sizeof(*pResDesc));
    mem2client(client, (void *)pTexDesc, sizeof(*pTexDesc));
    mem2client(client, (void *)pResViewDesc, sizeof(*pResViewDesc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
#ifdef DEBUG
    std::cout << "Hook: cudaDestroyTextureObject called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDestroyTextureObject);
    rpc_write(client, &texObject, sizeof(texObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaTextureObject_t texObject) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetTextureObjectResourceDesc called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pResDesc;
    mem2server(client, &_0pResDesc, (void *)pResDesc, sizeof(*pResDesc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetTextureObjectResourceDesc);
    rpc_write(client, &_0pResDesc, sizeof(_0pResDesc));
    rpc_write(client, &texObject, sizeof(texObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pResDesc, sizeof(*pResDesc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetTextureObjectTextureDesc called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pTexDesc;
    mem2server(client, &_0pTexDesc, (void *)pTexDesc, sizeof(*pTexDesc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetTextureObjectTextureDesc);
    rpc_write(client, &_0pTexDesc, sizeof(_0pTexDesc));
    rpc_write(client, &texObject, sizeof(texObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pTexDesc, sizeof(*pTexDesc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetTextureObjectResourceViewDesc called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pResViewDesc;
    mem2server(client, &_0pResViewDesc, (void *)pResViewDesc, sizeof(*pResViewDesc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetTextureObjectResourceViewDesc);
    rpc_write(client, &_0pResViewDesc, sizeof(_0pResViewDesc));
    rpc_write(client, &texObject, sizeof(texObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pResViewDesc, sizeof(*pResViewDesc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject, const struct cudaResourceDesc *pResDesc) {
#ifdef DEBUG
    std::cout << "Hook: cudaCreateSurfaceObject called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pSurfObject;
    mem2server(client, &_0pSurfObject, (void *)pSurfObject, sizeof(*pSurfObject));
    void *_0pResDesc;
    mem2server(client, &_0pResDesc, (void *)pResDesc, sizeof(*pResDesc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaCreateSurfaceObject);
    rpc_write(client, &_0pSurfObject, sizeof(_0pSurfObject));
    rpc_write(client, &_0pResDesc, sizeof(_0pResDesc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pSurfObject, sizeof(*pSurfObject));
    mem2client(client, (void *)pResDesc, sizeof(*pResDesc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
#ifdef DEBUG
    std::cout << "Hook: cudaDestroySurfaceObject called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDestroySurfaceObject);
    rpc_write(client, &surfObject, sizeof(surfObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetSurfaceObjectResourceDesc called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pResDesc;
    mem2server(client, &_0pResDesc, (void *)pResDesc, sizeof(*pResDesc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetSurfaceObjectResourceDesc);
    rpc_write(client, &_0pResDesc, sizeof(_0pResDesc));
    rpc_write(client, &surfObject, sizeof(surfObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pResDesc, sizeof(*pResDesc));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDriverGetVersion(int *driverVersion) {
#ifdef DEBUG
    std::cout << "Hook: cudaDriverGetVersion called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0driverVersion;
    mem2server(client, &_0driverVersion, (void *)driverVersion, sizeof(*driverVersion));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDriverGetVersion);
    rpc_write(client, &_0driverVersion, sizeof(_0driverVersion));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)driverVersion, sizeof(*driverVersion));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
#ifdef DEBUG
    std::cout << "Hook: cudaRuntimeGetVersion called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0runtimeVersion;
    mem2server(client, &_0runtimeVersion, (void *)runtimeVersion, sizeof(*runtimeVersion));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaRuntimeGetVersion);
    rpc_write(client, &_0runtimeVersion, sizeof(_0runtimeVersion));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)runtimeVersion, sizeof(*runtimeVersion));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphCreate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraph;
    mem2server(client, &_0pGraph, (void *)pGraph, sizeof(*pGraph));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphCreate);
    rpc_write(client, &_0pGraph, sizeof(_0pGraph));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraph, sizeof(*pGraph));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaKernelNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddKernelNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddKernelNode);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphKernelNodeGetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphKernelNodeGetParams);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphKernelNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphKernelNodeSetParams);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphKernelNodeCopyAttributes called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphKernelNodeCopyAttributes);
    rpc_write(client, &hSrc, sizeof(hSrc));
    rpc_write(client, &hDst, sizeof(hDst));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, enum cudaKernelNodeAttrID attr, union cudaKernelNodeAttrValue *value_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphKernelNodeGetAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0value_out;
    mem2server(client, &_0value_out, (void *)value_out, sizeof(*value_out));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphKernelNodeGetAttribute);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value_out, sizeof(_0value_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value_out, sizeof(*value_out));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, enum cudaKernelNodeAttrID attr, const union cudaKernelNodeAttrValue *value) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphKernelNodeSetAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0value;
    mem2server(client, &_0value, (void *)value, sizeof(*value));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphKernelNodeSetAttribute);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, sizeof(*value));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms *pCopyParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemcpyNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0pCopyParams;
    mem2server(client, &_0pCopyParams, (void *)pCopyParams, sizeof(*pCopyParams));
    void *_0sptr = nullptr;
    void *_0dptr = nullptr;
    if(pCopyParams != nullptr) {
        mem2server(client, &_0sptr, (void *)pCopyParams->srcPtr.ptr, sizeof(pCopyParams->srcPtr.pitch * pCopyParams->srcPtr.ysize));
        mem2server(client, &_0dptr, (void *)pCopyParams->dstPtr.ptr, sizeof(pCopyParams->dstPtr.pitch * pCopyParams->dstPtr.ysize));
    }
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddMemcpyNode);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0pCopyParams, sizeof(_0pCopyParams));
    rpc_write(client, &_0sptr, sizeof(_0sptr));
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    mem2client(client, (void *)pCopyParams, sizeof(*pCopyParams));
    if(pCopyParams != nullptr) {
        mem2client(client, (void *)pCopyParams->srcPtr.ptr, sizeof(pCopyParams->srcPtr.pitch * pCopyParams->srcPtr.ysize));
        mem2client(client, (void *)pCopyParams->dstPtr.ptr, sizeof(pCopyParams->dstPtr.pitch * pCopyParams->dstPtr.ysize));
    }
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemcpyNodeToSymbol called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0symbol;
    mem2server(client, &_0symbol, (void *)symbol, -1);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddMemcpyNodeToSymbol);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0symbol, sizeof(_0symbol));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    mem2client(client, (void *)symbol, -1);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemcpyNodeFromSymbol called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *_0symbol;
    mem2server(client, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddMemcpyNodeFromSymbol);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &_0symbol, sizeof(_0symbol));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    mem2client(client, (void *)dst, count);
    mem2client(client, (void *)symbol, -1);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemcpyNode1D called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddMemcpyNode1D);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    mem2client(client, (void *)dst, count);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, struct cudaMemcpy3DParms *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemcpyNodeGetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *_0sptr = nullptr;
    void *_0dptr = nullptr;
    if(pNodeParams != nullptr) {
        mem2server(client, &_0sptr, (void *)pNodeParams->srcPtr.ptr, sizeof(pNodeParams->srcPtr.pitch * pNodeParams->srcPtr.ysize));
        mem2server(client, &_0dptr, (void *)pNodeParams->dstPtr.ptr, sizeof(pNodeParams->dstPtr.pitch * pNodeParams->dstPtr.ysize));
    }
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphMemcpyNodeGetParams);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_write(client, &_0sptr, sizeof(_0sptr));
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(pNodeParams != nullptr) {
        mem2client(client, (void *)pNodeParams->srcPtr.ptr, sizeof(pNodeParams->srcPtr.pitch * pNodeParams->srcPtr.ysize));
        mem2client(client, (void *)pNodeParams->dstPtr.ptr, sizeof(pNodeParams->dstPtr.pitch * pNodeParams->dstPtr.ysize));
    }
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemcpyNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *_0sptr = nullptr;
    void *_0dptr = nullptr;
    if(pNodeParams != nullptr) {
        mem2server(client, &_0sptr, (void *)pNodeParams->srcPtr.ptr, sizeof(pNodeParams->srcPtr.pitch * pNodeParams->srcPtr.ysize));
        mem2server(client, &_0dptr, (void *)pNodeParams->dstPtr.ptr, sizeof(pNodeParams->dstPtr.pitch * pNodeParams->dstPtr.ysize));
    }
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphMemcpyNodeSetParams);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_write(client, &_0sptr, sizeof(_0sptr));
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(pNodeParams != nullptr) {
        mem2client(client, (void *)pNodeParams->srcPtr.ptr, sizeof(pNodeParams->srcPtr.pitch * pNodeParams->srcPtr.ysize));
        mem2client(client, (void *)pNodeParams->dstPtr.ptr, sizeof(pNodeParams->dstPtr.pitch * pNodeParams->dstPtr.ysize));
    }
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemcpyNodeSetParamsToSymbol called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0symbol;
    mem2server(client, &_0symbol, (void *)symbol, -1);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphMemcpyNodeSetParamsToSymbol);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0symbol, sizeof(_0symbol));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)symbol, -1);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemcpyNodeSetParamsFromSymbol called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *_0symbol;
    mem2server(client, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphMemcpyNodeSetParamsFromSymbol);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &_0symbol, sizeof(_0symbol));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, count);
    mem2client(client, (void *)symbol, -1);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemcpyNodeSetParams1D called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphMemcpyNodeSetParams1D);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, count);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemsetParams *pMemsetParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemsetNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0pMemsetParams;
    mem2server(client, &_0pMemsetParams, (void *)pMemsetParams, sizeof(*pMemsetParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddMemsetNode);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0pMemsetParams, sizeof(_0pMemsetParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    mem2client(client, (void *)pMemsetParams, sizeof(*pMemsetParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, struct cudaMemsetParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemsetNodeGetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphMemsetNodeGetParams);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemsetNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphMemsetNodeSetParams);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaHostNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddHostNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddHostNode);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, struct cudaHostNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphHostNodeGetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphHostNodeGetParams);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphHostNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphHostNodeSetParams);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaGraph_t childGraph) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddChildGraphNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddChildGraphNode);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &childGraph, sizeof(childGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t *pGraph) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphChildGraphNodeGetGraph called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraph;
    mem2server(client, &_0pGraph, (void *)pGraph, sizeof(*pGraph));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphChildGraphNodeGetGraph);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pGraph, sizeof(_0pGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraph, sizeof(*pGraph));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddEmptyNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddEmptyNode);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddEventRecordNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddEventRecordNode);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphEventRecordNodeGetEvent called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0event_out;
    mem2server(client, &_0event_out, (void *)event_out, sizeof(*event_out));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphEventRecordNodeGetEvent);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0event_out, sizeof(_0event_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)event_out, sizeof(*event_out));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphEventRecordNodeSetEvent called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphEventRecordNodeSetEvent);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddEventWaitNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddEventWaitNode);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphEventWaitNodeGetEvent called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0event_out;
    mem2server(client, &_0event_out, (void *)event_out, sizeof(*event_out));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphEventWaitNodeGetEvent);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0event_out, sizeof(_0event_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)event_out, sizeof(*event_out));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphEventWaitNodeSetEvent called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphEventWaitNodeSetEvent);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddExternalSemaphoresSignalNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0nodeParams;
    mem2server(client, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddExternalSemaphoresSignalNode);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreSignalNodeParams *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExternalSemaphoresSignalNodeGetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0params_out;
    mem2server(client, &_0params_out, (void *)params_out, sizeof(*params_out));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExternalSemaphoresSignalNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0params_out, sizeof(_0params_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)params_out, sizeof(*params_out));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExternalSemaphoresSignalNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0nodeParams;
    mem2server(client, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExternalSemaphoresSignalNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddExternalSemaphoresWaitNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0nodeParams;
    mem2server(client, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddExternalSemaphoresWaitNode);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreWaitNodeParams *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExternalSemaphoresWaitNodeGetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0params_out;
    mem2server(client, &_0params_out, (void *)params_out, sizeof(*params_out));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExternalSemaphoresWaitNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0params_out, sizeof(_0params_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)params_out, sizeof(*params_out));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExternalSemaphoresWaitNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0nodeParams;
    mem2server(client, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExternalSemaphoresWaitNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, struct cudaMemAllocNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemAllocNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0nodeParams;
    mem2server(client, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddMemAllocNode);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, struct cudaMemAllocNodeParams *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemAllocNodeGetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0params_out;
    mem2server(client, &_0params_out, (void *)params_out, sizeof(*params_out));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphMemAllocNodeGetParams);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0params_out, sizeof(_0params_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)params_out, sizeof(*params_out));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dptr) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddMemFreeNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphNode;
    mem2server(client, &_0pGraphNode, (void *)pGraphNode, sizeof(*pGraphNode));
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0dptr;
    mem2server(client, &_0dptr, (void *)dptr, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddMemFreeNode);
    rpc_write(client, &_0pGraphNode, sizeof(_0pGraphNode));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphNode, sizeof(*pGraphNode));
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    mem2client(client, (void *)dptr, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void *dptr_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemFreeNodeGetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dptr_out;
    mem2server(client, &_0dptr_out, (void *)dptr_out, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphMemFreeNodeGetParams);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0dptr_out, sizeof(_0dptr_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dptr_out, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGraphMemTrim(int device) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGraphMemTrim called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGraphMemTrim);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceGetGraphMemAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0value;
    mem2server(client, &_0value, (void *)value, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceGetGraphMemAttribute);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaDeviceSetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cudaDeviceSetGraphMemAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0value;
    mem2server(client, &_0value, (void *)value, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaDeviceSetGraphMemAttribute);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphClone called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphClone;
    mem2server(client, &_0pGraphClone, (void *)pGraphClone, sizeof(*pGraphClone));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphClone);
    rpc_write(client, &_0pGraphClone, sizeof(_0pGraphClone));
    rpc_write(client, &originalGraph, sizeof(originalGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphClone, sizeof(*pGraphClone));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphNodeFindInClone called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNode;
    mem2server(client, &_0pNode, (void *)pNode, sizeof(*pNode));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphNodeFindInClone);
    rpc_write(client, &_0pNode, sizeof(_0pNode));
    rpc_write(client, &originalNode, sizeof(originalNode));
    rpc_write(client, &clonedGraph, sizeof(clonedGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNode, sizeof(*pNode));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType *pType) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphNodeGetType called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pType;
    mem2server(client, &_0pType, (void *)pType, sizeof(*pType));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphNodeGetType);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pType, sizeof(_0pType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pType, sizeof(*pType));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphGetNodes called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0nodes;
    mem2server(client, &_0nodes, (void *)nodes, sizeof(*nodes));
    void *_0numNodes;
    mem2server(client, &_0numNodes, (void *)numNodes, sizeof(*numNodes));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphGetNodes);
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0nodes, sizeof(_0nodes));
    rpc_write(client, &_0numNodes, sizeof(_0numNodes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodes, sizeof(*nodes));
    mem2client(client, (void *)numNodes, sizeof(*numNodes));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphGetRootNodes called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pRootNodes;
    mem2server(client, &_0pRootNodes, (void *)pRootNodes, sizeof(*pRootNodes));
    void *_0pNumRootNodes;
    mem2server(client, &_0pNumRootNodes, (void *)pNumRootNodes, sizeof(*pNumRootNodes));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphGetRootNodes);
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pRootNodes, sizeof(_0pRootNodes));
    rpc_write(client, &_0pNumRootNodes, sizeof(_0pNumRootNodes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pRootNodes, sizeof(*pRootNodes));
    mem2client(client, (void *)pNumRootNodes, sizeof(*pNumRootNodes));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphGetEdges called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0from;
    mem2server(client, &_0from, (void *)from, sizeof(*from));
    void *_0to;
    mem2server(client, &_0to, (void *)to, sizeof(*to));
    void *_0numEdges;
    mem2server(client, &_0numEdges, (void *)numEdges, sizeof(*numEdges));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphGetEdges);
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0from, sizeof(_0from));
    rpc_write(client, &_0to, sizeof(_0to));
    rpc_write(client, &_0numEdges, sizeof(_0numEdges));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)from, sizeof(*from));
    mem2client(client, (void *)to, sizeof(*to));
    mem2client(client, (void *)numEdges, sizeof(*numEdges));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies, size_t *pNumDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphNodeGetDependencies called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pDependencies;
    mem2server(client, &_0pDependencies, (void *)pDependencies, sizeof(*pDependencies));
    void *_0pNumDependencies;
    mem2server(client, &_0pNumDependencies, (void *)pNumDependencies, sizeof(*pNumDependencies));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphNodeGetDependencies);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pDependencies, sizeof(_0pDependencies));
    rpc_write(client, &_0pNumDependencies, sizeof(_0pNumDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pDependencies, sizeof(*pDependencies));
    mem2client(client, (void *)pNumDependencies, sizeof(*pNumDependencies));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes, size_t *pNumDependentNodes) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphNodeGetDependentNodes called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pDependentNodes;
    mem2server(client, &_0pDependentNodes, (void *)pDependentNodes, sizeof(*pDependentNodes));
    void *_0pNumDependentNodes;
    mem2server(client, &_0pNumDependentNodes, (void *)pNumDependentNodes, sizeof(*pNumDependentNodes));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphNodeGetDependentNodes);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pDependentNodes, sizeof(_0pDependentNodes));
    rpc_write(client, &_0pNumDependentNodes, sizeof(_0pNumDependentNodes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pDependentNodes, sizeof(*pDependentNodes));
    mem2client(client, (void *)pNumDependentNodes, sizeof(*pNumDependentNodes));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphAddDependencies called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0from;
    mem2server(client, &_0from, (void *)from, sizeof(*from));
    void *_0to;
    mem2server(client, &_0to, (void *)to, sizeof(*to));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphAddDependencies);
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0from, sizeof(_0from));
    rpc_write(client, &_0to, sizeof(_0to));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)from, sizeof(*from));
    mem2client(client, (void *)to, sizeof(*to));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphRemoveDependencies called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0from;
    mem2server(client, &_0from, (void *)from, sizeof(*from));
    void *_0to;
    mem2server(client, &_0to, (void *)to, sizeof(*to));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphRemoveDependencies);
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0from, sizeof(_0from));
    rpc_write(client, &_0to, sizeof(_0to));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)from, sizeof(*from));
    mem2client(client, (void *)to, sizeof(*to));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphDestroyNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphDestroyNode);
    rpc_write(client, &node, sizeof(node));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphInstantiate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphExec;
    mem2server(client, &_0pGraphExec, (void *)pGraphExec, sizeof(*pGraphExec));
    void *_0pErrorNode;
    mem2server(client, &_0pErrorNode, (void *)pErrorNode, sizeof(*pErrorNode));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphInstantiate);
    rpc_write(client, &_0pGraphExec, sizeof(_0pGraphExec));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &_0pErrorNode, sizeof(_0pErrorNode));
    if(bufferSize > 0) {
        rpc_read(client, pLogBuffer, bufferSize, true);
    }
    rpc_write(client, &bufferSize, sizeof(bufferSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphExec, sizeof(*pGraphExec));
    mem2client(client, (void *)pErrorNode, sizeof(*pErrorNode));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphInstantiateWithFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGraphExec;
    mem2server(client, &_0pGraphExec, (void *)pGraphExec, sizeof(*pGraphExec));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphInstantiateWithFlags);
    rpc_write(client, &_0pGraphExec, sizeof(_0pGraphExec));
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGraphExec, sizeof(*pGraphExec));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecKernelNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecKernelNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecMemcpyNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *_0sptr = nullptr;
    void *_0dptr = nullptr;
    if(pNodeParams != nullptr) {
        mem2server(client, &_0sptr, (void *)pNodeParams->srcPtr.ptr, sizeof(pNodeParams->srcPtr.pitch * pNodeParams->srcPtr.ysize));
        mem2server(client, &_0dptr, (void *)pNodeParams->dstPtr.ptr, sizeof(pNodeParams->dstPtr.pitch * pNodeParams->dstPtr.ysize));
    }
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecMemcpyNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_write(client, &_0sptr, sizeof(_0sptr));
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(pNodeParams != nullptr) {
        mem2client(client, (void *)pNodeParams->srcPtr.ptr, sizeof(pNodeParams->srcPtr.pitch * pNodeParams->srcPtr.ysize));
        mem2client(client, (void *)pNodeParams->dstPtr.ptr, sizeof(pNodeParams->dstPtr.pitch * pNodeParams->dstPtr.ysize));
    }
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecMemcpyNodeSetParamsToSymbol called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0symbol;
    mem2server(client, &_0symbol, (void *)symbol, -1);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecMemcpyNodeSetParamsToSymbol);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0symbol, sizeof(_0symbol));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)symbol, -1);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecMemcpyNodeSetParamsFromSymbol called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *_0symbol;
    mem2server(client, &_0symbol, (void *)symbol, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecMemcpyNodeSetParamsFromSymbol);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &_0symbol, sizeof(_0symbol));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, count);
    mem2client(client, (void *)symbol, -1);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecMemcpyNodeSetParams1D called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, count);
    void *_0src;
    mem2server(client, &_0src, (void *)src, count);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecMemcpyNodeSetParams1D);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0dst, sizeof(_0dst));
    rpc_write(client, &_0src, sizeof(_0src));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &kind, sizeof(kind));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, count);
    mem2client(client, (void *)src, count);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecMemsetNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecMemsetNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecHostNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNodeParams;
    mem2server(client, &_0pNodeParams, (void *)pNodeParams, sizeof(*pNodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecHostNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_0pNodeParams, sizeof(_0pNodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNodeParams, sizeof(*pNodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecChildGraphNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecChildGraphNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &childGraph, sizeof(childGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecEventRecordNodeSetEvent called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecEventRecordNodeSetEvent);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecEventWaitNodeSetEvent called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecEventWaitNodeSetEvent);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecExternalSemaphoresSignalNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0nodeParams;
    mem2server(client, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecExternalSemaphoresSignalNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecExternalSemaphoresWaitNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0nodeParams;
    mem2server(client, &_0nodeParams, (void *)nodeParams, sizeof(*nodeParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecExternalSemaphoresWaitNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t *hErrorNode_out, enum cudaGraphExecUpdateResult *updateResult_out) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecUpdate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0hErrorNode_out;
    mem2server(client, &_0hErrorNode_out, (void *)hErrorNode_out, sizeof(*hErrorNode_out));
    void *_0updateResult_out;
    mem2server(client, &_0updateResult_out, (void *)updateResult_out, sizeof(*updateResult_out));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecUpdate);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0hErrorNode_out, sizeof(_0hErrorNode_out));
    rpc_write(client, &_0updateResult_out, sizeof(_0updateResult_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)hErrorNode_out, sizeof(*hErrorNode_out));
    mem2client(client, (void *)updateResult_out, sizeof(*updateResult_out));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphUpload called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphUpload);
    rpc_write(client, &graphExec, sizeof(graphExec));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphLaunch called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphLaunch);
    rpc_write(client, &graphExec, sizeof(graphExec));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphExecDestroy called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphExecDestroy);
    rpc_write(client, &graphExec, sizeof(graphExec));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphDestroy called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphDestroy);
    rpc_write(client, &graph, sizeof(graph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphDebugDotPrint called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphDebugDotPrint);
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, path, strlen(path) + 1, true);
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaUserObjectCreate(cudaUserObject_t *object_out, void *ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaUserObjectCreate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0object_out;
    mem2server(client, &_0object_out, (void *)object_out, sizeof(*object_out));
    void *_0ptr;
    mem2server(client, &_0ptr, (void *)ptr, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaUserObjectCreate);
    rpc_write(client, &_0object_out, sizeof(_0object_out));
    rpc_write(client, &_0ptr, sizeof(_0ptr));
    rpc_write(client, &destroy, sizeof(destroy));
    rpc_write(client, &initialRefcount, sizeof(initialRefcount));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)object_out, sizeof(*object_out));
    mem2client(client, (void *)ptr, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cudaUserObjectRetain called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaUserObjectRetain);
    rpc_write(client, &object, sizeof(object));
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cudaUserObjectRelease called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaUserObjectRelease);
    rpc_write(client, &object, sizeof(object));
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphRetainUserObject called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphRetainUserObject);
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &object, sizeof(object));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphReleaseUserObject called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGraphReleaseUserObject);
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &object, sizeof(object));
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetDriverEntryPoint called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetDriverEntryPoint);
    rpc_write(client, symbol, strlen(symbol) + 1, true);
    rpc_read(client, funcPtr, sizeof(void *));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetExportTable called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pExportTableId;
    mem2server(client, &_0pExportTableId, (void *)pExportTableId, sizeof(*pExportTableId));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetExportTable);
    rpc_read(client, ppExportTable, sizeof(*ppExportTable));
    rpc_write(client, &_0pExportTableId, sizeof(_0pExportTableId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pExportTableId, sizeof(*pExportTableId));
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetFuncBySymbol(cudaFunction_t *functionPtr, const void *symbolPtr) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetFuncBySymbol called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0functionPtr;
    mem2server(client, &_0functionPtr, (void *)functionPtr, sizeof(*functionPtr));
    void *_0symbolPtr;
    mem2server(client, &_0symbolPtr, (void *)symbolPtr, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    cudaError_t _result;
    rpc_prepare_request(client, RPC_cudaGetFuncBySymbol);
    rpc_write(client, &_0functionPtr, sizeof(_0functionPtr));
    rpc_write(client, &_0symbolPtr, sizeof(_0symbolPtr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)functionPtr, sizeof(*functionPtr));
    mem2client(client, (void *)symbolPtr, 0);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}
