#include <iostream>
#include <unordered_map>
#include "gen/hook_api.h"
#include "gen/handle_server.h"
#include "rpc.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

int handle_cuMemAllocHost_v2(void *args) {
    std::cout << "Handle function handle_cuMemAllocHost_v2 called" << std::endl;
    // TODO: Implement the function logic
    return 0;
}

int handle_cuMemFreeHost(void *args) {
    std::cout << "Handle function handle_cuMemFreeHost called" << std::endl;
    // TODO: Implement the function logic
    return 0;
}

int handle_cuMemHostAlloc(void *args) {
    std::cout << "Handle function handle_cuMemHostAlloc called" << std::endl;
    // TODO: Implement the function logic
    return 0;
}

int handle_cuMemAddressReserve(void *args) {
    std::cout << "Handle function handle_cuMemAddressReserve called" << std::endl;
    // TODO: Implement the function logic
    return 0;
}

int handle_cuMemCreate(void *args) {
    std::cout << "Handle function handle_cuMemCreate called" << std::endl;
    // TODO: Implement the function logic
    return 0;
}

int handle_cuMemRelease(void *args) {
    std::cout << "Handle function handle_cuMemRelease called" << std::endl;
    // TODO: Implement the function logic
    return 0;
}

int handle_cuMemMap(void *args) {
    std::cout << "Handle function handle_cuMemMap called" << std::endl;
    // TODO: Implement the function logic
    return 0;
}

int handle_cudaMallocManaged(void *args) {
    std::cout << "Handle function handle_cudaMallocManaged called" << std::endl;
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

int handle_cudaMallocHost(void *args) {
    std::cout << "Handle function handle_cudaMallocHost called" << std::endl;
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

int handle_cudaFree(void *args) {
    std::cout << "Handle function handle_cudaFree called" << std::endl;
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
    std::cout << "Handle function handle_cudaFreeHost called" << std::endl;
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

int handle_cudaHostAlloc(void *args) {
    std::cout << "Handle function handle_cudaHostAlloc called" << std::endl;
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

int handle_cuModuleGetGlobal_v2(void *args) {
    std::cout << "Handle function handle_cuModuleGetGlobal_v2 called" << std::endl;
    return 0;
}

int handle_cuMemAlloc_v2(void *args) {
    std::cout << "Handle function handle_cuMemAlloc_v2 called" << std::endl;
    return 0;
}

int handle_cuMemAllocPitch_v2(void *args) {
    std::cout << "Handle function handle_cuMemAllocPitch_v2 called" << std::endl;
    return 0;
}

int handle_cuMemGetAddressRange_v2(void *args) {
    std::cout << "Handle function handle_cuMemGetAddressRange_v2 called" << std::endl;
    return 0;
}

int handle_cuMemHostGetDevicePointer_v2(void *args) {
    std::cout << "Handle function handle_cuMemHostGetDevicePointer_v2 called" << std::endl;
    return 0;
}

int handle_cuMemAllocManaged(void *args) {
    std::cout << "Handle function handle_cuMemAllocManaged called" << std::endl;
    return 0;
}

int handle_cuIpcOpenMemHandle_v2(void *args) {
    std::cout << "Handle function handle_cuIpcOpenMemHandle_v2 called" << std::endl;
    return 0;
}

int handle_cuMemAllocAsync(void *args) {
    std::cout << "Handle function handle_cuMemAllocAsync called" << std::endl;
    return 0;
}

int handle_cuMemAllocFromPoolAsync(void *args) {
    std::cout << "Handle function handle_cuMemAllocFromPoolAsync called" << std::endl;
    return 0;
}

int handle_cuMemPoolImportPointer(void *args) {
    std::cout << "Handle function handle_cuMemPoolImportPointer called" << std::endl;
    return 0;
}

int handle_cuImportExternalMemory(void *args) {
    std::cout << "Handle function handle_cuImportExternalMemory called" << std::endl;
    return 0;
}

int handle_cuExternalMemoryGetMappedBuffer(void *args) {
    std::cout << "Handle function handle_cuExternalMemoryGetMappedBuffer called" << std::endl;
    return 0;
}

int handle_cuGraphMemFreeNodeGetParams(void *args) {
    std::cout << "Handle function handle_cuGraphMemFreeNodeGetParams called" << std::endl;
    return 0;
}

int handle_cuTexRefGetAddress_v2(void *args) {
    std::cout << "Handle function handle_cuTexRefGetAddress_v2 called" << std::endl;
    return 0;
}

int handle_cuGraphicsResourceGetMappedPointer_v2(void *args) {
    std::cout << "Handle function handle_cuGraphicsResourceGetMappedPointer_v2 called" << std::endl;
    return 0;
}

int handle_cudaMalloc(void *args) {
    std::cout << "Handle function handle_cudaMalloc called" << std::endl;
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

int handle_cudaMallocPitch(void *args) {
    std::cout << "Handle function handle_cudaMallocPitch called" << std::endl;
    return 0;
}

int handle_cudaMalloc3D(void *args) {
    std::cout << "Handle function handle_cudaMalloc3D called" << std::endl;
    return 0;
}

int handle_cudaGetSymbolAddress(void *args) {
    std::cout << "Handle function handle_cudaGetSymbolAddress called" << std::endl;
    return 0;
}

int handle_cuLibraryGetGlobal(void *args) {
    std::cout << "Handle function handle_cuLibraryGetGlobal called" << std::endl;
    return 0;
}

int handle_cuLibraryGetManaged(void *args) {
    std::cout << "Handle function handle_cuLibraryGetManaged called" << std::endl;
    return 0;
}

int handle_cuMemcpyBatchAsync(void *args) {
    std::cout << "Handle function handle_cuMemcpyBatchAsync called" << std::endl;
    return 0;
}

cudaMemoryType checkPointer(void *ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if(err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return cudaMemoryTypeUnregistered;
    }

    return attributes.type;
}

int handle_cudaMemcpy(void *args0) {
    std::cout << "Handle function cudaMemcpy called" << std::endl;
    RpcClient *client = (RpcClient *)args0;
    void *dst;
    void *src;
    void *toFree;
    size_t count;
    cudaError_t _result;
    enum cudaMemcpyKind kind;
    rpc_read(client, &dst, sizeof(dst));
    rpc_read(client, &src, sizeof(src));
    rpc_read(client, &count, sizeof(count));
    rpc_read(client, &kind, sizeof(kind));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return cudaErrorUnknown;
    }
    switch(kind) {
    case cudaMemcpyHostToDevice:
        toFree = nullptr;
        if(src == nullptr) {
            src = malloc(count);
            if(src == nullptr) {
                std::cerr << "Failed to allocate src" << std::endl;
                return cudaErrorMemoryAllocation;
            }
            toFree = src;
        }
        if(rpc_read_now(client, src, count) != 0) {
            std::cerr << "Failed to read src" << std::endl;
            return cudaErrorUnknown;
        }
        _result = cudaMemcpy(dst, src, count, kind);
        rpc_write(client, &_result, sizeof(_result));
        if(rpc_submit_response(client) != 0) {
            std::cerr << "Failed to submit response" << std::endl;
            if(toFree != nullptr) {
                free(toFree);
            }
            return cudaErrorUnknown;
        }
        if(toFree != nullptr) {
            free(toFree);
        }
        break;
    case cudaMemcpyDeviceToHost:
        toFree = nullptr;
        if(dst == nullptr) {
            dst = malloc(count);
            if(dst == nullptr) {
                std::cerr << "Failed to allocate dst" << std::endl;
                return cudaErrorMemoryAllocation;
            }
            toFree = dst;
        }
        _result = cudaMemcpy(dst, src, count, kind);
        if(_result != cudaSuccess) {
            count = 0;
        }
        rpc_write(client, dst, count);
        rpc_write(client, &_result, sizeof(_result));
        if(rpc_submit_response(client) != 0) {
            std::cerr << "Failed to submit response" << std::endl;
            if(toFree != nullptr) {
                free(toFree);
            }
            return cudaErrorUnknown;
        }
        if(toFree != nullptr) {
            free(toFree);
        }
        break;
    case cudaMemcpyDeviceToDevice:
        _result = cudaMemcpy(dst, src, count, kind);
        {
            bool dst_is_union = checkPointer(dst) == cudaMemoryTypeManaged;
            bool src_is_union = checkPointer(src) == cudaMemoryTypeManaged;
            if(src_is_union) {
                rpc_read_now(client, src, count);
            }
            if(src_is_union && dst_is_union) {
            }

            _result = cudaMemcpy(dst, src, count, kind);
            if(_result != cudaSuccess) {
                count = 0;
            }
            if(dst_is_union && !src_is_union) {
                rpc_write(client, dst, count);
            }
            rpc_write(client, &_result, sizeof(_result));
            if(rpc_submit_response(client) != 0) {
                std::cerr << "Failed to submit response" << std::endl;
                return cudaErrorUnknown;
            }
        }
        break;
    case cudaMemcpyHostToHost:
        toFree = nullptr;
        if(src == nullptr) {
            src = malloc(count);
            if(src == nullptr) {
                std::cerr << "Failed to allocate src" << std::endl;
                return cudaErrorMemoryAllocation;
            }
            toFree = src;
        }
        if(rpc_read_now(client, src, count) != 0) {
            std::cerr << "Failed to read src" << std::endl;
            return cudaErrorUnknown;
        }
        _result = cudaMemcpy(dst, src, count, kind);
        rpc_write(client, &_result, sizeof(_result));
        if(rpc_submit_response(client) != 0) {
            std::cerr << "Failed to submit response" << std::endl;
            if(toFree != nullptr) {
                free(toFree);
            }
            return cudaErrorUnknown;
        }
        if(toFree != nullptr) {
            free(toFree);
        }
        break;
    }
    return cudaSuccess;
}

int handle_cudaMemset(void *args0) {
    std::cout << "Handle function cudaMemset called" << std::endl;
    RpcClient *client = (RpcClient *)args0;
    void *devPtr;
    int value;
    size_t count;
    rpc_read(client, &devPtr, sizeof(devPtr));
    rpc_read(client, &value, sizeof(value));
    rpc_read(client, &count, sizeof(count));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    cudaError_t _result = cudaMemset(devPtr, value, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}
