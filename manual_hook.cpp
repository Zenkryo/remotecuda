#include <iostream>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>
#include <cstdlib>
#include "cuda_runtime.h"
#include <sys/mman.h>
#include "cuda.h"
#include "gen/hook_api.h"
#include "rpc.h"
#include "hidden_api.h"

extern std::unordered_map<std::string, void *> functionMap;

// 映射客户端主机内存地址到服务器主机内存地址
std::map<void *, std::pair<void *, size_t>> cs_host_mems;

// 映射客户端统一内存地址到服务器统一内存地址
std::map<void *, std::pair<void *, size_t>> cs_union_mems;

// 映射客户端设备内存地址到服务器设备内存地址
std::map<CUdeviceptr, CUdeviceptr> cs_dev_mems;

// 设备端内存指针（不包括上面的cs_dev_mems）
std::map<void *, size_t> server_dev_mems;

// 映射客户端内存保留地址到服务器内存保留地址
std::map<CUdeviceptr, CUdeviceptr> cs_reserve_mems;

// 服务器主机内存句柄
std::set<CUmemGenericAllocationHandle> host_handles;

void *getServerHostPtr(void *ptr);
void *getUnionPtr(void *ptr);
void *getServerDevPtr(void *ptr);
void freeDevPtr(void *ptr);

// 是不是只在客户端这边做客户端主机内存和服务器主机内存的映射就够了？
extern "C" cudaError_t cudaMallocHost(void **ptr, size_t size) {
    std::cout << "Hook: cudaMallocHost called" << std::endl;
    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *ptr = malloc(size);
    if(*ptr == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    rpc_prepare_request(client, RPC_cudaMallocHost);
    rpc_read(client, &serverPtr, sizeof(serverPtr));
    rpc_write(client, &size, sizeof(size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*ptr);
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        cs_host_mems[*ptr] = std::make_pair(serverPtr, size);
    } else {
        free(*ptr);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
    std::cout << "Hook: cudaHostAlloc called" << std::endl;
    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *pHost = malloc(size);
    if(*pHost == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    rpc_prepare_request(client, RPC_cudaHostAlloc);
    rpc_read(client, &serverPtr, sizeof(serverPtr));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*pHost);
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        cs_host_mems[*pHost] = std::make_pair(serverPtr, size);
    } else {
        free(*pHost);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
    std::cout << "Hook: cuMemAllocHost_v2 called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *pp = malloc(bytesize);
    if(*pp == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    rpc_prepare_request(client, RPC_cuMemAllocHost_v2);
    rpc_read(client, &serverPtr, sizeof(serverPtr));
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*pp);
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_host_mems[*pp] = std::make_pair(serverPtr, bytesize);
    } else {
        free(*pp);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    std::cout << "Hook: cuMemHostAlloc called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *pp = malloc(bytesize);
    if(*pp == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    rpc_prepare_request(client, RPC_cuMemHostAlloc);
    rpc_read(client, &serverPtr, sizeof(serverPtr));
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*pp);
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_host_mems[*pp] = std::make_pair(serverPtr, bytesize);
    } else {
        free(*pp);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags) {
    std::cout << "Hook: cuMemCreate called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemCreate);
    rpc_read(client, handle, sizeof(*handle));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, prop, sizeof(*prop));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
#ifdef CU_MEM_LOCATION_TYPE_HOST
        // 如果是主机内存,则记录句柄
        if(prop->location.type == CU_MEM_LOCATION_TYPE_HOST) {
            host_handles.insert(*handle);
        }
#endif
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
    std::cout << "Hook: cuMemAddressReserve called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    CUdeviceptr *serverPtr;
    // 客户端保留内存地址
    *ptr = (CUdeviceptr)mmap(NULL, size, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if(*ptr == 0) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    rpc_prepare_request(client, RPC_cuMemAddressReserve);
    rpc_read(client, serverPtr, sizeof(*serverPtr));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &alignment, sizeof(alignment));
    rpc_write(client, &addr, sizeof(addr));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        munmap((void *)*ptr, size);
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_reserve_mems[*ptr] = *serverPtr;
    } else {
        munmap((void *)*ptr, size);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
    std::cout << "Hook: cuMemMap called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    CUdeviceptr serverPtr;
    if(cs_reserve_mems.find(ptr) != cs_reserve_mems.end()) {
        serverPtr = cs_reserve_mems[ptr];
    } else {
        return CUDA_ERROR_INVALID_VALUE;
    }
    bool isHost = false;
    // 如何handle存在于host_handles中,则说明是在主机内存上映射
    if(host_handles.find(handle) != host_handles.end()) {
        isHost = true;
        // 在客户端实际映射主机内存到客户端保留地址
        mmap((void *)ptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, -1, 0);
    }
    rpc_prepare_request(client, RPC_cuMemMap);
    rpc_write(client, &serverPtr, sizeof(serverPtr));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        if(isHost) {
            munmap((void *)ptr, size);
        }
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        if(isHost) {
            cs_host_mems[(void *)ptr] = std::make_pair((void *)serverPtr, size);
        } else {
            cs_dev_mems[ptr] = serverPtr;
        }
        cs_reserve_mems.erase(ptr);
    } else {
        if(isHost) {
            munmap((void *)ptr, size);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    std::cout << "Hook: cudaMallocManaged called" << std::endl;
    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *devPtr = malloc(size);
    if(*devPtr == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    rpc_prepare_request(client, RPC_cudaMallocManaged);
    rpc_read(client, &serverPtr, sizeof(serverPtr));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*devPtr);
        exit(1);
    }
    if(_result == cudaSuccess) {
        cs_union_mems[*devPtr] = std::make_pair(serverPtr, size);
        rpc_release_client(client);
    } else {
        free(*devPtr);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaFreeHost(void *ptr) {
    std::cout << "Hook: cudaFreeHost called" << std::endl;
    cudaError_t _result;
    void *serverPtr;
    serverPtr = getServerHostPtr((void *)ptr);
    if(serverPtr == nullptr) {
        return cudaErrorInvalidValue;
    }
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaFreeHost);
    rpc_write(client, &serverPtr, sizeof(serverPtr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        free(ptr);
        cs_host_mems.erase(ptr);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemFreeHost(void *p) {
    std::cout << "Hook: cuMemFreeHost called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    if(cs_host_mems.find(p) == cs_host_mems.end()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    serverPtr = cs_host_mems[p].first;
    rpc_prepare_request(client, RPC_cuMemFreeHost);
    rpc_write(client, &serverPtr, sizeof(serverPtr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        free(p);
        cs_host_mems.erase(p);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaFree(void *devPtr) {
    std::cout << "Hook: cudaFree called" << std::endl;
    cudaError_t _result;
    void *serverPtr = getUnionPtr(devPtr);
    if(serverPtr == nullptr) {
        serverPtr = getServerDevPtr(devPtr);
        if(serverPtr == nullptr) {
            serverPtr = devPtr;
        }
    }
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaFree);
    rpc_write(client, &serverPtr, sizeof(serverPtr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        freeDevPtr(devPtr);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    std::cout << "Hook: cuMemRelease called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemRelease);
    rpc_write(client, &handle, sizeof(handle));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        host_handles.erase(handle);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
    std::cout << "Hook: cudaMalloc called" << std::endl;
    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaMalloc);
    rpc_read(client, devPtr, sizeof(*devPtr));
    rpc_write(client, &size, sizeof(size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        server_dev_mems[*devPtr] = size;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
    std::cout << "Hook: cudaMallocPitch called" << std::endl;
    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaMallocPitch);
    rpc_read(client, devPtr, sizeof(*devPtr));
    rpc_read(client, pitch, sizeof(*pitch));
    rpc_write(client, &width, sizeof(width));
    rpc_write(client, &height, sizeof(height));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        server_dev_mems[*devPtr] = *pitch * height;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr, struct cudaExtent extent) {
    std::cout << "Hook: cudaMalloc3D called" << std::endl;
    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaMalloc3D);
    rpc_read(client, pitchedDevPtr, sizeof(*pitchedDevPtr));
    rpc_write(client, &extent, sizeof(extent));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        server_dev_mems[pitchedDevPtr->ptr] = pitchedDevPtr->pitch * extent.height;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
    std::cout << "Hook: cudaGetSymbolAddress called" << std::endl;
    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaGetSymbolAddress);
    rpc_read(client, devPtr, sizeof(*devPtr));
    rpc_write(client, &symbol, sizeof(symbol));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        server_dev_mems[*devPtr] = 0;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    std::cout << "Hook: cuMemAlloc_v2 called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemAlloc_v2);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = bytesize;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    std::cout << "Hook: cuMemAllocPitch_v2 called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemAllocPitch_v2);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_read(client, pPitch, sizeof(*pPitch));
    rpc_write(client, &WidthInBytes, sizeof(WidthInBytes));
    rpc_write(client, &Height, sizeof(Height));
    rpc_write(client, &ElementSizeBytes, sizeof(ElementSizeBytes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = *pPitch * Height;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
    std::cout << "Hook: cuMemAllocAsync called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemAllocAsync);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = bytesize;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) {
    std::cout << "Hook: cuMemAllocFromPoolAsync called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemAllocFromPoolAsync);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_write(client, &pool, sizeof(pool));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = bytesize;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
    std::cout << "Hook: cuMemAllocManaged called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *dptr = (CUdeviceptr)malloc(bytesize);
    if(*dptr == 0) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    rpc_prepare_request(client, RPC_cuMemAllocManaged);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_union_mems[(void *)*dptr] = std::make_pair((void *)serverPtr, bytesize);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    std::cout << "Hook: cuMemGetAddressRange_v2 called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemGetAddressRange_v2);
    rpc_read(client, pbase, sizeof(*pbase));
    rpc_read(client, psize, sizeof(*psize));
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pbase] = *psize;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource) {
    std::cout << "Hook: cuGraphicsResourceGetMappedPointer_v2 called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphicsResourceGetMappedPointer_v2);
    rpc_read(client, pDevPtr, sizeof(*pDevPtr));
    rpc_read(client, pSize, sizeof(*pSize));
    rpc_write(client, &resource, sizeof(resource));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pDevPtr] = *pSize;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefGetAddress_v2(CUdeviceptr *pdptr, CUtexref hTexRef) {
    std::cout << "Hook: cuTexRefGetAddress_v2 called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefGetAddress_v2);
    rpc_read(client, pdptr, sizeof(*pdptr));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pdptr] = 0;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr *dptr_out) {
    std::cout << "Hook: cuGraphMemFreeNodeGetParams called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphMemFreeNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_read(client, dptr_out, sizeof(*dptr_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        if(cs_dev_mems.find(*dptr_out) != cs_dev_mems.end()) {
            server_dev_mems[(void *)*dptr_out] = 0;
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc) {
    std::cout << "Hook: cuExternalMemoryGetMappedBuffer called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuExternalMemoryGetMappedBuffer);
    rpc_read(client, devPtr, sizeof(*devPtr));
    rpc_write(client, &extMem, sizeof(extMem));
    rpc_write(client, bufferDesc, sizeof(*bufferDesc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*devPtr] = bufferDesc->size;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuImportExternalMemory(CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc) {
    std::cout << "Hook: cuImportExternalMemory called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuImportExternalMemory);
    rpc_read(client, extMem_out, sizeof(*extMem_out));
    rpc_write(client, memHandleDesc, sizeof(*memHandleDesc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*extMem_out] = memHandleDesc->size;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData *shareData) {
    std::cout << "Hook: cuMemPoolImportPointer called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPoolImportPointer);
    rpc_read(client, ptr_out, sizeof(*ptr_out));
    rpc_write(client, &pool, sizeof(pool));
    rpc_write(client, shareData, sizeof(*shareData));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        if(cs_dev_mems.find((CUdeviceptr)*ptr_out) != cs_dev_mems.end()) {
            server_dev_mems[(void *)*ptr_out] = 0;
        }
    }
    rpc_free_client(client);
    return _result;
}

// 用#ifdef来区分不同的cuda版本,如果没有CUDA_VERSION或其值大于11040定义下面的函数
#if CUDA_VERSION > 11040
extern "C" CUresult cuMemcpyBatchAsync(CUdeviceptr *dsts, CUdeviceptr *srcs, size_t *sizes, size_t count, CUmemcpyAttributes *attrs, size_t *attrsIdxs, size_t numAttrs, size_t *failIdx, CUstream hStream) {
    std::cout << "Hook: cuMemcpyBatchAsync called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyBatchAsync);
    CUdeviceptr *new_dsts = (CUdeviceptr *)malloc(count * sizeof(CUdeviceptr));
    CUdeviceptr *new_srcs = (CUdeviceptr *)malloc(count * sizeof(CUdeviceptr));
    if(new_dsts == nullptr || new_srcs == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    for(int i = 0; i < count; i++) {
        if(cs_union_mems.find((void *)dsts[i]) != cs_union_mems.end()) {
            std::pair<void *, size_t> mem = cs_union_mems[(void *)dsts[i]];
            new_dsts[i] = (CUdeviceptr)mem.first;
            rpc_write(client, &new_dsts[i], sizeof(new_dsts[i]));
            // 如果是拷贝到统一内存,还需要将数据读回到客户端, 服务器端需要调用cudaPointerGetAttributes来判断指针是否是统一内存
            rpc_read(client, (void *)dsts[i], sizes[i]);
        } else if(cs_dev_mems.find(dsts[i]) != cs_dev_mems.end()) {
            new_dsts[i] = (CUdeviceptr)cs_dev_mems[dsts[i]];
            rpc_write(client, &new_dsts[i], sizeof(new_dsts[i]));
        } else {
            rpc_write(client, (void *)&dsts[i], sizeof(dsts[i]));
        }
    }
    for(int i = 0; i < count; i++) {
        if(cs_union_mems.find((void *)srcs[i]) != cs_union_mems.end()) {
            std::pair<void *, size_t> mem = cs_union_mems[(void *)srcs[i]];
            new_srcs[i] = (CUdeviceptr)mem.first;
            rpc_write(client, &new_srcs[i], sizeof(new_srcs[i]));
            // 如果是从统一内存拷贝,还需要将数据写入到服务器端
            rpc_write(client, (void *)srcs[i], sizes[i]);
        } else if(cs_dev_mems.find(srcs[i]) != cs_dev_mems.end()) {
            new_srcs[i] = (CUdeviceptr)cs_dev_mems[srcs[i]];
            rpc_write(client, &new_srcs[i], sizeof(new_srcs[i]));
        } else {
            rpc_write(client, (void *)&srcs[i], sizeof(srcs[i]));
        }
    }
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, sizes, sizeof(*sizes) * count);
    rpc_read(client, attrs, sizeof(*attrs));
    rpc_read(client, attrsIdxs, sizeof(*attrsIdxs));
    rpc_write(client, &numAttrs, sizeof(numAttrs));
    rpc_read(client, failIdx, sizeof(*failIdx));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name) {
    std::cout << "Hook: cuLibraryGetManaged called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLibraryGetManaged);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_read(client, bytes, sizeof(*bytes));
    rpc_write(client, &library, sizeof(library));
    rpc_write(client, name, strlen(name) + 1);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        void *p = malloc(*bytes);
        if(p == nullptr) {
            // TODO: free memory on server
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        cs_union_mems[p] = std::make_pair((void *)*dptr, *bytes);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name) {
    std::cout << "Hook: cuLibraryGetGlobal called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLibraryGetGlobal);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_read(client, bytes, sizeof(*bytes));
    rpc_write(client, &library, sizeof(library));
    rpc_write(client, name, strlen(name) + 1);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = *bytes;
    }
    rpc_free_client(client);
    return _result;
}

#endif

extern "C" CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags) {
    std::cout << "Hook: cuIpcOpenMemHandle_v2 called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuIpcOpenMemHandle_v2);
    rpc_read(client, pdptr, sizeof(*pdptr));
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pdptr] = 0;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    std::cout << "Hook: cuMemHostGetDevicePointer_v2 called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemHostGetDevicePointer_v2);
    rpc_read(client, pdptr, sizeof(*pdptr));
    void *serverPtr = getServerHostPtr(p);
    if(serverPtr == nullptr) {
        rpc_write(client, &p, sizeof(p));
    } else {
        rpc_write(client, &serverPtr, sizeof(serverPtr));
    }
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pdptr] = 0;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    std::cout << "Hook: cuModuleGetGlobal_v2 called" << std::endl;
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuModuleGetGlobal_v2);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_read(client, bytes, sizeof(*bytes));
    rpc_write(client, &hmod, sizeof(hmod));
    rpc_write(client, name, strlen(name) + 1);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = *bytes;
    }
    rpc_free_client(client);
    return _result;
}

// 取得客户端主机内存地址对应的服务器主机内存地址
void *getServerHostPtr(void *ptr) {
    if(cs_host_mems.find(ptr) != cs_host_mems.end()) {
        return cs_host_mems[ptr].first;
    }
    return nullptr;
}

void *getUnionPtr(void *ptr) {
    if(cs_union_mems.find(ptr) != cs_union_mems.end()) {
        return cs_union_mems[ptr].first;
    }
    return nullptr;
}

void *getServerDevPtr(void *ptr) {
    if(cs_dev_mems.find((CUdeviceptr)ptr) != cs_dev_mems.end()) {
        return (void *)cs_dev_mems[(CUdeviceptr)ptr];
    }
    if(server_dev_mems.find(ptr) != server_dev_mems.end()) {
        return ptr;
    }
    return nullptr;
}

void freeDevPtr(void *ptr) {
    if(cs_union_mems.find(ptr) != cs_union_mems.end()) {
        free(ptr);
        cs_union_mems.erase(ptr);
    } else if(cs_dev_mems.find((CUdeviceptr)ptr) != cs_dev_mems.end()) {
        cs_dev_mems.erase((CUdeviceptr)ptr);
    } else if(server_dev_mems.find(ptr) != server_dev_mems.end()) {
        server_dev_mems.erase(ptr);
    }
}

extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    std::cout << "Hook: cudaMemcpy called" << std::endl;
    printf("dst: %p, src: %p, count: %ld, kind: %d\n", dst, src, count, kind);
    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaMemcpy);
    void *serverSrc;
    void *serverDst;
    bool src_is_union;
    bool dst_is_union;

    switch(kind) {
    case cudaMemcpyHostToDevice:
        serverSrc = getServerHostPtr((void *)src);
        if(serverSrc == nullptr) {
            serverSrc = getUnionPtr((void *)src);
        }
        serverDst = getUnionPtr(dst);
        if(serverDst == nullptr) {
            serverDst = getServerDevPtr(dst);
            if(serverDst == nullptr) {
                printf("device memory %p not in \n", dst);
                serverDst = dst;
            }
        } else {
            memcpy(dst, src, count);
        }
        rpc_write(client, &serverDst, sizeof(serverDst));
        rpc_write(client, &serverSrc, sizeof(serverSrc));
        rpc_write(client, &count, sizeof(count));
        rpc_write(client, &kind, sizeof(kind));
        rpc_write(client, src, count, false);
        break;
    case cudaMemcpyDeviceToHost:
        serverSrc = getUnionPtr((void *)src);
        if(serverSrc == nullptr) {
            serverSrc = getServerDevPtr((void *)src);
            if(serverSrc == nullptr) {
                printf("device memory %p not in \n", src);
                serverSrc = (void *)src;
            }
        }
        serverDst = getUnionPtr(dst);
        if(serverDst == nullptr) {
            serverDst = getServerHostPtr(dst);
        }
        rpc_write(client, &serverDst, sizeof(serverDst));
        rpc_write(client, &serverSrc, sizeof(serverSrc));
        rpc_write(client, &count, sizeof(count));
        rpc_write(client, &kind, sizeof(kind));
        rpc_read(client, dst, count, false);
        break;
    case cudaMemcpyDeviceToDevice:
        serverSrc = getUnionPtr((void *)src);
        if(serverSrc == nullptr) {
            src_is_union = false;
            serverSrc = getServerDevPtr((void *)src);
            if(serverSrc == nullptr) {
                printf("device memory %p not in \n", src);
                serverSrc = (void *)src;
            }
        } else {
            src_is_union = true;
        }
        serverDst = getUnionPtr(dst);
        if(serverDst == nullptr) {
            dst_is_union = false;
            serverDst = getServerDevPtr(dst);
            if(serverDst == nullptr) {
                printf("device memory %p not in \n", dst);
                serverDst = dst;
            }
        } else {
            dst_is_union = true;
        }
        rpc_write(client, &serverDst, sizeof(serverDst));
        rpc_write(client, &serverSrc, sizeof(serverSrc));
        rpc_write(client, &count, sizeof(count));
        rpc_write(client, &kind, sizeof(kind));
        if(src_is_union) { // 源地址为统一指针，需要将客户端数据发送到服务器
            rpc_write(client, src, count);
        }
        if(src_is_union && dst_is_union) { // 源地址和目的地址都是统一指针，无需从服务器读取数据
            memcpy(dst, src, count);
        } else if(dst_is_union) { // 目的地址为统一指针，需要从服务器读取数据
            rpc_read(client, dst, count, false);
        }
        break;
    case cudaMemcpyHostToHost:
        memcpy(dst, src, count);
        serverSrc = getServerHostPtr((void *)src);
        if(serverSrc == nullptr) {
            serverSrc = getUnionPtr((void *)src);
        }
        serverDst = getServerHostPtr(dst);
        if(serverDst == nullptr) {
            serverDst = getUnionPtr(dst);
        }
        if(serverDst == nullptr) {
            rpc_free_client(client);
            return cudaSuccess; // 本地内存拷贝,不需要服务器端处理
        }
        rpc_write(client, &serverDst, sizeof(serverDst));
        rpc_write(client, &serverSrc, sizeof(serverSrc));
        rpc_write(client, &count, sizeof(count));
        rpc_write(client, &kind, sizeof(kind));
        rpc_write(client, src, count, false);
        break;
    }
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    std::cout << "Hook: cudaMemset called" << std::endl;
    cudaError_t _result;
    bool isUnion = true;
    void *serverPtr;
    serverPtr = getServerHostPtr(devPtr);
    if(serverPtr == nullptr) {
        serverPtr = getUnionPtr(devPtr);
    }
    if(serverPtr != nullptr) {
        memset(devPtr, value, count);
    } else {
        serverPtr = getServerDevPtr(devPtr);
        if(serverPtr == nullptr) {
            serverPtr = devPtr;
        }
    }
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaMemset);
    rpc_write(client, &serverPtr, sizeof(serverPtr));
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

// extern "C" CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
//     // std::cout << "Hook: cuGetProcAddress called" << std::endl;
//     auto it = functionMap.find(symbol);
//     if(it != functionMap.end()) {
//         *pfn = (void *)it->second;
//         return CUDA_SUCCESS;
//     }
//     return CUDA_ERROR_NOT_FOUND;
// }

// Function to parse a PTX string and fill functions into a dynamically
// allocated array
#define MAX_FUNCTION_NAME 128
#define MAX_ARGS 32

struct Function {
    char *name;
    void *fat_cubin;       // the fat cubin that this function is a part of.
    const char *host_func; // if registered, points at the host function.
    int *arg_sizes;
    int arg_count;
};

std::vector<Function> functions;

static int get_type_size(const char *type) {
    if(*type == 'u' || *type == 's' || *type == 'f')
        type++;
    else
        return 0; // Unknown type
    if(*type == '8')
        return 1;
    if(*type == '1' && *(type + 1) == '6')
        return 2;
    if(*type == '3' && *(type + 1) == '2')
        return 4;
    if(*type == '6' && *(type + 1) == '4')
        return 8;
    return 0; // Unknown type
}

static void parse_ptx_string(void *fatCubin, const char *ptx_string, unsigned long long ptx_len) {
    for(unsigned long long i = 0; i < ptx_len; i++) {
        if(ptx_string[i] != '.' || i + 5 >= ptx_len || strncmp(ptx_string + i + 1, "entry", strlen("entry")) != 0)
            continue;

        char *name = (char *)malloc(MAX_FUNCTION_NAME);
        if(name == nullptr) {
            std::cerr << "Failed to allocate memory" << std::endl;
            exit(1);
        }
        int *arg_sizes = (int *)malloc(MAX_ARGS * sizeof(int));
        if(arg_sizes == nullptr) {
            std::cerr << "Failed to allocate memory" << std::endl;
            exit(1);
        }
        int arg_count = 0;

        i += strlen(".entry");
        while(i < ptx_len && !isalnum(ptx_string[i]) && ptx_string[i] != '_') {
            i++;
        }

        int j = 0;
        for(; j < MAX_FUNCTION_NAME - 1 && i < ptx_len && (isalnum(ptx_string[i]) || ptx_string[i] == '_');) {
            name[j++] = ptx_string[i++];
        }
        name[j] = '\0';

        while(i < ptx_len && ptx_string[i] != '(' && ptx_string[i] != '{') {
            i++;
        }

        if(ptx_string[i] == '(') {
            for(; arg_count < MAX_ARGS; arg_count++) {
                int arg_size = 0;

                while(i < ptx_len && (ptx_string[i] != '.' && ptx_string[i] != ')')) {
                    i++;
                }

                if(ptx_string[i] == ')') {
                    break;
                }

                // assert that the next token is "param"
                if(i + 5 >= ptx_len || strncmp(ptx_string + i, ".param", strlen(".param")) != 0) {
                    continue;
                }

                while(true) {
                    while(i < ptx_len && (ptx_string[i] != '.' && ptx_string[i] != ',' && ptx_string[i] != ')' && ptx_string[i] != '[')) {
                        i++;
                    }

                    if(ptx_string[i] == '.') {
                        // read the type, ignoring if it's not a valid type
                        int type_size = get_type_size(ptx_string + (++i));
                        if(type_size == 0) {
                            continue;
                        }
                        arg_size = type_size;
                    } else if(ptx_string[i] == '[') {
                        // this is an array type. read until the ]
                        int start = i + 1;
                        while(i < ptx_len && ptx_string[i] != ']') {
                            i++;
                        }

                        // parse the int value
                        int n = 0;
                        for(int j = start; j < i; j++) {
                            n = n * 10 + ptx_string[j] - '0';
                        }
                        arg_size *= n;
                    } else if(ptx_string[i] == ',' || ptx_string[i] == ')') {
                        break;
                    }
                }
                arg_sizes[arg_count] = arg_size;
            }
        }
        // add the function to the list
        functions.push_back(Function{
            .name = name,
            .fat_cubin = fatCubin,
            .host_func = nullptr,
            .arg_sizes = arg_sizes,
            .arg_count = arg_count,
        });
        // printf("function %s has %d arguments\n", name, arg_count);
    }
}

static size_t decompress(const uint8_t *input, size_t input_size, uint8_t *output, size_t output_size) {
    size_t ipos = 0, opos = 0;
    uint64_t next_nclen;  // length of next non-compressed segment
    uint64_t next_clen;   // length of next compressed segment
    uint64_t back_offset; // negative offset where redudant data is located,
                          // relative to current opos

    while(ipos < input_size) {
        next_nclen = (input[ipos] & 0xf0) >> 4;
        next_clen = 4 + (input[ipos] & 0xf);
        if(next_nclen == 0xf) {
            do {
                next_nclen += input[++ipos];
            } while(input[ipos] == 0xff);
        }

        if(memcpy(output + opos, input + (++ipos), next_nclen) == NULL) {
            fprintf(stderr, "Error copying data");
            return 0;
        }

        ipos += next_nclen;
        opos += next_nclen;
        if(ipos >= input_size || opos >= output_size) {
            break;
        }
        back_offset = input[ipos] + (input[ipos + 1] << 8);
        ipos += 2;
        if(next_clen == 0xf + 4) {
            do {
                next_clen += input[ipos++];
            } while(input[ipos - 1] == 0xff);
        }

        if(next_clen <= back_offset) {
            if(memcpy(output + opos, output + opos - back_offset, next_clen) == NULL) {
                fprintf(stderr, "Error copying data");
                return 0;
            }
        } else {
            if(memcpy(output + opos, output + opos - back_offset, back_offset) == NULL) {
                fprintf(stderr, "Error copying data");
                return 0;
            }
            for(size_t i = back_offset; i < next_clen; i++) {
                output[opos + i] = output[opos + i - back_offset];
            }
        }

        opos += next_clen;
    }
    return opos;
}

static ssize_t decompress_single_section(const uint8_t *input, uint8_t **output, size_t *output_size, struct __cudaFatCudaBinary2HeaderRec *eh, struct __cudaFatCudaBinary2EntryRec *th) {
    size_t padding;
    size_t input_read = 0;
    size_t output_written = 0;
    size_t decompress_ret = 0;
    const uint8_t zeroes[8] = {0};

    if(input == NULL || output == NULL || eh == NULL || th == NULL) {
        return 1;
    }

    uint8_t *mal = (uint8_t *)malloc(th->uncompressedBinarySize + 7);

    // add max padding of 7 bytes
    if((*output = mal) == NULL) {
        goto error;
    }

    decompress_ret = decompress(input, th->binarySize, *output, th->uncompressedBinarySize);

    // @brodey - keeping this temporarily so that we can compare the compression
    // returns
    if(decompress_ret != th->uncompressedBinarySize) {
        std::cout << "failed actual decompress..." << std::endl;
        goto error;
    }
    input_read += th->binarySize;
    output_written += th->uncompressedBinarySize;

    padding = ((8 - (size_t)(input + input_read)) % 8);
    if(memcmp(input + input_read, zeroes, padding) != 0) {
        goto error;
    }
    input_read += padding;

    padding = ((8 - (size_t)th->uncompressedBinarySize) % 8);
    // Because we always allocated enough memory for one more elf_header and this
    // is smaller than the maximal padding of 7, we do not have to reallocate
    // here.
    memset(*output, 0, padding);
    output_written += padding;

    *output_size = output_written;
    return input_read;
error:
    free(*output);
    *output = NULL;
    return -1;
}

static void parseFatBinary(void *fatCubin, __cudaFatCudaBinary2Header *header) {
    char *base = (char *)(header + 1);
    long long unsigned int offset = 0;
    __cudaFatCudaBinary2EntryRec *entry = (__cudaFatCudaBinary2EntryRec *)(base);

    while(offset < header->size) {
        entry = (__cudaFatCudaBinary2EntryRec *)(base + offset);
        offset += entry->binary + entry->binarySize;
        if(!(entry->type & FATBIN_2_PTX))
            continue;
        if(entry->flags & FATBIN_FLAG_COMPRESS) {
            uint8_t *text_data = NULL;
            size_t text_data_size = 0;
            if(decompress_single_section((const uint8_t *)entry + entry->binary, &text_data, &text_data_size, header, entry) < 0) {
                std::cout << "decompressing failed..." << std::endl;
                return;
            }
            parse_ptx_string(fatCubin, (char *)text_data, text_data_size);
        } else {
            parse_ptx_string(fatCubin, (char *)entry + entry->binary, entry->binarySize);
        }
    }
}

extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
    std::cout << "Hook: cudaLaunchKernel called" << std::endl;
    cudaError_t _result;
    Function *f = nullptr;
    for(auto &function : functions) {
        if(function.host_func == func) {
            f = &function;
            break;
        }
    }
    if(f == nullptr) {
        std::cerr << "Failed to find function" << std::endl;
        return cudaErrorInvalidDeviceFunction;
    }

    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaLaunchKernel);
    rpc_write(client, &func, sizeof(func));
    rpc_write(client, &gridDim, sizeof(gridDim));
    rpc_write(client, &blockDim, sizeof(blockDim));
    rpc_write(client, &sharedMem, sizeof(sharedMem));
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &f->arg_count, sizeof(f->arg_count));

    for(int i = 0; i < f->arg_count; i++) {
        rpc_write(client, args[i], f->arg_sizes[i]);
    }
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, struct CUstream_st *stream) {
    std::cout << "Hook: __cudaPushCallConfiguration called" << std::endl;
    unsigned _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaPushCallConfiguration);
    rpc_write(client, &gridDim, sizeof(gridDim));
    rpc_write(client, &blockDim, sizeof(blockDim));
    rpc_write(client, &sharedMem, sizeof(sharedMem));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream) {
    std::cout << "Hook: __cudaPopCallConfiguration called" << std::endl;
    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaPopCallConfiguration);
    rpc_read(client, gridDim, sizeof(*gridDim));
    rpc_read(client, blockDim, sizeof(*blockDim));
    rpc_read(client, sharedMem, sizeof(*sharedMem));
    rpc_read(client, stream, sizeof(cudaStream_t));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" void **__cudaRegisterFatBinary(void *fatCubin) {
    std::cout << "Hook: __cudaRegisterFatBinary called" << std::endl;
    void **_result = nullptr;
    __cudaFatCudaBinary2 *binary;
    __cudaFatCudaBinary2Header *header;
    unsigned long long size;

    binary = (__cudaFatCudaBinary2 *)fatCubin;
    header = (__cudaFatCudaBinary2Header *)binary->text;
    size = sizeof(__cudaFatCudaBinary2Header) + header->size;

    if(*(unsigned *)fatCubin != __cudaFatMAGIC2) {
        std::cerr << "Invalid fat binary magic" << std::endl;
        return nullptr;
    }
    parseFatBinary(fatCubin, header);

    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaRegisterFatBinary);
    rpc_write(client, binary, sizeof(__cudaFatCudaBinary2));
    rpc_write(client, header, size, true);
    printf("~~~~~~~~~~~~~~~~~~ %llu\n", size);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    std::cout << "Hook: __cudaRegisterFatBinaryEnd called" << std::endl;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaRegisterFatBinaryEnd);
    rpc_write(client, &fatCubinHandle, sizeof(fatCubinHandle));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    std::cout << "Hook: __cudaUnregisterFatBinary called" << std::endl;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaUnregisterFatBinary);
    rpc_write(client, &fatCubinHandle, sizeof(fatCubinHandle));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
}

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global) {
    std::cout << "Hook: __cudaRegisterVar called" << std::endl;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaRegisterVar);
    rpc_write(client, &fatCubinHandle, sizeof(fatCubinHandle));
    rpc_write(client, &hostVar, sizeof(hostVar));
    rpc_write(client, deviceAddress, strlen(deviceAddress) + 1);
    rpc_write(client, deviceName, strlen(deviceName) + 1);
    rpc_write(client, &ext, sizeof(ext));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &constant, sizeof(constant));
    rpc_write(client, &global, sizeof(global));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
}

extern "C" void __cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global) {
    std::cout << "Hook: __cudaRegisterManagedVar called" << std::endl;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaRegisterManagedVar);
    rpc_write(client, &fatCubinHandle, sizeof(fatCubinHandle));
    rpc_write(client, &hostVarPtrAddress, sizeof(hostVarPtrAddress));
    rpc_write(client, deviceAddress, strlen(deviceAddress) + 1);
    rpc_write(client, deviceName, strlen(deviceName) + 1);
    rpc_write(client, &ext, sizeof(ext));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &constant, sizeof(constant));
    rpc_write(client, &global, sizeof(global));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }

    rpc_free_client(client);
}

extern "C" char __cudaInitModule(void **fatCubinHandle) {
    std::cout << "Hook: __cudaInitModule called" << std::endl;
    char _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaInitModule);
    // PARAM void **fatCubinHandle
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **fatCubinHandle
    rpc_free_client(client);
    return _result;
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    std::cout << "Hook: __cudaRegisterFunction called" << std::endl;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaRegisterFunction);
    rpc_write(client, &fatCubinHandle, sizeof(fatCubinHandle));
    rpc_write(client, &hostFun, sizeof(hostFun));
    rpc_write(client, deviceFun, strlen(deviceFun) + 1, true);
    rpc_write(client, deviceName, strlen(deviceName) + 1, true);
    rpc_write(client, &thread_limit, sizeof(thread_limit));
    uint8_t mask = 0;
    if(tid != nullptr)
        mask |= 1 << 0;
    if(bid != nullptr)
        mask |= 1 << 1;
    if(bDim != nullptr)
        mask |= 1 << 2;
    if(gDim != nullptr)
        mask |= 1 << 3;
    if(wSize != nullptr)
        mask |= 1 << 4;
    rpc_write(client, &mask, sizeof(mask));
    if(tid != nullptr) {
        rpc_write(client, tid, sizeof(uint3));
    }
    if(bid != nullptr) {
        rpc_write(client, bid, sizeof(uint3));
    }
    if(bDim != nullptr) {
        rpc_write(client, bDim, sizeof(dim3));
    }
    if(gDim != nullptr) {
        rpc_write(client, gDim, sizeof(dim3));
    }
    if(wSize != nullptr) {
        rpc_write(client, wSize, sizeof(int));
    }
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    // also memorize the host pointer function
    for(auto &function : functions) {
        if(strcmp(function.name, deviceName) == 0) {
            function.host_func = hostFun;
            printf("register function %s %p\n", function.name, function.host_func);
        }
    }
    rpc_free_client(client);
}

extern "C" cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    std::cout << "Hook: cudaMemcpyToSymbol called" << std::endl;
    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaMemcpyToSymbol);
    rpc_write(client, &symbol, sizeof(symbol));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &kind, sizeof(kind));
    // 如果是统一内存，或设备内存都只需要将指针地址拷贝到server
    // 否则，写入null指针，然后再写入数据到server
    void *serverSrc = getUnionPtr((void *)src);
    if(serverSrc == nullptr) {
        serverSrc = getServerDevPtr((void *)src);
    }
    rpc_write(client, &serverSrc, sizeof(serverSrc));
    if(serverSrc == nullptr) {
        rpc_write(client, (uint8_t *)src, count);
    }
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    std::cout << "Hook: cudaMemcpyFromSymbol called" << std::endl;
    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaMemcpyFromSymbol);
    bool is_managed = true;
    void *serverDst = getUnionPtr(dst);
    if(serverDst == nullptr) {
        is_managed = false;
        serverDst = getServerDevPtr(dst);
    }
    rpc_write(client, &serverDst, sizeof(serverDst));
    rpc_write(client, &symbol, sizeof(symbol));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &kind, sizeof(kind));
    if(is_managed || serverDst == nullptr) {
        rpc_read(client, dst, count);
    }
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}
