#include <iostream>
#include <map>
#include <set>
#include <cstdlib>
#include "cuda_runtime.h"
#include <sys/mman.h>
#include "cuda.h"
#include "gen/hook_api.h"
#include "rpc.h"

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
        exit(1);
    }
    if(_result == cudaSuccess) {
        cs_host_mems[*ptr] = std::make_pair(serverPtr, size);
    } else {
        free(*ptr);
    }
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
        exit(1);
    }
    if(_result == cudaSuccess) {
        cs_host_mems[*pHost] = std::make_pair(serverPtr, size);
    } else {
        free(*pHost);
    }
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
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_host_mems[*pp] = std::make_pair(serverPtr, bytesize);
    } else {
        free(*pp);
    }
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
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_host_mems[*pp] = std::make_pair(serverPtr, bytesize);
    } else {
        free(*pp);
    }
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
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_reserve_mems[*ptr] = *serverPtr;
    } else {
        munmap((void *)*ptr, size);
    }
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
    } else {
        free(*devPtr);
    }
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
        exit(1);
    }
    if(_result == cudaSuccess) {
        free(ptr);
        cs_host_mems.erase(ptr);
    }
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
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        free(p);
        cs_host_mems.erase(p);
    }
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
    printf("1----------------------\n");
    if(rpc_submit_request(client) != 0) {
        printf("2----------------------\n");
        std::cerr << "Failed to submit request" << std::endl;
        exit(1);
    }
    if(_result == cudaSuccess) {
        freeDevPtr(devPtr);
    }
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
    return _result;
}

// 用#ifdef来区分不同的cuda版本,如果没有CUDA_VERSION或其值大于11040定义下面的函数
// #if CUDA_VERSION > 11040
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
    return _result;
}

// #endif

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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
    rpc_release_client(client);
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
        rpc_write(client, src, count);
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
        rpc_read(client, dst, count);
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
            rpc_read(client, dst, count);
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
        rpc_write(client, src, count);
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
