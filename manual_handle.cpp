#include <iostream>
#include <unordered_map>
#include "gen/hook_api.h"
#include "gen/handle_server.h"
#include "rpc.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "hidden_api.h"

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
    printf("---- size: %lu\n", size);
    printf("---- flags: %u\n", flags);
    cudaError_t _result = cudaHostAlloc(&pHost, size, flags);
    printf("pHost: %p\n", pHost);
    printf("result: %d\n", _result);
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

int handle_cuGetProcAddress(void *args0) {
    std::cout << "Handle function cuGetProcAddress called" << std::endl;
    return 0;
}

int handle___cudaPushCallConfiguration(void *args0) {
    std::cout << "Handle function __cudaPushCallConfiguration called" << std::endl;
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

int handle___cudaPopCallConfiguration(void *args0) {
    std::cout << "Handle function __cudaPopCallConfiguration called" << std::endl;
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

int handle___cudaRegisterFatBinary(void *args0) {
    std::cout << "Handle function __cudaRegisterFatBinary called" << std::endl;
    RpcClient *client = (RpcClient *)args0;
    __cudaFatCudaBinary2 *fatCubin = (__cudaFatCudaBinary2 *)malloc(sizeof(__cudaFatCudaBinary2));
    if(fatCubin == nullptr) {
        std::cerr << "Failed to allocate fatCubin" << std::endl;
        return 1;
    }
    unsigned long long size;
    rpc_read(client, fatCubin, sizeof(__cudaFatCudaBinary2));
    rpc_read(client, &size, sizeof(size));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    void *cubin = malloc(size);
    if(cubin == nullptr) {
        std::cerr << "Failed to allocate cubin" << std::endl;
        free(fatCubin);
        return 1;
    }
    rpc_read_now(client, cubin, size);
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
    std::cout << "Handle function __cudaRegisterFatBinaryEnd called" << std::endl;
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

int handle___cudaUnregisterFatBinary(void *args0) {
    std::cout << "Handle function __cudaUnregisterFatBinary called" << std::endl;
    RpcClient *client = (RpcClient *)args0;
    // PARAM void **fatCubinHandle
    void *fatCubinHandle;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    // PARAM void **fatCubinHandle
    __cudaUnregisterFatBinary(&fatCubinHandle);
    // PARAM void **fatCubinHandle
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle___cudaRegisterVar(void *args0) {
    std::cout << "Handle function __cudaRegisterVar called" << std::endl;
    RpcClient *client = (RpcClient *)args0;
    // PARAM void **fatCubinHandle
    void *fatCubinHandle;
    char hostVar[1024];
    char deviceAddress[1024];
    char deviceName[1024];
    rpc_read(client, deviceName, 1024);
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
        return 1;
    }
    // PARAM void **fatCubinHandle
    __cudaRegisterVar(&fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
    // PARAM void **fatCubinHandle
    rpc_write(client, hostVar, strlen(hostVar) + 1);
    rpc_write(client, deviceAddress, strlen(deviceAddress) + 1);
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle___cudaRegisterManagedVar(void *args0) {
    std::cout << "Handle function __cudaRegisterManagedVar called" << std::endl;
    RpcClient *client = (RpcClient *)args0;
    // PARAM void **fatCubinHandle
    void *fatCubinHandle;
    // PARAM void **hostVarPtrAddress
    void *hostVarPtrAddress;
    char deviceAddress[1024];
    char deviceName[1024];
    rpc_read(client, deviceName, 1024);
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
        return 1;
    }
    // PARAM void **fatCubinHandle
    // PARAM void **hostVarPtrAddress
    __cudaRegisterManagedVar(&fatCubinHandle, &hostVarPtrAddress, deviceAddress, deviceName, ext, size, constant, global);
    // PARAM void **fatCubinHandle
    // PARAM void **hostVarPtrAddress
    rpc_write(client, deviceAddress, strlen(deviceAddress) + 1);
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle___cudaInitModule(void *args0) {
    std::cout << "Handle function __cudaInitModule called" << std::endl;
    RpcClient *client = (RpcClient *)args0;
    // PARAM void **fatCubinHandle
    void *fatCubinHandle;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    // PARAM void **fatCubinHandle
    char _result = __cudaInitModule(&fatCubinHandle);
    // PARAM void **fatCubinHandle
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle___cudaRegisterTexture(void *args0) {
    std::cout << "Handle function __cudaRegisterTexture called" << std::endl;
    RpcClient *client = (RpcClient *)args0;
    // PARAM void **fatCubinHandle
    void *fatCubinHandle;
    struct textureReference hostVar;
    rpc_read(client, &hostVar, sizeof(hostVar));
    // PARAM const void **deviceAddress
    const void *deviceAddress;
    char deviceName[1024];
    rpc_read(client, deviceName, 1024);
    int dim;
    rpc_read(client, &dim, sizeof(dim));
    int norm;
    rpc_read(client, &norm, sizeof(norm));
    int ext;
    rpc_read(client, &ext, sizeof(ext));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    // PARAM void **fatCubinHandle
    // PARAM const void **deviceAddress
    __cudaRegisterTexture(&fatCubinHandle, &hostVar, &deviceAddress, deviceName, dim, norm, ext);
    // PARAM void **fatCubinHandle
    // PARAM const void **deviceAddress
    rpc_write(client, &deviceAddress, sizeof(deviceAddress));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle___cudaRegisterSurface(void *args0) {
    std::cout << "Handle function __cudaRegisterSurface called" << std::endl;
    RpcClient *client = (RpcClient *)args0;
    // PARAM void **fatCubinHandle
    void *fatCubinHandle;
    struct surfaceReference hostVar;
    rpc_read(client, &hostVar, sizeof(hostVar));
    // PARAM const void **deviceAddress
    const void *deviceAddress;
    char deviceName[1024];
    rpc_read(client, deviceName, 1024);
    int dim;
    rpc_read(client, &dim, sizeof(dim));
    int ext;
    rpc_read(client, &ext, sizeof(ext));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    // PARAM void **fatCubinHandle
    // PARAM const void **deviceAddress
    __cudaRegisterSurface(&fatCubinHandle, &hostVar, &deviceAddress, deviceName, dim, ext);
    // PARAM void **fatCubinHandle
    // PARAM const void **deviceAddress
    rpc_write(client, &deviceAddress, sizeof(deviceAddress));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }

    return 0;
}

int handle___cudaRegisterFunction(void *args0) {
    std::cout << "Handle function __cudaRegisterFunction called" << std::endl;
    RpcClient *client = (RpcClient *)args0;
    // PARAM void **fatCubinHandle
    void **fatCubinHandle;
    char *hostFun;
    char deviceFun[1024];
    char deviceName[1024];
    int thread_limit;
    uint3 tid;
    uint3 bid;
    dim3 bDim;
    dim3 gDim;
    int wSize;
    uint8_t mask;
    rpc_read(client, &fatCubinHandle, sizeof(fatCubinHandle));
    rpc_read(client, &hostFun, sizeof(hostFun));
    rpc_read(client, deviceFun, 1024);
    rpc_read(client, deviceName, 1024);
    rpc_read(client, &thread_limit, sizeof(thread_limit));
    rpc_read(client, &mask, sizeof(mask));

    if(mask & 1 << 0) {
        rpc_read(client, &tid, sizeof(uint3));
    }
    if(mask & 1 << 1) {
        rpc_read(client, &bid, sizeof(uint3));
    }
    if(mask & 1 << 2) {
        rpc_read(client, &bDim, sizeof(dim3));
    }
    if(mask & 1 << 3) {
        rpc_read(client, &gDim, sizeof(dim3));
    }
    if(mask & 1 << 4) {
        rpc_read(client, &wSize, sizeof(wSize));
    }
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        return 1;
    }
    __cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, mask & 1 << 0 ? &tid : nullptr, mask & 1 << 1 ? &bid : nullptr, mask & 1 << 2 ? &bDim : nullptr, mask & 1 << 3 ? &gDim : nullptr, mask & 1 << 4 ? &wSize : nullptr);
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        return 1;
    }
    return 0;
}
