#include <iostream>
#include <unordered_map>
#include "cuda.h"

#include "hook_api.h"
#include "../rpc.h"
extern void *(*real_dlsym)(void *, const char *);

extern "C" void *mem2server(void *clientPtr, size_t size);
extern "C" void mem2client(void *clientPtr, size_t size);
void *get_so_handle(const std::string &so_file);
extern "C" CUresult cuInit(unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuInit called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuInit);
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDriverGetVersion(int *driverVersion) {
#ifdef DEBUG
    std::cout << "Hook: cuDriverGetVersion called" << std::endl;
#endif
    void *_0driverVersion = mem2server((void *)driverVersion, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDriverGetVersion);
    rpc_write(client, &_0driverVersion, sizeof(_0driverVersion));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)driverVersion, 0);
    return _result;
}

extern "C" CUresult cuDeviceGet(CUdevice *device, int ordinal) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGet called" << std::endl;
#endif
    void *_0device = mem2server((void *)device, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGet);
    rpc_write(client, &_0device, sizeof(_0device));
    rpc_write(client, &ordinal, sizeof(ordinal));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)device, 0);
    return _result;
}

extern "C" CUresult cuDeviceGetCount(int *count) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetCount called" << std::endl;
#endif
    void *_0count = mem2server((void *)count, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetCount);
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)count, 0);
    return _result;
}

extern "C" CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetName called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetName);
    rpc_read(client, name, len, true);
    rpc_write(client, &len, sizeof(len));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetUuid called" << std::endl;
#endif
    void *_0uuid = mem2server((void *)uuid, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetUuid);
    rpc_write(client, &_0uuid, sizeof(_0uuid));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)uuid, 0);
    return _result;
}

extern "C" CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetUuid_v2 called" << std::endl;
#endif
    void *_0uuid = mem2server((void *)uuid, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetUuid_v2);
    rpc_write(client, &_0uuid, sizeof(_0uuid));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)uuid, 0);
    return _result;
}

extern "C" CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetLuid called" << std::endl;
#endif
    void *_0deviceNodeMask = mem2server((void *)deviceNodeMask, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetLuid);
    rpc_read(client, luid, 32, true);
    rpc_write(client, &_0deviceNodeMask, sizeof(_0deviceNodeMask));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)deviceNodeMask, 0);
    return _result;
}

extern "C" CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceTotalMem_v2 called" << std::endl;
#endif
    void *_0bytes = mem2server((void *)bytes, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceTotalMem_v2);
    rpc_write(client, &_0bytes, sizeof(_0bytes));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)bytes, 0);
    return _result;
}

extern "C" CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetTexture1DLinearMaxWidth called" << std::endl;
#endif
    void *_0maxWidthInElements = mem2server((void *)maxWidthInElements, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetTexture1DLinearMaxWidth);
    rpc_write(client, &_0maxWidthInElements, sizeof(_0maxWidthInElements));
    rpc_write(client, &format, sizeof(format));
    rpc_write(client, &numChannels, sizeof(numChannels));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)maxWidthInElements, 0);
    return _result;
}

extern "C" CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetAttribute called" << std::endl;
#endif
    void *_0pi = mem2server((void *)pi, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetAttribute);
    rpc_write(client, &_0pi, sizeof(_0pi));
    rpc_write(client, &attrib, sizeof(attrib));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pi, 0);
    return _result;
}

extern "C" CUresult cuDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, CUdevice dev, int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetNvSciSyncAttributes called" << std::endl;
#endif
    void *_0nvSciSyncAttrList = mem2server((void *)nvSciSyncAttrList, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetNvSciSyncAttributes);
    rpc_write(client, &_0nvSciSyncAttrList, sizeof(_0nvSciSyncAttrList));
    rpc_write(client, &dev, sizeof(dev));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nvSciSyncAttrList, 0);
    return _result;
}

extern "C" CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceSetMemPool called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceSetMemPool);
    rpc_write(client, &dev, sizeof(dev));
    rpc_write(client, &pool, sizeof(pool));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetMemPool called" << std::endl;
#endif
    void *_0pool = mem2server((void *)pool, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetMemPool);
    rpc_write(client, &_0pool, sizeof(_0pool));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pool, 0);
    return _result;
}

extern "C" CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetDefaultMemPool called" << std::endl;
#endif
    void *_0pool_out = mem2server((void *)pool_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetDefaultMemPool);
    rpc_write(client, &_0pool_out, sizeof(_0pool_out));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pool_out, 0);
    return _result;
}

extern "C" CUresult cuDeviceGetExecAffinitySupport(int *pi, CUexecAffinityType type, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetExecAffinitySupport called" << std::endl;
#endif
    void *_0pi = mem2server((void *)pi, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetExecAffinitySupport);
    rpc_write(client, &_0pi, sizeof(_0pi));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pi, 0);
    return _result;
}

extern "C" CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) {
#ifdef DEBUG
    std::cout << "Hook: cuFlushGPUDirectRDMAWrites called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuFlushGPUDirectRDMAWrites);
    rpc_write(client, &target, sizeof(target));
    rpc_write(client, &scope, sizeof(scope));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetProperties called" << std::endl;
#endif
    void *_0prop = mem2server((void *)prop, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetProperties);
    rpc_write(client, &_0prop, sizeof(_0prop));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)prop, 0);
    return _result;
}

extern "C" CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceComputeCapability called" << std::endl;
#endif
    void *_0major = mem2server((void *)major, 0);
    void *_0minor = mem2server((void *)minor, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceComputeCapability);
    rpc_write(client, &_0major, sizeof(_0major));
    rpc_write(client, &_0minor, sizeof(_0minor));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)major, 0);
    mem2client((void *)minor, 0);
    return _result;
}

extern "C" CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxRetain called" << std::endl;
#endif
    void *_0pctx = mem2server((void *)pctx, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDevicePrimaryCtxRetain);
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pctx, 0);
    return _result;
}

extern "C" CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxRelease_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDevicePrimaryCtxRelease_v2);
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxSetFlags_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDevicePrimaryCtxSetFlags_v2);
    rpc_write(client, &dev, sizeof(dev));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxGetState called" << std::endl;
#endif
    void *_0flags = mem2server((void *)flags, 0);
    void *_0active = mem2server((void *)active, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDevicePrimaryCtxGetState);
    rpc_write(client, &dev, sizeof(dev));
    rpc_write(client, &_0flags, sizeof(_0flags));
    rpc_write(client, &_0active, sizeof(_0active));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)flags, 0);
    mem2client((void *)active, 0);
    return _result;
}

extern "C" CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxReset_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDevicePrimaryCtxReset_v2);
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxCreate_v2 called" << std::endl;
#endif
    void *_0pctx = mem2server((void *)pctx, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxCreate_v2);
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    rpc_write(client, &flags, sizeof(flags));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pctx, 0);
    return _result;
}

extern "C" CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams, unsigned int flags, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxCreate_v3 called" << std::endl;
#endif
    void *_0pctx = mem2server((void *)pctx, 0);
    void *_0paramsArray = mem2server((void *)paramsArray, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxCreate_v3);
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    rpc_write(client, &_0paramsArray, sizeof(_0paramsArray));
    rpc_write(client, &numParams, sizeof(numParams));
    rpc_write(client, &flags, sizeof(flags));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pctx, 0);
    mem2client((void *)paramsArray, 0);
    return _result;
}

extern "C" CUresult cuCtxCreate_v4(CUcontext *pctx, CUctxCreateParams *ctxCreateParams, unsigned int flags, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxCreate_v4 called" << std::endl;
#endif
    void *_0pctx = mem2server((void *)pctx, 0);
    void *_0ctxCreateParams = mem2server((void *)ctxCreateParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxCreate_v4);
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    rpc_write(client, &_0ctxCreateParams, sizeof(_0ctxCreateParams));
    rpc_write(client, &flags, sizeof(flags));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pctx, 0);
    mem2client((void *)ctxCreateParams, 0);
    return _result;
}

extern "C" CUresult cuCtxDestroy_v2(CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxDestroy_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxDestroy_v2);
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxPushCurrent_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxPushCurrent_v2);
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxPopCurrent_v2 called" << std::endl;
#endif
    void *_0pctx = mem2server((void *)pctx, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxPopCurrent_v2);
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pctx, 0);
    return _result;
}

extern "C" CUresult cuCtxSetCurrent(CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSetCurrent called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxSetCurrent);
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxGetCurrent(CUcontext *pctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetCurrent called" << std::endl;
#endif
    void *_0pctx = mem2server((void *)pctx, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxGetCurrent);
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pctx, 0);
    return _result;
}

extern "C" CUresult cuCtxGetDevice(CUdevice *device) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetDevice called" << std::endl;
#endif
    void *_0device = mem2server((void *)device, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxGetDevice);
    rpc_write(client, &_0device, sizeof(_0device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)device, 0);
    return _result;
}

extern "C" CUresult cuCtxGetFlags(unsigned int *flags) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetFlags called" << std::endl;
#endif
    void *_0flags = mem2server((void *)flags, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxGetFlags);
    rpc_write(client, &_0flags, sizeof(_0flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)flags, 0);
    return _result;
}

extern "C" CUresult cuCtxSetFlags(unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSetFlags called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxSetFlags);
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxGetId(CUcontext ctx, unsigned long long *ctxId) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetId called" << std::endl;
#endif
    void *_0ctxId = mem2server((void *)ctxId, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxGetId);
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_write(client, &_0ctxId, sizeof(_0ctxId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)ctxId, 0);
    return _result;
}

extern "C" CUresult cuCtxSynchronize() {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSynchronize called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxSynchronize);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSetLimit called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxSetLimit);
    rpc_write(client, &limit, sizeof(limit));
    rpc_write(client, &value, sizeof(value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetLimit called" << std::endl;
#endif
    void *_0pvalue = mem2server((void *)pvalue, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxGetLimit);
    rpc_write(client, &_0pvalue, sizeof(_0pvalue));
    rpc_write(client, &limit, sizeof(limit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pvalue, 0);
    return _result;
}

extern "C" CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetCacheConfig called" << std::endl;
#endif
    void *_0pconfig = mem2server((void *)pconfig, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxGetCacheConfig);
    rpc_write(client, &_0pconfig, sizeof(_0pconfig));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pconfig, 0);
    return _result;
}

extern "C" CUresult cuCtxSetCacheConfig(CUfunc_cache config) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSetCacheConfig called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxSetCacheConfig);
    rpc_write(client, &config, sizeof(config));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetApiVersion called" << std::endl;
#endif
    void *_0version = mem2server((void *)version, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxGetApiVersion);
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_write(client, &_0version, sizeof(_0version));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)version, 0);
    return _result;
}

extern "C" CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetStreamPriorityRange called" << std::endl;
#endif
    void *_0leastPriority = mem2server((void *)leastPriority, 0);
    void *_0greatestPriority = mem2server((void *)greatestPriority, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxGetStreamPriorityRange);
    rpc_write(client, &_0leastPriority, sizeof(_0leastPriority));
    rpc_write(client, &_0greatestPriority, sizeof(_0greatestPriority));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)leastPriority, 0);
    mem2client((void *)greatestPriority, 0);
    return _result;
}

extern "C" CUresult cuCtxResetPersistingL2Cache() {
#ifdef DEBUG
    std::cout << "Hook: cuCtxResetPersistingL2Cache called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxResetPersistingL2Cache);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxGetExecAffinity(CUexecAffinityParam *pExecAffinity, CUexecAffinityType type) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetExecAffinity called" << std::endl;
#endif
    void *_0pExecAffinity = mem2server((void *)pExecAffinity, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxGetExecAffinity);
    rpc_write(client, &_0pExecAffinity, sizeof(_0pExecAffinity));
    rpc_write(client, &type, sizeof(type));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pExecAffinity, 0);
    return _result;
}

extern "C" CUresult cuCtxRecordEvent(CUcontext hCtx, CUevent hEvent) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxRecordEvent called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxRecordEvent);
    rpc_write(client, &hCtx, sizeof(hCtx));
    rpc_write(client, &hEvent, sizeof(hEvent));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxWaitEvent(CUcontext hCtx, CUevent hEvent) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxWaitEvent called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxWaitEvent);
    rpc_write(client, &hCtx, sizeof(hCtx));
    rpc_write(client, &hEvent, sizeof(hEvent));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxAttach called" << std::endl;
#endif
    void *_0pctx = mem2server((void *)pctx, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxAttach);
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pctx, 0);
    return _result;
}

extern "C" CUresult cuCtxDetach(CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxDetach called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxDetach);
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetSharedMemConfig called" << std::endl;
#endif
    void *_0pConfig = mem2server((void *)pConfig, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxGetSharedMemConfig);
    rpc_write(client, &_0pConfig, sizeof(_0pConfig));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pConfig, 0);
    return _result;
}

extern "C" CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSetSharedMemConfig called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxSetSharedMemConfig);
    rpc_write(client, &config, sizeof(config));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuModuleLoad(CUmodule *module, const char *fname) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleLoad called" << std::endl;
#endif
    void *_0module = mem2server((void *)module, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuModuleLoad);
    rpc_write(client, &_0module, sizeof(_0module));
    rpc_write(client, fname, strlen(fname) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)module, 0);
    return _result;
}

extern "C" CUresult cuModuleLoadData(CUmodule *module, const void *image) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleLoadData called" << std::endl;
#endif
    void *_0module = mem2server((void *)module, 0);
    void *_0image = mem2server((void *)image, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuModuleLoadData);
    rpc_write(client, &_0module, sizeof(_0module));
    rpc_write(client, &_0image, sizeof(_0image));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)module, 0);
    mem2client((void *)image, 0);
    return _result;
}

extern "C" CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleLoadDataEx called" << std::endl;
#endif
    void *_0module = mem2server((void *)module, 0);
    void *_0image = mem2server((void *)image, 0);
    void *_0options = mem2server((void *)options, 0);
    // PARAM void **optionValues
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuModuleLoadDataEx);
    rpc_write(client, &_0module, sizeof(_0module));
    rpc_write(client, &_0image, sizeof(_0image));
    rpc_write(client, &numOptions, sizeof(numOptions));
    rpc_write(client, &_0options, sizeof(_0options));
    // PARAM void **optionValues
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **optionValues
    rpc_free_client(client);
    mem2client((void *)module, 0);
    mem2client((void *)image, 0);
    mem2client((void *)options, 0);
    // PARAM void **optionValues
    return _result;
}

extern "C" CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleLoadFatBinary called" << std::endl;
#endif
    void *_0module = mem2server((void *)module, 0);
    void *_0fatCubin = mem2server((void *)fatCubin, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuModuleLoadFatBinary);
    rpc_write(client, &_0module, sizeof(_0module));
    rpc_write(client, &_0fatCubin, sizeof(_0fatCubin));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)module, 0);
    mem2client((void *)fatCubin, 0);
    return _result;
}

extern "C" CUresult cuModuleUnload(CUmodule hmod) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleUnload called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuModuleUnload);
    rpc_write(client, &hmod, sizeof(hmod));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode *mode) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleGetLoadingMode called" << std::endl;
#endif
    void *_0mode = mem2server((void *)mode, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuModuleGetLoadingMode);
    rpc_write(client, &_0mode, sizeof(_0mode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)mode, 0);
    return _result;
}

extern "C" CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleGetFunction called" << std::endl;
#endif
    void *_0hfunc = mem2server((void *)hfunc, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuModuleGetFunction);
    rpc_write(client, &_0hfunc, sizeof(_0hfunc));
    rpc_write(client, &hmod, sizeof(hmod));
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)hfunc, 0);
    return _result;
}

extern "C" CUresult cuModuleGetFunctionCount(unsigned int *count, CUmodule mod) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleGetFunctionCount called" << std::endl;
#endif
    void *_0count = mem2server((void *)count, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuModuleGetFunctionCount);
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_write(client, &mod, sizeof(mod));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)count, 0);
    return _result;
}

extern "C" CUresult cuModuleEnumerateFunctions(CUfunction *functions, unsigned int numFunctions, CUmodule mod) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleEnumerateFunctions called" << std::endl;
#endif
    void *_0functions = mem2server((void *)functions, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuModuleEnumerateFunctions);
    rpc_write(client, &_0functions, sizeof(_0functions));
    rpc_write(client, &numFunctions, sizeof(numFunctions));
    rpc_write(client, &mod, sizeof(mod));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)functions, 0);
    return _result;
}

extern "C" CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkCreate_v2 called" << std::endl;
#endif
    void *_0options = mem2server((void *)options, 0);
    // PARAM void **optionValues
    void *_0stateOut = mem2server((void *)stateOut, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLinkCreate_v2);
    rpc_write(client, &numOptions, sizeof(numOptions));
    rpc_write(client, &_0options, sizeof(_0options));
    // PARAM void **optionValues
    rpc_write(client, &_0stateOut, sizeof(_0stateOut));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **optionValues
    rpc_free_client(client);
    mem2client((void *)options, 0);
    // PARAM void **optionValues
    mem2client((void *)stateOut, 0);
    return _result;
}

extern "C" CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkAddData_v2 called" << std::endl;
#endif
    void *_0data = mem2server((void *)data, 0);
    void *_0options = mem2server((void *)options, 0);
    // PARAM void **optionValues
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLinkAddData_v2);
    rpc_write(client, &state, sizeof(state));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &_0data, sizeof(_0data));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_write(client, &numOptions, sizeof(numOptions));
    rpc_write(client, &_0options, sizeof(_0options));
    // PARAM void **optionValues
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **optionValues
    rpc_free_client(client);
    mem2client((void *)data, 0);
    mem2client((void *)options, 0);
    // PARAM void **optionValues
    return _result;
}

extern "C" CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkAddFile_v2 called" << std::endl;
#endif
    void *_0options = mem2server((void *)options, 0);
    // PARAM void **optionValues
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLinkAddFile_v2);
    rpc_write(client, &state, sizeof(state));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, path, strlen(path) + 1, true);
    rpc_write(client, &numOptions, sizeof(numOptions));
    rpc_write(client, &_0options, sizeof(_0options));
    // PARAM void **optionValues
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **optionValues
    rpc_free_client(client);
    mem2client((void *)options, 0);
    // PARAM void **optionValues
    return _result;
}

extern "C" CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkComplete called" << std::endl;
#endif
    // PARAM void **cubinOut
    void *_0sizeOut = mem2server((void *)sizeOut, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLinkComplete);
    rpc_write(client, &state, sizeof(state));
    // PARAM void **cubinOut
    rpc_write(client, &_0sizeOut, sizeof(_0sizeOut));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **cubinOut
    rpc_free_client(client);
    // PARAM void **cubinOut
    mem2client((void *)sizeOut, 0);
    return _result;
}

extern "C" CUresult cuLinkDestroy(CUlinkState state) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkDestroy called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLinkDestroy);
    rpc_write(client, &state, sizeof(state));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleGetTexRef called" << std::endl;
#endif
    void *_0pTexRef = mem2server((void *)pTexRef, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuModuleGetTexRef);
    rpc_write(client, &_0pTexRef, sizeof(_0pTexRef));
    rpc_write(client, &hmod, sizeof(hmod));
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pTexRef, 0);
    return _result;
}

extern "C" CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleGetSurfRef called" << std::endl;
#endif
    void *_0pSurfRef = mem2server((void *)pSurfRef, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuModuleGetSurfRef);
    rpc_write(client, &_0pSurfRef, sizeof(_0pSurfRef));
    rpc_write(client, &hmod, sizeof(hmod));
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pSurfRef, 0);
    return _result;
}

extern "C" CUresult cuLibraryLoadData(CUlibrary *library, const void *code, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions) {
#ifdef DEBUG
    std::cout << "Hook: cuLibraryLoadData called" << std::endl;
#endif
    void *_0library = mem2server((void *)library, 0);
    void *_0code = mem2server((void *)code, 0);
    void *_0jitOptions = mem2server((void *)jitOptions, 0);
    // PARAM void **jitOptionsValues
    void *_0libraryOptions = mem2server((void *)libraryOptions, 0);
    // PARAM void **libraryOptionValues
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLibraryLoadData);
    rpc_write(client, &_0library, sizeof(_0library));
    rpc_write(client, &_0code, sizeof(_0code));
    rpc_write(client, &_0jitOptions, sizeof(_0jitOptions));
    // PARAM void **jitOptionsValues
    rpc_write(client, &numJitOptions, sizeof(numJitOptions));
    rpc_write(client, &_0libraryOptions, sizeof(_0libraryOptions));
    // PARAM void **libraryOptionValues
    rpc_write(client, &numLibraryOptions, sizeof(numLibraryOptions));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **jitOptionsValues
    // PARAM void **libraryOptionValues
    rpc_free_client(client);
    mem2client((void *)library, 0);
    mem2client((void *)code, 0);
    mem2client((void *)jitOptions, 0);
    // PARAM void **jitOptionsValues
    mem2client((void *)libraryOptions, 0);
    // PARAM void **libraryOptionValues
    return _result;
}

extern "C" CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *fileName, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions) {
#ifdef DEBUG
    std::cout << "Hook: cuLibraryLoadFromFile called" << std::endl;
#endif
    void *_0library = mem2server((void *)library, 0);
    void *_0jitOptions = mem2server((void *)jitOptions, 0);
    // PARAM void **jitOptionsValues
    void *_0libraryOptions = mem2server((void *)libraryOptions, 0);
    // PARAM void **libraryOptionValues
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLibraryLoadFromFile);
    rpc_write(client, &_0library, sizeof(_0library));
    rpc_write(client, fileName, strlen(fileName) + 1, true);
    rpc_write(client, &_0jitOptions, sizeof(_0jitOptions));
    // PARAM void **jitOptionsValues
    rpc_write(client, &numJitOptions, sizeof(numJitOptions));
    rpc_write(client, &_0libraryOptions, sizeof(_0libraryOptions));
    // PARAM void **libraryOptionValues
    rpc_write(client, &numLibraryOptions, sizeof(numLibraryOptions));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **jitOptionsValues
    // PARAM void **libraryOptionValues
    rpc_free_client(client);
    mem2client((void *)library, 0);
    mem2client((void *)jitOptions, 0);
    // PARAM void **jitOptionsValues
    mem2client((void *)libraryOptions, 0);
    // PARAM void **libraryOptionValues
    return _result;
}

extern "C" CUresult cuLibraryUnload(CUlibrary library) {
#ifdef DEBUG
    std::cout << "Hook: cuLibraryUnload called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLibraryUnload);
    rpc_write(client, &library, sizeof(library));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuLibraryGetKernel called" << std::endl;
#endif
    void *_0pKernel = mem2server((void *)pKernel, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLibraryGetKernel);
    rpc_write(client, &_0pKernel, sizeof(_0pKernel));
    rpc_write(client, &library, sizeof(library));
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pKernel, 0);
    return _result;
}

extern "C" CUresult cuLibraryGetKernelCount(unsigned int *count, CUlibrary lib) {
#ifdef DEBUG
    std::cout << "Hook: cuLibraryGetKernelCount called" << std::endl;
#endif
    void *_0count = mem2server((void *)count, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLibraryGetKernelCount);
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_write(client, &lib, sizeof(lib));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)count, 0);
    return _result;
}

extern "C" CUresult cuLibraryEnumerateKernels(CUkernel *kernels, unsigned int numKernels, CUlibrary lib) {
#ifdef DEBUG
    std::cout << "Hook: cuLibraryEnumerateKernels called" << std::endl;
#endif
    void *_0kernels = mem2server((void *)kernels, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLibraryEnumerateKernels);
    rpc_write(client, &_0kernels, sizeof(_0kernels));
    rpc_write(client, &numKernels, sizeof(numKernels));
    rpc_write(client, &lib, sizeof(lib));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)kernels, 0);
    return _result;
}

extern "C" CUresult cuLibraryGetModule(CUmodule *pMod, CUlibrary library) {
#ifdef DEBUG
    std::cout << "Hook: cuLibraryGetModule called" << std::endl;
#endif
    void *_0pMod = mem2server((void *)pMod, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLibraryGetModule);
    rpc_write(client, &_0pMod, sizeof(_0pMod));
    rpc_write(client, &library, sizeof(library));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pMod, 0);
    return _result;
}

extern "C" CUresult cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel) {
#ifdef DEBUG
    std::cout << "Hook: cuKernelGetFunction called" << std::endl;
#endif
    void *_0pFunc = mem2server((void *)pFunc, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuKernelGetFunction);
    rpc_write(client, &_0pFunc, sizeof(_0pFunc));
    rpc_write(client, &kernel, sizeof(kernel));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pFunc, 0);
    return _result;
}

extern "C" CUresult cuKernelGetLibrary(CUlibrary *pLib, CUkernel kernel) {
#ifdef DEBUG
    std::cout << "Hook: cuKernelGetLibrary called" << std::endl;
#endif
    void *_0pLib = mem2server((void *)pLib, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuKernelGetLibrary);
    rpc_write(client, &_0pLib, sizeof(_0pLib));
    rpc_write(client, &kernel, sizeof(kernel));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pLib, 0);
    return _result;
}

extern "C" CUresult cuLibraryGetUnifiedFunction(void **fptr, CUlibrary library, const char *symbol) {
#ifdef DEBUG
    std::cout << "Hook: cuLibraryGetUnifiedFunction called" << std::endl;
#endif
    // PARAM void **fptr
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLibraryGetUnifiedFunction);
    // PARAM void **fptr
    rpc_write(client, &library, sizeof(library));
    rpc_write(client, symbol, strlen(symbol) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **fptr
    rpc_free_client(client);
    // PARAM void **fptr
    return _result;
}

extern "C" CUresult cuKernelGetAttribute(int *pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuKernelGetAttribute called" << std::endl;
#endif
    void *_0pi = mem2server((void *)pi, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuKernelGetAttribute);
    rpc_write(client, &_0pi, sizeof(_0pi));
    rpc_write(client, &attrib, sizeof(attrib));
    rpc_write(client, &kernel, sizeof(kernel));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pi, 0);
    return _result;
}

extern "C" CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuKernelSetAttribute called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuKernelSetAttribute);
    rpc_write(client, &attrib, sizeof(attrib));
    rpc_write(client, &val, sizeof(val));
    rpc_write(client, &kernel, sizeof(kernel));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuKernelSetCacheConfig called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuKernelSetCacheConfig);
    rpc_write(client, &kernel, sizeof(kernel));
    rpc_write(client, &config, sizeof(config));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuKernelGetName(const char **name, CUkernel hfunc) {
#ifdef DEBUG
    std::cout << "Hook: cuKernelGetName called" << std::endl;
#endif
    // PARAM const char **name
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuKernelGetName);
    // PARAM const char **name
    static char _cuKernelGetName_name[1024];
    rpc_read(client, _cuKernelGetName_name, 1024, true);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM const char **name
    *name = _cuKernelGetName_name;
    rpc_free_client(client);
    // PARAM const char **name
    return _result;
}

extern "C" CUresult cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex, size_t *paramOffset, size_t *paramSize) {
#ifdef DEBUG
    std::cout << "Hook: cuKernelGetParamInfo called" << std::endl;
#endif
    void *_0paramOffset = mem2server((void *)paramOffset, 0);
    void *_0paramSize = mem2server((void *)paramSize, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuKernelGetParamInfo);
    rpc_write(client, &kernel, sizeof(kernel));
    rpc_write(client, &paramIndex, sizeof(paramIndex));
    rpc_write(client, &_0paramOffset, sizeof(_0paramOffset));
    rpc_write(client, &_0paramSize, sizeof(_0paramSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)paramOffset, 0);
    mem2client((void *)paramSize, 0);
    return _result;
}

extern "C" CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetInfo_v2 called" << std::endl;
#endif
    void *_0free = mem2server((void *)free, 0);
    void *_0total = mem2server((void *)total, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemGetInfo_v2);
    rpc_write(client, &_0free, sizeof(_0free));
    rpc_write(client, &_0total, sizeof(_0total));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)free, 0);
    mem2client((void *)total, 0);
    return _result;
}

extern "C" CUresult cuMemFree_v2(CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemFree_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemFree_v2);
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
#ifdef DEBUG
    std::cout << "Hook: cuMemHostGetFlags called" << std::endl;
#endif
    void *_0pFlags = mem2server((void *)pFlags, 0);
    void *_0p = mem2server((void *)p, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemHostGetFlags);
    rpc_write(client, &_0pFlags, sizeof(_0pFlags));
    rpc_write(client, &_0p, sizeof(_0p));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pFlags, 0);
    mem2client((void *)p, 0);
    return _result;
}

extern "C" CUresult cuDeviceRegisterAsyncNotification(CUdevice device, CUasyncCallback callbackFunc, void *userData, CUasyncCallbackHandle *callback) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceRegisterAsyncNotification called" << std::endl;
#endif
    void *_0userData = mem2server((void *)userData, 0);
    void *_0callback = mem2server((void *)callback, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceRegisterAsyncNotification);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &callbackFunc, sizeof(callbackFunc));
    rpc_write(client, &_0userData, sizeof(_0userData));
    rpc_write(client, &_0callback, sizeof(_0callback));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)userData, 0);
    mem2client((void *)callback, 0);
    return _result;
}

extern "C" CUresult cuDeviceUnregisterAsyncNotification(CUdevice device, CUasyncCallbackHandle callback) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceUnregisterAsyncNotification called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceUnregisterAsyncNotification);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &callback, sizeof(callback));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetByPCIBusId called" << std::endl;
#endif
    void *_0dev = mem2server((void *)dev, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetByPCIBusId);
    rpc_write(client, &_0dev, sizeof(_0dev));
    rpc_write(client, pciBusId, strlen(pciBusId) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dev, 0);
    return _result;
}

extern "C" CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetPCIBusId called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetPCIBusId);
    rpc_read(client, pciBusId, len, true);
    rpc_write(client, &len, sizeof(len));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcGetEventHandle called" << std::endl;
#endif
    void *_0pHandle = mem2server((void *)pHandle, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuIpcGetEventHandle);
    rpc_write(client, &_0pHandle, sizeof(_0pHandle));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pHandle, 0);
    return _result;
}

extern "C" CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcOpenEventHandle called" << std::endl;
#endif
    void *_0phEvent = mem2server((void *)phEvent, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuIpcOpenEventHandle);
    rpc_write(client, &_0phEvent, sizeof(_0phEvent));
    rpc_write(client, &handle, sizeof(handle));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phEvent, 0);
    return _result;
}

extern "C" CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcGetMemHandle called" << std::endl;
#endif
    void *_0pHandle = mem2server((void *)pHandle, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuIpcGetMemHandle);
    rpc_write(client, &_0pHandle, sizeof(_0pHandle));
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pHandle, 0);
    return _result;
}

extern "C" CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcCloseMemHandle called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuIpcCloseMemHandle);
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemHostRegister_v2 called" << std::endl;
#endif
    void *_0p = mem2server((void *)p, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemHostRegister_v2);
    rpc_write(client, &_0p, sizeof(_0p));
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)p, 0);
    return _result;
}

extern "C" CUresult cuMemHostUnregister(void *p) {
#ifdef DEBUG
    std::cout << "Hook: cuMemHostUnregister called" << std::endl;
#endif
    void *_0p = mem2server((void *)p, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemHostUnregister);
    rpc_write(client, &_0p, sizeof(_0p));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)p, 0);
    return _result;
}

extern "C" CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpy);
    rpc_write(client, &dst, sizeof(dst));
    rpc_write(client, &src, sizeof(src));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyPeer called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyPeer);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &dstContext, sizeof(dstContext));
    rpc_write(client, &srcDevice, sizeof(srcDevice));
    rpc_write(client, &srcContext, sizeof(srcContext));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyHtoD_v2 called" << std::endl;
#endif
    void *_0srcHost = mem2server((void *)srcHost, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyHtoD_v2);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &_0srcHost, sizeof(_0srcHost));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)srcHost, 0);
    return _result;
}

extern "C" CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoH_v2 called" << std::endl;
#endif
    void *_0dstHost = mem2server((void *)dstHost, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyDtoH_v2);
    rpc_write(client, &_0dstHost, sizeof(_0dstHost));
    rpc_write(client, &srcDevice, sizeof(srcDevice));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dstHost, 0);
    return _result;
}

extern "C" CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoD_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyDtoD_v2);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &srcDevice, sizeof(srcDevice));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoA_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyDtoA_v2);
    rpc_write(client, &dstArray, sizeof(dstArray));
    rpc_write(client, &dstOffset, sizeof(dstOffset));
    rpc_write(client, &srcDevice, sizeof(srcDevice));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAtoD_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyAtoD_v2);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &srcArray, sizeof(srcArray));
    rpc_write(client, &srcOffset, sizeof(srcOffset));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyHtoA_v2 called" << std::endl;
#endif
    void *_0srcHost = mem2server((void *)srcHost, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyHtoA_v2);
    rpc_write(client, &dstArray, sizeof(dstArray));
    rpc_write(client, &dstOffset, sizeof(dstOffset));
    rpc_write(client, &_0srcHost, sizeof(_0srcHost));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)srcHost, 0);
    return _result;
}

extern "C" CUresult cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAtoH_v2 called" << std::endl;
#endif
    void *_0dstHost = mem2server((void *)dstHost, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyAtoH_v2);
    rpc_write(client, &_0dstHost, sizeof(_0dstHost));
    rpc_write(client, &srcArray, sizeof(srcArray));
    rpc_write(client, &srcOffset, sizeof(srcOffset));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dstHost, 0);
    return _result;
}

extern "C" CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAtoA_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyAtoA_v2);
    rpc_write(client, &dstArray, sizeof(dstArray));
    rpc_write(client, &dstOffset, sizeof(dstOffset));
    rpc_write(client, &srcArray, sizeof(srcArray));
    rpc_write(client, &srcOffset, sizeof(srcOffset));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy2D_v2 called" << std::endl;
#endif
    void *_0pCopy = mem2server((void *)pCopy, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpy2D_v2);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pCopy, 0);
    return _result;
}

extern "C" CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy2DUnaligned_v2 called" << std::endl;
#endif
    void *_0pCopy = mem2server((void *)pCopy, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpy2DUnaligned_v2);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pCopy, 0);
    return _result;
}

extern "C" CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy3D_v2 called" << std::endl;
#endif
    void *_0pCopy = mem2server((void *)pCopy, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpy3D_v2);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pCopy, 0);
    return _result;
}

extern "C" CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy3DPeer called" << std::endl;
#endif
    void *_0pCopy = mem2server((void *)pCopy, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpy3DPeer);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pCopy, 0);
    return _result;
}

extern "C" CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAsync called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyAsync);
    rpc_write(client, &dst, sizeof(dst));
    rpc_write(client, &src, sizeof(src));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
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

extern "C" CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyPeerAsync called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyPeerAsync);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &dstContext, sizeof(dstContext));
    rpc_write(client, &srcDevice, sizeof(srcDevice));
    rpc_write(client, &srcContext, sizeof(srcContext));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
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

extern "C" CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyHtoDAsync_v2 called" << std::endl;
#endif
    void *_0srcHost = mem2server((void *)srcHost, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyHtoDAsync_v2);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &_0srcHost, sizeof(_0srcHost));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)srcHost, 0);
    return _result;
}

extern "C" CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoHAsync_v2 called" << std::endl;
#endif
    void *_0dstHost = mem2server((void *)dstHost, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyDtoHAsync_v2);
    rpc_write(client, &_0dstHost, sizeof(_0dstHost));
    rpc_write(client, &srcDevice, sizeof(srcDevice));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dstHost, 0);
    return _result;
}

extern "C" CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoDAsync_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyDtoDAsync_v2);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &srcDevice, sizeof(srcDevice));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
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

extern "C" CUresult cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyHtoAAsync_v2 called" << std::endl;
#endif
    void *_0srcHost = mem2server((void *)srcHost, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyHtoAAsync_v2);
    rpc_write(client, &dstArray, sizeof(dstArray));
    rpc_write(client, &dstOffset, sizeof(dstOffset));
    rpc_write(client, &_0srcHost, sizeof(_0srcHost));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)srcHost, 0);
    return _result;
}

extern "C" CUresult cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAtoHAsync_v2 called" << std::endl;
#endif
    void *_0dstHost = mem2server((void *)dstHost, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpyAtoHAsync_v2);
    rpc_write(client, &_0dstHost, sizeof(_0dstHost));
    rpc_write(client, &srcArray, sizeof(srcArray));
    rpc_write(client, &srcOffset, sizeof(srcOffset));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dstHost, 0);
    return _result;
}

extern "C" CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy2DAsync_v2 called" << std::endl;
#endif
    void *_0pCopy = mem2server((void *)pCopy, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpy2DAsync_v2);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pCopy, 0);
    return _result;
}

extern "C" CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy3DAsync_v2 called" << std::endl;
#endif
    void *_0pCopy = mem2server((void *)pCopy, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpy3DAsync_v2);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pCopy, 0);
    return _result;
}

extern "C" CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy3DPeerAsync called" << std::endl;
#endif
    void *_0pCopy = mem2server((void *)pCopy, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpy3DPeerAsync);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pCopy, 0);
    return _result;
}

extern "C" CUresult cuMemcpy3DBatchAsync(size_t numOps, CUDA_MEMCPY3D_BATCH_OP *opList, size_t *failIdx, unsigned long long flags, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy3DBatchAsync called" << std::endl;
#endif
    void *_0opList = mem2server((void *)opList, 0);
    void *_0failIdx = mem2server((void *)failIdx, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemcpy3DBatchAsync);
    rpc_write(client, &numOps, sizeof(numOps));
    rpc_write(client, &_0opList, sizeof(_0opList));
    rpc_write(client, &_0failIdx, sizeof(_0failIdx));
    rpc_write(client, &flags, sizeof(flags));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)opList, 0);
    mem2client((void *)failIdx, 0);
    return _result;
}

extern "C" CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD8_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemsetD8_v2);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &uc, sizeof(uc));
    rpc_write(client, &N, sizeof(N));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD16_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemsetD16_v2);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &us, sizeof(us));
    rpc_write(client, &N, sizeof(N));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD32_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemsetD32_v2);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &ui, sizeof(ui));
    rpc_write(client, &N, sizeof(N));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D8_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemsetD2D8_v2);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &dstPitch, sizeof(dstPitch));
    rpc_write(client, &uc, sizeof(uc));
    rpc_write(client, &Width, sizeof(Width));
    rpc_write(client, &Height, sizeof(Height));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D16_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemsetD2D16_v2);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &dstPitch, sizeof(dstPitch));
    rpc_write(client, &us, sizeof(us));
    rpc_write(client, &Width, sizeof(Width));
    rpc_write(client, &Height, sizeof(Height));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D32_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemsetD2D32_v2);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &dstPitch, sizeof(dstPitch));
    rpc_write(client, &ui, sizeof(ui));
    rpc_write(client, &Width, sizeof(Width));
    rpc_write(client, &Height, sizeof(Height));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD8Async called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemsetD8Async);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &uc, sizeof(uc));
    rpc_write(client, &N, sizeof(N));
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

extern "C" CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD16Async called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemsetD16Async);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &us, sizeof(us));
    rpc_write(client, &N, sizeof(N));
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

extern "C" CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD32Async called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemsetD32Async);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &ui, sizeof(ui));
    rpc_write(client, &N, sizeof(N));
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

extern "C" CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D8Async called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemsetD2D8Async);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &dstPitch, sizeof(dstPitch));
    rpc_write(client, &uc, sizeof(uc));
    rpc_write(client, &Width, sizeof(Width));
    rpc_write(client, &Height, sizeof(Height));
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

extern "C" CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D16Async called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemsetD2D16Async);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &dstPitch, sizeof(dstPitch));
    rpc_write(client, &us, sizeof(us));
    rpc_write(client, &Width, sizeof(Width));
    rpc_write(client, &Height, sizeof(Height));
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

extern "C" CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D32Async called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemsetD2D32Async);
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &dstPitch, sizeof(dstPitch));
    rpc_write(client, &ui, sizeof(ui));
    rpc_write(client, &Width, sizeof(Width));
    rpc_write(client, &Height, sizeof(Height));
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

extern "C" CUresult cuArrayCreate_v2(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayCreate_v2 called" << std::endl;
#endif
    void *_0pHandle = mem2server((void *)pHandle, 0);
    void *_0pAllocateArray = mem2server((void *)pAllocateArray, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuArrayCreate_v2);
    rpc_write(client, &_0pHandle, sizeof(_0pHandle));
    rpc_write(client, &_0pAllocateArray, sizeof(_0pAllocateArray));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pHandle, 0);
    mem2client((void *)pAllocateArray, 0);
    return _result;
}

extern "C" CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayGetDescriptor_v2 called" << std::endl;
#endif
    void *_0pArrayDescriptor = mem2server((void *)pArrayDescriptor, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuArrayGetDescriptor_v2);
    rpc_write(client, &_0pArrayDescriptor, sizeof(_0pArrayDescriptor));
    rpc_write(client, &hArray, sizeof(hArray));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pArrayDescriptor, 0);
    return _result;
}

extern "C" CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUarray array) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayGetSparseProperties called" << std::endl;
#endif
    void *_0sparseProperties = mem2server((void *)sparseProperties, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuArrayGetSparseProperties);
    rpc_write(client, &_0sparseProperties, sizeof(_0sparseProperties));
    rpc_write(client, &array, sizeof(array));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)sparseProperties, 0);
    return _result;
}

extern "C" CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUmipmappedArray mipmap) {
#ifdef DEBUG
    std::cout << "Hook: cuMipmappedArrayGetSparseProperties called" << std::endl;
#endif
    void *_0sparseProperties = mem2server((void *)sparseProperties, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMipmappedArrayGetSparseProperties);
    rpc_write(client, &_0sparseProperties, sizeof(_0sparseProperties));
    rpc_write(client, &mipmap, sizeof(mipmap));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)sparseProperties, 0);
    return _result;
}

extern "C" CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUarray array, CUdevice device) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayGetMemoryRequirements called" << std::endl;
#endif
    void *_0memoryRequirements = mem2server((void *)memoryRequirements, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuArrayGetMemoryRequirements);
    rpc_write(client, &_0memoryRequirements, sizeof(_0memoryRequirements));
    rpc_write(client, &array, sizeof(array));
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)memoryRequirements, 0);
    return _result;
}

extern "C" CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUmipmappedArray mipmap, CUdevice device) {
#ifdef DEBUG
    std::cout << "Hook: cuMipmappedArrayGetMemoryRequirements called" << std::endl;
#endif
    void *_0memoryRequirements = mem2server((void *)memoryRequirements, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMipmappedArrayGetMemoryRequirements);
    rpc_write(client, &_0memoryRequirements, sizeof(_0memoryRequirements));
    rpc_write(client, &mipmap, sizeof(mipmap));
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)memoryRequirements, 0);
    return _result;
}

extern "C" CUresult cuArrayGetPlane(CUarray *pPlaneArray, CUarray hArray, unsigned int planeIdx) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayGetPlane called" << std::endl;
#endif
    void *_0pPlaneArray = mem2server((void *)pPlaneArray, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuArrayGetPlane);
    rpc_write(client, &_0pPlaneArray, sizeof(_0pPlaneArray));
    rpc_write(client, &hArray, sizeof(hArray));
    rpc_write(client, &planeIdx, sizeof(planeIdx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pPlaneArray, 0);
    return _result;
}

extern "C" CUresult cuArrayDestroy(CUarray hArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayDestroy called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuArrayDestroy);
    rpc_write(client, &hArray, sizeof(hArray));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuArray3DCreate_v2(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArray3DCreate_v2 called" << std::endl;
#endif
    void *_0pHandle = mem2server((void *)pHandle, 0);
    void *_0pAllocateArray = mem2server((void *)pAllocateArray, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuArray3DCreate_v2);
    rpc_write(client, &_0pHandle, sizeof(_0pHandle));
    rpc_write(client, &_0pAllocateArray, sizeof(_0pAllocateArray));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pHandle, 0);
    mem2client((void *)pAllocateArray, 0);
    return _result;
}

extern "C" CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArray3DGetDescriptor_v2 called" << std::endl;
#endif
    void *_0pArrayDescriptor = mem2server((void *)pArrayDescriptor, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuArray3DGetDescriptor_v2);
    rpc_write(client, &_0pArrayDescriptor, sizeof(_0pArrayDescriptor));
    rpc_write(client, &hArray, sizeof(hArray));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pArrayDescriptor, 0);
    return _result;
}

extern "C" CUresult cuMipmappedArrayCreate(CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels) {
#ifdef DEBUG
    std::cout << "Hook: cuMipmappedArrayCreate called" << std::endl;
#endif
    void *_0pHandle = mem2server((void *)pHandle, 0);
    void *_0pMipmappedArrayDesc = mem2server((void *)pMipmappedArrayDesc, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMipmappedArrayCreate);
    rpc_write(client, &_0pHandle, sizeof(_0pHandle));
    rpc_write(client, &_0pMipmappedArrayDesc, sizeof(_0pMipmappedArrayDesc));
    rpc_write(client, &numMipmapLevels, sizeof(numMipmapLevels));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pHandle, 0);
    mem2client((void *)pMipmappedArrayDesc, 0);
    return _result;
}

extern "C" CUresult cuMipmappedArrayGetLevel(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) {
#ifdef DEBUG
    std::cout << "Hook: cuMipmappedArrayGetLevel called" << std::endl;
#endif
    void *_0pLevelArray = mem2server((void *)pLevelArray, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMipmappedArrayGetLevel);
    rpc_write(client, &_0pLevelArray, sizeof(_0pLevelArray));
    rpc_write(client, &hMipmappedArray, sizeof(hMipmappedArray));
    rpc_write(client, &level, sizeof(level));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pLevelArray, 0);
    return _result;
}

extern "C" CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
#ifdef DEBUG
    std::cout << "Hook: cuMipmappedArrayDestroy called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMipmappedArrayDestroy);
    rpc_write(client, &hMipmappedArray, sizeof(hMipmappedArray));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemGetHandleForAddressRange(void *handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetHandleForAddressRange called" << std::endl;
#endif
    void *_0handle = mem2server((void *)handle, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemGetHandleForAddressRange);
    rpc_write(client, &_0handle, sizeof(_0handle));
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &handleType, sizeof(handleType));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)handle, 0);
    return _result;
}

extern "C" CUresult cuMemBatchDecompressAsync(CUmemDecompressParams *paramsArray, size_t count, unsigned int flags, size_t *errorIndex, CUstream stream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemBatchDecompressAsync called" << std::endl;
#endif
    void *_0paramsArray = mem2server((void *)paramsArray, 0);
    void *_0errorIndex = mem2server((void *)errorIndex, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemBatchDecompressAsync);
    rpc_write(client, &_0paramsArray, sizeof(_0paramsArray));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &flags, sizeof(flags));
    rpc_write(client, &_0errorIndex, sizeof(_0errorIndex));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)paramsArray, 0);
    mem2client((void *)errorIndex, 0);
    return _result;
}

extern "C" CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAddressFree called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemAddressFree);
    rpc_write(client, &ptr, sizeof(ptr));
    rpc_write(client, &size, sizeof(size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemMapArrayAsync called" << std::endl;
#endif
    void *_0mapInfoList = mem2server((void *)mapInfoList, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemMapArrayAsync);
    rpc_write(client, &_0mapInfoList, sizeof(_0mapInfoList));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)mapInfoList, 0);
    return _result;
}

extern "C" CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
#ifdef DEBUG
    std::cout << "Hook: cuMemUnmap called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemUnmap);
    rpc_write(client, &ptr, sizeof(ptr));
    rpc_write(client, &size, sizeof(size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cuMemSetAccess called" << std::endl;
#endif
    void *_0desc = mem2server((void *)desc, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemSetAccess);
    rpc_write(client, &ptr, sizeof(ptr));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &_0desc, sizeof(_0desc));
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)desc, 0);
    return _result;
}

extern "C" CUresult cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetAccess called" << std::endl;
#endif
    void *_0flags = mem2server((void *)flags, 0);
    void *_0location = mem2server((void *)location, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemGetAccess);
    rpc_write(client, &_0flags, sizeof(_0flags));
    rpc_write(client, &_0location, sizeof(_0location));
    rpc_write(client, &ptr, sizeof(ptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)flags, 0);
    mem2client((void *)location, 0);
    return _result;
}

extern "C" CUresult cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemExportToShareableHandle called" << std::endl;
#endif
    void *_0shareableHandle = mem2server((void *)shareableHandle, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemExportToShareableHandle);
    rpc_write(client, &_0shareableHandle, sizeof(_0shareableHandle));
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &handleType, sizeof(handleType));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)shareableHandle, 0);
    return _result;
}

extern "C" CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle, CUmemAllocationHandleType shHandleType) {
#ifdef DEBUG
    std::cout << "Hook: cuMemImportFromShareableHandle called" << std::endl;
#endif
    void *_0handle = mem2server((void *)handle, 0);
    void *_0osHandle = mem2server((void *)osHandle, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemImportFromShareableHandle);
    rpc_write(client, &_0handle, sizeof(_0handle));
    rpc_write(client, &_0osHandle, sizeof(_0osHandle));
    rpc_write(client, &shHandleType, sizeof(shHandleType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)handle, 0);
    mem2client((void *)osHandle, 0);
    return _result;
}

extern "C" CUresult cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetAllocationGranularity called" << std::endl;
#endif
    void *_0granularity = mem2server((void *)granularity, 0);
    void *_0prop = mem2server((void *)prop, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemGetAllocationGranularity);
    rpc_write(client, &_0granularity, sizeof(_0granularity));
    rpc_write(client, &_0prop, sizeof(_0prop));
    rpc_write(client, &option, sizeof(option));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)granularity, 0);
    mem2client((void *)prop, 0);
    return _result;
}

extern "C" CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetAllocationPropertiesFromHandle called" << std::endl;
#endif
    void *_0prop = mem2server((void *)prop, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemGetAllocationPropertiesFromHandle);
    rpc_write(client, &_0prop, sizeof(_0prop));
    rpc_write(client, &handle, sizeof(handle));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)prop, 0);
    return _result;
}

extern "C" CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemRetainAllocationHandle called" << std::endl;
#endif
    void *_0handle = mem2server((void *)handle, 0);
    void *_0addr = mem2server((void *)addr, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemRetainAllocationHandle);
    rpc_write(client, &_0handle, sizeof(_0handle));
    rpc_write(client, &_0addr, sizeof(_0addr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)handle, 0);
    mem2client((void *)addr, 0);
    return _result;
}

extern "C" CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemFreeAsync called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemFreeAsync);
    rpc_write(client, &dptr, sizeof(dptr));
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

extern "C" CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolTrimTo called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPoolTrimTo);
    rpc_write(client, &pool, sizeof(pool));
    rpc_write(client, &minBytesToKeep, sizeof(minBytesToKeep));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolSetAttribute called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPoolSetAttribute);
    rpc_write(client, &pool, sizeof(pool));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value, 0);
    return _result;
}

extern "C" CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolGetAttribute called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPoolGetAttribute);
    rpc_write(client, &pool, sizeof(pool));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value, 0);
    return _result;
}

extern "C" CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolSetAccess called" << std::endl;
#endif
    void *_0map = mem2server((void *)map, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPoolSetAccess);
    rpc_write(client, &pool, sizeof(pool));
    rpc_write(client, &_0map, sizeof(_0map));
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)map, 0);
    return _result;
}

extern "C" CUresult cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool, CUmemLocation *location) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolGetAccess called" << std::endl;
#endif
    void *_0flags = mem2server((void *)flags, 0);
    void *_0location = mem2server((void *)location, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPoolGetAccess);
    rpc_write(client, &_0flags, sizeof(_0flags));
    rpc_write(client, &memPool, sizeof(memPool));
    rpc_write(client, &_0location, sizeof(_0location));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)flags, 0);
    mem2client((void *)location, 0);
    return _result;
}

extern "C" CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolCreate called" << std::endl;
#endif
    void *_0pool = mem2server((void *)pool, 0);
    void *_0poolProps = mem2server((void *)poolProps, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPoolCreate);
    rpc_write(client, &_0pool, sizeof(_0pool));
    rpc_write(client, &_0poolProps, sizeof(_0poolProps));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pool, 0);
    mem2client((void *)poolProps, 0);
    return _result;
}

extern "C" CUresult cuMemPoolDestroy(CUmemoryPool pool) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolDestroy called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPoolDestroy);
    rpc_write(client, &pool, sizeof(pool));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolExportToShareableHandle called" << std::endl;
#endif
    void *_0handle_out = mem2server((void *)handle_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPoolExportToShareableHandle);
    rpc_write(client, &_0handle_out, sizeof(_0handle_out));
    rpc_write(client, &pool, sizeof(pool));
    rpc_write(client, &handleType, sizeof(handleType));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)handle_out, 0);
    return _result;
}

extern "C" CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool *pool_out, void *handle, CUmemAllocationHandleType handleType, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolImportFromShareableHandle called" << std::endl;
#endif
    void *_0pool_out = mem2server((void *)pool_out, 0);
    void *_0handle = mem2server((void *)handle, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPoolImportFromShareableHandle);
    rpc_write(client, &_0pool_out, sizeof(_0pool_out));
    rpc_write(client, &_0handle, sizeof(_0handle));
    rpc_write(client, &handleType, sizeof(handleType));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pool_out, 0);
    mem2client((void *)handle, 0);
    return _result;
}

extern "C" CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolExportPointer called" << std::endl;
#endif
    void *_0shareData_out = mem2server((void *)shareData_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPoolExportPointer);
    rpc_write(client, &_0shareData_out, sizeof(_0shareData_out));
    rpc_write(client, &ptr, sizeof(ptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)shareData_out, 0);
    return _result;
}

extern "C" CUresult cuMulticastCreate(CUmemGenericAllocationHandle *mcHandle, const CUmulticastObjectProp *prop) {
#ifdef DEBUG
    std::cout << "Hook: cuMulticastCreate called" << std::endl;
#endif
    void *_0mcHandle = mem2server((void *)mcHandle, 0);
    void *_0prop = mem2server((void *)prop, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMulticastCreate);
    rpc_write(client, &_0mcHandle, sizeof(_0mcHandle));
    rpc_write(client, &_0prop, sizeof(_0prop));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)mcHandle, 0);
    mem2client((void *)prop, 0);
    return _result;
}

extern "C" CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle mcHandle, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuMulticastAddDevice called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMulticastAddDevice);
    rpc_write(client, &mcHandle, sizeof(mcHandle));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMulticastBindMem called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMulticastBindMem);
    rpc_write(client, &mcHandle, sizeof(mcHandle));
    rpc_write(client, &mcOffset, sizeof(mcOffset));
    rpc_write(client, &memHandle, sizeof(memHandle));
    rpc_write(client, &memOffset, sizeof(memOffset));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMulticastBindAddr(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUdeviceptr memptr, size_t size, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMulticastBindAddr called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMulticastBindAddr);
    rpc_write(client, &mcHandle, sizeof(mcHandle));
    rpc_write(client, &mcOffset, sizeof(mcOffset));
    rpc_write(client, &memptr, sizeof(memptr));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMulticastUnbind(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, size_t size) {
#ifdef DEBUG
    std::cout << "Hook: cuMulticastUnbind called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMulticastUnbind);
    rpc_write(client, &mcHandle, sizeof(mcHandle));
    rpc_write(client, &dev, sizeof(dev));
    rpc_write(client, &mcOffset, sizeof(mcOffset));
    rpc_write(client, &size, sizeof(size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMulticastGetGranularity(size_t *granularity, const CUmulticastObjectProp *prop, CUmulticastGranularity_flags option) {
#ifdef DEBUG
    std::cout << "Hook: cuMulticastGetGranularity called" << std::endl;
#endif
    void *_0granularity = mem2server((void *)granularity, 0);
    void *_0prop = mem2server((void *)prop, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMulticastGetGranularity);
    rpc_write(client, &_0granularity, sizeof(_0granularity));
    rpc_write(client, &_0prop, sizeof(_0prop));
    rpc_write(client, &option, sizeof(option));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)granularity, 0);
    mem2client((void *)prop, 0);
    return _result;
}

extern "C" CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuPointerGetAttribute called" << std::endl;
#endif
    void *_0data = mem2server((void *)data, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuPointerGetAttribute);
    rpc_write(client, &_0data, sizeof(_0data));
    rpc_write(client, &attribute, sizeof(attribute));
    rpc_write(client, &ptr, sizeof(ptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)data, 0);
    return _result;
}

extern "C" CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPrefetchAsync called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPrefetchAsync);
    rpc_write(client, &devPtr, sizeof(devPtr));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &dstDevice, sizeof(dstDevice));
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

extern "C" CUresult cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPrefetchAsync_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPrefetchAsync_v2);
    rpc_write(client, &devPtr, sizeof(devPtr));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &location, sizeof(location));
    rpc_write(client, &flags, sizeof(flags));
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

extern "C" CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAdvise called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemAdvise);
    rpc_write(client, &devPtr, sizeof(devPtr));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &advice, sizeof(advice));
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAdvise_v2(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUmemLocation location) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAdvise_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemAdvise_v2);
    rpc_write(client, &devPtr, sizeof(devPtr));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &advice, sizeof(advice));
    rpc_write(client, &location, sizeof(location));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cuMemRangeGetAttribute called" << std::endl;
#endif
    void *_0data = mem2server((void *)data, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemRangeGetAttribute);
    rpc_write(client, &_0data, sizeof(_0data));
    rpc_write(client, &dataSize, sizeof(dataSize));
    rpc_write(client, &attribute, sizeof(attribute));
    rpc_write(client, &devPtr, sizeof(devPtr));
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)data, 0);
    return _result;
}

extern "C" CUresult cuMemRangeGetAttributes(void **data, size_t *dataSizes, CUmem_range_attribute *attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cuMemRangeGetAttributes called" << std::endl;
#endif
    // PARAM void **data
    void *_0dataSizes = mem2server((void *)dataSizes, 0);
    void *_0attributes = mem2server((void *)attributes, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemRangeGetAttributes);
    // PARAM void **data
    rpc_write(client, &_0dataSizes, sizeof(_0dataSizes));
    rpc_write(client, &_0attributes, sizeof(_0attributes));
    rpc_write(client, &numAttributes, sizeof(numAttributes));
    rpc_write(client, &devPtr, sizeof(devPtr));
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **data
    rpc_free_client(client);
    // PARAM void **data
    mem2client((void *)dataSizes, 0);
    mem2client((void *)attributes, 0);
    return _result;
}

extern "C" CUresult cuPointerSetAttribute(const void *value, CUpointer_attribute attribute, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuPointerSetAttribute called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuPointerSetAttribute);
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_write(client, &attribute, sizeof(attribute));
    rpc_write(client, &ptr, sizeof(ptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value, 0);
    return _result;
}

extern "C" CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuPointerGetAttributes called" << std::endl;
#endif
    void *_0attributes = mem2server((void *)attributes, 0);
    // PARAM void **data
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuPointerGetAttributes);
    rpc_write(client, &numAttributes, sizeof(numAttributes));
    rpc_write(client, &_0attributes, sizeof(_0attributes));
    // PARAM void **data
    rpc_write(client, &ptr, sizeof(ptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **data
    rpc_free_client(client);
    mem2client((void *)attributes, 0);
    // PARAM void **data
    return _result;
}

extern "C" CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamCreate called" << std::endl;
#endif
    void *_0phStream = mem2server((void *)phStream, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamCreate);
    rpc_write(client, &_0phStream, sizeof(_0phStream));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phStream, 0);
    return _result;
}

extern "C" CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamCreateWithPriority called" << std::endl;
#endif
    void *_0phStream = mem2server((void *)phStream, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamCreateWithPriority);
    rpc_write(client, &_0phStream, sizeof(_0phStream));
    rpc_write(client, &flags, sizeof(flags));
    rpc_write(client, &priority, sizeof(priority));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phStream, 0);
    return _result;
}

extern "C" CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetPriority called" << std::endl;
#endif
    void *_0priority = mem2server((void *)priority, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamGetPriority);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0priority, sizeof(_0priority));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)priority, 0);
    return _result;
}

extern "C" CUresult cuStreamGetDevice(CUstream hStream, CUdevice *device) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetDevice called" << std::endl;
#endif
    void *_0device = mem2server((void *)device, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamGetDevice);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0device, sizeof(_0device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)device, 0);
    return _result;
}

extern "C" CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetFlags called" << std::endl;
#endif
    void *_0flags = mem2server((void *)flags, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamGetFlags);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0flags, sizeof(_0flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)flags, 0);
    return _result;
}

extern "C" CUresult cuStreamGetId(CUstream hStream, unsigned long long *streamId) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetId called" << std::endl;
#endif
    void *_0streamId = mem2server((void *)streamId, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamGetId);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0streamId, sizeof(_0streamId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)streamId, 0);
    return _result;
}

extern "C" CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetCtx called" << std::endl;
#endif
    void *_0pctx = mem2server((void *)pctx, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamGetCtx);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pctx, 0);
    return _result;
}

extern "C" CUresult cuStreamGetCtx_v2(CUstream hStream, CUcontext *pCtx, CUgreenCtx *pGreenCtx) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetCtx_v2 called" << std::endl;
#endif
    void *_0pCtx = mem2server((void *)pCtx, 0);
    void *_0pGreenCtx = mem2server((void *)pGreenCtx, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamGetCtx_v2);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0pCtx, sizeof(_0pCtx));
    rpc_write(client, &_0pGreenCtx, sizeof(_0pGreenCtx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pCtx, 0);
    mem2client((void *)pGreenCtx, 0);
    return _result;
}

extern "C" CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWaitEvent called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamWaitEvent);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &hEvent, sizeof(hEvent));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamAddCallback called" << std::endl;
#endif
    void *_0userData = mem2server((void *)userData, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamAddCallback);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &callback, sizeof(callback));
    rpc_write(client, &_0userData, sizeof(_0userData));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)userData, 0);
    return _result;
}

extern "C" CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamBeginCapture_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamBeginCapture_v2);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &mode, sizeof(mode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamBeginCaptureToGraph(CUstream hStream, CUgraph hGraph, const CUgraphNode *dependencies, const CUgraphEdgeData *dependencyData, size_t numDependencies, CUstreamCaptureMode mode) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamBeginCaptureToGraph called" << std::endl;
#endif
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0dependencyData = mem2server((void *)dependencyData, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamBeginCaptureToGraph);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &_0dependencyData, sizeof(_0dependencyData));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &mode, sizeof(mode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dependencies, 0);
    mem2client((void *)dependencyData, 0);
    return _result;
}

extern "C" CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode) {
#ifdef DEBUG
    std::cout << "Hook: cuThreadExchangeStreamCaptureMode called" << std::endl;
#endif
    void *_0mode = mem2server((void *)mode, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuThreadExchangeStreamCaptureMode);
    rpc_write(client, &_0mode, sizeof(_0mode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)mode, 0);
    return _result;
}

extern "C" CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamEndCapture called" << std::endl;
#endif
    void *_0phGraph = mem2server((void *)phGraph, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamEndCapture);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0phGraph, sizeof(_0phGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraph, 0);
    return _result;
}

extern "C" CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus *captureStatus) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamIsCapturing called" << std::endl;
#endif
    void *_0captureStatus = mem2server((void *)captureStatus, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamIsCapturing);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0captureStatus, sizeof(_0captureStatus));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)captureStatus, 0);
    return _result;
}

extern "C" CUresult cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetCaptureInfo_v2 called" << std::endl;
#endif
    void *_0captureStatus_out = mem2server((void *)captureStatus_out, 0);
    void *_0id_out = mem2server((void *)id_out, 0);
    void *_0graph_out = mem2server((void *)graph_out, 0);
    // PARAM const CUgraphNode **dependencies_out
    void *_0numDependencies_out = mem2server((void *)numDependencies_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamGetCaptureInfo_v2);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0captureStatus_out, sizeof(_0captureStatus_out));
    rpc_write(client, &_0id_out, sizeof(_0id_out));
    rpc_write(client, &_0graph_out, sizeof(_0graph_out));
    // PARAM const CUgraphNode **dependencies_out
    static CUgraphNode _cuStreamGetCaptureInfo_v2_dependencies_out;
    rpc_read(client, &_cuStreamGetCaptureInfo_v2_dependencies_out, sizeof(CUgraphNode));
    rpc_write(client, &_0numDependencies_out, sizeof(_0numDependencies_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM const CUgraphNode **dependencies_out
    *dependencies_out = &_cuStreamGetCaptureInfo_v2_dependencies_out;
    rpc_free_client(client);
    mem2client((void *)captureStatus_out, 0);
    mem2client((void *)id_out, 0);
    mem2client((void *)graph_out, 0);
    // PARAM const CUgraphNode **dependencies_out
    mem2client((void *)numDependencies_out, 0);
    return _result;
}

extern "C" CUresult cuStreamGetCaptureInfo_v3(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, const CUgraphEdgeData **edgeData_out, size_t *numDependencies_out) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetCaptureInfo_v3 called" << std::endl;
#endif
    void *_0captureStatus_out = mem2server((void *)captureStatus_out, 0);
    void *_0id_out = mem2server((void *)id_out, 0);
    void *_0graph_out = mem2server((void *)graph_out, 0);
    // PARAM const CUgraphNode **dependencies_out
    // PARAM const CUgraphEdgeData **edgeData_out
    void *_0numDependencies_out = mem2server((void *)numDependencies_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamGetCaptureInfo_v3);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0captureStatus_out, sizeof(_0captureStatus_out));
    rpc_write(client, &_0id_out, sizeof(_0id_out));
    rpc_write(client, &_0graph_out, sizeof(_0graph_out));
    // PARAM const CUgraphNode **dependencies_out
    static CUgraphNode _cuStreamGetCaptureInfo_v3_dependencies_out;
    rpc_read(client, &_cuStreamGetCaptureInfo_v3_dependencies_out, sizeof(CUgraphNode));
    // PARAM const CUgraphEdgeData **edgeData_out
    static CUgraphEdgeData _cuStreamGetCaptureInfo_v3_edgeData_out;
    rpc_read(client, &_cuStreamGetCaptureInfo_v3_edgeData_out, sizeof(CUgraphEdgeData));
    rpc_write(client, &_0numDependencies_out, sizeof(_0numDependencies_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM const CUgraphNode **dependencies_out
    *dependencies_out = &_cuStreamGetCaptureInfo_v3_dependencies_out;
    // PARAM const CUgraphEdgeData **edgeData_out
    *edgeData_out = &_cuStreamGetCaptureInfo_v3_edgeData_out;
    rpc_free_client(client);
    mem2client((void *)captureStatus_out, 0);
    mem2client((void *)id_out, 0);
    mem2client((void *)graph_out, 0);
    // PARAM const CUgraphNode **dependencies_out
    // PARAM const CUgraphEdgeData **edgeData_out
    mem2client((void *)numDependencies_out, 0);
    return _result;
}

extern "C" CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode *dependencies, size_t numDependencies, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamUpdateCaptureDependencies called" << std::endl;
#endif
    void *_0dependencies = mem2server((void *)dependencies, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamUpdateCaptureDependencies);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dependencies, 0);
    return _result;
}

extern "C" CUresult cuStreamUpdateCaptureDependencies_v2(CUstream hStream, CUgraphNode *dependencies, const CUgraphEdgeData *dependencyData, size_t numDependencies, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamUpdateCaptureDependencies_v2 called" << std::endl;
#endif
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0dependencyData = mem2server((void *)dependencyData, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamUpdateCaptureDependencies_v2);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &_0dependencyData, sizeof(_0dependencyData));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dependencies, 0);
    mem2client((void *)dependencyData, 0);
    return _result;
}

extern "C" CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamAttachMemAsync called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamAttachMemAsync);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_write(client, &length, sizeof(length));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamQuery(CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamQuery called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamQuery);
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

extern "C" CUresult cuStreamSynchronize(CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamSynchronize called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamSynchronize);
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

extern "C" CUresult cuStreamDestroy_v2(CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamDestroy_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamDestroy_v2);
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

extern "C" CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamCopyAttributes called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamCopyAttributes);
    rpc_write(client, &dst, sizeof(dst));
    rpc_write(client, &src, sizeof(src));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue *value_out) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetAttribute called" << std::endl;
#endif
    void *_0value_out = mem2server((void *)value_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamGetAttribute);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value_out, sizeof(_0value_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value_out, 0);
    return _result;
}

extern "C" CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue *value) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamSetAttribute called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamSetAttribute);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value, 0);
    return _result;
}

extern "C" CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuEventCreate called" << std::endl;
#endif
    void *_0phEvent = mem2server((void *)phEvent, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuEventCreate);
    rpc_write(client, &_0phEvent, sizeof(_0phEvent));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phEvent, 0);
    return _result;
}

extern "C" CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuEventRecord called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuEventRecord);
    rpc_write(client, &hEvent, sizeof(hEvent));
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

extern "C" CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuEventRecordWithFlags called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuEventRecordWithFlags);
    rpc_write(client, &hEvent, sizeof(hEvent));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuEventQuery(CUevent hEvent) {
#ifdef DEBUG
    std::cout << "Hook: cuEventQuery called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuEventQuery);
    rpc_write(client, &hEvent, sizeof(hEvent));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuEventSynchronize(CUevent hEvent) {
#ifdef DEBUG
    std::cout << "Hook: cuEventSynchronize called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuEventSynchronize);
    rpc_write(client, &hEvent, sizeof(hEvent));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuEventDestroy_v2(CUevent hEvent) {
#ifdef DEBUG
    std::cout << "Hook: cuEventDestroy_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuEventDestroy_v2);
    rpc_write(client, &hEvent, sizeof(hEvent));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
#ifdef DEBUG
    std::cout << "Hook: cuEventElapsedTime called" << std::endl;
#endif
    void *_0pMilliseconds = mem2server((void *)pMilliseconds, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuEventElapsedTime);
    rpc_write(client, &_0pMilliseconds, sizeof(_0pMilliseconds));
    rpc_write(client, &hStart, sizeof(hStart));
    rpc_write(client, &hEnd, sizeof(hEnd));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pMilliseconds, 0);
    return _result;
}

extern "C" CUresult cuEventElapsedTime_v2(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
#ifdef DEBUG
    std::cout << "Hook: cuEventElapsedTime_v2 called" << std::endl;
#endif
    void *_0pMilliseconds = mem2server((void *)pMilliseconds, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuEventElapsedTime_v2);
    rpc_write(client, &_0pMilliseconds, sizeof(_0pMilliseconds));
    rpc_write(client, &hStart, sizeof(hStart));
    rpc_write(client, &hEnd, sizeof(hEnd));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pMilliseconds, 0);
    return _result;
}

extern "C" CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuExternalMemoryGetMappedMipmappedArray called" << std::endl;
#endif
    void *_0mipmap = mem2server((void *)mipmap, 0);
    void *_0mipmapDesc = mem2server((void *)mipmapDesc, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuExternalMemoryGetMappedMipmappedArray);
    rpc_write(client, &_0mipmap, sizeof(_0mipmap));
    rpc_write(client, &extMem, sizeof(extMem));
    rpc_write(client, &_0mipmapDesc, sizeof(_0mipmapDesc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)mipmap, 0);
    mem2client((void *)mipmapDesc, 0);
    return _result;
}

extern "C" CUresult cuDestroyExternalMemory(CUexternalMemory extMem) {
#ifdef DEBUG
    std::cout << "Hook: cuDestroyExternalMemory called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDestroyExternalMemory);
    rpc_write(client, &extMem, sizeof(extMem));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuImportExternalSemaphore(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuImportExternalSemaphore called" << std::endl;
#endif
    void *_0extSem_out = mem2server((void *)extSem_out, 0);
    void *_0semHandleDesc = mem2server((void *)semHandleDesc, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuImportExternalSemaphore);
    rpc_write(client, &_0extSem_out, sizeof(_0extSem_out));
    rpc_write(client, &_0semHandleDesc, sizeof(_0semHandleDesc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)extSem_out, 0);
    mem2client((void *)semHandleDesc, 0);
    return _result;
}

extern "C" CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream) {
#ifdef DEBUG
    std::cout << "Hook: cuSignalExternalSemaphoresAsync called" << std::endl;
#endif
    void *_0extSemArray = mem2server((void *)extSemArray, 0);
    void *_0paramsArray = mem2server((void *)paramsArray, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuSignalExternalSemaphoresAsync);
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
    rpc_free_client(client);
    mem2client((void *)extSemArray, 0);
    mem2client((void *)paramsArray, 0);
    return _result;
}

extern "C" CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream) {
#ifdef DEBUG
    std::cout << "Hook: cuWaitExternalSemaphoresAsync called" << std::endl;
#endif
    void *_0extSemArray = mem2server((void *)extSemArray, 0);
    void *_0paramsArray = mem2server((void *)paramsArray, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuWaitExternalSemaphoresAsync);
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
    rpc_free_client(client);
    mem2client((void *)extSemArray, 0);
    mem2client((void *)paramsArray, 0);
    return _result;
}

extern "C" CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
#ifdef DEBUG
    std::cout << "Hook: cuDestroyExternalSemaphore called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDestroyExternalSemaphore);
    rpc_write(client, &extSem, sizeof(extSem));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWaitValue32_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamWaitValue32_v2);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &addr, sizeof(addr));
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWaitValue64_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamWaitValue64_v2);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &addr, sizeof(addr));
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWriteValue32_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamWriteValue32_v2);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &addr, sizeof(addr));
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWriteValue64_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamWriteValue64_v2);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &addr, sizeof(addr));
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamBatchMemOp_v2 called" << std::endl;
#endif
    void *_0paramArray = mem2server((void *)paramArray, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamBatchMemOp_v2);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_0paramArray, sizeof(_0paramArray));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)paramArray, 0);
    return _result;
}

extern "C" CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncGetAttribute called" << std::endl;
#endif
    void *_0pi = mem2server((void *)pi, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuFuncGetAttribute);
    rpc_write(client, &_0pi, sizeof(_0pi));
    rpc_write(client, &attrib, sizeof(attrib));
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pi, 0);
    return _result;
}

extern "C" CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetAttribute called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuFuncSetAttribute);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_write(client, &attrib, sizeof(attrib));
    rpc_write(client, &value, sizeof(value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetCacheConfig called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuFuncSetCacheConfig);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_write(client, &config, sizeof(config));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuFuncGetModule(CUmodule *hmod, CUfunction hfunc) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncGetModule called" << std::endl;
#endif
    void *_0hmod = mem2server((void *)hmod, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuFuncGetModule);
    rpc_write(client, &_0hmod, sizeof(_0hmod));
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)hmod, 0);
    return _result;
}

extern "C" CUresult cuFuncGetName(const char **name, CUfunction hfunc) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncGetName called" << std::endl;
#endif
    // PARAM const char **name
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuFuncGetName);
    // PARAM const char **name
    static char _cuFuncGetName_name[1024];
    rpc_read(client, _cuFuncGetName_name, 1024, true);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM const char **name
    *name = _cuFuncGetName_name;
    rpc_free_client(client);
    // PARAM const char **name
    return _result;
}

extern "C" CUresult cuFuncGetParamInfo(CUfunction func, size_t paramIndex, size_t *paramOffset, size_t *paramSize) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncGetParamInfo called" << std::endl;
#endif
    void *_0paramOffset = mem2server((void *)paramOffset, 0);
    void *_0paramSize = mem2server((void *)paramSize, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuFuncGetParamInfo);
    rpc_write(client, &func, sizeof(func));
    rpc_write(client, &paramIndex, sizeof(paramIndex));
    rpc_write(client, &_0paramOffset, sizeof(_0paramOffset));
    rpc_write(client, &_0paramSize, sizeof(_0paramSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)paramOffset, 0);
    mem2client((void *)paramSize, 0);
    return _result;
}

extern "C" CUresult cuFuncIsLoaded(CUfunctionLoadingState *state, CUfunction function) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncIsLoaded called" << std::endl;
#endif
    void *_0state = mem2server((void *)state, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuFuncIsLoaded);
    rpc_write(client, &_0state, sizeof(_0state));
    rpc_write(client, &function, sizeof(function));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)state, 0);
    return _result;
}

extern "C" CUresult cuFuncLoad(CUfunction function) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncLoad called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuFuncLoad);
    rpc_write(client, &function, sizeof(function));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchKernel called" << std::endl;
#endif
    // PARAM void **kernelParams
    // PARAM void **extra
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLaunchKernel);
    rpc_write(client, &f, sizeof(f));
    rpc_write(client, &gridDimX, sizeof(gridDimX));
    rpc_write(client, &gridDimY, sizeof(gridDimY));
    rpc_write(client, &gridDimZ, sizeof(gridDimZ));
    rpc_write(client, &blockDimX, sizeof(blockDimX));
    rpc_write(client, &blockDimY, sizeof(blockDimY));
    rpc_write(client, &blockDimZ, sizeof(blockDimZ));
    rpc_write(client, &sharedMemBytes, sizeof(sharedMemBytes));
    rpc_write(client, &hStream, sizeof(hStream));
    // PARAM void **kernelParams
    // PARAM void **extra
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **kernelParams
    // PARAM void **extra
    rpc_free_client(client);
    // PARAM void **kernelParams
    // PARAM void **extra
    return _result;
}

extern "C" CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchKernelEx called" << std::endl;
#endif
    void *_0config = mem2server((void *)config, 0);
    // PARAM void **kernelParams
    // PARAM void **extra
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLaunchKernelEx);
    rpc_write(client, &_0config, sizeof(_0config));
    rpc_write(client, &f, sizeof(f));
    // PARAM void **kernelParams
    // PARAM void **extra
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **kernelParams
    // PARAM void **extra
    rpc_free_client(client);
    mem2client((void *)config, 0);
    // PARAM void **kernelParams
    // PARAM void **extra
    return _result;
}

extern "C" CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchCooperativeKernel called" << std::endl;
#endif
    // PARAM void **kernelParams
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLaunchCooperativeKernel);
    rpc_write(client, &f, sizeof(f));
    rpc_write(client, &gridDimX, sizeof(gridDimX));
    rpc_write(client, &gridDimY, sizeof(gridDimY));
    rpc_write(client, &gridDimZ, sizeof(gridDimZ));
    rpc_write(client, &blockDimX, sizeof(blockDimX));
    rpc_write(client, &blockDimY, sizeof(blockDimY));
    rpc_write(client, &blockDimZ, sizeof(blockDimZ));
    rpc_write(client, &sharedMemBytes, sizeof(sharedMemBytes));
    rpc_write(client, &hStream, sizeof(hStream));
    // PARAM void **kernelParams
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **kernelParams
    rpc_free_client(client);
    // PARAM void **kernelParams
    return _result;
}

extern "C" CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchCooperativeKernelMultiDevice called" << std::endl;
#endif
    void *_0launchParamsList = mem2server((void *)launchParamsList, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLaunchCooperativeKernelMultiDevice);
    rpc_write(client, &_0launchParamsList, sizeof(_0launchParamsList));
    rpc_write(client, &numDevices, sizeof(numDevices));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)launchParamsList, 0);
    return _result;
}

extern "C" CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchHostFunc called" << std::endl;
#endif
    void *_0userData = mem2server((void *)userData, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLaunchHostFunc);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &fn, sizeof(fn));
    rpc_write(client, &_0userData, sizeof(_0userData));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)userData, 0);
    return _result;
}

extern "C" CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetBlockShape called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuFuncSetBlockShape);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_write(client, &x, sizeof(x));
    rpc_write(client, &y, sizeof(y));
    rpc_write(client, &z, sizeof(z));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetSharedSize called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuFuncSetSharedSize);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_write(client, &bytes, sizeof(bytes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSetSize called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuParamSetSize);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_write(client, &numbytes, sizeof(numbytes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSeti called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuParamSeti);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &value, sizeof(value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuParamSetf(CUfunction hfunc, int offset, float value) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSetf called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuParamSetf);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &value, sizeof(value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSetv called" << std::endl;
#endif
    void *_0ptr = mem2server((void *)ptr, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuParamSetv);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &_0ptr, sizeof(_0ptr));
    rpc_write(client, &numbytes, sizeof(numbytes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)ptr, 0);
    return _result;
}

extern "C" CUresult cuLaunch(CUfunction f) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunch called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLaunch);
    rpc_write(client, &f, sizeof(f));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchGrid called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLaunchGrid);
    rpc_write(client, &f, sizeof(f));
    rpc_write(client, &grid_width, sizeof(grid_width));
    rpc_write(client, &grid_height, sizeof(grid_height));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchGridAsync called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLaunchGridAsync);
    rpc_write(client, &f, sizeof(f));
    rpc_write(client, &grid_width, sizeof(grid_width));
    rpc_write(client, &grid_height, sizeof(grid_height));
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

extern "C" CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSetTexRef called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuParamSetTexRef);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_write(client, &texunit, sizeof(texunit));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetSharedMemConfig called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuFuncSetSharedMemConfig);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_write(client, &config, sizeof(config));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphCreate called" << std::endl;
#endif
    void *_0phGraph = mem2server((void *)phGraph, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphCreate);
    rpc_write(client, &_0phGraph, sizeof(_0phGraph));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraph, 0);
    return _result;
}

extern "C" CUresult cuGraphAddKernelNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddKernelNode_v2 called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddKernelNode_v2);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeGetParams_v2 called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphKernelNodeGetParams_v2);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphKernelNodeSetParams_v2(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeSetParams_v2 called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphKernelNodeSetParams_v2);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddMemcpyNode called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0copyParams = mem2server((void *)copyParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddMemcpyNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0copyParams, sizeof(_0copyParams));
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    mem2client((void *)copyParams, 0);
    return _result;
}

extern "C" CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemcpyNodeGetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphMemcpyNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemcpyNodeSetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphMemcpyNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddMemsetNode called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0memsetParams = mem2server((void *)memsetParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddMemsetNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0memsetParams, sizeof(_0memsetParams));
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    mem2client((void *)memsetParams, 0);
    return _result;
}

extern "C" CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemsetNodeGetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphMemsetNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemsetNodeSetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphMemsetNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddHostNode called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddHostNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphHostNodeGetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphHostNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphHostNodeSetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphHostNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddChildGraphNode called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddChildGraphNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &childGraph, sizeof(childGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    return _result;
}

extern "C" CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphChildGraphNodeGetGraph called" << std::endl;
#endif
    void *_0phGraph = mem2server((void *)phGraph, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphChildGraphNodeGetGraph);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0phGraph, sizeof(_0phGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraph, 0);
    return _result;
}

extern "C" CUresult cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddEmptyNode called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddEmptyNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    return _result;
}

extern "C" CUresult cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddEventRecordNode called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddEventRecordNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    return _result;
}

extern "C" CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphEventRecordNodeGetEvent called" << std::endl;
#endif
    void *_0event_out = mem2server((void *)event_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphEventRecordNodeGetEvent);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0event_out, sizeof(_0event_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)event_out, 0);
    return _result;
}

extern "C" CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphEventRecordNodeSetEvent called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphEventRecordNodeSetEvent);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddEventWaitNode called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddEventWaitNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    return _result;
}

extern "C" CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphEventWaitNodeGetEvent called" << std::endl;
#endif
    void *_0event_out = mem2server((void *)event_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphEventWaitNodeGetEvent);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0event_out, sizeof(_0event_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)event_out, 0);
    return _result;
}

extern "C" CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphEventWaitNodeSetEvent called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphEventWaitNodeSetEvent);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddExternalSemaphoresSignalNode called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddExternalSemaphoresSignalNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExternalSemaphoresSignalNodeGetParams called" << std::endl;
#endif
    void *_0params_out = mem2server((void *)params_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExternalSemaphoresSignalNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0params_out, sizeof(_0params_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)params_out, 0);
    return _result;
}

extern "C" CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExternalSemaphoresSignalNodeSetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExternalSemaphoresSignalNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddExternalSemaphoresWaitNode called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddExternalSemaphoresWaitNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExternalSemaphoresWaitNodeGetParams called" << std::endl;
#endif
    void *_0params_out = mem2server((void *)params_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExternalSemaphoresWaitNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0params_out, sizeof(_0params_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)params_out, 0);
    return _result;
}

extern "C" CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExternalSemaphoresWaitNodeSetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExternalSemaphoresWaitNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphAddBatchMemOpNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddBatchMemOpNode called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddBatchMemOpNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphBatchMemOpNodeGetParams called" << std::endl;
#endif
    void *_0nodeParams_out = mem2server((void *)nodeParams_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphBatchMemOpNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams_out, sizeof(_0nodeParams_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams_out, 0);
    return _result;
}

extern "C" CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphBatchMemOpNodeSetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphBatchMemOpNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecBatchMemOpNodeSetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecBatchMemOpNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddMemAllocNode called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddMemAllocNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemAllocNodeGetParams called" << std::endl;
#endif
    void *_0params_out = mem2server((void *)params_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphMemAllocNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0params_out, sizeof(_0params_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)params_out, 0);
    return _result;
}

extern "C" CUresult cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddMemFreeNode called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddMemFreeNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    return _result;
}

extern "C" CUresult cuDeviceGraphMemTrim(CUdevice device) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGraphMemTrim called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGraphMemTrim);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetGraphMemAttribute called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetGraphMemAttribute);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value, 0);
    return _result;
}

extern "C" CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceSetGraphMemAttribute called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceSetGraphMemAttribute);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value, 0);
    return _result;
}

extern "C" CUresult cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphClone called" << std::endl;
#endif
    void *_0phGraphClone = mem2server((void *)phGraphClone, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphClone);
    rpc_write(client, &_0phGraphClone, sizeof(_0phGraphClone));
    rpc_write(client, &originalGraph, sizeof(originalGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphClone, 0);
    return _result;
}

extern "C" CUresult cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeFindInClone called" << std::endl;
#endif
    void *_0phNode = mem2server((void *)phNode, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphNodeFindInClone);
    rpc_write(client, &_0phNode, sizeof(_0phNode));
    rpc_write(client, &hOriginalNode, sizeof(hOriginalNode));
    rpc_write(client, &hClonedGraph, sizeof(hClonedGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phNode, 0);
    return _result;
}

extern "C" CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeGetType called" << std::endl;
#endif
    void *_0type = mem2server((void *)type, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphNodeGetType);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0type, sizeof(_0type));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)type, 0);
    return _result;
}

extern "C" CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphGetNodes called" << std::endl;
#endif
    void *_0nodes = mem2server((void *)nodes, 0);
    void *_0numNodes = mem2server((void *)numNodes, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphGetNodes);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0nodes, sizeof(_0nodes));
    rpc_write(client, &_0numNodes, sizeof(_0numNodes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodes, 0);
    mem2client((void *)numNodes, 0);
    return _result;
}

extern "C" CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphGetRootNodes called" << std::endl;
#endif
    void *_0rootNodes = mem2server((void *)rootNodes, 0);
    void *_0numRootNodes = mem2server((void *)numRootNodes, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphGetRootNodes);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0rootNodes, sizeof(_0rootNodes));
    rpc_write(client, &_0numRootNodes, sizeof(_0numRootNodes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)rootNodes, 0);
    mem2client((void *)numRootNodes, 0);
    return _result;
}

extern "C" CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphGetEdges called" << std::endl;
#endif
    void *_0from = mem2server((void *)from, 0);
    void *_0to = mem2server((void *)to, 0);
    void *_0numEdges = mem2server((void *)numEdges, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphGetEdges);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0from, sizeof(_0from));
    rpc_write(client, &_0to, sizeof(_0to));
    rpc_write(client, &_0numEdges, sizeof(_0numEdges));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)from, 0);
    mem2client((void *)to, 0);
    mem2client((void *)numEdges, 0);
    return _result;
}

extern "C" CUresult cuGraphGetEdges_v2(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, CUgraphEdgeData *edgeData, size_t *numEdges) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphGetEdges_v2 called" << std::endl;
#endif
    void *_0from = mem2server((void *)from, 0);
    void *_0to = mem2server((void *)to, 0);
    void *_0edgeData = mem2server((void *)edgeData, 0);
    void *_0numEdges = mem2server((void *)numEdges, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphGetEdges_v2);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0from, sizeof(_0from));
    rpc_write(client, &_0to, sizeof(_0to));
    rpc_write(client, &_0edgeData, sizeof(_0edgeData));
    rpc_write(client, &_0numEdges, sizeof(_0numEdges));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)from, 0);
    mem2client((void *)to, 0);
    mem2client((void *)edgeData, 0);
    mem2client((void *)numEdges, 0);
    return _result;
}

extern "C" CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeGetDependencies called" << std::endl;
#endif
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0numDependencies = mem2server((void *)numDependencies, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphNodeGetDependencies);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &_0numDependencies, sizeof(_0numDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dependencies, 0);
    mem2client((void *)numDependencies, 0);
    return _result;
}

extern "C" CUresult cuGraphNodeGetDependencies_v2(CUgraphNode hNode, CUgraphNode *dependencies, CUgraphEdgeData *edgeData, size_t *numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeGetDependencies_v2 called" << std::endl;
#endif
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0edgeData = mem2server((void *)edgeData, 0);
    void *_0numDependencies = mem2server((void *)numDependencies, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphNodeGetDependencies_v2);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &_0edgeData, sizeof(_0edgeData));
    rpc_write(client, &_0numDependencies, sizeof(_0numDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dependencies, 0);
    mem2client((void *)edgeData, 0);
    mem2client((void *)numDependencies, 0);
    return _result;
}

extern "C" CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeGetDependentNodes called" << std::endl;
#endif
    void *_0dependentNodes = mem2server((void *)dependentNodes, 0);
    void *_0numDependentNodes = mem2server((void *)numDependentNodes, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphNodeGetDependentNodes);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0dependentNodes, sizeof(_0dependentNodes));
    rpc_write(client, &_0numDependentNodes, sizeof(_0numDependentNodes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dependentNodes, 0);
    mem2client((void *)numDependentNodes, 0);
    return _result;
}

extern "C" CUresult cuGraphNodeGetDependentNodes_v2(CUgraphNode hNode, CUgraphNode *dependentNodes, CUgraphEdgeData *edgeData, size_t *numDependentNodes) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeGetDependentNodes_v2 called" << std::endl;
#endif
    void *_0dependentNodes = mem2server((void *)dependentNodes, 0);
    void *_0edgeData = mem2server((void *)edgeData, 0);
    void *_0numDependentNodes = mem2server((void *)numDependentNodes, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphNodeGetDependentNodes_v2);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0dependentNodes, sizeof(_0dependentNodes));
    rpc_write(client, &_0edgeData, sizeof(_0edgeData));
    rpc_write(client, &_0numDependentNodes, sizeof(_0numDependentNodes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dependentNodes, 0);
    mem2client((void *)edgeData, 0);
    mem2client((void *)numDependentNodes, 0);
    return _result;
}

extern "C" CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddDependencies called" << std::endl;
#endif
    void *_0from = mem2server((void *)from, 0);
    void *_0to = mem2server((void *)to, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddDependencies);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0from, sizeof(_0from));
    rpc_write(client, &_0to, sizeof(_0to));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)from, 0);
    mem2client((void *)to, 0);
    return _result;
}

extern "C" CUresult cuGraphAddDependencies_v2(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, const CUgraphEdgeData *edgeData, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddDependencies_v2 called" << std::endl;
#endif
    void *_0from = mem2server((void *)from, 0);
    void *_0to = mem2server((void *)to, 0);
    void *_0edgeData = mem2server((void *)edgeData, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddDependencies_v2);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0from, sizeof(_0from));
    rpc_write(client, &_0to, sizeof(_0to));
    rpc_write(client, &_0edgeData, sizeof(_0edgeData));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)from, 0);
    mem2client((void *)to, 0);
    mem2client((void *)edgeData, 0);
    return _result;
}

extern "C" CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphRemoveDependencies called" << std::endl;
#endif
    void *_0from = mem2server((void *)from, 0);
    void *_0to = mem2server((void *)to, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphRemoveDependencies);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0from, sizeof(_0from));
    rpc_write(client, &_0to, sizeof(_0to));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)from, 0);
    mem2client((void *)to, 0);
    return _result;
}

extern "C" CUresult cuGraphRemoveDependencies_v2(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, const CUgraphEdgeData *edgeData, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphRemoveDependencies_v2 called" << std::endl;
#endif
    void *_0from = mem2server((void *)from, 0);
    void *_0to = mem2server((void *)to, 0);
    void *_0edgeData = mem2server((void *)edgeData, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphRemoveDependencies_v2);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0from, sizeof(_0from));
    rpc_write(client, &_0to, sizeof(_0to));
    rpc_write(client, &_0edgeData, sizeof(_0edgeData));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)from, 0);
    mem2client((void *)to, 0);
    mem2client((void *)edgeData, 0);
    return _result;
}

extern "C" CUresult cuGraphDestroyNode(CUgraphNode hNode) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphDestroyNode called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphDestroyNode);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphInstantiateWithFlags called" << std::endl;
#endif
    void *_0phGraphExec = mem2server((void *)phGraphExec, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphInstantiateWithFlags);
    rpc_write(client, &_0phGraphExec, sizeof(_0phGraphExec));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphExec, 0);
    return _result;
}

extern "C" CUresult cuGraphInstantiateWithParams(CUgraphExec *phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS *instantiateParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphInstantiateWithParams called" << std::endl;
#endif
    void *_0phGraphExec = mem2server((void *)phGraphExec, 0);
    void *_0instantiateParams = mem2server((void *)instantiateParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphInstantiateWithParams);
    rpc_write(client, &_0phGraphExec, sizeof(_0phGraphExec));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0instantiateParams, sizeof(_0instantiateParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphExec, 0);
    mem2client((void *)instantiateParams, 0);
    return _result;
}

extern "C" CUresult cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t *flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecGetFlags called" << std::endl;
#endif
    void *_0flags = mem2server((void *)flags, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecGetFlags);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &_0flags, sizeof(_0flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)flags, 0);
    return _result;
}

extern "C" CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecKernelNodeSetParams_v2 called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecKernelNodeSetParams_v2);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecMemcpyNodeSetParams called" << std::endl;
#endif
    void *_0copyParams = mem2server((void *)copyParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecMemcpyNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0copyParams, sizeof(_0copyParams));
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)copyParams, 0);
    return _result;
}

extern "C" CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecMemsetNodeSetParams called" << std::endl;
#endif
    void *_0memsetParams = mem2server((void *)memsetParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecMemsetNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0memsetParams, sizeof(_0memsetParams));
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)memsetParams, 0);
    return _result;
}

extern "C" CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecHostNodeSetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecHostNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecChildGraphNodeSetParams called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecChildGraphNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &childGraph, sizeof(childGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecEventRecordNodeSetEvent called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecEventRecordNodeSetEvent);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecEventWaitNodeSetEvent called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecEventWaitNodeSetEvent);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecExternalSemaphoresSignalNodeSetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecExternalSemaphoresSignalNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecExternalSemaphoresWaitNodeSetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecExternalSemaphoresWaitNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeSetEnabled called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphNodeSetEnabled);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &isEnabled, sizeof(isEnabled));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int *isEnabled) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeGetEnabled called" << std::endl;
#endif
    void *_0isEnabled = mem2server((void *)isEnabled, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphNodeGetEnabled);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0isEnabled, sizeof(_0isEnabled));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)isEnabled, 0);
    return _result;
}

extern "C" CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphUpload called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphUpload);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
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

extern "C" CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphLaunch called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphLaunch);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
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

extern "C" CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecDestroy called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecDestroy);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphDestroy(CUgraph hGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphDestroy called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphDestroy);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo *resultInfo) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecUpdate_v2 called" << std::endl;
#endif
    void *_0resultInfo = mem2server((void *)resultInfo, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecUpdate_v2);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0resultInfo, sizeof(_0resultInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)resultInfo, 0);
    return _result;
}

extern "C" CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeCopyAttributes called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphKernelNodeCopyAttributes);
    rpc_write(client, &dst, sizeof(dst));
    rpc_write(client, &src, sizeof(src));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue *value_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeGetAttribute called" << std::endl;
#endif
    void *_0value_out = mem2server((void *)value_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphKernelNodeGetAttribute);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value_out, sizeof(_0value_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value_out, 0);
    return _result;
}

extern "C" CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue *value) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeSetAttribute called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphKernelNodeSetAttribute);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value, 0);
    return _result;
}

extern "C" CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char *path, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphDebugDotPrint called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphDebugDotPrint);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, path, strlen(path) + 1, true);
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuUserObjectCreate(CUuserObject *object_out, void *ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuUserObjectCreate called" << std::endl;
#endif
    void *_0object_out = mem2server((void *)object_out, 0);
    void *_0ptr = mem2server((void *)ptr, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuUserObjectCreate);
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
    rpc_free_client(client);
    mem2client((void *)object_out, 0);
    mem2client((void *)ptr, 0);
    return _result;
}

extern "C" CUresult cuUserObjectRetain(CUuserObject object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cuUserObjectRetain called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuUserObjectRetain);
    rpc_write(client, &object, sizeof(object));
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

extern "C" CUresult cuUserObjectRelease(CUuserObject object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cuUserObjectRelease called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuUserObjectRelease);
    rpc_write(client, &object, sizeof(object));
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

extern "C" CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphRetainUserObject called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphRetainUserObject);
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
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphReleaseUserObject called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphReleaseUserObject);
    rpc_write(client, &graph, sizeof(graph));
    rpc_write(client, &object, sizeof(object));
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

extern "C" CUresult cuGraphAddNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraphNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddNode called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphAddNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, const CUgraphEdgeData *dependencyData, size_t numDependencies, CUgraphNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddNode_v2 called" << std::endl;
#endif
    void *_0phGraphNode = mem2server((void *)phGraphNode, 0);
    void *_0dependencies = mem2server((void *)dependencies, 0);
    void *_0dependencyData = mem2server((void *)dependencyData, 0);
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphAddNode_v2);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    rpc_write(client, &_0dependencyData, sizeof(_0dependencyData));
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phGraphNode, 0);
    mem2client((void *)dependencies, 0);
    mem2client((void *)dependencyData, 0);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphNodeSetParams(CUgraphNode hNode, CUgraphNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeSetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphExecNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraphNodeParams *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecNodeSetParams called" << std::endl;
#endif
    void *_0nodeParams = mem2server((void *)nodeParams, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphExecNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeParams, 0);
    return _result;
}

extern "C" CUresult cuGraphConditionalHandleCreate(CUgraphConditionalHandle *pHandle_out, CUgraph hGraph, CUcontext ctx, unsigned int defaultLaunchValue, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphConditionalHandleCreate called" << std::endl;
#endif
    void *_0pHandle_out = mem2server((void *)pHandle_out, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphConditionalHandleCreate);
    rpc_write(client, &_0pHandle_out, sizeof(_0pHandle_out));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_write(client, &defaultLaunchValue, sizeof(defaultLaunchValue));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pHandle_out, 0);
    return _result;
}

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxActiveBlocksPerMultiprocessor called" << std::endl;
#endif
    void *_0numBlocks = mem2server((void *)numBlocks, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor);
    rpc_write(client, &_0numBlocks, sizeof(_0numBlocks));
    rpc_write(client, &func, sizeof(func));
    rpc_write(client, &blockSize, sizeof(blockSize));
    rpc_write(client, &dynamicSMemSize, sizeof(dynamicSMemSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)numBlocks, 0);
    return _result;
}

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags called" << std::endl;
#endif
    void *_0numBlocks = mem2server((void *)numBlocks, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    rpc_write(client, &_0numBlocks, sizeof(_0numBlocks));
    rpc_write(client, &func, sizeof(func));
    rpc_write(client, &blockSize, sizeof(blockSize));
    rpc_write(client, &dynamicSMemSize, sizeof(dynamicSMemSize));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)numBlocks, 0);
    return _result;
}

extern "C" CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxPotentialBlockSize called" << std::endl;
#endif
    void *_0minGridSize = mem2server((void *)minGridSize, 0);
    void *_0blockSize = mem2server((void *)blockSize, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuOccupancyMaxPotentialBlockSize);
    rpc_write(client, &_0minGridSize, sizeof(_0minGridSize));
    rpc_write(client, &_0blockSize, sizeof(_0blockSize));
    rpc_write(client, &func, sizeof(func));
    rpc_write(client, &blockSizeToDynamicSMemSize, sizeof(blockSizeToDynamicSMemSize));
    rpc_write(client, &dynamicSMemSize, sizeof(dynamicSMemSize));
    rpc_write(client, &blockSizeLimit, sizeof(blockSizeLimit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)minGridSize, 0);
    mem2client((void *)blockSize, 0);
    return _result;
}

extern "C" CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxPotentialBlockSizeWithFlags called" << std::endl;
#endif
    void *_0minGridSize = mem2server((void *)minGridSize, 0);
    void *_0blockSize = mem2server((void *)blockSize, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuOccupancyMaxPotentialBlockSizeWithFlags);
    rpc_write(client, &_0minGridSize, sizeof(_0minGridSize));
    rpc_write(client, &_0blockSize, sizeof(_0blockSize));
    rpc_write(client, &func, sizeof(func));
    rpc_write(client, &blockSizeToDynamicSMemSize, sizeof(blockSizeToDynamicSMemSize));
    rpc_write(client, &dynamicSMemSize, sizeof(dynamicSMemSize));
    rpc_write(client, &blockSizeLimit, sizeof(blockSizeLimit));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)minGridSize, 0);
    mem2client((void *)blockSize, 0);
    return _result;
}

extern "C" CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyAvailableDynamicSMemPerBlock called" << std::endl;
#endif
    void *_0dynamicSmemSize = mem2server((void *)dynamicSmemSize, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuOccupancyAvailableDynamicSMemPerBlock);
    rpc_write(client, &_0dynamicSmemSize, sizeof(_0dynamicSmemSize));
    rpc_write(client, &func, sizeof(func));
    rpc_write(client, &numBlocks, sizeof(numBlocks));
    rpc_write(client, &blockSize, sizeof(blockSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dynamicSmemSize, 0);
    return _result;
}

extern "C" CUresult cuOccupancyMaxPotentialClusterSize(int *clusterSize, CUfunction func, const CUlaunchConfig *config) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxPotentialClusterSize called" << std::endl;
#endif
    void *_0clusterSize = mem2server((void *)clusterSize, 0);
    void *_0config = mem2server((void *)config, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuOccupancyMaxPotentialClusterSize);
    rpc_write(client, &_0clusterSize, sizeof(_0clusterSize));
    rpc_write(client, &func, sizeof(func));
    rpc_write(client, &_0config, sizeof(_0config));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)clusterSize, 0);
    mem2client((void *)config, 0);
    return _result;
}

extern "C" CUresult cuOccupancyMaxActiveClusters(int *numClusters, CUfunction func, const CUlaunchConfig *config) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxActiveClusters called" << std::endl;
#endif
    void *_0numClusters = mem2server((void *)numClusters, 0);
    void *_0config = mem2server((void *)config, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuOccupancyMaxActiveClusters);
    rpc_write(client, &_0numClusters, sizeof(_0numClusters));
    rpc_write(client, &func, sizeof(func));
    rpc_write(client, &_0config, sizeof(_0config));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)numClusters, 0);
    mem2client((void *)config, 0);
    return _result;
}

extern "C" CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetArray called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefSetArray);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &hArray, sizeof(hArray));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMipmappedArray called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefSetMipmappedArray);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &hMipmappedArray, sizeof(hMipmappedArray));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefSetAddress_v2(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetAddress_v2 called" << std::endl;
#endif
    void *_0ByteOffset = mem2server((void *)ByteOffset, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefSetAddress_v2);
    rpc_write(client, &_0ByteOffset, sizeof(_0ByteOffset));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_write(client, &bytes, sizeof(bytes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)ByteOffset, 0);
    return _result;
}

extern "C" CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetAddress2D_v3 called" << std::endl;
#endif
    void *_0desc = mem2server((void *)desc, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefSetAddress2D_v3);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &_0desc, sizeof(_0desc));
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_write(client, &Pitch, sizeof(Pitch));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)desc, 0);
    return _result;
}

extern "C" CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetFormat called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefSetFormat);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &fmt, sizeof(fmt));
    rpc_write(client, &NumPackedComponents, sizeof(NumPackedComponents));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetAddressMode called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefSetAddressMode);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &dim, sizeof(dim));
    rpc_write(client, &am, sizeof(am));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetFilterMode called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefSetFilterMode);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &fm, sizeof(fm));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMipmapFilterMode called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefSetMipmapFilterMode);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &fm, sizeof(fm));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMipmapLevelBias called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefSetMipmapLevelBias);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &bias, sizeof(bias));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMipmapLevelClamp called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefSetMipmapLevelClamp);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &minMipmapLevelClamp, sizeof(minMipmapLevelClamp));
    rpc_write(client, &maxMipmapLevelClamp, sizeof(maxMipmapLevelClamp));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMaxAnisotropy called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefSetMaxAnisotropy);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &maxAniso, sizeof(maxAniso));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetBorderColor called" << std::endl;
#endif
    void *_0pBorderColor = mem2server((void *)pBorderColor, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefSetBorderColor);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &_0pBorderColor, sizeof(_0pBorderColor));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pBorderColor, 0);
    return _result;
}

extern "C" CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetFlags called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefSetFlags);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetArray called" << std::endl;
#endif
    void *_0phArray = mem2server((void *)phArray, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefGetArray);
    rpc_write(client, &_0phArray, sizeof(_0phArray));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phArray, 0);
    return _result;
}

extern "C" CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMipmappedArray called" << std::endl;
#endif
    void *_0phMipmappedArray = mem2server((void *)phMipmappedArray, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefGetMipmappedArray);
    rpc_write(client, &_0phMipmappedArray, sizeof(_0phMipmappedArray));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phMipmappedArray, 0);
    return _result;
}

extern "C" CUresult cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetAddressMode called" << std::endl;
#endif
    void *_0pam = mem2server((void *)pam, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefGetAddressMode);
    rpc_write(client, &_0pam, sizeof(_0pam));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &dim, sizeof(dim));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pam, 0);
    return _result;
}

extern "C" CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetFilterMode called" << std::endl;
#endif
    void *_0pfm = mem2server((void *)pfm, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefGetFilterMode);
    rpc_write(client, &_0pfm, sizeof(_0pfm));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pfm, 0);
    return _result;
}

extern "C" CUresult cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetFormat called" << std::endl;
#endif
    void *_0pFormat = mem2server((void *)pFormat, 0);
    void *_0pNumChannels = mem2server((void *)pNumChannels, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefGetFormat);
    rpc_write(client, &_0pFormat, sizeof(_0pFormat));
    rpc_write(client, &_0pNumChannels, sizeof(_0pNumChannels));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pFormat, 0);
    mem2client((void *)pNumChannels, 0);
    return _result;
}

extern "C" CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMipmapFilterMode called" << std::endl;
#endif
    void *_0pfm = mem2server((void *)pfm, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefGetMipmapFilterMode);
    rpc_write(client, &_0pfm, sizeof(_0pfm));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pfm, 0);
    return _result;
}

extern "C" CUresult cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMipmapLevelBias called" << std::endl;
#endif
    void *_0pbias = mem2server((void *)pbias, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefGetMipmapLevelBias);
    rpc_write(client, &_0pbias, sizeof(_0pbias));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pbias, 0);
    return _result;
}

extern "C" CUresult cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMipmapLevelClamp called" << std::endl;
#endif
    void *_0pminMipmapLevelClamp = mem2server((void *)pminMipmapLevelClamp, 0);
    void *_0pmaxMipmapLevelClamp = mem2server((void *)pmaxMipmapLevelClamp, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefGetMipmapLevelClamp);
    rpc_write(client, &_0pminMipmapLevelClamp, sizeof(_0pminMipmapLevelClamp));
    rpc_write(client, &_0pmaxMipmapLevelClamp, sizeof(_0pmaxMipmapLevelClamp));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pminMipmapLevelClamp, 0);
    mem2client((void *)pmaxMipmapLevelClamp, 0);
    return _result;
}

extern "C" CUresult cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMaxAnisotropy called" << std::endl;
#endif
    void *_0pmaxAniso = mem2server((void *)pmaxAniso, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefGetMaxAnisotropy);
    rpc_write(client, &_0pmaxAniso, sizeof(_0pmaxAniso));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pmaxAniso, 0);
    return _result;
}

extern "C" CUresult cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetBorderColor called" << std::endl;
#endif
    void *_0pBorderColor = mem2server((void *)pBorderColor, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefGetBorderColor);
    rpc_write(client, &_0pBorderColor, sizeof(_0pBorderColor));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pBorderColor, 0);
    return _result;
}

extern "C" CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetFlags called" << std::endl;
#endif
    void *_0pFlags = mem2server((void *)pFlags, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefGetFlags);
    rpc_write(client, &_0pFlags, sizeof(_0pFlags));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pFlags, 0);
    return _result;
}

extern "C" CUresult cuTexRefCreate(CUtexref *pTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefCreate called" << std::endl;
#endif
    void *_0pTexRef = mem2server((void *)pTexRef, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefCreate);
    rpc_write(client, &_0pTexRef, sizeof(_0pTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pTexRef, 0);
    return _result;
}

extern "C" CUresult cuTexRefDestroy(CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefDestroy called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefDestroy);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfRefSetArray called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuSurfRefSetArray);
    rpc_write(client, &hSurfRef, sizeof(hSurfRef));
    rpc_write(client, &hArray, sizeof(hArray));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfRefGetArray called" << std::endl;
#endif
    void *_0phArray = mem2server((void *)phArray, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuSurfRefGetArray);
    rpc_write(client, &_0phArray, sizeof(_0phArray));
    rpc_write(client, &hSurfRef, sizeof(hSurfRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phArray, 0);
    return _result;
}

extern "C" CUresult cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc, const CUDA_TEXTURE_DESC *pTexDesc, const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectCreate called" << std::endl;
#endif
    void *_0pTexObject = mem2server((void *)pTexObject, 0);
    void *_0pResDesc = mem2server((void *)pResDesc, 0);
    void *_0pTexDesc = mem2server((void *)pTexDesc, 0);
    void *_0pResViewDesc = mem2server((void *)pResViewDesc, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexObjectCreate);
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
    rpc_free_client(client);
    mem2client((void *)pTexObject, 0);
    mem2client((void *)pResDesc, 0);
    mem2client((void *)pTexDesc, 0);
    mem2client((void *)pResViewDesc, 0);
    return _result;
}

extern "C" CUresult cuTexObjectDestroy(CUtexObject texObject) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectDestroy called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexObjectDestroy);
    rpc_write(client, &texObject, sizeof(texObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectGetResourceDesc called" << std::endl;
#endif
    void *_0pResDesc = mem2server((void *)pResDesc, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexObjectGetResourceDesc);
    rpc_write(client, &_0pResDesc, sizeof(_0pResDesc));
    rpc_write(client, &texObject, sizeof(texObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pResDesc, 0);
    return _result;
}

extern "C" CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectGetTextureDesc called" << std::endl;
#endif
    void *_0pTexDesc = mem2server((void *)pTexDesc, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexObjectGetTextureDesc);
    rpc_write(client, &_0pTexDesc, sizeof(_0pTexDesc));
    rpc_write(client, &texObject, sizeof(texObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pTexDesc, 0);
    return _result;
}

extern "C" CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectGetResourceViewDesc called" << std::endl;
#endif
    void *_0pResViewDesc = mem2server((void *)pResViewDesc, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexObjectGetResourceViewDesc);
    rpc_write(client, &_0pResViewDesc, sizeof(_0pResViewDesc));
    rpc_write(client, &texObject, sizeof(texObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pResViewDesc, 0);
    return _result;
}

extern "C" CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfObjectCreate called" << std::endl;
#endif
    void *_0pSurfObject = mem2server((void *)pSurfObject, 0);
    void *_0pResDesc = mem2server((void *)pResDesc, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuSurfObjectCreate);
    rpc_write(client, &_0pSurfObject, sizeof(_0pSurfObject));
    rpc_write(client, &_0pResDesc, sizeof(_0pResDesc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pSurfObject, 0);
    mem2client((void *)pResDesc, 0);
    return _result;
}

extern "C" CUresult cuSurfObjectDestroy(CUsurfObject surfObject) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfObjectDestroy called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuSurfObjectDestroy);
    rpc_write(client, &surfObject, sizeof(surfObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUsurfObject surfObject) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfObjectGetResourceDesc called" << std::endl;
#endif
    void *_0pResDesc = mem2server((void *)pResDesc, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuSurfObjectGetResourceDesc);
    rpc_write(client, &_0pResDesc, sizeof(_0pResDesc));
    rpc_write(client, &surfObject, sizeof(surfObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pResDesc, 0);
    return _result;
}

extern "C" CUresult cuTensorMapEncodeTiled(CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim, const cuuint64_t *globalStrides, const cuuint32_t *boxDim, const cuuint32_t *elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
#ifdef DEBUG
    std::cout << "Hook: cuTensorMapEncodeTiled called" << std::endl;
#endif
    void *_0tensorMap = mem2server((void *)tensorMap, 0);
    void *_0globalAddress = mem2server((void *)globalAddress, 0);
    void *_0globalDim = mem2server((void *)globalDim, 0);
    void *_0globalStrides = mem2server((void *)globalStrides, 0);
    void *_0boxDim = mem2server((void *)boxDim, 0);
    void *_0elementStrides = mem2server((void *)elementStrides, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTensorMapEncodeTiled);
    rpc_write(client, &_0tensorMap, sizeof(_0tensorMap));
    rpc_write(client, &tensorDataType, sizeof(tensorDataType));
    rpc_write(client, &tensorRank, sizeof(tensorRank));
    rpc_write(client, &_0globalAddress, sizeof(_0globalAddress));
    rpc_write(client, &_0globalDim, sizeof(_0globalDim));
    rpc_write(client, &_0globalStrides, sizeof(_0globalStrides));
    rpc_write(client, &_0boxDim, sizeof(_0boxDim));
    rpc_write(client, &_0elementStrides, sizeof(_0elementStrides));
    rpc_write(client, &interleave, sizeof(interleave));
    rpc_write(client, &swizzle, sizeof(swizzle));
    rpc_write(client, &l2Promotion, sizeof(l2Promotion));
    rpc_write(client, &oobFill, sizeof(oobFill));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)tensorMap, 0);
    mem2client((void *)globalAddress, 0);
    mem2client((void *)globalDim, 0);
    mem2client((void *)globalStrides, 0);
    mem2client((void *)boxDim, 0);
    mem2client((void *)elementStrides, 0);
    return _result;
}

extern "C" CUresult cuTensorMapEncodeIm2col(CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim, const cuuint64_t *globalStrides, const int *pixelBoxLowerCorner, const int *pixelBoxUpperCorner, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t *elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
#ifdef DEBUG
    std::cout << "Hook: cuTensorMapEncodeIm2col called" << std::endl;
#endif
    void *_0tensorMap = mem2server((void *)tensorMap, 0);
    void *_0globalAddress = mem2server((void *)globalAddress, 0);
    void *_0globalDim = mem2server((void *)globalDim, 0);
    void *_0globalStrides = mem2server((void *)globalStrides, 0);
    void *_0pixelBoxLowerCorner = mem2server((void *)pixelBoxLowerCorner, 0);
    void *_0pixelBoxUpperCorner = mem2server((void *)pixelBoxUpperCorner, 0);
    void *_0elementStrides = mem2server((void *)elementStrides, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTensorMapEncodeIm2col);
    rpc_write(client, &_0tensorMap, sizeof(_0tensorMap));
    rpc_write(client, &tensorDataType, sizeof(tensorDataType));
    rpc_write(client, &tensorRank, sizeof(tensorRank));
    rpc_write(client, &_0globalAddress, sizeof(_0globalAddress));
    rpc_write(client, &_0globalDim, sizeof(_0globalDim));
    rpc_write(client, &_0globalStrides, sizeof(_0globalStrides));
    rpc_write(client, &_0pixelBoxLowerCorner, sizeof(_0pixelBoxLowerCorner));
    rpc_write(client, &_0pixelBoxUpperCorner, sizeof(_0pixelBoxUpperCorner));
    rpc_write(client, &channelsPerPixel, sizeof(channelsPerPixel));
    rpc_write(client, &pixelsPerColumn, sizeof(pixelsPerColumn));
    rpc_write(client, &_0elementStrides, sizeof(_0elementStrides));
    rpc_write(client, &interleave, sizeof(interleave));
    rpc_write(client, &swizzle, sizeof(swizzle));
    rpc_write(client, &l2Promotion, sizeof(l2Promotion));
    rpc_write(client, &oobFill, sizeof(oobFill));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)tensorMap, 0);
    mem2client((void *)globalAddress, 0);
    mem2client((void *)globalDim, 0);
    mem2client((void *)globalStrides, 0);
    mem2client((void *)pixelBoxLowerCorner, 0);
    mem2client((void *)pixelBoxUpperCorner, 0);
    mem2client((void *)elementStrides, 0);
    return _result;
}

extern "C" CUresult cuTensorMapEncodeIm2colWide(CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim, const cuuint64_t *globalStrides, int pixelBoxLowerCornerWidth, int pixelBoxUpperCornerWidth, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t *elementStrides, CUtensorMapInterleave interleave, CUtensorMapIm2ColWideMode mode, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
#ifdef DEBUG
    std::cout << "Hook: cuTensorMapEncodeIm2colWide called" << std::endl;
#endif
    void *_0tensorMap = mem2server((void *)tensorMap, 0);
    void *_0globalAddress = mem2server((void *)globalAddress, 0);
    void *_0globalDim = mem2server((void *)globalDim, 0);
    void *_0globalStrides = mem2server((void *)globalStrides, 0);
    void *_0elementStrides = mem2server((void *)elementStrides, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTensorMapEncodeIm2colWide);
    rpc_write(client, &_0tensorMap, sizeof(_0tensorMap));
    rpc_write(client, &tensorDataType, sizeof(tensorDataType));
    rpc_write(client, &tensorRank, sizeof(tensorRank));
    rpc_write(client, &_0globalAddress, sizeof(_0globalAddress));
    rpc_write(client, &_0globalDim, sizeof(_0globalDim));
    rpc_write(client, &_0globalStrides, sizeof(_0globalStrides));
    rpc_write(client, &pixelBoxLowerCornerWidth, sizeof(pixelBoxLowerCornerWidth));
    rpc_write(client, &pixelBoxUpperCornerWidth, sizeof(pixelBoxUpperCornerWidth));
    rpc_write(client, &channelsPerPixel, sizeof(channelsPerPixel));
    rpc_write(client, &pixelsPerColumn, sizeof(pixelsPerColumn));
    rpc_write(client, &_0elementStrides, sizeof(_0elementStrides));
    rpc_write(client, &interleave, sizeof(interleave));
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &swizzle, sizeof(swizzle));
    rpc_write(client, &l2Promotion, sizeof(l2Promotion));
    rpc_write(client, &oobFill, sizeof(oobFill));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)tensorMap, 0);
    mem2client((void *)globalAddress, 0);
    mem2client((void *)globalDim, 0);
    mem2client((void *)globalStrides, 0);
    mem2client((void *)elementStrides, 0);
    return _result;
}

extern "C" CUresult cuTensorMapReplaceAddress(CUtensorMap *tensorMap, void *globalAddress) {
#ifdef DEBUG
    std::cout << "Hook: cuTensorMapReplaceAddress called" << std::endl;
#endif
    void *_0tensorMap = mem2server((void *)tensorMap, 0);
    void *_0globalAddress = mem2server((void *)globalAddress, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTensorMapReplaceAddress);
    rpc_write(client, &_0tensorMap, sizeof(_0tensorMap));
    rpc_write(client, &_0globalAddress, sizeof(_0globalAddress));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)tensorMap, 0);
    mem2client((void *)globalAddress, 0);
    return _result;
}

extern "C" CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceCanAccessPeer called" << std::endl;
#endif
    void *_0canAccessPeer = mem2server((void *)canAccessPeer, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceCanAccessPeer);
    rpc_write(client, &_0canAccessPeer, sizeof(_0canAccessPeer));
    rpc_write(client, &dev, sizeof(dev));
    rpc_write(client, &peerDev, sizeof(peerDev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)canAccessPeer, 0);
    return _result;
}

extern "C" CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxEnablePeerAccess called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxEnablePeerAccess);
    rpc_write(client, &peerContext, sizeof(peerContext));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxDisablePeerAccess called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxDisablePeerAccess);
    rpc_write(client, &peerContext, sizeof(peerContext));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetP2PAttribute called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetP2PAttribute);
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_write(client, &attrib, sizeof(attrib));
    rpc_write(client, &srcDevice, sizeof(srcDevice));
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value, 0);
    return _result;
}

extern "C" CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsUnregisterResource called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphicsUnregisterResource);
    rpc_write(client, &resource, sizeof(resource));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsSubResourceGetMappedArray called" << std::endl;
#endif
    void *_0pArray = mem2server((void *)pArray, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphicsSubResourceGetMappedArray);
    rpc_write(client, &_0pArray, sizeof(_0pArray));
    rpc_write(client, &resource, sizeof(resource));
    rpc_write(client, &arrayIndex, sizeof(arrayIndex));
    rpc_write(client, &mipLevel, sizeof(mipLevel));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pArray, 0);
    return _result;
}

extern "C" CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsResourceGetMappedMipmappedArray called" << std::endl;
#endif
    void *_0pMipmappedArray = mem2server((void *)pMipmappedArray, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphicsResourceGetMappedMipmappedArray);
    rpc_write(client, &_0pMipmappedArray, sizeof(_0pMipmappedArray));
    rpc_write(client, &resource, sizeof(resource));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pMipmappedArray, 0);
    return _result;
}

extern "C" CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsResourceSetMapFlags_v2 called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphicsResourceSetMapFlags_v2);
    rpc_write(client, &resource, sizeof(resource));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsMapResources called" << std::endl;
#endif
    void *_0resources = mem2server((void *)resources, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphicsMapResources);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_0resources, sizeof(_0resources));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)resources, 0);
    return _result;
}

extern "C" CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsUnmapResources called" << std::endl;
#endif
    void *_0resources = mem2server((void *)resources, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphicsUnmapResources);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_0resources, sizeof(_0resources));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)resources, 0);
    return _result;
}

extern "C" CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
#ifdef DEBUG
    std::cout << "Hook: cuGetProcAddress_v2 called" << std::endl;
#endif
    // PARAM void **pfn
    void *_0symbolStatus = mem2server((void *)symbolStatus, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGetProcAddress_v2);
    rpc_write(client, symbol, strlen(symbol) + 1, true);
    // PARAM void **pfn
    rpc_write(client, &cudaVersion, sizeof(cudaVersion));
    rpc_write(client, &flags, sizeof(flags));
    rpc_write(client, &_0symbolStatus, sizeof(_0symbolStatus));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **pfn
    rpc_free_client(client);
    // PARAM void **pfn
    mem2client((void *)symbolStatus, 0);
    return _result;
}

extern "C" CUresult cuCoredumpGetAttribute(CUcoredumpSettings attrib, void *value, size_t *size) {
#ifdef DEBUG
    std::cout << "Hook: cuCoredumpGetAttribute called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    void *_0size = mem2server((void *)size, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCoredumpGetAttribute);
    rpc_write(client, &attrib, sizeof(attrib));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_write(client, &_0size, sizeof(_0size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value, 0);
    mem2client((void *)size, 0);
    return _result;
}

extern "C" CUresult cuCoredumpGetAttributeGlobal(CUcoredumpSettings attrib, void *value, size_t *size) {
#ifdef DEBUG
    std::cout << "Hook: cuCoredumpGetAttributeGlobal called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    void *_0size = mem2server((void *)size, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCoredumpGetAttributeGlobal);
    rpc_write(client, &attrib, sizeof(attrib));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_write(client, &_0size, sizeof(_0size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value, 0);
    mem2client((void *)size, 0);
    return _result;
}

extern "C" CUresult cuCoredumpSetAttribute(CUcoredumpSettings attrib, void *value, size_t *size) {
#ifdef DEBUG
    std::cout << "Hook: cuCoredumpSetAttribute called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    void *_0size = mem2server((void *)size, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCoredumpSetAttribute);
    rpc_write(client, &attrib, sizeof(attrib));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_write(client, &_0size, sizeof(_0size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value, 0);
    mem2client((void *)size, 0);
    return _result;
}

extern "C" CUresult cuCoredumpSetAttributeGlobal(CUcoredumpSettings attrib, void *value, size_t *size) {
#ifdef DEBUG
    std::cout << "Hook: cuCoredumpSetAttributeGlobal called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    void *_0size = mem2server((void *)size, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCoredumpSetAttributeGlobal);
    rpc_write(client, &attrib, sizeof(attrib));
    rpc_write(client, &_0value, sizeof(_0value));
    rpc_write(client, &_0size, sizeof(_0size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)value, 0);
    mem2client((void *)size, 0);
    return _result;
}

extern "C" CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
#ifdef DEBUG
    std::cout << "Hook: cuGetExportTable called" << std::endl;
#endif
    // PARAM const void **ppExportTable
    void *_0pExportTableId = mem2server((void *)pExportTableId, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGetExportTable);
    // PARAM const void **ppExportTable
    rpc_read(client, ppExportTable, sizeof(*ppExportTable));
    rpc_write(client, &_0pExportTableId, sizeof(_0pExportTableId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM const void **ppExportTable
    rpc_free_client(client);
    // PARAM const void **ppExportTable
    mem2client((void *)pExportTableId, 0);
    return _result;
}

extern "C" CUresult cuGreenCtxCreate(CUgreenCtx *phCtx, CUdevResourceDesc desc, CUdevice dev, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGreenCtxCreate called" << std::endl;
#endif
    void *_0phCtx = mem2server((void *)phCtx, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGreenCtxCreate);
    rpc_write(client, &_0phCtx, sizeof(_0phCtx));
    rpc_write(client, &desc, sizeof(desc));
    rpc_write(client, &dev, sizeof(dev));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phCtx, 0);
    return _result;
}

extern "C" CUresult cuGreenCtxDestroy(CUgreenCtx hCtx) {
#ifdef DEBUG
    std::cout << "Hook: cuGreenCtxDestroy called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGreenCtxDestroy);
    rpc_write(client, &hCtx, sizeof(hCtx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxFromGreenCtx(CUcontext *pContext, CUgreenCtx hCtx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxFromGreenCtx called" << std::endl;
#endif
    void *_0pContext = mem2server((void *)pContext, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxFromGreenCtx);
    rpc_write(client, &_0pContext, sizeof(_0pContext));
    rpc_write(client, &hCtx, sizeof(hCtx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pContext, 0);
    return _result;
}

extern "C" CUresult cuDeviceGetDevResource(CUdevice device, CUdevResource *resource, CUdevResourceType type) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetDevResource called" << std::endl;
#endif
    void *_0resource = mem2server((void *)resource, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDeviceGetDevResource);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0resource, sizeof(_0resource));
    rpc_write(client, &type, sizeof(type));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)resource, 0);
    return _result;
}

extern "C" CUresult cuCtxGetDevResource(CUcontext hCtx, CUdevResource *resource, CUdevResourceType type) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetDevResource called" << std::endl;
#endif
    void *_0resource = mem2server((void *)resource, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCtxGetDevResource);
    rpc_write(client, &hCtx, sizeof(hCtx));
    rpc_write(client, &_0resource, sizeof(_0resource));
    rpc_write(client, &type, sizeof(type));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)resource, 0);
    return _result;
}

extern "C" CUresult cuGreenCtxGetDevResource(CUgreenCtx hCtx, CUdevResource *resource, CUdevResourceType type) {
#ifdef DEBUG
    std::cout << "Hook: cuGreenCtxGetDevResource called" << std::endl;
#endif
    void *_0resource = mem2server((void *)resource, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGreenCtxGetDevResource);
    rpc_write(client, &hCtx, sizeof(hCtx));
    rpc_write(client, &_0resource, sizeof(_0resource));
    rpc_write(client, &type, sizeof(type));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)resource, 0);
    return _result;
}

extern "C" CUresult cuDevSmResourceSplitByCount(CUdevResource *result, unsigned int *nbGroups, const CUdevResource *input, CUdevResource *remaining, unsigned int useFlags, unsigned int minCount) {
#ifdef DEBUG
    std::cout << "Hook: cuDevSmResourceSplitByCount called" << std::endl;
#endif
    void *_0result = mem2server((void *)result, 0);
    void *_0nbGroups = mem2server((void *)nbGroups, 0);
    void *_0input = mem2server((void *)input, 0);
    void *_0remaining = mem2server((void *)remaining, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDevSmResourceSplitByCount);
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_write(client, &_0nbGroups, sizeof(_0nbGroups));
    rpc_write(client, &_0input, sizeof(_0input));
    rpc_write(client, &_0remaining, sizeof(_0remaining));
    rpc_write(client, &useFlags, sizeof(useFlags));
    rpc_write(client, &minCount, sizeof(minCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)result, 0);
    mem2client((void *)nbGroups, 0);
    mem2client((void *)input, 0);
    mem2client((void *)remaining, 0);
    return _result;
}

extern "C" CUresult cuDevResourceGenerateDesc(CUdevResourceDesc *phDesc, CUdevResource *resources, unsigned int nbResources) {
#ifdef DEBUG
    std::cout << "Hook: cuDevResourceGenerateDesc called" << std::endl;
#endif
    void *_0phDesc = mem2server((void *)phDesc, 0);
    void *_0resources = mem2server((void *)resources, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuDevResourceGenerateDesc);
    rpc_write(client, &_0phDesc, sizeof(_0phDesc));
    rpc_write(client, &_0resources, sizeof(_0resources));
    rpc_write(client, &nbResources, sizeof(nbResources));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phDesc, 0);
    mem2client((void *)resources, 0);
    return _result;
}

extern "C" CUresult cuGreenCtxRecordEvent(CUgreenCtx hCtx, CUevent hEvent) {
#ifdef DEBUG
    std::cout << "Hook: cuGreenCtxRecordEvent called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGreenCtxRecordEvent);
    rpc_write(client, &hCtx, sizeof(hCtx));
    rpc_write(client, &hEvent, sizeof(hEvent));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGreenCtxWaitEvent(CUgreenCtx hCtx, CUevent hEvent) {
#ifdef DEBUG
    std::cout << "Hook: cuGreenCtxWaitEvent called" << std::endl;
#endif
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGreenCtxWaitEvent);
    rpc_write(client, &hCtx, sizeof(hCtx));
    rpc_write(client, &hEvent, sizeof(hEvent));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamGetGreenCtx(CUstream hStream, CUgreenCtx *phCtx) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetGreenCtx called" << std::endl;
#endif
    void *_0phCtx = mem2server((void *)phCtx, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuStreamGetGreenCtx);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0phCtx, sizeof(_0phCtx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phCtx, 0);
    return _result;
}

extern "C" CUresult cuGreenCtxStreamCreate(CUstream *phStream, CUgreenCtx greenCtx, unsigned int flags, int priority) {
#ifdef DEBUG
    std::cout << "Hook: cuGreenCtxStreamCreate called" << std::endl;
#endif
    void *_0phStream = mem2server((void *)phStream, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGreenCtxStreamCreate);
    rpc_write(client, &_0phStream, sizeof(_0phStream));
    rpc_write(client, &greenCtx, sizeof(greenCtx));
    rpc_write(client, &flags, sizeof(flags));
    rpc_write(client, &priority, sizeof(priority));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)phStream, 0);
    return _result;
}

extern "C" CUresult cuCheckpointProcessGetRestoreThreadId(int pid, int *tid) {
#ifdef DEBUG
    std::cout << "Hook: cuCheckpointProcessGetRestoreThreadId called" << std::endl;
#endif
    void *_0tid = mem2server((void *)tid, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCheckpointProcessGetRestoreThreadId);
    rpc_write(client, &pid, sizeof(pid));
    rpc_write(client, &_0tid, sizeof(_0tid));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)tid, 0);
    return _result;
}

extern "C" CUresult cuCheckpointProcessGetState(int pid, CUprocessState *state) {
#ifdef DEBUG
    std::cout << "Hook: cuCheckpointProcessGetState called" << std::endl;
#endif
    void *_0state = mem2server((void *)state, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCheckpointProcessGetState);
    rpc_write(client, &pid, sizeof(pid));
    rpc_write(client, &_0state, sizeof(_0state));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)state, 0);
    return _result;
}

extern "C" CUresult cuCheckpointProcessLock(int pid, CUcheckpointLockArgs *args) {
#ifdef DEBUG
    std::cout << "Hook: cuCheckpointProcessLock called" << std::endl;
#endif
    void *_0args = mem2server((void *)args, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCheckpointProcessLock);
    rpc_write(client, &pid, sizeof(pid));
    rpc_write(client, &_0args, sizeof(_0args));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)args, 0);
    return _result;
}

extern "C" CUresult cuCheckpointProcessCheckpoint(int pid, CUcheckpointCheckpointArgs *args) {
#ifdef DEBUG
    std::cout << "Hook: cuCheckpointProcessCheckpoint called" << std::endl;
#endif
    void *_0args = mem2server((void *)args, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCheckpointProcessCheckpoint);
    rpc_write(client, &pid, sizeof(pid));
    rpc_write(client, &_0args, sizeof(_0args));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)args, 0);
    return _result;
}

extern "C" CUresult cuCheckpointProcessRestore(int pid, CUcheckpointRestoreArgs *args) {
#ifdef DEBUG
    std::cout << "Hook: cuCheckpointProcessRestore called" << std::endl;
#endif
    void *_0args = mem2server((void *)args, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCheckpointProcessRestore);
    rpc_write(client, &pid, sizeof(pid));
    rpc_write(client, &_0args, sizeof(_0args));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)args, 0);
    return _result;
}

extern "C" CUresult cuCheckpointProcessUnlock(int pid, CUcheckpointUnlockArgs *args) {
#ifdef DEBUG
    std::cout << "Hook: cuCheckpointProcessUnlock called" << std::endl;
#endif
    void *_0args = mem2server((void *)args, 0);
    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuCheckpointProcessUnlock);
    rpc_write(client, &pid, sizeof(pid));
    rpc_write(client, &_0args, sizeof(_0args));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)args, 0);
    return _result;
}

