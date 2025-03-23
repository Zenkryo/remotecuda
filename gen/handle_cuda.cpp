#include <iostream>
#include <unordered_map>
#include "hook_api.h"
#include "handle_server.h"
#include "../rpc.h"
#include "cuda.h"

int handle_cuInit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuInit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int Flags;
    rpc_read(client, &Flags, sizeof(Flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuInit(Flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDriverGetVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDriverGetVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*driverVersion;
    rpc_read(client, &driverVersion, sizeof(driverVersion));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDriverGetVersion(driverVersion);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGet(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGet called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice*device;
    rpc_read(client, &device, sizeof(device));
    int ordinal;
    rpc_read(client, &ordinal, sizeof(ordinal));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGet(device, ordinal);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetCount(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetCount called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*count;
    rpc_read(client, &count, sizeof(count));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetCount(count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetName(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetName called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    char name[1024];
    int len;
    rpc_read(client, &len, sizeof(len));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetName(name, len, dev);
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetUuid(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetUuid called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUuuid*uuid;
    rpc_read(client, &uuid, sizeof(uuid));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetUuid(uuid, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetUuid_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetUuid_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUuuid*uuid;
    rpc_read(client, &uuid, sizeof(uuid));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetUuid_v2(uuid, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetLuid(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetLuid called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    char luid[1024];
    unsigned int*deviceNodeMask;
    rpc_read(client, &deviceNodeMask, sizeof(deviceNodeMask));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetLuid(luid, deviceNodeMask, dev);
    rpc_write(client, luid, strlen(luid) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceTotalMem_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceTotalMem_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t*bytes;
    rpc_read(client, &bytes, sizeof(bytes));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceTotalMem_v2(bytes, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetTexture1DLinearMaxWidth(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetTexture1DLinearMaxWidth called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t*maxWidthInElements;
    rpc_read(client, &maxWidthInElements, sizeof(maxWidthInElements));
    CUarray_format format;
    rpc_read(client, &format, sizeof(format));
    unsigned numChannels;
    rpc_read(client, &numChannels, sizeof(numChannels));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, format, numChannels, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*pi;
    rpc_read(client, &pi, sizeof(pi));
    CUdevice_attribute attrib;
    rpc_read(client, &attrib, sizeof(attrib));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetAttribute(pi, attrib, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetNvSciSyncAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetNvSciSyncAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *nvSciSyncAttrList;
    rpc_read(client, &nvSciSyncAttrList, sizeof(nvSciSyncAttrList));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, dev, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceSetMemPool(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceSetMemPool called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUmemoryPool pool;
    rpc_read(client, &pool, sizeof(pool));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceSetMemPool(dev, pool);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetMemPool(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetMemPool called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemoryPool*pool;
    rpc_read(client, &pool, sizeof(pool));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetMemPool(pool, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetDefaultMemPool(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetDefaultMemPool called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemoryPool*pool_out;
    rpc_read(client, &pool_out, sizeof(pool_out));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetDefaultMemPool(pool_out, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetExecAffinitySupport(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetExecAffinitySupport called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*pi;
    rpc_read(client, &pi, sizeof(pi));
    CUexecAffinityType type;
    rpc_read(client, &type, sizeof(type));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetExecAffinitySupport(pi, type, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuFlushGPUDirectRDMAWrites(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuFlushGPUDirectRDMAWrites called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUflushGPUDirectRDMAWritesTarget target;
    rpc_read(client, &target, sizeof(target));
    CUflushGPUDirectRDMAWritesScope scope;
    rpc_read(client, &scope, sizeof(scope));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFlushGPUDirectRDMAWrites(target, scope);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetProperties(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetProperties called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevprop*prop;
    rpc_read(client, &prop, sizeof(prop));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetProperties(prop, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceComputeCapability(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceComputeCapability called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*major;
    rpc_read(client, &major, sizeof(major));
    int*minor;
    rpc_read(client, &minor, sizeof(minor));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceComputeCapability(major, minor, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDevicePrimaryCtxRetain(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDevicePrimaryCtxRetain called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext*pctx;
    rpc_read(client, &pctx, sizeof(pctx));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDevicePrimaryCtxRetain(pctx, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDevicePrimaryCtxRelease_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDevicePrimaryCtxRelease_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDevicePrimaryCtxRelease_v2(dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDevicePrimaryCtxSetFlags_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDevicePrimaryCtxSetFlags_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDevicePrimaryCtxSetFlags_v2(dev, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDevicePrimaryCtxGetState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDevicePrimaryCtxGetState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    unsigned int*flags;
    rpc_read(client, &flags, sizeof(flags));
    int*active;
    rpc_read(client, &active, sizeof(active));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDevicePrimaryCtxGetState(dev, flags, active);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDevicePrimaryCtxReset_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDevicePrimaryCtxReset_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDevicePrimaryCtxReset_v2(dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxCreate_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxCreate_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext*pctx;
    rpc_read(client, &pctx, sizeof(pctx));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxCreate_v2(pctx, flags, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxCreate_v3(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxCreate_v3 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext*pctx;
    rpc_read(client, &pctx, sizeof(pctx));
    CUexecAffinityParam*paramsArray;
    rpc_read(client, &paramsArray, sizeof(paramsArray));
    int numParams;
    rpc_read(client, &numParams, sizeof(numParams));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxCreate_v3(pctx, paramsArray, numParams, flags, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxCreate_v4(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxCreate_v4 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext*pctx;
    rpc_read(client, &pctx, sizeof(pctx));
    CUctxCreateParams*ctxCreateParams;
    rpc_read(client, &ctxCreateParams, sizeof(ctxCreateParams));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxCreate_v4(pctx, ctxCreateParams, flags, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxDestroy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxDestroy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext ctx;
    rpc_read(client, &ctx, sizeof(ctx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxDestroy_v2(ctx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxPushCurrent_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxPushCurrent_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext ctx;
    rpc_read(client, &ctx, sizeof(ctx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxPushCurrent_v2(ctx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxPopCurrent_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxPopCurrent_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext*pctx;
    rpc_read(client, &pctx, sizeof(pctx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxPopCurrent_v2(pctx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxSetCurrent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxSetCurrent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext ctx;
    rpc_read(client, &ctx, sizeof(ctx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxSetCurrent(ctx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxGetCurrent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxGetCurrent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext*pctx;
    rpc_read(client, &pctx, sizeof(pctx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetCurrent(pctx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxGetDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxGetDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice*device;
    rpc_read(client, &device, sizeof(device));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetDevice(device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxGetFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxGetFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int*flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetFlags(flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxSetFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxSetFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxSetFlags(flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxGetId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxGetId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext ctx;
    rpc_read(client, &ctx, sizeof(ctx));
    unsigned long long*ctxId;
    rpc_read(client, &ctxId, sizeof(ctxId));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetId(ctx, ctxId);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxSynchronize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxSynchronize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxSynchronize();
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxSetLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxSetLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUlimit limit;
    rpc_read(client, &limit, sizeof(limit));
    size_t value;
    rpc_read(client, &value, sizeof(value));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxSetLimit(limit, value);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxGetLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxGetLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t*pvalue;
    rpc_read(client, &pvalue, sizeof(pvalue));
    CUlimit limit;
    rpc_read(client, &limit, sizeof(limit));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetLimit(pvalue, limit);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxGetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxGetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunc_cache*pconfig;
    rpc_read(client, &pconfig, sizeof(pconfig));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetCacheConfig(pconfig);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxSetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxSetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunc_cache config;
    rpc_read(client, &config, sizeof(config));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxSetCacheConfig(config);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxGetApiVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxGetApiVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext ctx;
    rpc_read(client, &ctx, sizeof(ctx));
    unsigned int*version;
    rpc_read(client, &version, sizeof(version));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetApiVersion(ctx, version);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxGetStreamPriorityRange(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxGetStreamPriorityRange called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*leastPriority;
    rpc_read(client, &leastPriority, sizeof(leastPriority));
    int*greatestPriority;
    rpc_read(client, &greatestPriority, sizeof(greatestPriority));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetStreamPriorityRange(leastPriority, greatestPriority);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxResetPersistingL2Cache(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxResetPersistingL2Cache called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxResetPersistingL2Cache();
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxGetExecAffinity(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxGetExecAffinity called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUexecAffinityParam*pExecAffinity;
    rpc_read(client, &pExecAffinity, sizeof(pExecAffinity));
    CUexecAffinityType type;
    rpc_read(client, &type, sizeof(type));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetExecAffinity(pExecAffinity, type);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxRecordEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxRecordEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext hCtx;
    rpc_read(client, &hCtx, sizeof(hCtx));
    CUevent hEvent;
    rpc_read(client, &hEvent, sizeof(hEvent));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxRecordEvent(hCtx, hEvent);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxWaitEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxWaitEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext hCtx;
    rpc_read(client, &hCtx, sizeof(hCtx));
    CUevent hEvent;
    rpc_read(client, &hEvent, sizeof(hEvent));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxWaitEvent(hCtx, hEvent);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxAttach(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxAttach called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext*pctx;
    rpc_read(client, &pctx, sizeof(pctx));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxAttach(pctx, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxDetach(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxDetach called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext ctx;
    rpc_read(client, &ctx, sizeof(ctx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxDetach(ctx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxGetSharedMemConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxGetSharedMemConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUsharedconfig*pConfig;
    rpc_read(client, &pConfig, sizeof(pConfig));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetSharedMemConfig(pConfig);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxSetSharedMemConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxSetSharedMemConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUsharedconfig config;
    rpc_read(client, &config, sizeof(config));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxSetSharedMemConfig(config);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuModuleLoad(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuModuleLoad called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmodule*module;
    rpc_read(client, &module, sizeof(module));
    char *fname = nullptr;
    rpc_read(client, &fname, 0, true);
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(fname);
    _result = cuModuleLoad(module, fname);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuModuleLoadData(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuModuleLoadData called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmodule*module;
    rpc_read(client, &module, sizeof(module));
    void *image;
    rpc_read(client, &image, sizeof(image));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuModuleLoadData(module, image);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuModuleLoadDataEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuModuleLoadDataEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmodule*module;
    rpc_read(client, &module, sizeof(module));
    void *image;
    rpc_read(client, &image, sizeof(image));
    unsigned int numOptions;
    rpc_read(client, &numOptions, sizeof(numOptions));
    CUjit_option*options;
    rpc_read(client, &options, sizeof(options));
    // PARAM void **optionValues
    void *optionValues;
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **optionValues
    _result = cuModuleLoadDataEx(module, image, numOptions, options, &optionValues);
    // PARAM void **optionValues
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **optionValues
    return rtn;
}

int handle_cuModuleLoadFatBinary(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuModuleLoadFatBinary called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmodule*module;
    rpc_read(client, &module, sizeof(module));
    void *fatCubin;
    rpc_read(client, &fatCubin, sizeof(fatCubin));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuModuleLoadFatBinary(module, fatCubin);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuModuleUnload(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuModuleUnload called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmodule hmod;
    rpc_read(client, &hmod, sizeof(hmod));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuModuleUnload(hmod);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuModuleGetLoadingMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuModuleGetLoadingMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmoduleLoadingMode*mode;
    rpc_read(client, &mode, sizeof(mode));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuModuleGetLoadingMode(mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuModuleGetFunction(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuModuleGetFunction called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction*hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    CUmodule hmod;
    rpc_read(client, &hmod, sizeof(hmod));
    char *name = nullptr;
    rpc_read(client, &name, 0, true);
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(name);
    _result = cuModuleGetFunction(hfunc, hmod, name);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuModuleGetFunctionCount(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuModuleGetFunctionCount called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    CUmodule mod;
    rpc_read(client, &mod, sizeof(mod));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuModuleGetFunctionCount(count, mod);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuModuleEnumerateFunctions(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuModuleEnumerateFunctions called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction*functions;
    rpc_read(client, &functions, sizeof(functions));
    unsigned int numFunctions;
    rpc_read(client, &numFunctions, sizeof(numFunctions));
    CUmodule mod;
    rpc_read(client, &mod, sizeof(mod));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuModuleEnumerateFunctions(functions, numFunctions, mod);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuLinkCreate_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLinkCreate_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int numOptions;
    rpc_read(client, &numOptions, sizeof(numOptions));
    CUjit_option*options;
    rpc_read(client, &options, sizeof(options));
    // PARAM void **optionValues
    void *optionValues;
    CUlinkState*stateOut;
    rpc_read(client, &stateOut, sizeof(stateOut));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **optionValues
    _result = cuLinkCreate_v2(numOptions, options, &optionValues, stateOut);
    // PARAM void **optionValues
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **optionValues
    return rtn;
}

int handle_cuLinkAddData_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLinkAddData_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUlinkState state;
    rpc_read(client, &state, sizeof(state));
    CUjitInputType type;
    rpc_read(client, &type, sizeof(type));
    void *data;
    rpc_read(client, &data, sizeof(data));
    size_t size;
    rpc_read(client, &size, sizeof(size));
    char *name = nullptr;
    rpc_read(client, &name, 0, true);
    unsigned int numOptions;
    rpc_read(client, &numOptions, sizeof(numOptions));
    CUjit_option*options;
    rpc_read(client, &options, sizeof(options));
    // PARAM void **optionValues
    void *optionValues;
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(name);
    // PARAM void **optionValues
    _result = cuLinkAddData_v2(state, type, data, size, name, numOptions, options, &optionValues);
    // PARAM void **optionValues
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **optionValues
    return rtn;
}

int handle_cuLinkAddFile_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLinkAddFile_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUlinkState state;
    rpc_read(client, &state, sizeof(state));
    CUjitInputType type;
    rpc_read(client, &type, sizeof(type));
    char *path = nullptr;
    rpc_read(client, &path, 0, true);
    unsigned int numOptions;
    rpc_read(client, &numOptions, sizeof(numOptions));
    CUjit_option*options;
    rpc_read(client, &options, sizeof(options));
    // PARAM void **optionValues
    void *optionValues;
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(path);
    // PARAM void **optionValues
    _result = cuLinkAddFile_v2(state, type, path, numOptions, options, &optionValues);
    // PARAM void **optionValues
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **optionValues
    return rtn;
}

int handle_cuLinkComplete(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLinkComplete called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUlinkState state;
    rpc_read(client, &state, sizeof(state));
    // PARAM void **cubinOut
    void *cubinOut;
    size_t*sizeOut;
    rpc_read(client, &sizeOut, sizeof(sizeOut));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **cubinOut
    _result = cuLinkComplete(state, &cubinOut, sizeOut);
    // PARAM void **cubinOut
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **cubinOut
    return rtn;
}

int handle_cuLinkDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLinkDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUlinkState state;
    rpc_read(client, &state, sizeof(state));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLinkDestroy(state);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuModuleGetTexRef(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuModuleGetTexRef called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref*pTexRef;
    rpc_read(client, &pTexRef, sizeof(pTexRef));
    CUmodule hmod;
    rpc_read(client, &hmod, sizeof(hmod));
    char *name = nullptr;
    rpc_read(client, &name, 0, true);
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(name);
    _result = cuModuleGetTexRef(pTexRef, hmod, name);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuModuleGetSurfRef(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuModuleGetSurfRef called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUsurfref*pSurfRef;
    rpc_read(client, &pSurfRef, sizeof(pSurfRef));
    CUmodule hmod;
    rpc_read(client, &hmod, sizeof(hmod));
    char *name = nullptr;
    rpc_read(client, &name, 0, true);
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(name);
    _result = cuModuleGetSurfRef(pSurfRef, hmod, name);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuLibraryLoadData(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLibraryLoadData called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUlibrary*library;
    rpc_read(client, &library, sizeof(library));
    void *code;
    rpc_read(client, &code, sizeof(code));
    CUjit_option*jitOptions;
    rpc_read(client, &jitOptions, sizeof(jitOptions));
    // PARAM void **jitOptionsValues
    void *jitOptionsValues;
    unsigned int numJitOptions;
    rpc_read(client, &numJitOptions, sizeof(numJitOptions));
    CUlibraryOption*libraryOptions;
    rpc_read(client, &libraryOptions, sizeof(libraryOptions));
    // PARAM void **libraryOptionValues
    void *libraryOptionValues;
    unsigned int numLibraryOptions;
    rpc_read(client, &numLibraryOptions, sizeof(numLibraryOptions));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **jitOptionsValues
    // PARAM void **libraryOptionValues
    _result = cuLibraryLoadData(library, code, jitOptions, &jitOptionsValues, numJitOptions, libraryOptions, &libraryOptionValues, numLibraryOptions);
    // PARAM void **jitOptionsValues
    // PARAM void **libraryOptionValues
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **jitOptionsValues
    // PARAM void **libraryOptionValues
    return rtn;
}

int handle_cuLibraryLoadFromFile(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLibraryLoadFromFile called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUlibrary*library;
    rpc_read(client, &library, sizeof(library));
    char *fileName = nullptr;
    rpc_read(client, &fileName, 0, true);
    CUjit_option*jitOptions;
    rpc_read(client, &jitOptions, sizeof(jitOptions));
    // PARAM void **jitOptionsValues
    void *jitOptionsValues;
    unsigned int numJitOptions;
    rpc_read(client, &numJitOptions, sizeof(numJitOptions));
    CUlibraryOption*libraryOptions;
    rpc_read(client, &libraryOptions, sizeof(libraryOptions));
    // PARAM void **libraryOptionValues
    void *libraryOptionValues;
    unsigned int numLibraryOptions;
    rpc_read(client, &numLibraryOptions, sizeof(numLibraryOptions));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(fileName);
    // PARAM void **jitOptionsValues
    // PARAM void **libraryOptionValues
    _result = cuLibraryLoadFromFile(library, fileName, jitOptions, &jitOptionsValues, numJitOptions, libraryOptions, &libraryOptionValues, numLibraryOptions);
    // PARAM void **jitOptionsValues
    // PARAM void **libraryOptionValues
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **jitOptionsValues
    // PARAM void **libraryOptionValues
    return rtn;
}

int handle_cuLibraryUnload(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLibraryUnload called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUlibrary library;
    rpc_read(client, &library, sizeof(library));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLibraryUnload(library);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuLibraryGetKernel(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLibraryGetKernel called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUkernel*pKernel;
    rpc_read(client, &pKernel, sizeof(pKernel));
    CUlibrary library;
    rpc_read(client, &library, sizeof(library));
    char *name = nullptr;
    rpc_read(client, &name, 0, true);
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(name);
    _result = cuLibraryGetKernel(pKernel, library, name);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuLibraryGetKernelCount(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLibraryGetKernelCount called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    CUlibrary lib;
    rpc_read(client, &lib, sizeof(lib));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLibraryGetKernelCount(count, lib);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuLibraryEnumerateKernels(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLibraryEnumerateKernels called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUkernel*kernels;
    rpc_read(client, &kernels, sizeof(kernels));
    unsigned int numKernels;
    rpc_read(client, &numKernels, sizeof(numKernels));
    CUlibrary lib;
    rpc_read(client, &lib, sizeof(lib));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLibraryEnumerateKernels(kernels, numKernels, lib);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuLibraryGetModule(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLibraryGetModule called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmodule*pMod;
    rpc_read(client, &pMod, sizeof(pMod));
    CUlibrary library;
    rpc_read(client, &library, sizeof(library));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLibraryGetModule(pMod, library);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuKernelGetFunction(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuKernelGetFunction called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction*pFunc;
    rpc_read(client, &pFunc, sizeof(pFunc));
    CUkernel kernel;
    rpc_read(client, &kernel, sizeof(kernel));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuKernelGetFunction(pFunc, kernel);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuKernelGetLibrary(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuKernelGetLibrary called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUlibrary*pLib;
    rpc_read(client, &pLib, sizeof(pLib));
    CUkernel kernel;
    rpc_read(client, &kernel, sizeof(kernel));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuKernelGetLibrary(pLib, kernel);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuLibraryGetUnifiedFunction(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLibraryGetUnifiedFunction called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    // PARAM void **fptr
    void *fptr;
    CUlibrary library;
    rpc_read(client, &library, sizeof(library));
    char *symbol = nullptr;
    rpc_read(client, &symbol, 0, true);
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **fptr
    buffers.insert(symbol);
    _result = cuLibraryGetUnifiedFunction(&fptr, library, symbol);
    // PARAM void **fptr
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **fptr
    return rtn;
}

int handle_cuKernelGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuKernelGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*pi;
    rpc_read(client, &pi, sizeof(pi));
    CUfunction_attribute attrib;
    rpc_read(client, &attrib, sizeof(attrib));
    CUkernel kernel;
    rpc_read(client, &kernel, sizeof(kernel));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuKernelGetAttribute(pi, attrib, kernel, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuKernelSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuKernelSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction_attribute attrib;
    rpc_read(client, &attrib, sizeof(attrib));
    int val;
    rpc_read(client, &val, sizeof(val));
    CUkernel kernel;
    rpc_read(client, &kernel, sizeof(kernel));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuKernelSetAttribute(attrib, val, kernel, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuKernelSetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuKernelSetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUkernel kernel;
    rpc_read(client, &kernel, sizeof(kernel));
    CUfunc_cache config;
    rpc_read(client, &config, sizeof(config));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuKernelSetCacheConfig(kernel, config, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuKernelGetName(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuKernelGetName called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    // PARAM const char **name
    const char *name;
    CUkernel hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM const char **name
    _result = cuKernelGetName(&name, hfunc);
    // PARAM const char **name
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM const char **name
    return rtn;
}

int handle_cuKernelGetParamInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuKernelGetParamInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUkernel kernel;
    rpc_read(client, &kernel, sizeof(kernel));
    size_t paramIndex;
    rpc_read(client, &paramIndex, sizeof(paramIndex));
    size_t*paramOffset;
    rpc_read(client, &paramOffset, sizeof(paramOffset));
    size_t*paramSize;
    rpc_read(client, &paramSize, sizeof(paramSize));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuKernelGetParamInfo(kernel, paramIndex, paramOffset, paramSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemGetInfo_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemGetInfo_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t*free;
    rpc_read(client, &free, sizeof(free));
    size_t*total;
    rpc_read(client, &total, sizeof(total));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemGetInfo_v2(free, total);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemFree_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemFree_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dptr;
    rpc_read(client, &dptr, sizeof(dptr));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemFree_v2(dptr);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemHostGetFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemHostGetFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int*pFlags;
    rpc_read(client, &pFlags, sizeof(pFlags));
    void *p;
    rpc_read(client, &p, sizeof(p));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemHostGetFlags(pFlags, p);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceRegisterAsyncNotification(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceRegisterAsyncNotification called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice device;
    rpc_read(client, &device, sizeof(device));
    CUasyncCallback callbackFunc;
    rpc_read(client, &callbackFunc, sizeof(callbackFunc));
    void *userData;
    rpc_read(client, &userData, sizeof(userData));
    CUasyncCallbackHandle*callback;
    rpc_read(client, &callback, sizeof(callback));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceRegisterAsyncNotification(device, callbackFunc, userData, callback);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceUnregisterAsyncNotification(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceUnregisterAsyncNotification called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice device;
    rpc_read(client, &device, sizeof(device));
    CUasyncCallbackHandle callback;
    rpc_read(client, &callback, sizeof(callback));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceUnregisterAsyncNotification(device, callback);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetByPCIBusId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetByPCIBusId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice*dev;
    rpc_read(client, &dev, sizeof(dev));
    char *pciBusId = nullptr;
    rpc_read(client, &pciBusId, 0, true);
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(pciBusId);
    _result = cuDeviceGetByPCIBusId(dev, pciBusId);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetPCIBusId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetPCIBusId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    char pciBusId[1024];
    int len;
    rpc_read(client, &len, sizeof(len));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetPCIBusId(pciBusId, len, dev);
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

int handle_cuIpcGetEventHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuIpcGetEventHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUipcEventHandle*pHandle;
    rpc_read(client, &pHandle, sizeof(pHandle));
    CUevent event;
    rpc_read(client, &event, sizeof(event));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuIpcGetEventHandle(pHandle, event);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuIpcOpenEventHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuIpcOpenEventHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUevent*phEvent;
    rpc_read(client, &phEvent, sizeof(phEvent));
    CUipcEventHandle handle;
    rpc_read(client, &handle, sizeof(handle));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuIpcOpenEventHandle(phEvent, handle);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuIpcGetMemHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuIpcGetMemHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUipcMemHandle*pHandle;
    rpc_read(client, &pHandle, sizeof(pHandle));
    CUdeviceptr dptr;
    rpc_read(client, &dptr, sizeof(dptr));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuIpcGetMemHandle(pHandle, dptr);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuIpcCloseMemHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuIpcCloseMemHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dptr;
    rpc_read(client, &dptr, sizeof(dptr));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuIpcCloseMemHandle(dptr);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemHostRegister_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemHostRegister_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *p;
    rpc_read(client, &p, sizeof(p));
    size_t bytesize;
    rpc_read(client, &bytesize, sizeof(bytesize));
    unsigned int Flags;
    rpc_read(client, &Flags, sizeof(Flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemHostRegister_v2(p, bytesize, Flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemHostUnregister(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemHostUnregister called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *p;
    rpc_read(client, &p, sizeof(p));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemHostUnregister(p);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dst;
    rpc_read(client, &dst, sizeof(dst));
    CUdeviceptr src;
    rpc_read(client, &src, sizeof(src));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy(dst, src, ByteCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyPeer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyPeer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    CUcontext dstContext;
    rpc_read(client, &dstContext, sizeof(dstContext));
    CUdeviceptr srcDevice;
    rpc_read(client, &srcDevice, sizeof(srcDevice));
    CUcontext srcContext;
    rpc_read(client, &srcContext, sizeof(srcContext));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyHtoD_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyHtoD_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    void *srcHost;
    rpc_read(client, &srcHost, sizeof(srcHost));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyDtoH_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyDtoH_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dstHost;
    rpc_read(client, &dstHost, sizeof(dstHost));
    CUdeviceptr srcDevice;
    rpc_read(client, &srcDevice, sizeof(srcDevice));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyDtoD_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyDtoD_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    CUdeviceptr srcDevice;
    rpc_read(client, &srcDevice, sizeof(srcDevice));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyDtoA_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyDtoA_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarray dstArray;
    rpc_read(client, &dstArray, sizeof(dstArray));
    size_t dstOffset;
    rpc_read(client, &dstOffset, sizeof(dstOffset));
    CUdeviceptr srcDevice;
    rpc_read(client, &srcDevice, sizeof(srcDevice));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyAtoD_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyAtoD_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    CUarray srcArray;
    rpc_read(client, &srcArray, sizeof(srcArray));
    size_t srcOffset;
    rpc_read(client, &srcOffset, sizeof(srcOffset));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyHtoA_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyHtoA_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarray dstArray;
    rpc_read(client, &dstArray, sizeof(dstArray));
    size_t dstOffset;
    rpc_read(client, &dstOffset, sizeof(dstOffset));
    void *srcHost;
    rpc_read(client, &srcHost, sizeof(srcHost));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyAtoH_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyAtoH_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dstHost;
    rpc_read(client, &dstHost, sizeof(dstHost));
    CUarray srcArray;
    rpc_read(client, &srcArray, sizeof(srcArray));
    size_t srcOffset;
    rpc_read(client, &srcOffset, sizeof(srcOffset));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyAtoA_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyAtoA_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarray dstArray;
    rpc_read(client, &dstArray, sizeof(dstArray));
    size_t dstOffset;
    rpc_read(client, &dstOffset, sizeof(dstOffset));
    CUarray srcArray;
    rpc_read(client, &srcArray, sizeof(srcArray));
    size_t srcOffset;
    rpc_read(client, &srcOffset, sizeof(srcOffset));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpy2D_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpy2D_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_MEMCPY2D *pCopy;
    rpc_read(client, &pCopy, sizeof(pCopy));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy2D_v2(pCopy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpy2DUnaligned_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpy2DUnaligned_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_MEMCPY2D *pCopy;
    rpc_read(client, &pCopy, sizeof(pCopy));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy2DUnaligned_v2(pCopy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpy3D_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpy3D_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_MEMCPY3D *pCopy;
    rpc_read(client, &pCopy, sizeof(pCopy));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy3D_v2(pCopy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpy3DPeer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpy3DPeer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_MEMCPY3D_PEER *pCopy;
    rpc_read(client, &pCopy, sizeof(pCopy));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy3DPeer(pCopy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dst;
    rpc_read(client, &dst, sizeof(dst));
    CUdeviceptr src;
    rpc_read(client, &src, sizeof(src));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyAsync(dst, src, ByteCount, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyPeerAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyPeerAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    CUcontext dstContext;
    rpc_read(client, &dstContext, sizeof(dstContext));
    CUdeviceptr srcDevice;
    rpc_read(client, &srcDevice, sizeof(srcDevice));
    CUcontext srcContext;
    rpc_read(client, &srcContext, sizeof(srcContext));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyHtoDAsync_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyHtoDAsync_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    void *srcHost;
    rpc_read(client, &srcHost, sizeof(srcHost));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyDtoHAsync_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyDtoHAsync_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dstHost;
    rpc_read(client, &dstHost, sizeof(dstHost));
    CUdeviceptr srcDevice;
    rpc_read(client, &srcDevice, sizeof(srcDevice));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyDtoDAsync_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyDtoDAsync_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    CUdeviceptr srcDevice;
    rpc_read(client, &srcDevice, sizeof(srcDevice));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyHtoAAsync_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyHtoAAsync_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarray dstArray;
    rpc_read(client, &dstArray, sizeof(dstArray));
    size_t dstOffset;
    rpc_read(client, &dstOffset, sizeof(dstOffset));
    void *srcHost;
    rpc_read(client, &srcHost, sizeof(srcHost));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpyAtoHAsync_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpyAtoHAsync_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *dstHost;
    rpc_read(client, &dstHost, sizeof(dstHost));
    CUarray srcArray;
    rpc_read(client, &srcArray, sizeof(srcArray));
    size_t srcOffset;
    rpc_read(client, &srcOffset, sizeof(srcOffset));
    size_t ByteCount;
    rpc_read(client, &ByteCount, sizeof(ByteCount));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpy2DAsync_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpy2DAsync_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_MEMCPY2D *pCopy;
    rpc_read(client, &pCopy, sizeof(pCopy));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy2DAsync_v2(pCopy, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpy3DAsync_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpy3DAsync_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_MEMCPY3D *pCopy;
    rpc_read(client, &pCopy, sizeof(pCopy));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy3DAsync_v2(pCopy, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpy3DPeerAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpy3DPeerAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_MEMCPY3D_PEER *pCopy;
    rpc_read(client, &pCopy, sizeof(pCopy));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy3DPeerAsync(pCopy, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemcpy3DBatchAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemcpy3DBatchAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t numOps;
    rpc_read(client, &numOps, sizeof(numOps));
    CUDA_MEMCPY3D_BATCH_OP*opList;
    rpc_read(client, &opList, sizeof(opList));
    size_t*failIdx;
    rpc_read(client, &failIdx, sizeof(failIdx));
    unsigned long long flags;
    rpc_read(client, &flags, sizeof(flags));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy3DBatchAsync(numOps, opList, failIdx, flags, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemsetD8_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemsetD8_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    unsigned char uc;
    rpc_read(client, &uc, sizeof(uc));
    size_t N;
    rpc_read(client, &N, sizeof(N));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD8_v2(dstDevice, uc, N);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemsetD16_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemsetD16_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    unsigned short us;
    rpc_read(client, &us, sizeof(us));
    size_t N;
    rpc_read(client, &N, sizeof(N));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD16_v2(dstDevice, us, N);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemsetD32_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemsetD32_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    unsigned int ui;
    rpc_read(client, &ui, sizeof(ui));
    size_t N;
    rpc_read(client, &N, sizeof(N));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD32_v2(dstDevice, ui, N);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemsetD2D8_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemsetD2D8_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    size_t dstPitch;
    rpc_read(client, &dstPitch, sizeof(dstPitch));
    unsigned char uc;
    rpc_read(client, &uc, sizeof(uc));
    size_t Width;
    rpc_read(client, &Width, sizeof(Width));
    size_t Height;
    rpc_read(client, &Height, sizeof(Height));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemsetD2D16_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemsetD2D16_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    size_t dstPitch;
    rpc_read(client, &dstPitch, sizeof(dstPitch));
    unsigned short us;
    rpc_read(client, &us, sizeof(us));
    size_t Width;
    rpc_read(client, &Width, sizeof(Width));
    size_t Height;
    rpc_read(client, &Height, sizeof(Height));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemsetD2D32_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemsetD2D32_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    size_t dstPitch;
    rpc_read(client, &dstPitch, sizeof(dstPitch));
    unsigned int ui;
    rpc_read(client, &ui, sizeof(ui));
    size_t Width;
    rpc_read(client, &Width, sizeof(Width));
    size_t Height;
    rpc_read(client, &Height, sizeof(Height));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemsetD8Async(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemsetD8Async called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    unsigned char uc;
    rpc_read(client, &uc, sizeof(uc));
    size_t N;
    rpc_read(client, &N, sizeof(N));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD8Async(dstDevice, uc, N, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemsetD16Async(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemsetD16Async called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    unsigned short us;
    rpc_read(client, &us, sizeof(us));
    size_t N;
    rpc_read(client, &N, sizeof(N));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD16Async(dstDevice, us, N, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemsetD32Async(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemsetD32Async called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    unsigned int ui;
    rpc_read(client, &ui, sizeof(ui));
    size_t N;
    rpc_read(client, &N, sizeof(N));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD32Async(dstDevice, ui, N, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemsetD2D8Async(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemsetD2D8Async called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    size_t dstPitch;
    rpc_read(client, &dstPitch, sizeof(dstPitch));
    unsigned char uc;
    rpc_read(client, &uc, sizeof(uc));
    size_t Width;
    rpc_read(client, &Width, sizeof(Width));
    size_t Height;
    rpc_read(client, &Height, sizeof(Height));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemsetD2D16Async(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemsetD2D16Async called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    size_t dstPitch;
    rpc_read(client, &dstPitch, sizeof(dstPitch));
    unsigned short us;
    rpc_read(client, &us, sizeof(us));
    size_t Width;
    rpc_read(client, &Width, sizeof(Width));
    size_t Height;
    rpc_read(client, &Height, sizeof(Height));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemsetD2D32Async(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemsetD2D32Async called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    size_t dstPitch;
    rpc_read(client, &dstPitch, sizeof(dstPitch));
    unsigned int ui;
    rpc_read(client, &ui, sizeof(ui));
    size_t Width;
    rpc_read(client, &Width, sizeof(Width));
    size_t Height;
    rpc_read(client, &Height, sizeof(Height));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuArrayCreate_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuArrayCreate_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarray*pHandle;
    rpc_read(client, &pHandle, sizeof(pHandle));
    CUDA_ARRAY_DESCRIPTOR *pAllocateArray;
    rpc_read(client, &pAllocateArray, sizeof(pAllocateArray));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArrayCreate_v2(pHandle, pAllocateArray);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuArrayGetDescriptor_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuArrayGetDescriptor_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_ARRAY_DESCRIPTOR*pArrayDescriptor;
    rpc_read(client, &pArrayDescriptor, sizeof(pArrayDescriptor));
    CUarray hArray;
    rpc_read(client, &hArray, sizeof(hArray));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArrayGetDescriptor_v2(pArrayDescriptor, hArray);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuArrayGetSparseProperties(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuArrayGetSparseProperties called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_ARRAY_SPARSE_PROPERTIES*sparseProperties;
    rpc_read(client, &sparseProperties, sizeof(sparseProperties));
    CUarray array;
    rpc_read(client, &array, sizeof(array));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArrayGetSparseProperties(sparseProperties, array);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMipmappedArrayGetSparseProperties(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMipmappedArrayGetSparseProperties called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_ARRAY_SPARSE_PROPERTIES*sparseProperties;
    rpc_read(client, &sparseProperties, sizeof(sparseProperties));
    CUmipmappedArray mipmap;
    rpc_read(client, &mipmap, sizeof(mipmap));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMipmappedArrayGetSparseProperties(sparseProperties, mipmap);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuArrayGetMemoryRequirements(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuArrayGetMemoryRequirements called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_ARRAY_MEMORY_REQUIREMENTS*memoryRequirements;
    rpc_read(client, &memoryRequirements, sizeof(memoryRequirements));
    CUarray array;
    rpc_read(client, &array, sizeof(array));
    CUdevice device;
    rpc_read(client, &device, sizeof(device));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArrayGetMemoryRequirements(memoryRequirements, array, device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMipmappedArrayGetMemoryRequirements(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMipmappedArrayGetMemoryRequirements called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_ARRAY_MEMORY_REQUIREMENTS*memoryRequirements;
    rpc_read(client, &memoryRequirements, sizeof(memoryRequirements));
    CUmipmappedArray mipmap;
    rpc_read(client, &mipmap, sizeof(mipmap));
    CUdevice device;
    rpc_read(client, &device, sizeof(device));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuArrayGetPlane(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuArrayGetPlane called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarray*pPlaneArray;
    rpc_read(client, &pPlaneArray, sizeof(pPlaneArray));
    CUarray hArray;
    rpc_read(client, &hArray, sizeof(hArray));
    unsigned int planeIdx;
    rpc_read(client, &planeIdx, sizeof(planeIdx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArrayGetPlane(pPlaneArray, hArray, planeIdx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuArrayDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuArrayDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarray hArray;
    rpc_read(client, &hArray, sizeof(hArray));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArrayDestroy(hArray);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuArray3DCreate_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuArray3DCreate_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarray*pHandle;
    rpc_read(client, &pHandle, sizeof(pHandle));
    CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray;
    rpc_read(client, &pAllocateArray, sizeof(pAllocateArray));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArray3DCreate_v2(pHandle, pAllocateArray);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuArray3DGetDescriptor_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuArray3DGetDescriptor_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_ARRAY3D_DESCRIPTOR*pArrayDescriptor;
    rpc_read(client, &pArrayDescriptor, sizeof(pArrayDescriptor));
    CUarray hArray;
    rpc_read(client, &hArray, sizeof(hArray));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMipmappedArrayCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMipmappedArrayCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmipmappedArray*pHandle;
    rpc_read(client, &pHandle, sizeof(pHandle));
    CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc;
    rpc_read(client, &pMipmappedArrayDesc, sizeof(pMipmappedArrayDesc));
    unsigned int numMipmapLevels;
    rpc_read(client, &numMipmapLevels, sizeof(numMipmapLevels));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMipmappedArrayGetLevel(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMipmappedArrayGetLevel called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarray*pLevelArray;
    rpc_read(client, &pLevelArray, sizeof(pLevelArray));
    CUmipmappedArray hMipmappedArray;
    rpc_read(client, &hMipmappedArray, sizeof(hMipmappedArray));
    unsigned int level;
    rpc_read(client, &level, sizeof(level));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMipmappedArrayDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMipmappedArrayDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmipmappedArray hMipmappedArray;
    rpc_read(client, &hMipmappedArray, sizeof(hMipmappedArray));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMipmappedArrayDestroy(hMipmappedArray);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemGetHandleForAddressRange(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemGetHandleForAddressRange called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *handle;
    rpc_read(client, &handle, sizeof(handle));
    CUdeviceptr dptr;
    rpc_read(client, &dptr, sizeof(dptr));
    size_t size;
    rpc_read(client, &size, sizeof(size));
    CUmemRangeHandleType handleType;
    rpc_read(client, &handleType, sizeof(handleType));
    unsigned long long flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemGetHandleForAddressRange(handle, dptr, size, handleType, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemBatchDecompressAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemBatchDecompressAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemDecompressParams*paramsArray;
    rpc_read(client, &paramsArray, sizeof(paramsArray));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    size_t*errorIndex;
    rpc_read(client, &errorIndex, sizeof(errorIndex));
    CUstream stream;
    rpc_read(client, &stream, sizeof(stream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemBatchDecompressAsync(paramsArray, count, flags, errorIndex, stream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemAddressFree(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemAddressFree called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    size_t size;
    rpc_read(client, &size, sizeof(size));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemAddressFree(ptr, size);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemMapArrayAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemMapArrayAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarrayMapInfo*mapInfoList;
    rpc_read(client, &mapInfoList, sizeof(mapInfoList));
    unsigned int count;
    rpc_read(client, &count, sizeof(count));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemMapArrayAsync(mapInfoList, count, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemUnmap(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemUnmap called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    size_t size;
    rpc_read(client, &size, sizeof(size));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemUnmap(ptr, size);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemSetAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemSetAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    size_t size;
    rpc_read(client, &size, sizeof(size));
    CUmemAccessDesc *desc;
    rpc_read(client, &desc, sizeof(desc));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemSetAccess(ptr, size, desc, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemGetAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemGetAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned long long*flags;
    rpc_read(client, &flags, sizeof(flags));
    CUmemLocation *location;
    rpc_read(client, &location, sizeof(location));
    CUdeviceptr ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemGetAccess(flags, location, ptr);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemExportToShareableHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemExportToShareableHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *shareableHandle;
    rpc_read(client, &shareableHandle, sizeof(shareableHandle));
    CUmemGenericAllocationHandle handle;
    rpc_read(client, &handle, sizeof(handle));
    CUmemAllocationHandleType handleType;
    rpc_read(client, &handleType, sizeof(handleType));
    unsigned long long flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemExportToShareableHandle(shareableHandle, handle, handleType, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemImportFromShareableHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemImportFromShareableHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemGenericAllocationHandle*handle;
    rpc_read(client, &handle, sizeof(handle));
    void *osHandle;
    rpc_read(client, &osHandle, sizeof(osHandle));
    CUmemAllocationHandleType shHandleType;
    rpc_read(client, &shHandleType, sizeof(shHandleType));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemImportFromShareableHandle(handle, osHandle, shHandleType);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemGetAllocationGranularity(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemGetAllocationGranularity called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t*granularity;
    rpc_read(client, &granularity, sizeof(granularity));
    CUmemAllocationProp *prop;
    rpc_read(client, &prop, sizeof(prop));
    CUmemAllocationGranularity_flags option;
    rpc_read(client, &option, sizeof(option));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemGetAllocationGranularity(granularity, prop, option);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemGetAllocationPropertiesFromHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemGetAllocationPropertiesFromHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemAllocationProp*prop;
    rpc_read(client, &prop, sizeof(prop));
    CUmemGenericAllocationHandle handle;
    rpc_read(client, &handle, sizeof(handle));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemGetAllocationPropertiesFromHandle(prop, handle);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemRetainAllocationHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemRetainAllocationHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemGenericAllocationHandle*handle;
    rpc_read(client, &handle, sizeof(handle));
    void *addr;
    rpc_read(client, &addr, sizeof(addr));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemRetainAllocationHandle(handle, addr);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemFreeAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemFreeAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr dptr;
    rpc_read(client, &dptr, sizeof(dptr));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemFreeAsync(dptr, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemPoolTrimTo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemPoolTrimTo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemoryPool pool;
    rpc_read(client, &pool, sizeof(pool));
    size_t minBytesToKeep;
    rpc_read(client, &minBytesToKeep, sizeof(minBytesToKeep));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolTrimTo(pool, minBytesToKeep);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemPoolSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemPoolSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemoryPool pool;
    rpc_read(client, &pool, sizeof(pool));
    CUmemPool_attribute attr;
    rpc_read(client, &attr, sizeof(attr));
    void *value;
    rpc_read(client, &value, sizeof(value));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolSetAttribute(pool, attr, value);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemPoolGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemPoolGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemoryPool pool;
    rpc_read(client, &pool, sizeof(pool));
    CUmemPool_attribute attr;
    rpc_read(client, &attr, sizeof(attr));
    void *value;
    rpc_read(client, &value, sizeof(value));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolGetAttribute(pool, attr, value);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemPoolSetAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemPoolSetAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemoryPool pool;
    rpc_read(client, &pool, sizeof(pool));
    CUmemAccessDesc *map;
    rpc_read(client, &map, sizeof(map));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolSetAccess(pool, map, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemPoolGetAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemPoolGetAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemAccess_flags*flags;
    rpc_read(client, &flags, sizeof(flags));
    CUmemoryPool memPool;
    rpc_read(client, &memPool, sizeof(memPool));
    CUmemLocation*location;
    rpc_read(client, &location, sizeof(location));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolGetAccess(flags, memPool, location);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemPoolCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemPoolCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemoryPool*pool;
    rpc_read(client, &pool, sizeof(pool));
    CUmemPoolProps *poolProps;
    rpc_read(client, &poolProps, sizeof(poolProps));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolCreate(pool, poolProps);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemPoolDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemPoolDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemoryPool pool;
    rpc_read(client, &pool, sizeof(pool));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolDestroy(pool);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemPoolExportToShareableHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemPoolExportToShareableHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *handle_out;
    rpc_read(client, &handle_out, sizeof(handle_out));
    CUmemoryPool pool;
    rpc_read(client, &pool, sizeof(pool));
    CUmemAllocationHandleType handleType;
    rpc_read(client, &handleType, sizeof(handleType));
    unsigned long long flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolExportToShareableHandle(handle_out, pool, handleType, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemPoolImportFromShareableHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemPoolImportFromShareableHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemoryPool*pool_out;
    rpc_read(client, &pool_out, sizeof(pool_out));
    void *handle;
    rpc_read(client, &handle, sizeof(handle));
    CUmemAllocationHandleType handleType;
    rpc_read(client, &handleType, sizeof(handleType));
    unsigned long long flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolImportFromShareableHandle(pool_out, handle, handleType, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemPoolExportPointer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemPoolExportPointer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemPoolPtrExportData*shareData_out;
    rpc_read(client, &shareData_out, sizeof(shareData_out));
    CUdeviceptr ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolExportPointer(shareData_out, ptr);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMulticastCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMulticastCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemGenericAllocationHandle*mcHandle;
    rpc_read(client, &mcHandle, sizeof(mcHandle));
    CUmulticastObjectProp *prop;
    rpc_read(client, &prop, sizeof(prop));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMulticastCreate(mcHandle, prop);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMulticastAddDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMulticastAddDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemGenericAllocationHandle mcHandle;
    rpc_read(client, &mcHandle, sizeof(mcHandle));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMulticastAddDevice(mcHandle, dev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMulticastBindMem(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMulticastBindMem called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemGenericAllocationHandle mcHandle;
    rpc_read(client, &mcHandle, sizeof(mcHandle));
    size_t mcOffset;
    rpc_read(client, &mcOffset, sizeof(mcOffset));
    CUmemGenericAllocationHandle memHandle;
    rpc_read(client, &memHandle, sizeof(memHandle));
    size_t memOffset;
    rpc_read(client, &memOffset, sizeof(memOffset));
    size_t size;
    rpc_read(client, &size, sizeof(size));
    unsigned long long flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMulticastBindMem(mcHandle, mcOffset, memHandle, memOffset, size, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMulticastBindAddr(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMulticastBindAddr called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemGenericAllocationHandle mcHandle;
    rpc_read(client, &mcHandle, sizeof(mcHandle));
    size_t mcOffset;
    rpc_read(client, &mcOffset, sizeof(mcOffset));
    CUdeviceptr memptr;
    rpc_read(client, &memptr, sizeof(memptr));
    size_t size;
    rpc_read(client, &size, sizeof(size));
    unsigned long long flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMulticastBindAddr(mcHandle, mcOffset, memptr, size, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMulticastUnbind(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMulticastUnbind called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmemGenericAllocationHandle mcHandle;
    rpc_read(client, &mcHandle, sizeof(mcHandle));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    size_t mcOffset;
    rpc_read(client, &mcOffset, sizeof(mcOffset));
    size_t size;
    rpc_read(client, &size, sizeof(size));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMulticastUnbind(mcHandle, dev, mcOffset, size);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMulticastGetGranularity(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMulticastGetGranularity called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t*granularity;
    rpc_read(client, &granularity, sizeof(granularity));
    CUmulticastObjectProp *prop;
    rpc_read(client, &prop, sizeof(prop));
    CUmulticastGranularity_flags option;
    rpc_read(client, &option, sizeof(option));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMulticastGetGranularity(granularity, prop, option);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuPointerGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuPointerGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *data;
    rpc_read(client, &data, sizeof(data));
    CUpointer_attribute attribute;
    rpc_read(client, &attribute, sizeof(attribute));
    CUdeviceptr ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuPointerGetAttribute(data, attribute, ptr);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemPrefetchAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemPrefetchAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    CUdevice dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPrefetchAsync(devPtr, count, dstDevice, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemPrefetchAsync_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemPrefetchAsync_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    CUmemLocation location;
    rpc_read(client, &location, sizeof(location));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPrefetchAsync_v2(devPtr, count, location, flags, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemAdvise(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemAdvise called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    CUmem_advise advice;
    rpc_read(client, &advice, sizeof(advice));
    CUdevice device;
    rpc_read(client, &device, sizeof(device));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemAdvise(devPtr, count, advice, device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemAdvise_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemAdvise_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdeviceptr devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    CUmem_advise advice;
    rpc_read(client, &advice, sizeof(advice));
    CUmemLocation location;
    rpc_read(client, &location, sizeof(location));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemAdvise_v2(devPtr, count, advice, location);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemRangeGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemRangeGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *data;
    rpc_read(client, &data, sizeof(data));
    size_t dataSize;
    rpc_read(client, &dataSize, sizeof(dataSize));
    CUmem_range_attribute attribute;
    rpc_read(client, &attribute, sizeof(attribute));
    CUdeviceptr devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemRangeGetAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemRangeGetAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    // PARAM void **data
    void *data;
    size_t*dataSizes;
    rpc_read(client, &dataSizes, sizeof(dataSizes));
    CUmem_range_attribute*attributes;
    rpc_read(client, &attributes, sizeof(attributes));
    size_t numAttributes;
    rpc_read(client, &numAttributes, sizeof(numAttributes));
    CUdeviceptr devPtr;
    rpc_read(client, &devPtr, sizeof(devPtr));
    size_t count;
    rpc_read(client, &count, sizeof(count));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **data
    _result = cuMemRangeGetAttributes(&data, dataSizes, attributes, numAttributes, devPtr, count);
    // PARAM void **data
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **data
    return rtn;
}

int handle_cuPointerSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuPointerSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    void *value;
    rpc_read(client, &value, sizeof(value));
    CUpointer_attribute attribute;
    rpc_read(client, &attribute, sizeof(attribute));
    CUdeviceptr ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuPointerSetAttribute(value, attribute, ptr);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuPointerGetAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuPointerGetAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int numAttributes;
    rpc_read(client, &numAttributes, sizeof(numAttributes));
    CUpointer_attribute*attributes;
    rpc_read(client, &attributes, sizeof(attributes));
    // PARAM void **data
    void *data;
    CUdeviceptr ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **data
    _result = cuPointerGetAttributes(numAttributes, attributes, &data, ptr);
    // PARAM void **data
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **data
    return rtn;
}

int handle_cuStreamCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream*phStream;
    rpc_read(client, &phStream, sizeof(phStream));
    unsigned int Flags;
    rpc_read(client, &Flags, sizeof(Flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamCreate(phStream, Flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamCreateWithPriority(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamCreateWithPriority called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream*phStream;
    rpc_read(client, &phStream, sizeof(phStream));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    int priority;
    rpc_read(client, &priority, sizeof(priority));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamCreateWithPriority(phStream, flags, priority);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamGetPriority(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamGetPriority called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    int*priority;
    rpc_read(client, &priority, sizeof(priority));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetPriority(hStream, priority);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamGetDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamGetDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUdevice*device;
    rpc_read(client, &device, sizeof(device));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetDevice(hStream, device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamGetFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamGetFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    unsigned int*flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetFlags(hStream, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamGetId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamGetId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    unsigned long long*streamId;
    rpc_read(client, &streamId, sizeof(streamId));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetId(hStream, streamId);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamGetCtx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamGetCtx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUcontext*pctx;
    rpc_read(client, &pctx, sizeof(pctx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetCtx(hStream, pctx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamGetCtx_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamGetCtx_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUcontext*pCtx;
    rpc_read(client, &pCtx, sizeof(pCtx));
    CUgreenCtx*pGreenCtx;
    rpc_read(client, &pGreenCtx, sizeof(pGreenCtx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetCtx_v2(hStream, pCtx, pGreenCtx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamWaitEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamWaitEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUevent hEvent;
    rpc_read(client, &hEvent, sizeof(hEvent));
    unsigned int Flags;
    rpc_read(client, &Flags, sizeof(Flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamWaitEvent(hStream, hEvent, Flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamAddCallback(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamAddCallback called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUstreamCallback callback;
    rpc_read(client, &callback, sizeof(callback));
    void *userData;
    rpc_read(client, &userData, sizeof(userData));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamAddCallback(hStream, callback, userData, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamBeginCapture_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamBeginCapture_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUstreamCaptureMode mode;
    rpc_read(client, &mode, sizeof(mode));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamBeginCapture_v2(hStream, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamBeginCaptureToGraph(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamBeginCaptureToGraph called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    CUgraphEdgeData *dependencyData;
    rpc_read(client, &dependencyData, sizeof(dependencyData));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUstreamCaptureMode mode;
    rpc_read(client, &mode, sizeof(mode));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamBeginCaptureToGraph(hStream, hGraph, dependencies, dependencyData, numDependencies, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuThreadExchangeStreamCaptureMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuThreadExchangeStreamCaptureMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstreamCaptureMode*mode;
    rpc_read(client, &mode, sizeof(mode));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuThreadExchangeStreamCaptureMode(mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamEndCapture(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamEndCapture called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUgraph*phGraph;
    rpc_read(client, &phGraph, sizeof(phGraph));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamEndCapture(hStream, phGraph);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamIsCapturing(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamIsCapturing called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUstreamCaptureStatus*captureStatus;
    rpc_read(client, &captureStatus, sizeof(captureStatus));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamIsCapturing(hStream, captureStatus);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamGetCaptureInfo_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamGetCaptureInfo_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUstreamCaptureStatus*captureStatus_out;
    rpc_read(client, &captureStatus_out, sizeof(captureStatus_out));
    cuuint64_t*id_out;
    rpc_read(client, &id_out, sizeof(id_out));
    CUgraph*graph_out;
    rpc_read(client, &graph_out, sizeof(graph_out));
    // PARAM const CUgraphNode **dependencies_out
    const CUgraphNode *dependencies_out;
    size_t*numDependencies_out;
    rpc_read(client, &numDependencies_out, sizeof(numDependencies_out));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM const CUgraphNode **dependencies_out
    _result = cuStreamGetCaptureInfo_v2(hStream, captureStatus_out, id_out, graph_out, &dependencies_out, numDependencies_out);
    // PARAM const CUgraphNode **dependencies_out
    rpc_write(client, dependencies_out, sizeof(CUgraphNode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM const CUgraphNode **dependencies_out
    return rtn;
}

int handle_cuStreamGetCaptureInfo_v3(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamGetCaptureInfo_v3 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUstreamCaptureStatus*captureStatus_out;
    rpc_read(client, &captureStatus_out, sizeof(captureStatus_out));
    cuuint64_t*id_out;
    rpc_read(client, &id_out, sizeof(id_out));
    CUgraph*graph_out;
    rpc_read(client, &graph_out, sizeof(graph_out));
    // PARAM const CUgraphNode **dependencies_out
    const CUgraphNode *dependencies_out;
    // PARAM const CUgraphEdgeData **edgeData_out
    const CUgraphEdgeData *edgeData_out;
    size_t*numDependencies_out;
    rpc_read(client, &numDependencies_out, sizeof(numDependencies_out));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM const CUgraphNode **dependencies_out
    // PARAM const CUgraphEdgeData **edgeData_out
    _result = cuStreamGetCaptureInfo_v3(hStream, captureStatus_out, id_out, graph_out, &dependencies_out, &edgeData_out, numDependencies_out);
    // PARAM const CUgraphNode **dependencies_out
    rpc_write(client, dependencies_out, sizeof(CUgraphNode));
    // PARAM const CUgraphEdgeData **edgeData_out
    rpc_write(client, edgeData_out, sizeof(CUgraphEdgeData));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM const CUgraphNode **dependencies_out
    // PARAM const CUgraphEdgeData **edgeData_out
    return rtn;
}

int handle_cuStreamUpdateCaptureDependencies(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamUpdateCaptureDependencies called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUgraphNode*dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamUpdateCaptureDependencies(hStream, dependencies, numDependencies, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamUpdateCaptureDependencies_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamUpdateCaptureDependencies_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUgraphNode*dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    CUgraphEdgeData *dependencyData;
    rpc_read(client, &dependencyData, sizeof(dependencyData));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamUpdateCaptureDependencies_v2(hStream, dependencies, dependencyData, numDependencies, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamAttachMemAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamAttachMemAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUdeviceptr dptr;
    rpc_read(client, &dptr, sizeof(dptr));
    size_t length;
    rpc_read(client, &length, sizeof(length));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamAttachMemAsync(hStream, dptr, length, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamQuery(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamQuery called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamQuery(hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamSynchronize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamSynchronize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamSynchronize(hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamDestroy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamDestroy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamDestroy_v2(hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamCopyAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamCopyAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream dst;
    rpc_read(client, &dst, sizeof(dst));
    CUstream src;
    rpc_read(client, &src, sizeof(src));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamCopyAttributes(dst, src);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUstreamAttrID attr;
    rpc_read(client, &attr, sizeof(attr));
    CUstreamAttrValue*value_out;
    rpc_read(client, &value_out, sizeof(value_out));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetAttribute(hStream, attr, value_out);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUstreamAttrID attr;
    rpc_read(client, &attr, sizeof(attr));
    CUstreamAttrValue *value;
    rpc_read(client, &value, sizeof(value));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamSetAttribute(hStream, attr, value);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuEventCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuEventCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUevent*phEvent;
    rpc_read(client, &phEvent, sizeof(phEvent));
    unsigned int Flags;
    rpc_read(client, &Flags, sizeof(Flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventCreate(phEvent, Flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuEventRecord(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuEventRecord called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUevent hEvent;
    rpc_read(client, &hEvent, sizeof(hEvent));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventRecord(hEvent, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuEventRecordWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuEventRecordWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUevent hEvent;
    rpc_read(client, &hEvent, sizeof(hEvent));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventRecordWithFlags(hEvent, hStream, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuEventQuery(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuEventQuery called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUevent hEvent;
    rpc_read(client, &hEvent, sizeof(hEvent));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventQuery(hEvent);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuEventSynchronize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuEventSynchronize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUevent hEvent;
    rpc_read(client, &hEvent, sizeof(hEvent));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventSynchronize(hEvent);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuEventDestroy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuEventDestroy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUevent hEvent;
    rpc_read(client, &hEvent, sizeof(hEvent));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventDestroy_v2(hEvent);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuEventElapsedTime(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuEventElapsedTime called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    float*pMilliseconds;
    rpc_read(client, &pMilliseconds, sizeof(pMilliseconds));
    CUevent hStart;
    rpc_read(client, &hStart, sizeof(hStart));
    CUevent hEnd;
    rpc_read(client, &hEnd, sizeof(hEnd));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventElapsedTime(pMilliseconds, hStart, hEnd);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuEventElapsedTime_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuEventElapsedTime_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    float*pMilliseconds;
    rpc_read(client, &pMilliseconds, sizeof(pMilliseconds));
    CUevent hStart;
    rpc_read(client, &hStart, sizeof(hStart));
    CUevent hEnd;
    rpc_read(client, &hEnd, sizeof(hEnd));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventElapsedTime_v2(pMilliseconds, hStart, hEnd);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuExternalMemoryGetMappedMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuExternalMemoryGetMappedMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmipmappedArray*mipmap;
    rpc_read(client, &mipmap, sizeof(mipmap));
    CUexternalMemory extMem;
    rpc_read(client, &extMem, sizeof(extMem));
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc;
    rpc_read(client, &mipmapDesc, sizeof(mipmapDesc));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDestroyExternalMemory(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDestroyExternalMemory called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUexternalMemory extMem;
    rpc_read(client, &extMem, sizeof(extMem));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDestroyExternalMemory(extMem);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuImportExternalSemaphore(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuImportExternalSemaphore called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUexternalSemaphore*extSem_out;
    rpc_read(client, &extSem_out, sizeof(extSem_out));
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc;
    rpc_read(client, &semHandleDesc, sizeof(semHandleDesc));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuImportExternalSemaphore(extSem_out, semHandleDesc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuSignalExternalSemaphoresAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuSignalExternalSemaphoresAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUexternalSemaphore *extSemArray;
    rpc_read(client, &extSemArray, sizeof(extSemArray));
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray;
    rpc_read(client, &paramsArray, sizeof(paramsArray));
    unsigned int numExtSems;
    rpc_read(client, &numExtSems, sizeof(numExtSems));
    CUstream stream;
    rpc_read(client, &stream, sizeof(stream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuWaitExternalSemaphoresAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuWaitExternalSemaphoresAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUexternalSemaphore *extSemArray;
    rpc_read(client, &extSemArray, sizeof(extSemArray));
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray;
    rpc_read(client, &paramsArray, sizeof(paramsArray));
    unsigned int numExtSems;
    rpc_read(client, &numExtSems, sizeof(numExtSems));
    CUstream stream;
    rpc_read(client, &stream, sizeof(stream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDestroyExternalSemaphore(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDestroyExternalSemaphore called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUexternalSemaphore extSem;
    rpc_read(client, &extSem, sizeof(extSem));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDestroyExternalSemaphore(extSem);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamWaitValue32_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamWaitValue32_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream stream;
    rpc_read(client, &stream, sizeof(stream));
    CUdeviceptr addr;
    rpc_read(client, &addr, sizeof(addr));
    cuuint32_t value;
    rpc_read(client, &value, sizeof(value));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamWaitValue32_v2(stream, addr, value, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamWaitValue64_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamWaitValue64_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream stream;
    rpc_read(client, &stream, sizeof(stream));
    CUdeviceptr addr;
    rpc_read(client, &addr, sizeof(addr));
    cuuint64_t value;
    rpc_read(client, &value, sizeof(value));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamWaitValue64_v2(stream, addr, value, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamWriteValue32_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamWriteValue32_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream stream;
    rpc_read(client, &stream, sizeof(stream));
    CUdeviceptr addr;
    rpc_read(client, &addr, sizeof(addr));
    cuuint32_t value;
    rpc_read(client, &value, sizeof(value));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamWriteValue32_v2(stream, addr, value, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamWriteValue64_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamWriteValue64_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream stream;
    rpc_read(client, &stream, sizeof(stream));
    CUdeviceptr addr;
    rpc_read(client, &addr, sizeof(addr));
    cuuint64_t value;
    rpc_read(client, &value, sizeof(value));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamWriteValue64_v2(stream, addr, value, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamBatchMemOp_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamBatchMemOp_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream stream;
    rpc_read(client, &stream, sizeof(stream));
    unsigned int count;
    rpc_read(client, &count, sizeof(count));
    CUstreamBatchMemOpParams*paramArray;
    rpc_read(client, &paramArray, sizeof(paramArray));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamBatchMemOp_v2(stream, count, paramArray, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuFuncGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuFuncGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*pi;
    rpc_read(client, &pi, sizeof(pi));
    CUfunction_attribute attrib;
    rpc_read(client, &attrib, sizeof(attrib));
    CUfunction hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncGetAttribute(pi, attrib, hfunc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuFuncSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuFuncSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    CUfunction_attribute attrib;
    rpc_read(client, &attrib, sizeof(attrib));
    int value;
    rpc_read(client, &value, sizeof(value));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncSetAttribute(hfunc, attrib, value);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuFuncSetCacheConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuFuncSetCacheConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    CUfunc_cache config;
    rpc_read(client, &config, sizeof(config));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncSetCacheConfig(hfunc, config);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuFuncGetModule(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuFuncGetModule called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmodule*hmod;
    rpc_read(client, &hmod, sizeof(hmod));
    CUfunction hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncGetModule(hmod, hfunc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuFuncGetName(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuFuncGetName called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    // PARAM const char **name
    const char *name;
    CUfunction hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM const char **name
    _result = cuFuncGetName(&name, hfunc);
    // PARAM const char **name
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM const char **name
    return rtn;
}

int handle_cuFuncGetParamInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuFuncGetParamInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction func;
    rpc_read(client, &func, sizeof(func));
    size_t paramIndex;
    rpc_read(client, &paramIndex, sizeof(paramIndex));
    size_t*paramOffset;
    rpc_read(client, &paramOffset, sizeof(paramOffset));
    size_t*paramSize;
    rpc_read(client, &paramSize, sizeof(paramSize));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncGetParamInfo(func, paramIndex, paramOffset, paramSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuFuncIsLoaded(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuFuncIsLoaded called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunctionLoadingState*state;
    rpc_read(client, &state, sizeof(state));
    CUfunction function;
    rpc_read(client, &function, sizeof(function));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncIsLoaded(state, function);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuFuncLoad(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuFuncLoad called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction function;
    rpc_read(client, &function, sizeof(function));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncLoad(function);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuLaunchKernel(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLaunchKernel called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction f;
    rpc_read(client, &f, sizeof(f));
    unsigned int gridDimX;
    rpc_read(client, &gridDimX, sizeof(gridDimX));
    unsigned int gridDimY;
    rpc_read(client, &gridDimY, sizeof(gridDimY));
    unsigned int gridDimZ;
    rpc_read(client, &gridDimZ, sizeof(gridDimZ));
    unsigned int blockDimX;
    rpc_read(client, &blockDimX, sizeof(blockDimX));
    unsigned int blockDimY;
    rpc_read(client, &blockDimY, sizeof(blockDimY));
    unsigned int blockDimZ;
    rpc_read(client, &blockDimZ, sizeof(blockDimZ));
    unsigned int sharedMemBytes;
    rpc_read(client, &sharedMemBytes, sizeof(sharedMemBytes));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    // PARAM void **kernelParams
    void *kernelParams;
    // PARAM void **extra
    void *extra;
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **kernelParams
    // PARAM void **extra
    _result = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, &kernelParams, &extra);
    // PARAM void **kernelParams
    // PARAM void **extra
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **kernelParams
    // PARAM void **extra
    return rtn;
}

int handle_cuLaunchKernelEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLaunchKernelEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUlaunchConfig *config;
    rpc_read(client, &config, sizeof(config));
    CUfunction f;
    rpc_read(client, &f, sizeof(f));
    // PARAM void **kernelParams
    void *kernelParams;
    // PARAM void **extra
    void *extra;
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **kernelParams
    // PARAM void **extra
    _result = cuLaunchKernelEx(config, f, &kernelParams, &extra);
    // PARAM void **kernelParams
    // PARAM void **extra
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **kernelParams
    // PARAM void **extra
    return rtn;
}

int handle_cuLaunchCooperativeKernel(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLaunchCooperativeKernel called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction f;
    rpc_read(client, &f, sizeof(f));
    unsigned int gridDimX;
    rpc_read(client, &gridDimX, sizeof(gridDimX));
    unsigned int gridDimY;
    rpc_read(client, &gridDimY, sizeof(gridDimY));
    unsigned int gridDimZ;
    rpc_read(client, &gridDimZ, sizeof(gridDimZ));
    unsigned int blockDimX;
    rpc_read(client, &blockDimX, sizeof(blockDimX));
    unsigned int blockDimY;
    rpc_read(client, &blockDimY, sizeof(blockDimY));
    unsigned int blockDimZ;
    rpc_read(client, &blockDimZ, sizeof(blockDimZ));
    unsigned int sharedMemBytes;
    rpc_read(client, &sharedMemBytes, sizeof(sharedMemBytes));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    // PARAM void **kernelParams
    void *kernelParams;
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **kernelParams
    _result = cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, &kernelParams);
    // PARAM void **kernelParams
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **kernelParams
    return rtn;
}

int handle_cuLaunchCooperativeKernelMultiDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLaunchCooperativeKernelMultiDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_LAUNCH_PARAMS*launchParamsList;
    rpc_read(client, &launchParamsList, sizeof(launchParamsList));
    unsigned int numDevices;
    rpc_read(client, &numDevices, sizeof(numDevices));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuLaunchHostFunc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLaunchHostFunc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUhostFn fn;
    rpc_read(client, &fn, sizeof(fn));
    void *userData;
    rpc_read(client, &userData, sizeof(userData));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLaunchHostFunc(hStream, fn, userData);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuFuncSetBlockShape(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuFuncSetBlockShape called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    int x;
    rpc_read(client, &x, sizeof(x));
    int y;
    rpc_read(client, &y, sizeof(y));
    int z;
    rpc_read(client, &z, sizeof(z));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncSetBlockShape(hfunc, x, y, z);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuFuncSetSharedSize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuFuncSetSharedSize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    unsigned int bytes;
    rpc_read(client, &bytes, sizeof(bytes));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncSetSharedSize(hfunc, bytes);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuParamSetSize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuParamSetSize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    unsigned int numbytes;
    rpc_read(client, &numbytes, sizeof(numbytes));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuParamSetSize(hfunc, numbytes);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuParamSeti(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuParamSeti called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    int offset;
    rpc_read(client, &offset, sizeof(offset));
    unsigned int value;
    rpc_read(client, &value, sizeof(value));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuParamSeti(hfunc, offset, value);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuParamSetf(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuParamSetf called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    int offset;
    rpc_read(client, &offset, sizeof(offset));
    float value;
    rpc_read(client, &value, sizeof(value));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuParamSetf(hfunc, offset, value);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuParamSetv(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuParamSetv called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    int offset;
    rpc_read(client, &offset, sizeof(offset));
    void *ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    unsigned int numbytes;
    rpc_read(client, &numbytes, sizeof(numbytes));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuParamSetv(hfunc, offset, ptr, numbytes);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuLaunch(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLaunch called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction f;
    rpc_read(client, &f, sizeof(f));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLaunch(f);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuLaunchGrid(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLaunchGrid called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction f;
    rpc_read(client, &f, sizeof(f));
    int grid_width;
    rpc_read(client, &grid_width, sizeof(grid_width));
    int grid_height;
    rpc_read(client, &grid_height, sizeof(grid_height));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLaunchGrid(f, grid_width, grid_height);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuLaunchGridAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLaunchGridAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction f;
    rpc_read(client, &f, sizeof(f));
    int grid_width;
    rpc_read(client, &grid_width, sizeof(grid_width));
    int grid_height;
    rpc_read(client, &grid_height, sizeof(grid_height));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLaunchGridAsync(f, grid_width, grid_height, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuParamSetTexRef(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuParamSetTexRef called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    int texunit;
    rpc_read(client, &texunit, sizeof(texunit));
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuParamSetTexRef(hfunc, texunit, hTexRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuFuncSetSharedMemConfig(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuFuncSetSharedMemConfig called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfunction hfunc;
    rpc_read(client, &hfunc, sizeof(hfunc));
    CUsharedconfig config;
    rpc_read(client, &config, sizeof(config));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncSetSharedMemConfig(hfunc, config);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph*phGraph;
    rpc_read(client, &phGraph, sizeof(phGraph));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphCreate(phGraph, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddKernelNode_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddKernelNode_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUDA_KERNEL_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddKernelNode_v2(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphKernelNodeGetParams_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphKernelNodeGetParams_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_KERNEL_NODE_PARAMS*nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphKernelNodeGetParams_v2(hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphKernelNodeSetParams_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphKernelNodeSetParams_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_KERNEL_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphKernelNodeSetParams_v2(hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddMemcpyNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddMemcpyNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUDA_MEMCPY3D *copyParams;
    rpc_read(client, &copyParams, sizeof(copyParams));
    CUcontext ctx;
    rpc_read(client, &ctx, sizeof(ctx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphMemcpyNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphMemcpyNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_MEMCPY3D*nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphMemcpyNodeGetParams(hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphMemcpyNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphMemcpyNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_MEMCPY3D *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphMemcpyNodeSetParams(hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddMemsetNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddMemsetNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUDA_MEMSET_NODE_PARAMS *memsetParams;
    rpc_read(client, &memsetParams, sizeof(memsetParams));
    CUcontext ctx;
    rpc_read(client, &ctx, sizeof(ctx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphMemsetNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphMemsetNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_MEMSET_NODE_PARAMS*nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphMemsetNodeGetParams(hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphMemsetNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphMemsetNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_MEMSET_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphMemsetNodeSetParams(hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddHostNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddHostNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUDA_HOST_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphHostNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphHostNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_HOST_NODE_PARAMS*nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphHostNodeGetParams(hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphHostNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphHostNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_HOST_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphHostNodeSetParams(hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddChildGraphNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddChildGraphNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUgraph childGraph;
    rpc_read(client, &childGraph, sizeof(childGraph));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddChildGraphNode(phGraphNode, hGraph, dependencies, numDependencies, childGraph);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphChildGraphNodeGetGraph(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphChildGraphNodeGetGraph called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUgraph*phGraph;
    rpc_read(client, &phGraph, sizeof(phGraph));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphChildGraphNodeGetGraph(hNode, phGraph);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddEmptyNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddEmptyNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddEmptyNode(phGraphNode, hGraph, dependencies, numDependencies);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddEventRecordNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddEventRecordNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUevent event;
    rpc_read(client, &event, sizeof(event));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddEventRecordNode(phGraphNode, hGraph, dependencies, numDependencies, event);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphEventRecordNodeGetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphEventRecordNodeGetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUevent*event_out;
    rpc_read(client, &event_out, sizeof(event_out));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphEventRecordNodeGetEvent(hNode, event_out);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphEventRecordNodeSetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphEventRecordNodeSetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUevent event;
    rpc_read(client, &event, sizeof(event));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphEventRecordNodeSetEvent(hNode, event);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddEventWaitNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddEventWaitNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUevent event;
    rpc_read(client, &event, sizeof(event));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddEventWaitNode(phGraphNode, hGraph, dependencies, numDependencies, event);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphEventWaitNodeGetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphEventWaitNodeGetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUevent*event_out;
    rpc_read(client, &event_out, sizeof(event_out));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphEventWaitNodeGetEvent(hNode, event_out);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphEventWaitNodeSetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphEventWaitNodeSetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUevent event;
    rpc_read(client, &event, sizeof(event));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphEventWaitNodeSetEvent(hNode, event);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddExternalSemaphoresSignalNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddExternalSemaphoresSignalNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddExternalSemaphoresSignalNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExternalSemaphoresSignalNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExternalSemaphoresSignalNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*params_out;
    rpc_read(client, &params_out, sizeof(params_out));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExternalSemaphoresSignalNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExternalSemaphoresSignalNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddExternalSemaphoresWaitNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddExternalSemaphoresWaitNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddExternalSemaphoresWaitNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExternalSemaphoresWaitNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExternalSemaphoresWaitNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_EXT_SEM_WAIT_NODE_PARAMS*params_out;
    rpc_read(client, &params_out, sizeof(params_out));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExternalSemaphoresWaitNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExternalSemaphoresWaitNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddBatchMemOpNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddBatchMemOpNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddBatchMemOpNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphBatchMemOpNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphBatchMemOpNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_BATCH_MEM_OP_NODE_PARAMS*nodeParams_out;
    rpc_read(client, &nodeParams_out, sizeof(nodeParams_out));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphBatchMemOpNodeGetParams(hNode, nodeParams_out);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphBatchMemOpNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphBatchMemOpNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphBatchMemOpNodeSetParams(hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecBatchMemOpNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecBatchMemOpNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecBatchMemOpNodeSetParams(hGraphExec, hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddMemAllocNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddMemAllocNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUDA_MEM_ALLOC_NODE_PARAMS*nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddMemAllocNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphMemAllocNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphMemAllocNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_MEM_ALLOC_NODE_PARAMS*params_out;
    rpc_read(client, &params_out, sizeof(params_out));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphMemAllocNodeGetParams(hNode, params_out);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddMemFreeNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddMemFreeNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUdeviceptr dptr;
    rpc_read(client, &dptr, sizeof(dptr));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddMemFreeNode(phGraphNode, hGraph, dependencies, numDependencies, dptr);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGraphMemTrim(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGraphMemTrim called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice device;
    rpc_read(client, &device, sizeof(device));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGraphMemTrim(device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetGraphMemAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetGraphMemAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice device;
    rpc_read(client, &device, sizeof(device));
    CUgraphMem_attribute attr;
    rpc_read(client, &attr, sizeof(attr));
    void *value;
    rpc_read(client, &value, sizeof(value));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetGraphMemAttribute(device, attr, value);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceSetGraphMemAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceSetGraphMemAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice device;
    rpc_read(client, &device, sizeof(device));
    CUgraphMem_attribute attr;
    rpc_read(client, &attr, sizeof(attr));
    void *value;
    rpc_read(client, &value, sizeof(value));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceSetGraphMemAttribute(device, attr, value);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphClone(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphClone called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph*phGraphClone;
    rpc_read(client, &phGraphClone, sizeof(phGraphClone));
    CUgraph originalGraph;
    rpc_read(client, &originalGraph, sizeof(originalGraph));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphClone(phGraphClone, originalGraph);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphNodeFindInClone(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphNodeFindInClone called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phNode;
    rpc_read(client, &phNode, sizeof(phNode));
    CUgraphNode hOriginalNode;
    rpc_read(client, &hOriginalNode, sizeof(hOriginalNode));
    CUgraph hClonedGraph;
    rpc_read(client, &hClonedGraph, sizeof(hClonedGraph));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphNodeGetType(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphNodeGetType called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUgraphNodeType*type;
    rpc_read(client, &type, sizeof(type));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphNodeGetType(hNode, type);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphGetNodes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphGetNodes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode*nodes;
    rpc_read(client, &nodes, sizeof(nodes));
    size_t*numNodes;
    rpc_read(client, &numNodes, sizeof(numNodes));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphGetNodes(hGraph, nodes, numNodes);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphGetRootNodes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphGetRootNodes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode*rootNodes;
    rpc_read(client, &rootNodes, sizeof(rootNodes));
    size_t*numRootNodes;
    rpc_read(client, &numRootNodes, sizeof(numRootNodes));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphGetRootNodes(hGraph, rootNodes, numRootNodes);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphGetEdges(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphGetEdges called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode*from;
    rpc_read(client, &from, sizeof(from));
    CUgraphNode*to;
    rpc_read(client, &to, sizeof(to));
    size_t*numEdges;
    rpc_read(client, &numEdges, sizeof(numEdges));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphGetEdges(hGraph, from, to, numEdges);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphGetEdges_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphGetEdges_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode*from;
    rpc_read(client, &from, sizeof(from));
    CUgraphNode*to;
    rpc_read(client, &to, sizeof(to));
    CUgraphEdgeData*edgeData;
    rpc_read(client, &edgeData, sizeof(edgeData));
    size_t*numEdges;
    rpc_read(client, &numEdges, sizeof(numEdges));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphGetEdges_v2(hGraph, from, to, edgeData, numEdges);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphNodeGetDependencies(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphNodeGetDependencies called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUgraphNode*dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t*numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphNodeGetDependencies(hNode, dependencies, numDependencies);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphNodeGetDependencies_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphNodeGetDependencies_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUgraphNode*dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    CUgraphEdgeData*edgeData;
    rpc_read(client, &edgeData, sizeof(edgeData));
    size_t*numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphNodeGetDependencies_v2(hNode, dependencies, edgeData, numDependencies);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphNodeGetDependentNodes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphNodeGetDependentNodes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUgraphNode*dependentNodes;
    rpc_read(client, &dependentNodes, sizeof(dependentNodes));
    size_t*numDependentNodes;
    rpc_read(client, &numDependentNodes, sizeof(numDependentNodes));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphNodeGetDependentNodes_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphNodeGetDependentNodes_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUgraphNode*dependentNodes;
    rpc_read(client, &dependentNodes, sizeof(dependentNodes));
    CUgraphEdgeData*edgeData;
    rpc_read(client, &edgeData, sizeof(edgeData));
    size_t*numDependentNodes;
    rpc_read(client, &numDependentNodes, sizeof(numDependentNodes));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphNodeGetDependentNodes_v2(hNode, dependentNodes, edgeData, numDependentNodes);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddDependencies(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddDependencies called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *from;
    rpc_read(client, &from, sizeof(from));
    CUgraphNode *to;
    rpc_read(client, &to, sizeof(to));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddDependencies(hGraph, from, to, numDependencies);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddDependencies_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddDependencies_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *from;
    rpc_read(client, &from, sizeof(from));
    CUgraphNode *to;
    rpc_read(client, &to, sizeof(to));
    CUgraphEdgeData *edgeData;
    rpc_read(client, &edgeData, sizeof(edgeData));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddDependencies_v2(hGraph, from, to, edgeData, numDependencies);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphRemoveDependencies(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphRemoveDependencies called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *from;
    rpc_read(client, &from, sizeof(from));
    CUgraphNode *to;
    rpc_read(client, &to, sizeof(to));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphRemoveDependencies(hGraph, from, to, numDependencies);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphRemoveDependencies_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphRemoveDependencies_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *from;
    rpc_read(client, &from, sizeof(from));
    CUgraphNode *to;
    rpc_read(client, &to, sizeof(to));
    CUgraphEdgeData *edgeData;
    rpc_read(client, &edgeData, sizeof(edgeData));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphRemoveDependencies_v2(hGraph, from, to, edgeData, numDependencies);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphDestroyNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphDestroyNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphDestroyNode(hNode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphInstantiateWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphInstantiateWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec*phGraphExec;
    rpc_read(client, &phGraphExec, sizeof(phGraphExec));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    unsigned long long flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphInstantiateWithFlags(phGraphExec, hGraph, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphInstantiateWithParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphInstantiateWithParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec*phGraphExec;
    rpc_read(client, &phGraphExec, sizeof(phGraphExec));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUDA_GRAPH_INSTANTIATE_PARAMS*instantiateParams;
    rpc_read(client, &instantiateParams, sizeof(instantiateParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphInstantiateWithParams(phGraphExec, hGraph, instantiateParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecGetFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecGetFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    cuuint64_t*flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecGetFlags(hGraphExec, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecKernelNodeSetParams_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecKernelNodeSetParams_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_KERNEL_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecKernelNodeSetParams_v2(hGraphExec, hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecMemcpyNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecMemcpyNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_MEMCPY3D *copyParams;
    rpc_read(client, &copyParams, sizeof(copyParams));
    CUcontext ctx;
    rpc_read(client, &ctx, sizeof(ctx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecMemsetNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecMemsetNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_MEMSET_NODE_PARAMS *memsetParams;
    rpc_read(client, &memsetParams, sizeof(memsetParams));
    CUcontext ctx;
    rpc_read(client, &ctx, sizeof(ctx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecHostNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecHostNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_HOST_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecHostNodeSetParams(hGraphExec, hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecChildGraphNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecChildGraphNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUgraph childGraph;
    rpc_read(client, &childGraph, sizeof(childGraph));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecChildGraphNodeSetParams(hGraphExec, hNode, childGraph);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecEventRecordNodeSetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecEventRecordNodeSetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUevent event;
    rpc_read(client, &event, sizeof(event));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecEventWaitNodeSetEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecEventWaitNodeSetEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUevent event;
    rpc_read(client, &event, sizeof(event));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecExternalSemaphoresSignalNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecExternalSemaphoresSignalNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecExternalSemaphoresWaitNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecExternalSemaphoresWaitNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphNodeSetEnabled(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphNodeSetEnabled called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    unsigned int isEnabled;
    rpc_read(client, &isEnabled, sizeof(isEnabled));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphNodeSetEnabled(hGraphExec, hNode, isEnabled);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphNodeGetEnabled(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphNodeGetEnabled called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    unsigned int*isEnabled;
    rpc_read(client, &isEnabled, sizeof(isEnabled));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphNodeGetEnabled(hGraphExec, hNode, isEnabled);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphUpload(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphUpload called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphUpload(hGraphExec, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphLaunch(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphLaunch called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphLaunch(hGraphExec, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecDestroy(hGraphExec);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphDestroy(hGraph);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecUpdate_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecUpdate_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphExecUpdateResultInfo*resultInfo;
    rpc_read(client, &resultInfo, sizeof(resultInfo));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecUpdate_v2(hGraphExec, hGraph, resultInfo);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphKernelNodeCopyAttributes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphKernelNodeCopyAttributes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode dst;
    rpc_read(client, &dst, sizeof(dst));
    CUgraphNode src;
    rpc_read(client, &src, sizeof(src));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphKernelNodeCopyAttributes(dst, src);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphKernelNodeGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphKernelNodeGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUkernelNodeAttrID attr;
    rpc_read(client, &attr, sizeof(attr));
    CUkernelNodeAttrValue*value_out;
    rpc_read(client, &value_out, sizeof(value_out));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphKernelNodeGetAttribute(hNode, attr, value_out);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphKernelNodeSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphKernelNodeSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUkernelNodeAttrID attr;
    rpc_read(client, &attr, sizeof(attr));
    CUkernelNodeAttrValue *value;
    rpc_read(client, &value, sizeof(value));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphKernelNodeSetAttribute(hNode, attr, value);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphDebugDotPrint(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphDebugDotPrint called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    char *path = nullptr;
    rpc_read(client, &path, 0, true);
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(path);
    _result = cuGraphDebugDotPrint(hGraph, path, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuUserObjectCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuUserObjectCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUuserObject*object_out;
    rpc_read(client, &object_out, sizeof(object_out));
    void *ptr;
    rpc_read(client, &ptr, sizeof(ptr));
    CUhostFn destroy;
    rpc_read(client, &destroy, sizeof(destroy));
    unsigned int initialRefcount;
    rpc_read(client, &initialRefcount, sizeof(initialRefcount));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuUserObjectRetain(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuUserObjectRetain called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUuserObject object;
    rpc_read(client, &object, sizeof(object));
    unsigned int count;
    rpc_read(client, &count, sizeof(count));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuUserObjectRetain(object, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuUserObjectRelease(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuUserObjectRelease called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUuserObject object;
    rpc_read(client, &object, sizeof(object));
    unsigned int count;
    rpc_read(client, &count, sizeof(count));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuUserObjectRelease(object, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphRetainUserObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphRetainUserObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph graph;
    rpc_read(client, &graph, sizeof(graph));
    CUuserObject object;
    rpc_read(client, &object, sizeof(object));
    unsigned int count;
    rpc_read(client, &count, sizeof(count));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphRetainUserObject(graph, object, count, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphReleaseUserObject(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphReleaseUserObject called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraph graph;
    rpc_read(client, &graph, sizeof(graph));
    CUuserObject object;
    rpc_read(client, &object, sizeof(object));
    unsigned int count;
    rpc_read(client, &count, sizeof(count));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphReleaseUserObject(graph, object, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUgraphNodeParams*nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddNode_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddNode_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode*phGraphNode;
    rpc_read(client, &phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUgraphNode *dependencies;
    rpc_read(client, &dependencies, sizeof(dependencies));
    CUgraphEdgeData *dependencyData;
    rpc_read(client, &dependencyData, sizeof(dependencyData));
    size_t numDependencies;
    rpc_read(client, &numDependencies, sizeof(numDependencies));
    CUgraphNodeParams*nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddNode_v2(phGraphNode, hGraph, dependencies, dependencyData, numDependencies, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUgraphNodeParams*nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphNodeSetParams(hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphExec hGraphExec;
    rpc_read(client, &hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    rpc_read(client, &hNode, sizeof(hNode));
    CUgraphNodeParams*nodeParams;
    rpc_read(client, &nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecNodeSetParams(hGraphExec, hNode, nodeParams);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphConditionalHandleCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphConditionalHandleCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphConditionalHandle*pHandle_out;
    rpc_read(client, &pHandle_out, sizeof(pHandle_out));
    CUgraph hGraph;
    rpc_read(client, &hGraph, sizeof(hGraph));
    CUcontext ctx;
    rpc_read(client, &ctx, sizeof(ctx));
    unsigned int defaultLaunchValue;
    rpc_read(client, &defaultLaunchValue, sizeof(defaultLaunchValue));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphConditionalHandleCreate(pHandle_out, hGraph, ctx, defaultLaunchValue, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuOccupancyMaxActiveBlocksPerMultiprocessor(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuOccupancyMaxActiveBlocksPerMultiprocessor called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*numBlocks;
    rpc_read(client, &numBlocks, sizeof(numBlocks));
    CUfunction func;
    rpc_read(client, &func, sizeof(func));
    int blockSize;
    rpc_read(client, &blockSize, sizeof(blockSize));
    size_t dynamicSMemSize;
    rpc_read(client, &dynamicSMemSize, sizeof(dynamicSMemSize));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*numBlocks;
    rpc_read(client, &numBlocks, sizeof(numBlocks));
    CUfunction func;
    rpc_read(client, &func, sizeof(func));
    int blockSize;
    rpc_read(client, &blockSize, sizeof(blockSize));
    size_t dynamicSMemSize;
    rpc_read(client, &dynamicSMemSize, sizeof(dynamicSMemSize));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuOccupancyMaxPotentialBlockSize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuOccupancyMaxPotentialBlockSize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*minGridSize;
    rpc_read(client, &minGridSize, sizeof(minGridSize));
    int*blockSize;
    rpc_read(client, &blockSize, sizeof(blockSize));
    CUfunction func;
    rpc_read(client, &func, sizeof(func));
    CUoccupancyB2DSize blockSizeToDynamicSMemSize;
    rpc_read(client, &blockSizeToDynamicSMemSize, sizeof(blockSizeToDynamicSMemSize));
    size_t dynamicSMemSize;
    rpc_read(client, &dynamicSMemSize, sizeof(dynamicSMemSize));
    int blockSizeLimit;
    rpc_read(client, &blockSizeLimit, sizeof(blockSizeLimit));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuOccupancyMaxPotentialBlockSizeWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuOccupancyMaxPotentialBlockSizeWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*minGridSize;
    rpc_read(client, &minGridSize, sizeof(minGridSize));
    int*blockSize;
    rpc_read(client, &blockSize, sizeof(blockSize));
    CUfunction func;
    rpc_read(client, &func, sizeof(func));
    CUoccupancyB2DSize blockSizeToDynamicSMemSize;
    rpc_read(client, &blockSizeToDynamicSMemSize, sizeof(blockSizeToDynamicSMemSize));
    size_t dynamicSMemSize;
    rpc_read(client, &dynamicSMemSize, sizeof(dynamicSMemSize));
    int blockSizeLimit;
    rpc_read(client, &blockSizeLimit, sizeof(blockSizeLimit));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuOccupancyAvailableDynamicSMemPerBlock(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuOccupancyAvailableDynamicSMemPerBlock called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t*dynamicSmemSize;
    rpc_read(client, &dynamicSmemSize, sizeof(dynamicSmemSize));
    CUfunction func;
    rpc_read(client, &func, sizeof(func));
    int numBlocks;
    rpc_read(client, &numBlocks, sizeof(numBlocks));
    int blockSize;
    rpc_read(client, &blockSize, sizeof(blockSize));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuOccupancyMaxPotentialClusterSize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuOccupancyMaxPotentialClusterSize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*clusterSize;
    rpc_read(client, &clusterSize, sizeof(clusterSize));
    CUfunction func;
    rpc_read(client, &func, sizeof(func));
    CUlaunchConfig *config;
    rpc_read(client, &config, sizeof(config));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuOccupancyMaxPotentialClusterSize(clusterSize, func, config);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuOccupancyMaxActiveClusters(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuOccupancyMaxActiveClusters called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*numClusters;
    rpc_read(client, &numClusters, sizeof(numClusters));
    CUfunction func;
    rpc_read(client, &func, sizeof(func));
    CUlaunchConfig *config;
    rpc_read(client, &config, sizeof(config));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuOccupancyMaxActiveClusters(numClusters, func, config);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefSetArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefSetArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUarray hArray;
    rpc_read(client, &hArray, sizeof(hArray));
    unsigned int Flags;
    rpc_read(client, &Flags, sizeof(Flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetArray(hTexRef, hArray, Flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefSetMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefSetMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUmipmappedArray hMipmappedArray;
    rpc_read(client, &hMipmappedArray, sizeof(hMipmappedArray));
    unsigned int Flags;
    rpc_read(client, &Flags, sizeof(Flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefSetAddress_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefSetAddress_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t*ByteOffset;
    rpc_read(client, &ByteOffset, sizeof(ByteOffset));
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUdeviceptr dptr;
    rpc_read(client, &dptr, sizeof(dptr));
    size_t bytes;
    rpc_read(client, &bytes, sizeof(bytes));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefSetAddress2D_v3(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefSetAddress2D_v3 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUDA_ARRAY_DESCRIPTOR *desc;
    rpc_read(client, &desc, sizeof(desc));
    CUdeviceptr dptr;
    rpc_read(client, &dptr, sizeof(dptr));
    size_t Pitch;
    rpc_read(client, &Pitch, sizeof(Pitch));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefSetFormat(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefSetFormat called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUarray_format fmt;
    rpc_read(client, &fmt, sizeof(fmt));
    int NumPackedComponents;
    rpc_read(client, &NumPackedComponents, sizeof(NumPackedComponents));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefSetAddressMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefSetAddressMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    int dim;
    rpc_read(client, &dim, sizeof(dim));
    CUaddress_mode am;
    rpc_read(client, &am, sizeof(am));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetAddressMode(hTexRef, dim, am);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefSetFilterMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefSetFilterMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUfilter_mode fm;
    rpc_read(client, &fm, sizeof(fm));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetFilterMode(hTexRef, fm);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefSetMipmapFilterMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefSetMipmapFilterMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUfilter_mode fm;
    rpc_read(client, &fm, sizeof(fm));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetMipmapFilterMode(hTexRef, fm);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefSetMipmapLevelBias(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefSetMipmapLevelBias called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    float bias;
    rpc_read(client, &bias, sizeof(bias));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetMipmapLevelBias(hTexRef, bias);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefSetMipmapLevelClamp(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefSetMipmapLevelClamp called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    float minMipmapLevelClamp;
    rpc_read(client, &minMipmapLevelClamp, sizeof(minMipmapLevelClamp));
    float maxMipmapLevelClamp;
    rpc_read(client, &maxMipmapLevelClamp, sizeof(maxMipmapLevelClamp));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefSetMaxAnisotropy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefSetMaxAnisotropy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    unsigned int maxAniso;
    rpc_read(client, &maxAniso, sizeof(maxAniso));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetMaxAnisotropy(hTexRef, maxAniso);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefSetBorderColor(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefSetBorderColor called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    float*pBorderColor;
    rpc_read(client, &pBorderColor, sizeof(pBorderColor));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetBorderColor(hTexRef, pBorderColor);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefSetFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefSetFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    unsigned int Flags;
    rpc_read(client, &Flags, sizeof(Flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetFlags(hTexRef, Flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefGetArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefGetArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarray*phArray;
    rpc_read(client, &phArray, sizeof(phArray));
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetArray(phArray, hTexRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefGetMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefGetMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmipmappedArray*phMipmappedArray;
    rpc_read(client, &phMipmappedArray, sizeof(phMipmappedArray));
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetMipmappedArray(phMipmappedArray, hTexRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefGetAddressMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefGetAddressMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUaddress_mode*pam;
    rpc_read(client, &pam, sizeof(pam));
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    int dim;
    rpc_read(client, &dim, sizeof(dim));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetAddressMode(pam, hTexRef, dim);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefGetFilterMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefGetFilterMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfilter_mode*pfm;
    rpc_read(client, &pfm, sizeof(pfm));
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetFilterMode(pfm, hTexRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefGetFormat(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefGetFormat called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarray_format*pFormat;
    rpc_read(client, &pFormat, sizeof(pFormat));
    int*pNumChannels;
    rpc_read(client, &pNumChannels, sizeof(pNumChannels));
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetFormat(pFormat, pNumChannels, hTexRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefGetMipmapFilterMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefGetMipmapFilterMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUfilter_mode*pfm;
    rpc_read(client, &pfm, sizeof(pfm));
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetMipmapFilterMode(pfm, hTexRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefGetMipmapLevelBias(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefGetMipmapLevelBias called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    float*pbias;
    rpc_read(client, &pbias, sizeof(pbias));
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetMipmapLevelBias(pbias, hTexRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefGetMipmapLevelClamp(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefGetMipmapLevelClamp called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    float*pminMipmapLevelClamp;
    rpc_read(client, &pminMipmapLevelClamp, sizeof(pminMipmapLevelClamp));
    float*pmaxMipmapLevelClamp;
    rpc_read(client, &pmaxMipmapLevelClamp, sizeof(pmaxMipmapLevelClamp));
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefGetMaxAnisotropy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefGetMaxAnisotropy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*pmaxAniso;
    rpc_read(client, &pmaxAniso, sizeof(pmaxAniso));
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefGetBorderColor(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefGetBorderColor called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    float*pBorderColor;
    rpc_read(client, &pBorderColor, sizeof(pBorderColor));
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetBorderColor(pBorderColor, hTexRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefGetFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefGetFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int*pFlags;
    rpc_read(client, &pFlags, sizeof(pFlags));
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetFlags(pFlags, hTexRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref*pTexRef;
    rpc_read(client, &pTexRef, sizeof(pTexRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefCreate(pTexRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexRefDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexRefDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexref hTexRef;
    rpc_read(client, &hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefDestroy(hTexRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuSurfRefSetArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuSurfRefSetArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUsurfref hSurfRef;
    rpc_read(client, &hSurfRef, sizeof(hSurfRef));
    CUarray hArray;
    rpc_read(client, &hArray, sizeof(hArray));
    unsigned int Flags;
    rpc_read(client, &Flags, sizeof(Flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuSurfRefSetArray(hSurfRef, hArray, Flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuSurfRefGetArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuSurfRefGetArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarray*phArray;
    rpc_read(client, &phArray, sizeof(phArray));
    CUsurfref hSurfRef;
    rpc_read(client, &hSurfRef, sizeof(hSurfRef));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuSurfRefGetArray(phArray, hSurfRef);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexObjectCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexObjectCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexObject*pTexObject;
    rpc_read(client, &pTexObject, sizeof(pTexObject));
    CUDA_RESOURCE_DESC *pResDesc;
    rpc_read(client, &pResDesc, sizeof(pResDesc));
    CUDA_TEXTURE_DESC *pTexDesc;
    rpc_read(client, &pTexDesc, sizeof(pTexDesc));
    CUDA_RESOURCE_VIEW_DESC *pResViewDesc;
    rpc_read(client, &pResViewDesc, sizeof(pResViewDesc));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexObjectDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexObjectDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtexObject texObject;
    rpc_read(client, &texObject, sizeof(texObject));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexObjectDestroy(texObject);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexObjectGetResourceDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexObjectGetResourceDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_RESOURCE_DESC*pResDesc;
    rpc_read(client, &pResDesc, sizeof(pResDesc));
    CUtexObject texObject;
    rpc_read(client, &texObject, sizeof(texObject));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexObjectGetResourceDesc(pResDesc, texObject);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexObjectGetTextureDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexObjectGetTextureDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_TEXTURE_DESC*pTexDesc;
    rpc_read(client, &pTexDesc, sizeof(pTexDesc));
    CUtexObject texObject;
    rpc_read(client, &texObject, sizeof(texObject));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexObjectGetTextureDesc(pTexDesc, texObject);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTexObjectGetResourceViewDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTexObjectGetResourceViewDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_RESOURCE_VIEW_DESC*pResViewDesc;
    rpc_read(client, &pResViewDesc, sizeof(pResViewDesc));
    CUtexObject texObject;
    rpc_read(client, &texObject, sizeof(texObject));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexObjectGetResourceViewDesc(pResViewDesc, texObject);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuSurfObjectCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuSurfObjectCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUsurfObject*pSurfObject;
    rpc_read(client, &pSurfObject, sizeof(pSurfObject));
    CUDA_RESOURCE_DESC *pResDesc;
    rpc_read(client, &pResDesc, sizeof(pResDesc));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuSurfObjectCreate(pSurfObject, pResDesc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuSurfObjectDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuSurfObjectDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUsurfObject surfObject;
    rpc_read(client, &surfObject, sizeof(surfObject));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuSurfObjectDestroy(surfObject);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuSurfObjectGetResourceDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuSurfObjectGetResourceDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUDA_RESOURCE_DESC*pResDesc;
    rpc_read(client, &pResDesc, sizeof(pResDesc));
    CUsurfObject surfObject;
    rpc_read(client, &surfObject, sizeof(surfObject));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuSurfObjectGetResourceDesc(pResDesc, surfObject);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTensorMapEncodeTiled(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTensorMapEncodeTiled called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtensorMap*tensorMap;
    rpc_read(client, &tensorMap, sizeof(tensorMap));
    CUtensorMapDataType tensorDataType;
    rpc_read(client, &tensorDataType, sizeof(tensorDataType));
    cuuint32_t tensorRank;
    rpc_read(client, &tensorRank, sizeof(tensorRank));
    void *globalAddress;
    rpc_read(client, &globalAddress, sizeof(globalAddress));
    cuuint64_t *globalDim;
    rpc_read(client, &globalDim, sizeof(globalDim));
    cuuint64_t *globalStrides;
    rpc_read(client, &globalStrides, sizeof(globalStrides));
    cuuint32_t *boxDim;
    rpc_read(client, &boxDim, sizeof(boxDim));
    cuuint32_t *elementStrides;
    rpc_read(client, &elementStrides, sizeof(elementStrides));
    CUtensorMapInterleave interleave;
    rpc_read(client, &interleave, sizeof(interleave));
    CUtensorMapSwizzle swizzle;
    rpc_read(client, &swizzle, sizeof(swizzle));
    CUtensorMapL2promotion l2Promotion;
    rpc_read(client, &l2Promotion, sizeof(l2Promotion));
    CUtensorMapFloatOOBfill oobFill;
    rpc_read(client, &oobFill, sizeof(oobFill));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTensorMapEncodeTiled(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTensorMapEncodeIm2col(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTensorMapEncodeIm2col called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtensorMap*tensorMap;
    rpc_read(client, &tensorMap, sizeof(tensorMap));
    CUtensorMapDataType tensorDataType;
    rpc_read(client, &tensorDataType, sizeof(tensorDataType));
    cuuint32_t tensorRank;
    rpc_read(client, &tensorRank, sizeof(tensorRank));
    void *globalAddress;
    rpc_read(client, &globalAddress, sizeof(globalAddress));
    cuuint64_t *globalDim;
    rpc_read(client, &globalDim, sizeof(globalDim));
    cuuint64_t *globalStrides;
    rpc_read(client, &globalStrides, sizeof(globalStrides));
    int *pixelBoxLowerCorner;
    rpc_read(client, &pixelBoxLowerCorner, sizeof(pixelBoxLowerCorner));
    int *pixelBoxUpperCorner;
    rpc_read(client, &pixelBoxUpperCorner, sizeof(pixelBoxUpperCorner));
    cuuint32_t channelsPerPixel;
    rpc_read(client, &channelsPerPixel, sizeof(channelsPerPixel));
    cuuint32_t pixelsPerColumn;
    rpc_read(client, &pixelsPerColumn, sizeof(pixelsPerColumn));
    cuuint32_t *elementStrides;
    rpc_read(client, &elementStrides, sizeof(elementStrides));
    CUtensorMapInterleave interleave;
    rpc_read(client, &interleave, sizeof(interleave));
    CUtensorMapSwizzle swizzle;
    rpc_read(client, &swizzle, sizeof(swizzle));
    CUtensorMapL2promotion l2Promotion;
    rpc_read(client, &l2Promotion, sizeof(l2Promotion));
    CUtensorMapFloatOOBfill oobFill;
    rpc_read(client, &oobFill, sizeof(oobFill));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTensorMapEncodeIm2col(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion, oobFill);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTensorMapEncodeIm2colWide(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTensorMapEncodeIm2colWide called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtensorMap*tensorMap;
    rpc_read(client, &tensorMap, sizeof(tensorMap));
    CUtensorMapDataType tensorDataType;
    rpc_read(client, &tensorDataType, sizeof(tensorDataType));
    cuuint32_t tensorRank;
    rpc_read(client, &tensorRank, sizeof(tensorRank));
    void *globalAddress;
    rpc_read(client, &globalAddress, sizeof(globalAddress));
    cuuint64_t *globalDim;
    rpc_read(client, &globalDim, sizeof(globalDim));
    cuuint64_t *globalStrides;
    rpc_read(client, &globalStrides, sizeof(globalStrides));
    int pixelBoxLowerCornerWidth;
    rpc_read(client, &pixelBoxLowerCornerWidth, sizeof(pixelBoxLowerCornerWidth));
    int pixelBoxUpperCornerWidth;
    rpc_read(client, &pixelBoxUpperCornerWidth, sizeof(pixelBoxUpperCornerWidth));
    cuuint32_t channelsPerPixel;
    rpc_read(client, &channelsPerPixel, sizeof(channelsPerPixel));
    cuuint32_t pixelsPerColumn;
    rpc_read(client, &pixelsPerColumn, sizeof(pixelsPerColumn));
    cuuint32_t *elementStrides;
    rpc_read(client, &elementStrides, sizeof(elementStrides));
    CUtensorMapInterleave interleave;
    rpc_read(client, &interleave, sizeof(interleave));
    CUtensorMapIm2ColWideMode mode;
    rpc_read(client, &mode, sizeof(mode));
    CUtensorMapSwizzle swizzle;
    rpc_read(client, &swizzle, sizeof(swizzle));
    CUtensorMapL2promotion l2Promotion;
    rpc_read(client, &l2Promotion, sizeof(l2Promotion));
    CUtensorMapFloatOOBfill oobFill;
    rpc_read(client, &oobFill, sizeof(oobFill));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTensorMapEncodeIm2colWide(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCornerWidth, pixelBoxUpperCornerWidth, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, mode, swizzle, l2Promotion, oobFill);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuTensorMapReplaceAddress(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuTensorMapReplaceAddress called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUtensorMap*tensorMap;
    rpc_read(client, &tensorMap, sizeof(tensorMap));
    void *globalAddress;
    rpc_read(client, &globalAddress, sizeof(globalAddress));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTensorMapReplaceAddress(tensorMap, globalAddress);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceCanAccessPeer(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceCanAccessPeer called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*canAccessPeer;
    rpc_read(client, &canAccessPeer, sizeof(canAccessPeer));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    CUdevice peerDev;
    rpc_read(client, &peerDev, sizeof(peerDev));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceCanAccessPeer(canAccessPeer, dev, peerDev);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxEnablePeerAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxEnablePeerAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext peerContext;
    rpc_read(client, &peerContext, sizeof(peerContext));
    unsigned int Flags;
    rpc_read(client, &Flags, sizeof(Flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxEnablePeerAccess(peerContext, Flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxDisablePeerAccess(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxDisablePeerAccess called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext peerContext;
    rpc_read(client, &peerContext, sizeof(peerContext));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxDisablePeerAccess(peerContext);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetP2PAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetP2PAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*value;
    rpc_read(client, &value, sizeof(value));
    CUdevice_P2PAttribute attrib;
    rpc_read(client, &attrib, sizeof(attrib));
    CUdevice srcDevice;
    rpc_read(client, &srcDevice, sizeof(srcDevice));
    CUdevice dstDevice;
    rpc_read(client, &dstDevice, sizeof(dstDevice));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphicsUnregisterResource(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphicsUnregisterResource called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphicsResource resource;
    rpc_read(client, &resource, sizeof(resource));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphicsUnregisterResource(resource);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphicsSubResourceGetMappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphicsSubResourceGetMappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUarray*pArray;
    rpc_read(client, &pArray, sizeof(pArray));
    CUgraphicsResource resource;
    rpc_read(client, &resource, sizeof(resource));
    unsigned int arrayIndex;
    rpc_read(client, &arrayIndex, sizeof(arrayIndex));
    unsigned int mipLevel;
    rpc_read(client, &mipLevel, sizeof(mipLevel));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphicsResourceGetMappedMipmappedArray(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphicsResourceGetMappedMipmappedArray called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUmipmappedArray*pMipmappedArray;
    rpc_read(client, &pMipmappedArray, sizeof(pMipmappedArray));
    CUgraphicsResource resource;
    rpc_read(client, &resource, sizeof(resource));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphicsResourceSetMapFlags_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphicsResourceSetMapFlags_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgraphicsResource resource;
    rpc_read(client, &resource, sizeof(resource));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphicsResourceSetMapFlags_v2(resource, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphicsMapResources(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphicsMapResources called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int count;
    rpc_read(client, &count, sizeof(count));
    CUgraphicsResource*resources;
    rpc_read(client, &resources, sizeof(resources));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphicsMapResources(count, resources, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphicsUnmapResources(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphicsUnmapResources called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int count;
    rpc_read(client, &count, sizeof(count));
    CUgraphicsResource*resources;
    rpc_read(client, &resources, sizeof(resources));
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphicsUnmapResources(count, resources, hStream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGetProcAddress_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGetProcAddress_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    char *symbol = nullptr;
    rpc_read(client, &symbol, 0, true);
    // PARAM void **pfn
    void *pfn;
    int cudaVersion;
    rpc_read(client, &cudaVersion, sizeof(cudaVersion));
    cuuint64_t flags;
    rpc_read(client, &flags, sizeof(flags));
    CUdriverProcAddressQueryResult*symbolStatus;
    rpc_read(client, &symbolStatus, sizeof(symbolStatus));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(symbol);
    // PARAM void **pfn
    _result = cuGetProcAddress_v2(symbol, &pfn, cudaVersion, flags, symbolStatus);
    // PARAM void **pfn
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    // PARAM void **pfn
    return rtn;
}

int handle_cuCoredumpGetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCoredumpGetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcoredumpSettings attrib;
    rpc_read(client, &attrib, sizeof(attrib));
    void *value;
    rpc_read(client, &value, sizeof(value));
    size_t*size;
    rpc_read(client, &size, sizeof(size));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCoredumpGetAttribute(attrib, value, size);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCoredumpGetAttributeGlobal(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCoredumpGetAttributeGlobal called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcoredumpSettings attrib;
    rpc_read(client, &attrib, sizeof(attrib));
    void *value;
    rpc_read(client, &value, sizeof(value));
    size_t*size;
    rpc_read(client, &size, sizeof(size));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCoredumpGetAttributeGlobal(attrib, value, size);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCoredumpSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCoredumpSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcoredumpSettings attrib;
    rpc_read(client, &attrib, sizeof(attrib));
    void *value;
    rpc_read(client, &value, sizeof(value));
    size_t*size;
    rpc_read(client, &size, sizeof(size));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCoredumpSetAttribute(attrib, value, size);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCoredumpSetAttributeGlobal(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCoredumpSetAttributeGlobal called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcoredumpSettings attrib;
    rpc_read(client, &attrib, sizeof(attrib));
    void *value;
    rpc_read(client, &value, sizeof(value));
    size_t*size;
    rpc_read(client, &size, sizeof(size));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCoredumpSetAttributeGlobal(attrib, value, size);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGetExportTable(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGetExportTable called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    // PARAM const void **ppExportTable
    const void *ppExportTable;
    CUuuid *pExportTableId;
    rpc_read(client, &pExportTableId, sizeof(pExportTableId));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM const void **ppExportTable
    _result = cuGetExportTable(&ppExportTable, pExportTableId);
    // PARAM const void **ppExportTable
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
    // PARAM const void **ppExportTable
    return rtn;
}

int handle_cuGreenCtxCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGreenCtxCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgreenCtx*phCtx;
    rpc_read(client, &phCtx, sizeof(phCtx));
    CUdevResourceDesc desc;
    rpc_read(client, &desc, sizeof(desc));
    CUdevice dev;
    rpc_read(client, &dev, sizeof(dev));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGreenCtxCreate(phCtx, desc, dev, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGreenCtxDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGreenCtxDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgreenCtx hCtx;
    rpc_read(client, &hCtx, sizeof(hCtx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGreenCtxDestroy(hCtx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxFromGreenCtx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxFromGreenCtx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext*pContext;
    rpc_read(client, &pContext, sizeof(pContext));
    CUgreenCtx hCtx;
    rpc_read(client, &hCtx, sizeof(hCtx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxFromGreenCtx(pContext, hCtx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDeviceGetDevResource(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetDevResource called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevice device;
    rpc_read(client, &device, sizeof(device));
    CUdevResource*resource;
    rpc_read(client, &resource, sizeof(resource));
    CUdevResourceType type;
    rpc_read(client, &type, sizeof(type));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetDevResource(device, resource, type);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCtxGetDevResource(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCtxGetDevResource called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUcontext hCtx;
    rpc_read(client, &hCtx, sizeof(hCtx));
    CUdevResource*resource;
    rpc_read(client, &resource, sizeof(resource));
    CUdevResourceType type;
    rpc_read(client, &type, sizeof(type));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetDevResource(hCtx, resource, type);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGreenCtxGetDevResource(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGreenCtxGetDevResource called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgreenCtx hCtx;
    rpc_read(client, &hCtx, sizeof(hCtx));
    CUdevResource*resource;
    rpc_read(client, &resource, sizeof(resource));
    CUdevResourceType type;
    rpc_read(client, &type, sizeof(type));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGreenCtxGetDevResource(hCtx, resource, type);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDevSmResourceSplitByCount(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDevSmResourceSplitByCount called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevResource*result;
    rpc_read(client, &result, sizeof(result));
    unsigned int*nbGroups;
    rpc_read(client, &nbGroups, sizeof(nbGroups));
    CUdevResource *input;
    rpc_read(client, &input, sizeof(input));
    CUdevResource*remaining;
    rpc_read(client, &remaining, sizeof(remaining));
    unsigned int useFlags;
    rpc_read(client, &useFlags, sizeof(useFlags));
    unsigned int minCount;
    rpc_read(client, &minCount, sizeof(minCount));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDevSmResourceSplitByCount(result, nbGroups, input, remaining, useFlags, minCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuDevResourceGenerateDesc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDevResourceGenerateDesc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUdevResourceDesc*phDesc;
    rpc_read(client, &phDesc, sizeof(phDesc));
    CUdevResource*resources;
    rpc_read(client, &resources, sizeof(resources));
    unsigned int nbResources;
    rpc_read(client, &nbResources, sizeof(nbResources));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDevResourceGenerateDesc(phDesc, resources, nbResources);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGreenCtxRecordEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGreenCtxRecordEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgreenCtx hCtx;
    rpc_read(client, &hCtx, sizeof(hCtx));
    CUevent hEvent;
    rpc_read(client, &hEvent, sizeof(hEvent));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGreenCtxRecordEvent(hCtx, hEvent);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGreenCtxWaitEvent(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGreenCtxWaitEvent called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUgreenCtx hCtx;
    rpc_read(client, &hCtx, sizeof(hCtx));
    CUevent hEvent;
    rpc_read(client, &hEvent, sizeof(hEvent));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGreenCtxWaitEvent(hCtx, hEvent);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamGetGreenCtx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamGetGreenCtx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream hStream;
    rpc_read(client, &hStream, sizeof(hStream));
    CUgreenCtx*phCtx;
    rpc_read(client, &phCtx, sizeof(phCtx));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetGreenCtx(hStream, phCtx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGreenCtxStreamCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGreenCtxStreamCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    CUstream*phStream;
    rpc_read(client, &phStream, sizeof(phStream));
    CUgreenCtx greenCtx;
    rpc_read(client, &greenCtx, sizeof(greenCtx));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    int priority;
    rpc_read(client, &priority, sizeof(priority));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGreenCtxStreamCreate(phStream, greenCtx, flags, priority);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCheckpointProcessGetRestoreThreadId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCheckpointProcessGetRestoreThreadId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int pid;
    rpc_read(client, &pid, sizeof(pid));
    int*tid;
    rpc_read(client, &tid, sizeof(tid));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCheckpointProcessGetRestoreThreadId(pid, tid);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCheckpointProcessGetState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCheckpointProcessGetState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int pid;
    rpc_read(client, &pid, sizeof(pid));
    CUprocessState*state;
    rpc_read(client, &state, sizeof(state));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCheckpointProcessGetState(pid, state);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCheckpointProcessLock(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCheckpointProcessLock called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int pid;
    rpc_read(client, &pid, sizeof(pid));
    CUcheckpointLockArgs*args;
    rpc_read(client, &args, sizeof(args));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCheckpointProcessLock(pid, args);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCheckpointProcessCheckpoint(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCheckpointProcessCheckpoint called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int pid;
    rpc_read(client, &pid, sizeof(pid));
    CUcheckpointCheckpointArgs*args;
    rpc_read(client, &args, sizeof(args));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCheckpointProcessCheckpoint(pid, args);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCheckpointProcessRestore(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCheckpointProcessRestore called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int pid;
    rpc_read(client, &pid, sizeof(pid));
    CUcheckpointRestoreArgs*args;
    rpc_read(client, &args, sizeof(args));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCheckpointProcessRestore(pid, args);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuCheckpointProcessUnlock(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuCheckpointProcessUnlock called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int pid;
    rpc_read(client, &pid, sizeof(pid));
    CUcheckpointUnlockArgs*args;
    rpc_read(client, &args, sizeof(args));
    CUresult _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCheckpointProcessUnlock(pid, args);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

