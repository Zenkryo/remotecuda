#include <iostream>
#include <map>
#include <string.h>
#include "hook_api.h"
#include "handle_server.h"
#include "rpc/rpc_core.h"
#include "cuda.h"

using namespace rpc;
int handle_cuInit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuInit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    unsigned int Flags;
    conn->read(&Flags, sizeof(Flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuInit(Flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *driverVersion;
    conn->read(&driverVersion, sizeof(driverVersion));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDriverGetVersion(driverVersion);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdevice *device;
    conn->read(&device, sizeof(device));
    int ordinal;
    conn->read(&ordinal, sizeof(ordinal));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGet(device, ordinal);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *count;
    conn->read(&count, sizeof(count));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetCount(count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    char name[1024];
    int len;
    conn->read(&len, sizeof(len));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetName(name, len, dev);
    if(len > 0) {
        conn->write(name, strlen(name) + 1, true);
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

int handle_cuDeviceGetUuid(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceGetUuid called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUuuid *uuid;
    conn->read(&uuid, sizeof(uuid));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetUuid(uuid, dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUuuid *uuid;
    conn->read(&uuid, sizeof(uuid));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetUuid_v2(uuid, dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    char luid[1024];
    unsigned int *deviceNodeMask;
    conn->read(&deviceNodeMask, sizeof(deviceNodeMask));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetLuid(luid, deviceNodeMask, dev);
    if(32 > 0) {
        conn->write(luid, strlen(luid) + 1, true);
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

int handle_cuDeviceTotalMem_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuDeviceTotalMem_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    size_t *bytes;
    conn->read(&bytes, sizeof(bytes));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceTotalMem_v2(bytes, dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    size_t *maxWidthInElements;
    conn->read(&maxWidthInElements, sizeof(maxWidthInElements));
    CUarray_format format;
    conn->read(&format, sizeof(format));
    unsigned numChannels;
    conn->read(&numChannels, sizeof(numChannels));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, format, numChannels, dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *pi;
    conn->read(&pi, sizeof(pi));
    CUdevice_attribute attrib;
    conn->read(&attrib, sizeof(attrib));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetAttribute(pi, attrib, dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    void *nvSciSyncAttrList;
    conn->read(&nvSciSyncAttrList, sizeof(nvSciSyncAttrList));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, dev, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUmemoryPool pool;
    conn->read(&pool, sizeof(pool));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceSetMemPool(dev, pool);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemoryPool *pool;
    conn->read(&pool, sizeof(pool));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetMemPool(pool, dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemoryPool *pool_out;
    conn->read(&pool_out, sizeof(pool_out));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetDefaultMemPool(pool_out, dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdevprop *prop;
    conn->read(&prop, sizeof(prop));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetProperties(prop, dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *major;
    conn->read(&major, sizeof(major));
    int *minor;
    conn->read(&minor, sizeof(minor));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceComputeCapability(major, minor, dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUcontext *pctx;
    conn->read(&pctx, sizeof(pctx));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDevicePrimaryCtxRetain(pctx, dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDevicePrimaryCtxRelease_v2(dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDevicePrimaryCtxSetFlags_v2(dev, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    unsigned int *flags;
    conn->read(&flags, sizeof(flags));
    int *active;
    conn->read(&active, sizeof(active));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDevicePrimaryCtxGetState(dev, flags, active);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDevicePrimaryCtxReset_v2(dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *pi;
    conn->read(&pi, sizeof(pi));
    CUexecAffinityType type;
    conn->read(&type, sizeof(type));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetExecAffinitySupport(pi, type, dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUcontext *pctx;
    conn->read(&pctx, sizeof(pctx));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxCreate_v2(pctx, flags, dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUcontext *pctx;
    conn->read(&pctx, sizeof(pctx));
    CUexecAffinityParam *paramsArray;
    conn->read(&paramsArray, sizeof(paramsArray));
    int numParams;
    conn->read(&numParams, sizeof(numParams));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxCreate_v3(pctx, paramsArray, numParams, flags, dev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUcontext ctx;
    conn->read(&ctx, sizeof(ctx));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxDestroy_v2(ctx);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUcontext ctx;
    conn->read(&ctx, sizeof(ctx));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxPushCurrent_v2(ctx);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUcontext *pctx;
    conn->read(&pctx, sizeof(pctx));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxPopCurrent_v2(pctx);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUcontext ctx;
    conn->read(&ctx, sizeof(ctx));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxSetCurrent(ctx);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUcontext *pctx;
    conn->read(&pctx, sizeof(pctx));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetCurrent(pctx);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdevice *device;
    conn->read(&device, sizeof(device));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetDevice(device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int *flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetFlags(flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxSynchronize();
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUlimit limit;
    conn->read(&limit, sizeof(limit));
    size_t value;
    conn->read(&value, sizeof(value));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxSetLimit(limit, value);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    size_t *pvalue;
    conn->read(&pvalue, sizeof(pvalue));
    CUlimit limit;
    conn->read(&limit, sizeof(limit));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetLimit(pvalue, limit);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunc_cache *pconfig;
    conn->read(&pconfig, sizeof(pconfig));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetCacheConfig(pconfig);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunc_cache config;
    conn->read(&config, sizeof(config));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxSetCacheConfig(config);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUsharedconfig *pConfig;
    conn->read(&pConfig, sizeof(pConfig));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetSharedMemConfig(pConfig);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUsharedconfig config;
    conn->read(&config, sizeof(config));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxSetSharedMemConfig(config);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUcontext ctx;
    conn->read(&ctx, sizeof(ctx));
    unsigned int *version;
    conn->read(&version, sizeof(version));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetApiVersion(ctx, version);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *leastPriority;
    conn->read(&leastPriority, sizeof(leastPriority));
    int *greatestPriority;
    conn->read(&greatestPriority, sizeof(greatestPriority));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetStreamPriorityRange(leastPriority, greatestPriority);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxResetPersistingL2Cache();
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUexecAffinityParam *pExecAffinity;
    conn->read(&pExecAffinity, sizeof(pExecAffinity));
    CUexecAffinityType type;
    conn->read(&type, sizeof(type));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxGetExecAffinity(pExecAffinity, type);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUcontext *pctx;
    conn->read(&pctx, sizeof(pctx));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxAttach(pctx, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUcontext ctx;
    conn->read(&ctx, sizeof(ctx));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxDetach(ctx);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmodule *module;
    conn->read(&module, sizeof(module));
    char *fname = nullptr;
    conn->read(&fname, 0, true);
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(fname);
    _result = cuModuleLoad(module, fname);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmodule *module;
    conn->read(&module, sizeof(module));
    void *image;
    conn->read(&image, sizeof(image));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuModuleLoadData(module, image);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmodule *module;
    conn->read(&module, sizeof(module));
    void *image;
    conn->read(&image, sizeof(image));
    unsigned int numOptions;
    conn->read(&numOptions, sizeof(numOptions));
    CUjit_option *options;
    conn->read(&options, sizeof(options));
    // PARAM void **optionValues
    void *optionValues;
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **optionValues
    _result = cuModuleLoadDataEx(module, image, numOptions, options, &optionValues);
    // PARAM void **optionValues
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    RpcConn *conn = (RpcConn *)args0;
    CUmodule *module;
    conn->read(&module, sizeof(module));
    void *fatCubin;
    conn->read(&fatCubin, sizeof(fatCubin));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuModuleLoadFatBinary(module, fatCubin);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmodule hmod;
    conn->read(&hmod, sizeof(hmod));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuModuleUnload(hmod);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction *hfunc;
    conn->read(&hfunc, sizeof(hfunc));
    CUmodule hmod;
    conn->read(&hmod, sizeof(hmod));
    char *name = nullptr;
    conn->read(&name, 0, true);
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(name);
    _result = cuModuleGetFunction(hfunc, hmod, name);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref *pTexRef;
    conn->read(&pTexRef, sizeof(pTexRef));
    CUmodule hmod;
    conn->read(&hmod, sizeof(hmod));
    char *name = nullptr;
    conn->read(&name, 0, true);
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(name);
    _result = cuModuleGetTexRef(pTexRef, hmod, name);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUsurfref *pSurfRef;
    conn->read(&pSurfRef, sizeof(pSurfRef));
    CUmodule hmod;
    conn->read(&hmod, sizeof(hmod));
    char *name = nullptr;
    conn->read(&name, 0, true);
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(name);
    _result = cuModuleGetSurfRef(pSurfRef, hmod, name);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int numOptions;
    conn->read(&numOptions, sizeof(numOptions));
    CUjit_option *options;
    conn->read(&options, sizeof(options));
    // PARAM void **optionValues
    void *optionValues;
    CUlinkState *stateOut;
    conn->read(&stateOut, sizeof(stateOut));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **optionValues
    _result = cuLinkCreate_v2(numOptions, options, &optionValues, stateOut);
    // PARAM void **optionValues
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    RpcConn *conn = (RpcConn *)args0;
    CUlinkState state;
    conn->read(&state, sizeof(state));
    CUjitInputType type;
    conn->read(&type, sizeof(type));
    void *data;
    conn->read(&data, sizeof(data));
    size_t size;
    conn->read(&size, sizeof(size));
    char *name = nullptr;
    conn->read(&name, 0, true);
    unsigned int numOptions;
    conn->read(&numOptions, sizeof(numOptions));
    CUjit_option *options;
    conn->read(&options, sizeof(options));
    // PARAM void **optionValues
    void *optionValues;
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(name);
    // PARAM void **optionValues
    _result = cuLinkAddData_v2(state, type, data, size, name, numOptions, options, &optionValues);
    // PARAM void **optionValues
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    RpcConn *conn = (RpcConn *)args0;
    CUlinkState state;
    conn->read(&state, sizeof(state));
    CUjitInputType type;
    conn->read(&type, sizeof(type));
    char *path = nullptr;
    conn->read(&path, 0, true);
    unsigned int numOptions;
    conn->read(&numOptions, sizeof(numOptions));
    CUjit_option *options;
    conn->read(&options, sizeof(options));
    // PARAM void **optionValues
    void *optionValues;
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(path);
    // PARAM void **optionValues
    _result = cuLinkAddFile_v2(state, type, path, numOptions, options, &optionValues);
    // PARAM void **optionValues
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    RpcConn *conn = (RpcConn *)args0;
    CUlinkState state;
    conn->read(&state, sizeof(state));
    // PARAM void **cubinOut
    void *cubinOut;
    size_t *sizeOut;
    conn->read(&sizeOut, sizeof(sizeOut));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **cubinOut
    _result = cuLinkComplete(state, &cubinOut, sizeOut);
    // PARAM void **cubinOut
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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
    RpcConn *conn = (RpcConn *)args0;
    CUlinkState state;
    conn->read(&state, sizeof(state));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLinkDestroy(state);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    size_t *free;
    conn->read(&free, sizeof(free));
    size_t *total;
    conn->read(&total, sizeof(total));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemGetInfo_v2(free, total);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dptr;
    conn->read(&dptr, sizeof(dptr));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemFree_v2(dptr);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int *pFlags;
    conn->read(&pFlags, sizeof(pFlags));
    void *p;
    conn->read(&p, sizeof(p));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemHostGetFlags(pFlags, p);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdevice *dev;
    conn->read(&dev, sizeof(dev));
    char *pciBusId = nullptr;
    conn->read(&pciBusId, 0, true);
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(pciBusId);
    _result = cuDeviceGetByPCIBusId(dev, pciBusId);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    char pciBusId[1024];
    int len;
    conn->read(&len, sizeof(len));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetPCIBusId(pciBusId, len, dev);
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

int handle_cuIpcGetEventHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuIpcGetEventHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUipcEventHandle *pHandle;
    conn->read(&pHandle, sizeof(pHandle));
    CUevent event;
    conn->read(&event, sizeof(event));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuIpcGetEventHandle(pHandle, event);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUevent *phEvent;
    conn->read(&phEvent, sizeof(phEvent));
    CUipcEventHandle handle;
    conn->read(&handle, sizeof(handle));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuIpcOpenEventHandle(phEvent, handle);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUipcMemHandle *pHandle;
    conn->read(&pHandle, sizeof(pHandle));
    CUdeviceptr dptr;
    conn->read(&dptr, sizeof(dptr));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuIpcGetMemHandle(pHandle, dptr);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dptr;
    conn->read(&dptr, sizeof(dptr));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuIpcCloseMemHandle(dptr);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    void *p;
    conn->read(&p, sizeof(p));
    size_t bytesize;
    conn->read(&bytesize, sizeof(bytesize));
    unsigned int Flags;
    conn->read(&Flags, sizeof(Flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemHostRegister_v2(p, bytesize, Flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    void *p;
    conn->read(&p, sizeof(p));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemHostUnregister(p);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dst;
    conn->read(&dst, sizeof(dst));
    CUdeviceptr src;
    conn->read(&src, sizeof(src));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy(dst, src, ByteCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    CUcontext dstContext;
    conn->read(&dstContext, sizeof(dstContext));
    CUdeviceptr srcDevice;
    conn->read(&srcDevice, sizeof(srcDevice));
    CUcontext srcContext;
    conn->read(&srcContext, sizeof(srcContext));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    void *srcHost;
    conn->read(&srcHost, sizeof(srcHost));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    void *dstHost;
    conn->read(&dstHost, sizeof(dstHost));
    CUdeviceptr srcDevice;
    conn->read(&srcDevice, sizeof(srcDevice));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    CUdeviceptr srcDevice;
    conn->read(&srcDevice, sizeof(srcDevice));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarray dstArray;
    conn->read(&dstArray, sizeof(dstArray));
    size_t dstOffset;
    conn->read(&dstOffset, sizeof(dstOffset));
    CUdeviceptr srcDevice;
    conn->read(&srcDevice, sizeof(srcDevice));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    CUarray srcArray;
    conn->read(&srcArray, sizeof(srcArray));
    size_t srcOffset;
    conn->read(&srcOffset, sizeof(srcOffset));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarray dstArray;
    conn->read(&dstArray, sizeof(dstArray));
    size_t dstOffset;
    conn->read(&dstOffset, sizeof(dstOffset));
    void *srcHost;
    conn->read(&srcHost, sizeof(srcHost));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    void *dstHost;
    conn->read(&dstHost, sizeof(dstHost));
    CUarray srcArray;
    conn->read(&srcArray, sizeof(srcArray));
    size_t srcOffset;
    conn->read(&srcOffset, sizeof(srcOffset));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarray dstArray;
    conn->read(&dstArray, sizeof(dstArray));
    size_t dstOffset;
    conn->read(&dstOffset, sizeof(dstOffset));
    CUarray srcArray;
    conn->read(&srcArray, sizeof(srcArray));
    size_t srcOffset;
    conn->read(&srcOffset, sizeof(srcOffset));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_MEMCPY2D *pCopy = nullptr;
    conn->read(&pCopy, sizeof(pCopy));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy2D_v2(pCopy);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_MEMCPY2D *pCopy = nullptr;
    conn->read(&pCopy, sizeof(pCopy));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy2DUnaligned_v2(pCopy);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_MEMCPY3D *pCopy = nullptr;
    conn->read(&pCopy, sizeof(pCopy));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy3D_v2(pCopy);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_MEMCPY3D_PEER *pCopy = nullptr;
    conn->read(&pCopy, sizeof(pCopy));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy3DPeer(pCopy);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dst;
    conn->read(&dst, sizeof(dst));
    CUdeviceptr src;
    conn->read(&src, sizeof(src));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyAsync(dst, src, ByteCount, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    CUcontext dstContext;
    conn->read(&dstContext, sizeof(dstContext));
    CUdeviceptr srcDevice;
    conn->read(&srcDevice, sizeof(srcDevice));
    CUcontext srcContext;
    conn->read(&srcContext, sizeof(srcContext));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    void *srcHost;
    conn->read(&srcHost, sizeof(srcHost));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    void *dstHost;
    conn->read(&dstHost, sizeof(dstHost));
    CUdeviceptr srcDevice;
    conn->read(&srcDevice, sizeof(srcDevice));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    CUdeviceptr srcDevice;
    conn->read(&srcDevice, sizeof(srcDevice));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarray dstArray;
    conn->read(&dstArray, sizeof(dstArray));
    size_t dstOffset;
    conn->read(&dstOffset, sizeof(dstOffset));
    void *srcHost;
    conn->read(&srcHost, sizeof(srcHost));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    void *dstHost;
    conn->read(&dstHost, sizeof(dstHost));
    CUarray srcArray;
    conn->read(&srcArray, sizeof(srcArray));
    size_t srcOffset;
    conn->read(&srcOffset, sizeof(srcOffset));
    size_t ByteCount;
    conn->read(&ByteCount, sizeof(ByteCount));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_MEMCPY2D *pCopy = nullptr;
    conn->read(&pCopy, sizeof(pCopy));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy2DAsync_v2(pCopy, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_MEMCPY3D *pCopy = nullptr;
    conn->read(&pCopy, sizeof(pCopy));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy3DAsync_v2(pCopy, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_MEMCPY3D_PEER *pCopy = nullptr;
    conn->read(&pCopy, sizeof(pCopy));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemcpy3DPeerAsync(pCopy, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    unsigned char uc;
    conn->read(&uc, sizeof(uc));
    size_t N;
    conn->read(&N, sizeof(N));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD8_v2(dstDevice, uc, N);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    unsigned short us;
    conn->read(&us, sizeof(us));
    size_t N;
    conn->read(&N, sizeof(N));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD16_v2(dstDevice, us, N);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    unsigned int ui;
    conn->read(&ui, sizeof(ui));
    size_t N;
    conn->read(&N, sizeof(N));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD32_v2(dstDevice, ui, N);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    size_t dstPitch;
    conn->read(&dstPitch, sizeof(dstPitch));
    unsigned char uc;
    conn->read(&uc, sizeof(uc));
    size_t Width;
    conn->read(&Width, sizeof(Width));
    size_t Height;
    conn->read(&Height, sizeof(Height));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    size_t dstPitch;
    conn->read(&dstPitch, sizeof(dstPitch));
    unsigned short us;
    conn->read(&us, sizeof(us));
    size_t Width;
    conn->read(&Width, sizeof(Width));
    size_t Height;
    conn->read(&Height, sizeof(Height));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    size_t dstPitch;
    conn->read(&dstPitch, sizeof(dstPitch));
    unsigned int ui;
    conn->read(&ui, sizeof(ui));
    size_t Width;
    conn->read(&Width, sizeof(Width));
    size_t Height;
    conn->read(&Height, sizeof(Height));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    unsigned char uc;
    conn->read(&uc, sizeof(uc));
    size_t N;
    conn->read(&N, sizeof(N));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD8Async(dstDevice, uc, N, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    unsigned short us;
    conn->read(&us, sizeof(us));
    size_t N;
    conn->read(&N, sizeof(N));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD16Async(dstDevice, us, N, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    unsigned int ui;
    conn->read(&ui, sizeof(ui));
    size_t N;
    conn->read(&N, sizeof(N));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD32Async(dstDevice, ui, N, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    size_t dstPitch;
    conn->read(&dstPitch, sizeof(dstPitch));
    unsigned char uc;
    conn->read(&uc, sizeof(uc));
    size_t Width;
    conn->read(&Width, sizeof(Width));
    size_t Height;
    conn->read(&Height, sizeof(Height));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    size_t dstPitch;
    conn->read(&dstPitch, sizeof(dstPitch));
    unsigned short us;
    conn->read(&us, sizeof(us));
    size_t Width;
    conn->read(&Width, sizeof(Width));
    size_t Height;
    conn->read(&Height, sizeof(Height));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    size_t dstPitch;
    conn->read(&dstPitch, sizeof(dstPitch));
    unsigned int ui;
    conn->read(&ui, sizeof(ui));
    size_t Width;
    conn->read(&Width, sizeof(Width));
    size_t Height;
    conn->read(&Height, sizeof(Height));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarray *pHandle;
    conn->read(&pHandle, sizeof(pHandle));
    CUDA_ARRAY_DESCRIPTOR *pAllocateArray = nullptr;
    conn->read(&pAllocateArray, sizeof(pAllocateArray));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArrayCreate_v2(pHandle, pAllocateArray);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor;
    conn->read(&pArrayDescriptor, sizeof(pArrayDescriptor));
    CUarray hArray;
    conn->read(&hArray, sizeof(hArray));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArrayGetDescriptor_v2(pArrayDescriptor, hArray);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties;
    conn->read(&sparseProperties, sizeof(sparseProperties));
    CUarray array;
    conn->read(&array, sizeof(array));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArrayGetSparseProperties(sparseProperties, array);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties;
    conn->read(&sparseProperties, sizeof(sparseProperties));
    CUmipmappedArray mipmap;
    conn->read(&mipmap, sizeof(mipmap));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMipmappedArrayGetSparseProperties(sparseProperties, mipmap);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarray *pPlaneArray;
    conn->read(&pPlaneArray, sizeof(pPlaneArray));
    CUarray hArray;
    conn->read(&hArray, sizeof(hArray));
    unsigned int planeIdx;
    conn->read(&planeIdx, sizeof(planeIdx));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArrayGetPlane(pPlaneArray, hArray, planeIdx);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarray hArray;
    conn->read(&hArray, sizeof(hArray));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArrayDestroy(hArray);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarray *pHandle;
    conn->read(&pHandle, sizeof(pHandle));
    CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray = nullptr;
    conn->read(&pAllocateArray, sizeof(pAllocateArray));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArray3DCreate_v2(pHandle, pAllocateArray);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor;
    conn->read(&pArrayDescriptor, sizeof(pArrayDescriptor));
    CUarray hArray;
    conn->read(&hArray, sizeof(hArray));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmipmappedArray *pHandle;
    conn->read(&pHandle, sizeof(pHandle));
    CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc = nullptr;
    conn->read(&pMipmappedArrayDesc, sizeof(pMipmappedArrayDesc));
    unsigned int numMipmapLevels;
    conn->read(&numMipmapLevels, sizeof(numMipmapLevels));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarray *pLevelArray;
    conn->read(&pLevelArray, sizeof(pLevelArray));
    CUmipmappedArray hMipmappedArray;
    conn->read(&hMipmappedArray, sizeof(hMipmappedArray));
    unsigned int level;
    conn->read(&level, sizeof(level));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmipmappedArray hMipmappedArray;
    conn->read(&hMipmappedArray, sizeof(hMipmappedArray));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMipmappedArrayDestroy(hMipmappedArray);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr ptr;
    conn->read(&ptr, sizeof(ptr));
    size_t size;
    conn->read(&size, sizeof(size));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemAddressFree(ptr, size);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarrayMapInfo *mapInfoList;
    conn->read(&mapInfoList, sizeof(mapInfoList));
    unsigned int count;
    conn->read(&count, sizeof(count));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemMapArrayAsync(mapInfoList, count, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr ptr;
    conn->read(&ptr, sizeof(ptr));
    size_t size;
    conn->read(&size, sizeof(size));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemUnmap(ptr, size);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr ptr;
    conn->read(&ptr, sizeof(ptr));
    size_t size;
    conn->read(&size, sizeof(size));
    CUmemAccessDesc *desc = nullptr;
    conn->read(&desc, sizeof(desc));
    size_t count;
    conn->read(&count, sizeof(count));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemSetAccess(ptr, size, desc, count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned long long *flags;
    conn->read(&flags, sizeof(flags));
    CUmemLocation *location = nullptr;
    conn->read(&location, sizeof(location));
    CUdeviceptr ptr;
    conn->read(&ptr, sizeof(ptr));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemGetAccess(flags, location, ptr);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    void *shareableHandle;
    conn->read(&shareableHandle, sizeof(shareableHandle));
    CUmemGenericAllocationHandle handle;
    conn->read(&handle, sizeof(handle));
    CUmemAllocationHandleType handleType;
    conn->read(&handleType, sizeof(handleType));
    unsigned long long flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemExportToShareableHandle(shareableHandle, handle, handleType, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemGenericAllocationHandle *handle;
    conn->read(&handle, sizeof(handle));
    void *osHandle;
    conn->read(&osHandle, sizeof(osHandle));
    CUmemAllocationHandleType shHandleType;
    conn->read(&shHandleType, sizeof(shHandleType));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemImportFromShareableHandle(handle, osHandle, shHandleType);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    size_t *granularity;
    conn->read(&granularity, sizeof(granularity));
    CUmemAllocationProp *prop = nullptr;
    conn->read(&prop, sizeof(prop));
    CUmemAllocationGranularity_flags option;
    conn->read(&option, sizeof(option));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemGetAllocationGranularity(granularity, prop, option);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemAllocationProp *prop;
    conn->read(&prop, sizeof(prop));
    CUmemGenericAllocationHandle handle;
    conn->read(&handle, sizeof(handle));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemGetAllocationPropertiesFromHandle(prop, handle);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemGenericAllocationHandle *handle;
    conn->read(&handle, sizeof(handle));
    void *addr;
    conn->read(&addr, sizeof(addr));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemRetainAllocationHandle(handle, addr);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr dptr;
    conn->read(&dptr, sizeof(dptr));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemFreeAsync(dptr, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemAllocAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemAllocAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr *dptr;
    conn->read(&dptr, sizeof(dptr));
    size_t bytesize;
    conn->read(&bytesize, sizeof(bytesize));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemAllocAsync(dptr, bytesize, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemoryPool pool;
    conn->read(&pool, sizeof(pool));
    size_t minBytesToKeep;
    conn->read(&minBytesToKeep, sizeof(minBytesToKeep));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolTrimTo(pool, minBytesToKeep);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemoryPool pool;
    conn->read(&pool, sizeof(pool));
    CUmemPool_attribute attr;
    conn->read(&attr, sizeof(attr));
    void *value;
    conn->read(&value, sizeof(value));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolSetAttribute(pool, attr, value);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemoryPool pool;
    conn->read(&pool, sizeof(pool));
    CUmemPool_attribute attr;
    conn->read(&attr, sizeof(attr));
    void *value;
    conn->read(&value, sizeof(value));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolGetAttribute(pool, attr, value);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemoryPool pool;
    conn->read(&pool, sizeof(pool));
    CUmemAccessDesc *map = nullptr;
    conn->read(&map, sizeof(map));
    size_t count;
    conn->read(&count, sizeof(count));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolSetAccess(pool, map, count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemAccess_flags *flags;
    conn->read(&flags, sizeof(flags));
    CUmemoryPool memPool;
    conn->read(&memPool, sizeof(memPool));
    CUmemLocation *location;
    conn->read(&location, sizeof(location));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolGetAccess(flags, memPool, location);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemoryPool *pool;
    conn->read(&pool, sizeof(pool));
    CUmemPoolProps *poolProps = nullptr;
    conn->read(&poolProps, sizeof(poolProps));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolCreate(pool, poolProps);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemoryPool pool;
    conn->read(&pool, sizeof(pool));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolDestroy(pool);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuMemAllocFromPoolAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuMemAllocFromPoolAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr *dptr;
    conn->read(&dptr, sizeof(dptr));
    size_t bytesize;
    conn->read(&bytesize, sizeof(bytesize));
    CUmemoryPool pool;
    conn->read(&pool, sizeof(pool));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    void *handle_out;
    conn->read(&handle_out, sizeof(handle_out));
    CUmemoryPool pool;
    conn->read(&pool, sizeof(pool));
    CUmemAllocationHandleType handleType;
    conn->read(&handleType, sizeof(handleType));
    unsigned long long flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolExportToShareableHandle(handle_out, pool, handleType, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemoryPool *pool_out;
    conn->read(&pool_out, sizeof(pool_out));
    void *handle;
    conn->read(&handle, sizeof(handle));
    CUmemAllocationHandleType handleType;
    conn->read(&handleType, sizeof(handleType));
    unsigned long long flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolImportFromShareableHandle(pool_out, handle, handleType, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmemPoolPtrExportData *shareData_out;
    conn->read(&shareData_out, sizeof(shareData_out));
    CUdeviceptr ptr;
    conn->read(&ptr, sizeof(ptr));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPoolExportPointer(shareData_out, ptr);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    void *data;
    conn->read(&data, sizeof(data));
    CUpointer_attribute attribute;
    conn->read(&attribute, sizeof(attribute));
    CUdeviceptr ptr;
    conn->read(&ptr, sizeof(ptr));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuPointerGetAttribute(data, attribute, ptr);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    size_t count;
    conn->read(&count, sizeof(count));
    CUdevice dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemPrefetchAsync(devPtr, count, dstDevice, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdeviceptr devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    size_t count;
    conn->read(&count, sizeof(count));
    CUmem_advise advice;
    conn->read(&advice, sizeof(advice));
    CUdevice device;
    conn->read(&device, sizeof(device));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemAdvise(devPtr, count, advice, device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    void *data;
    conn->read(&data, sizeof(data));
    size_t dataSize;
    conn->read(&dataSize, sizeof(dataSize));
    CUmem_range_attribute attribute;
    conn->read(&attribute, sizeof(attribute));
    CUdeviceptr devPtr;
    conn->read(&devPtr, sizeof(devPtr));
    size_t count;
    conn->read(&count, sizeof(count));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuPointerSetAttribute(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuPointerSetAttribute called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    void *value;
    conn->read(&value, sizeof(value));
    CUpointer_attribute attribute;
    conn->read(&attribute, sizeof(attribute));
    CUdeviceptr ptr;
    conn->read(&ptr, sizeof(ptr));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuPointerSetAttribute(value, attribute, ptr);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUstream *phStream;
    conn->read(&phStream, sizeof(phStream));
    unsigned int Flags;
    conn->read(&Flags, sizeof(Flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamCreate(phStream, Flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream *phStream;
    conn->read(&phStream, sizeof(phStream));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    int priority;
    conn->read(&priority, sizeof(priority));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamCreateWithPriority(phStream, flags, priority);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    int *priority;
    conn->read(&priority, sizeof(priority));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetPriority(hStream, priority);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    unsigned int *flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetFlags(hStream, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUcontext *pctx;
    conn->read(&pctx, sizeof(pctx));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetCtx(hStream, pctx);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUevent hEvent;
    conn->read(&hEvent, sizeof(hEvent));
    unsigned int Flags;
    conn->read(&Flags, sizeof(Flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamWaitEvent(hStream, hEvent, Flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUstreamCallback callback;
    conn->read(&callback, sizeof(callback));
    void *userData;
    conn->read(&userData, sizeof(userData));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamAddCallback(hStream, callback, userData, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUstreamCaptureMode mode;
    conn->read(&mode, sizeof(mode));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamBeginCapture_v2(hStream, mode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstreamCaptureMode *mode;
    conn->read(&mode, sizeof(mode));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuThreadExchangeStreamCaptureMode(mode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUgraph *phGraph;
    conn->read(&phGraph, sizeof(phGraph));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamEndCapture(hStream, phGraph);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUstreamCaptureStatus *captureStatus;
    conn->read(&captureStatus, sizeof(captureStatus));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamIsCapturing(hStream, captureStatus);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamGetCaptureInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamGetCaptureInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUstreamCaptureStatus *captureStatus_out;
    conn->read(&captureStatus_out, sizeof(captureStatus_out));
    cuuint64_t *id_out;
    conn->read(&id_out, sizeof(id_out));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetCaptureInfo(hStream, captureStatus_out, id_out);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUstreamCaptureStatus *captureStatus_out;
    conn->read(&captureStatus_out, sizeof(captureStatus_out));
    cuuint64_t *id_out;
    conn->read(&id_out, sizeof(id_out));
    CUgraph *graph_out;
    conn->read(&graph_out, sizeof(graph_out));
    const CUgraphNode *dependencies_out;
    size_t *numDependencies_out;
    conn->read(&numDependencies_out, sizeof(numDependencies_out));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetCaptureInfo_v2(hStream, captureStatus_out, id_out, graph_out, &dependencies_out, numDependencies_out);
    conn->write(dependencies_out, sizeof(CUgraphNode));
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamUpdateCaptureDependencies(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamUpdateCaptureDependencies called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUgraphNode *dependencies;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamUpdateCaptureDependencies(hStream, dependencies, numDependencies, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUdeviceptr dptr;
    conn->read(&dptr, sizeof(dptr));
    size_t length;
    conn->read(&length, sizeof(length));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamAttachMemAsync(hStream, dptr, length, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamQuery(hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamSynchronize(hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamDestroy_v2(hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream dst;
    conn->read(&dst, sizeof(dst));
    CUstream src;
    conn->read(&src, sizeof(src));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamCopyAttributes(dst, src);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUstreamAttrID attr;
    conn->read(&attr, sizeof(attr));
    CUstreamAttrValue *value_out;
    conn->read(&value_out, sizeof(value_out));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamGetAttribute(hStream, attr, value_out);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUstreamAttrID attr;
    conn->read(&attr, sizeof(attr));
    CUstreamAttrValue *value = nullptr;
    conn->read(&value, sizeof(value));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamSetAttribute(hStream, attr, value);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUevent *phEvent;
    conn->read(&phEvent, sizeof(phEvent));
    unsigned int Flags;
    conn->read(&Flags, sizeof(Flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventCreate(phEvent, Flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUevent hEvent;
    conn->read(&hEvent, sizeof(hEvent));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventRecord(hEvent, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUevent hEvent;
    conn->read(&hEvent, sizeof(hEvent));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventRecordWithFlags(hEvent, hStream, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUevent hEvent;
    conn->read(&hEvent, sizeof(hEvent));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventQuery(hEvent);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUevent hEvent;
    conn->read(&hEvent, sizeof(hEvent));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventSynchronize(hEvent);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUevent hEvent;
    conn->read(&hEvent, sizeof(hEvent));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventDestroy_v2(hEvent);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    float *pMilliseconds;
    conn->read(&pMilliseconds, sizeof(pMilliseconds));
    CUevent hStart;
    conn->read(&hStart, sizeof(hStart));
    CUevent hEnd;
    conn->read(&hEnd, sizeof(hEnd));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuEventElapsedTime(pMilliseconds, hStart, hEnd);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmipmappedArray *mipmap;
    conn->read(&mipmap, sizeof(mipmap));
    CUexternalMemory extMem;
    conn->read(&extMem, sizeof(extMem));
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc = nullptr;
    conn->read(&mipmapDesc, sizeof(mipmapDesc));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUexternalMemory extMem;
    conn->read(&extMem, sizeof(extMem));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDestroyExternalMemory(extMem);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUexternalSemaphore *extSem_out;
    conn->read(&extSem_out, sizeof(extSem_out));
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc = nullptr;
    conn->read(&semHandleDesc, sizeof(semHandleDesc));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuImportExternalSemaphore(extSem_out, semHandleDesc);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUexternalSemaphore *extSemArray = nullptr;
    conn->read(&extSemArray, sizeof(extSemArray));
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray = nullptr;
    conn->read(&paramsArray, sizeof(paramsArray));
    unsigned int numExtSems;
    conn->read(&numExtSems, sizeof(numExtSems));
    CUstream stream;
    conn->read(&stream, sizeof(stream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUexternalSemaphore *extSemArray = nullptr;
    conn->read(&extSemArray, sizeof(extSemArray));
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray = nullptr;
    conn->read(&paramsArray, sizeof(paramsArray));
    unsigned int numExtSems;
    conn->read(&numExtSems, sizeof(numExtSems));
    CUstream stream;
    conn->read(&stream, sizeof(stream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUexternalSemaphore extSem;
    conn->read(&extSem, sizeof(extSem));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDestroyExternalSemaphore(extSem);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamWaitValue32(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamWaitValue32 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUstream stream;
    conn->read(&stream, sizeof(stream));
    CUdeviceptr addr;
    conn->read(&addr, sizeof(addr));
    cuuint32_t value;
    conn->read(&value, sizeof(value));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamWaitValue32(stream, addr, value, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamWaitValue64(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamWaitValue64 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUstream stream;
    conn->read(&stream, sizeof(stream));
    CUdeviceptr addr;
    conn->read(&addr, sizeof(addr));
    cuuint64_t value;
    conn->read(&value, sizeof(value));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamWaitValue64(stream, addr, value, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamWriteValue32(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamWriteValue32 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUstream stream;
    conn->read(&stream, sizeof(stream));
    CUdeviceptr addr;
    conn->read(&addr, sizeof(addr));
    cuuint32_t value;
    conn->read(&value, sizeof(value));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamWriteValue32(stream, addr, value, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamWriteValue64(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamWriteValue64 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUstream stream;
    conn->read(&stream, sizeof(stream));
    CUdeviceptr addr;
    conn->read(&addr, sizeof(addr));
    cuuint64_t value;
    conn->read(&value, sizeof(value));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamWriteValue64(stream, addr, value, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuStreamBatchMemOp(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuStreamBatchMemOp called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUstream stream;
    conn->read(&stream, sizeof(stream));
    unsigned int count;
    conn->read(&count, sizeof(count));
    CUstreamBatchMemOpParams *paramArray;
    conn->read(&paramArray, sizeof(paramArray));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuStreamBatchMemOp(stream, count, paramArray, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *pi;
    conn->read(&pi, sizeof(pi));
    CUfunction_attribute attrib;
    conn->read(&attrib, sizeof(attrib));
    CUfunction hfunc;
    conn->read(&hfunc, sizeof(hfunc));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncGetAttribute(pi, attrib, hfunc);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction hfunc;
    conn->read(&hfunc, sizeof(hfunc));
    CUfunction_attribute attrib;
    conn->read(&attrib, sizeof(attrib));
    int value;
    conn->read(&value, sizeof(value));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncSetAttribute(hfunc, attrib, value);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction hfunc;
    conn->read(&hfunc, sizeof(hfunc));
    CUfunc_cache config;
    conn->read(&config, sizeof(config));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncSetCacheConfig(hfunc, config);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction hfunc;
    conn->read(&hfunc, sizeof(hfunc));
    CUsharedconfig config;
    conn->read(&config, sizeof(config));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncSetSharedMemConfig(hfunc, config);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmodule *hmod;
    conn->read(&hmod, sizeof(hmod));
    CUfunction hfunc;
    conn->read(&hfunc, sizeof(hfunc));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncGetModule(hmod, hfunc);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction f;
    conn->read(&f, sizeof(f));
    unsigned int gridDimX;
    conn->read(&gridDimX, sizeof(gridDimX));
    unsigned int gridDimY;
    conn->read(&gridDimY, sizeof(gridDimY));
    unsigned int gridDimZ;
    conn->read(&gridDimZ, sizeof(gridDimZ));
    unsigned int blockDimX;
    conn->read(&blockDimX, sizeof(blockDimX));
    unsigned int blockDimY;
    conn->read(&blockDimY, sizeof(blockDimY));
    unsigned int blockDimZ;
    conn->read(&blockDimZ, sizeof(blockDimZ));
    unsigned int sharedMemBytes;
    conn->read(&sharedMemBytes, sizeof(sharedMemBytes));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    // PARAM void **kernelParams
    void *kernelParams;
    // PARAM void **extra
    void *extra;
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    // PARAM void **kernelParams
    // PARAM void **extra
    _result = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, &kernelParams, &extra);
    // PARAM void **kernelParams
    // PARAM void **extra
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
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

int handle_cuLaunchCooperativeKernelMultiDevice(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuLaunchCooperativeKernelMultiDevice called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUDA_LAUNCH_PARAMS *launchParamsList;
    conn->read(&launchParamsList, sizeof(launchParamsList));
    unsigned int numDevices;
    conn->read(&numDevices, sizeof(numDevices));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUhostFn fn;
    conn->read(&fn, sizeof(fn));
    void *userData;
    conn->read(&userData, sizeof(userData));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLaunchHostFunc(hStream, fn, userData);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction hfunc;
    conn->read(&hfunc, sizeof(hfunc));
    int x;
    conn->read(&x, sizeof(x));
    int y;
    conn->read(&y, sizeof(y));
    int z;
    conn->read(&z, sizeof(z));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncSetBlockShape(hfunc, x, y, z);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction hfunc;
    conn->read(&hfunc, sizeof(hfunc));
    unsigned int bytes;
    conn->read(&bytes, sizeof(bytes));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFuncSetSharedSize(hfunc, bytes);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction hfunc;
    conn->read(&hfunc, sizeof(hfunc));
    unsigned int numbytes;
    conn->read(&numbytes, sizeof(numbytes));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuParamSetSize(hfunc, numbytes);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction hfunc;
    conn->read(&hfunc, sizeof(hfunc));
    int offset;
    conn->read(&offset, sizeof(offset));
    unsigned int value;
    conn->read(&value, sizeof(value));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuParamSeti(hfunc, offset, value);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction hfunc;
    conn->read(&hfunc, sizeof(hfunc));
    int offset;
    conn->read(&offset, sizeof(offset));
    float value;
    conn->read(&value, sizeof(value));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuParamSetf(hfunc, offset, value);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction hfunc;
    conn->read(&hfunc, sizeof(hfunc));
    int offset;
    conn->read(&offset, sizeof(offset));
    void *ptr;
    conn->read(&ptr, sizeof(ptr));
    unsigned int numbytes;
    conn->read(&numbytes, sizeof(numbytes));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuParamSetv(hfunc, offset, ptr, numbytes);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction f;
    conn->read(&f, sizeof(f));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLaunch(f);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction f;
    conn->read(&f, sizeof(f));
    int grid_width;
    conn->read(&grid_width, sizeof(grid_width));
    int grid_height;
    conn->read(&grid_height, sizeof(grid_height));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLaunchGrid(f, grid_width, grid_height);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction f;
    conn->read(&f, sizeof(f));
    int grid_width;
    conn->read(&grid_width, sizeof(grid_width));
    int grid_height;
    conn->read(&grid_height, sizeof(grid_height));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuLaunchGridAsync(f, grid_width, grid_height, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfunction hfunc;
    conn->read(&hfunc, sizeof(hfunc));
    int texunit;
    conn->read(&texunit, sizeof(texunit));
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuParamSetTexRef(hfunc, texunit, hTexRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraph *phGraph;
    conn->read(&phGraph, sizeof(phGraph));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphCreate(phGraph, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphAddKernelNode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphAddKernelNode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode *phGraphNode;
    conn->read(&phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *dependencies = nullptr;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUDA_KERNEL_NODE_PARAMS *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddKernelNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphKernelNodeGetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphKernelNodeGetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_KERNEL_NODE_PARAMS *nodeParams;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphKernelNodeGetParams(hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphKernelNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphKernelNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_KERNEL_NODE_PARAMS *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphKernelNodeSetParams(hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode *phGraphNode;
    conn->read(&phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *dependencies = nullptr;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUDA_MEMCPY3D *copyParams = nullptr;
    conn->read(&copyParams, sizeof(copyParams));
    CUcontext ctx;
    conn->read(&ctx, sizeof(ctx));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_MEMCPY3D *nodeParams;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphMemcpyNodeGetParams(hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_MEMCPY3D *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphMemcpyNodeSetParams(hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode *phGraphNode;
    conn->read(&phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *dependencies = nullptr;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUDA_MEMSET_NODE_PARAMS *memsetParams = nullptr;
    conn->read(&memsetParams, sizeof(memsetParams));
    CUcontext ctx;
    conn->read(&ctx, sizeof(ctx));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_MEMSET_NODE_PARAMS *nodeParams;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphMemsetNodeGetParams(hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_MEMSET_NODE_PARAMS *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphMemsetNodeSetParams(hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode *phGraphNode;
    conn->read(&phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *dependencies = nullptr;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUDA_HOST_NODE_PARAMS *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_HOST_NODE_PARAMS *nodeParams;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphHostNodeGetParams(hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_HOST_NODE_PARAMS *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphHostNodeSetParams(hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode *phGraphNode;
    conn->read(&phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *dependencies = nullptr;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUgraph childGraph;
    conn->read(&childGraph, sizeof(childGraph));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddChildGraphNode(phGraphNode, hGraph, dependencies, numDependencies, childGraph);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUgraph *phGraph;
    conn->read(&phGraph, sizeof(phGraph));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphChildGraphNodeGetGraph(hNode, phGraph);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode *phGraphNode;
    conn->read(&phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *dependencies = nullptr;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddEmptyNode(phGraphNode, hGraph, dependencies, numDependencies);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode *phGraphNode;
    conn->read(&phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *dependencies = nullptr;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUevent event;
    conn->read(&event, sizeof(event));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddEventRecordNode(phGraphNode, hGraph, dependencies, numDependencies, event);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUevent *event_out;
    conn->read(&event_out, sizeof(event_out));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphEventRecordNodeGetEvent(hNode, event_out);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUevent event;
    conn->read(&event, sizeof(event));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphEventRecordNodeSetEvent(hNode, event);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode *phGraphNode;
    conn->read(&phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *dependencies = nullptr;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUevent event;
    conn->read(&event, sizeof(event));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddEventWaitNode(phGraphNode, hGraph, dependencies, numDependencies, event);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUevent *event_out;
    conn->read(&event_out, sizeof(event_out));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphEventWaitNodeGetEvent(hNode, event_out);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUevent event;
    conn->read(&event, sizeof(event));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphEventWaitNodeSetEvent(hNode, event);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode *phGraphNode;
    conn->read(&phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *dependencies = nullptr;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddExternalSemaphoresSignalNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out;
    conn->read(&params_out, sizeof(params_out));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode *phGraphNode;
    conn->read(&phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *dependencies = nullptr;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddExternalSemaphoresWaitNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out;
    conn->read(&params_out, sizeof(params_out));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode *phGraphNode;
    conn->read(&phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *dependencies = nullptr;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddMemAllocNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_MEM_ALLOC_NODE_PARAMS *params_out;
    conn->read(&params_out, sizeof(params_out));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphMemAllocNodeGetParams(hNode, params_out);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode *phGraphNode;
    conn->read(&phGraphNode, sizeof(phGraphNode));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *dependencies = nullptr;
    conn->read(&dependencies, sizeof(dependencies));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUdeviceptr dptr;
    conn->read(&dptr, sizeof(dptr));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddMemFreeNode(phGraphNode, hGraph, dependencies, numDependencies, dptr);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdevice device;
    conn->read(&device, sizeof(device));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGraphMemTrim(device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdevice device;
    conn->read(&device, sizeof(device));
    CUgraphMem_attribute attr;
    conn->read(&attr, sizeof(attr));
    void *value;
    conn->read(&value, sizeof(value));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetGraphMemAttribute(device, attr, value);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUdevice device;
    conn->read(&device, sizeof(device));
    CUgraphMem_attribute attr;
    conn->read(&attr, sizeof(attr));
    void *value;
    conn->read(&value, sizeof(value));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceSetGraphMemAttribute(device, attr, value);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraph *phGraphClone;
    conn->read(&phGraphClone, sizeof(phGraphClone));
    CUgraph originalGraph;
    conn->read(&originalGraph, sizeof(originalGraph));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphClone(phGraphClone, originalGraph);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode *phNode;
    conn->read(&phNode, sizeof(phNode));
    CUgraphNode hOriginalNode;
    conn->read(&hOriginalNode, sizeof(hOriginalNode));
    CUgraph hClonedGraph;
    conn->read(&hClonedGraph, sizeof(hClonedGraph));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUgraphNodeType *type;
    conn->read(&type, sizeof(type));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphNodeGetType(hNode, type);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *nodes;
    conn->read(&nodes, sizeof(nodes));
    size_t *numNodes;
    conn->read(&numNodes, sizeof(numNodes));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphGetNodes(hGraph, nodes, numNodes);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *rootNodes;
    conn->read(&rootNodes, sizeof(rootNodes));
    size_t *numRootNodes;
    conn->read(&numRootNodes, sizeof(numRootNodes));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphGetRootNodes(hGraph, rootNodes, numRootNodes);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *from;
    conn->read(&from, sizeof(from));
    CUgraphNode *to;
    conn->read(&to, sizeof(to));
    size_t *numEdges;
    conn->read(&numEdges, sizeof(numEdges));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphGetEdges(hGraph, from, to, numEdges);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUgraphNode *dependencies;
    conn->read(&dependencies, sizeof(dependencies));
    size_t *numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphNodeGetDependencies(hNode, dependencies, numDependencies);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUgraphNode *dependentNodes;
    conn->read(&dependentNodes, sizeof(dependentNodes));
    size_t *numDependentNodes;
    conn->read(&numDependentNodes, sizeof(numDependentNodes));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *from = nullptr;
    conn->read(&from, sizeof(from));
    CUgraphNode *to = nullptr;
    conn->read(&to, sizeof(to));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphAddDependencies(hGraph, from, to, numDependencies);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *from = nullptr;
    conn->read(&from, sizeof(from));
    CUgraphNode *to = nullptr;
    conn->read(&to, sizeof(to));
    size_t numDependencies;
    conn->read(&numDependencies, sizeof(numDependencies));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphRemoveDependencies(hGraph, from, to, numDependencies);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphDestroyNode(hNode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphInstantiate_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphInstantiate_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec *phGraphExec;
    conn->read(&phGraphExec, sizeof(phGraphExec));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *phErrorNode;
    conn->read(&phErrorNode, sizeof(phErrorNode));
    char logBuffer[1024];
    size_t bufferSize;
    conn->read(&bufferSize, sizeof(bufferSize));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphInstantiate_v2(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize);
    if(bufferSize > 0) {
        conn->write(logBuffer, strlen(logBuffer) + 1, true);
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

int handle_cuGraphInstantiateWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphInstantiateWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec *phGraphExec;
    conn->read(&phGraphExec, sizeof(phGraphExec));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    unsigned long long flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphInstantiateWithFlags(phGraphExec, hGraph, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecKernelNodeSetParams(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecKernelNodeSetParams called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_KERNEL_NODE_PARAMS *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecKernelNodeSetParams(hGraphExec, hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_MEMCPY3D *copyParams = nullptr;
    conn->read(&copyParams, sizeof(copyParams));
    CUcontext ctx;
    conn->read(&ctx, sizeof(ctx));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_MEMSET_NODE_PARAMS *memsetParams = nullptr;
    conn->read(&memsetParams, sizeof(memsetParams));
    CUcontext ctx;
    conn->read(&ctx, sizeof(ctx));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_HOST_NODE_PARAMS *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecHostNodeSetParams(hGraphExec, hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUgraph childGraph;
    conn->read(&childGraph, sizeof(childGraph));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecChildGraphNodeSetParams(hGraphExec, hNode, childGraph);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUevent event;
    conn->read(&event, sizeof(event));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUevent event;
    conn->read(&event, sizeof(event));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams = nullptr;
    conn->read(&nodeParams, sizeof(nodeParams));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphUpload(hGraphExec, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphLaunch(hGraphExec, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecDestroy(hGraphExec);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphDestroy(hGraph);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cuGraphExecUpdate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuGraphExecUpdate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUgraphExec hGraphExec;
    conn->read(&hGraphExec, sizeof(hGraphExec));
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    CUgraphNode *hErrorNode_out;
    conn->read(&hErrorNode_out, sizeof(hErrorNode_out));
    CUgraphExecUpdateResult *updateResult_out;
    conn->read(&updateResult_out, sizeof(updateResult_out));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode dst;
    conn->read(&dst, sizeof(dst));
    CUgraphNode src;
    conn->read(&src, sizeof(src));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphKernelNodeCopyAttributes(dst, src);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUkernelNodeAttrID attr;
    conn->read(&attr, sizeof(attr));
    CUkernelNodeAttrValue *value_out;
    conn->read(&value_out, sizeof(value_out));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphKernelNodeGetAttribute(hNode, attr, value_out);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphNode hNode;
    conn->read(&hNode, sizeof(hNode));
    CUkernelNodeAttrID attr;
    conn->read(&attr, sizeof(attr));
    CUkernelNodeAttrValue *value = nullptr;
    conn->read(&value, sizeof(value));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphKernelNodeSetAttribute(hNode, attr, value);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraph hGraph;
    conn->read(&hGraph, sizeof(hGraph));
    char *path = nullptr;
    conn->read(&path, 0, true);
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(path);
    _result = cuGraphDebugDotPrint(hGraph, path, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUuserObject *object_out;
    conn->read(&object_out, sizeof(object_out));
    void *ptr;
    conn->read(&ptr, sizeof(ptr));
    CUhostFn destroy;
    conn->read(&destroy, sizeof(destroy));
    unsigned int initialRefcount;
    conn->read(&initialRefcount, sizeof(initialRefcount));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUuserObject object;
    conn->read(&object, sizeof(object));
    unsigned int count;
    conn->read(&count, sizeof(count));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuUserObjectRetain(object, count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUuserObject object;
    conn->read(&object, sizeof(object));
    unsigned int count;
    conn->read(&count, sizeof(count));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuUserObjectRelease(object, count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraph graph;
    conn->read(&graph, sizeof(graph));
    CUuserObject object;
    conn->read(&object, sizeof(object));
    unsigned int count;
    conn->read(&count, sizeof(count));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphRetainUserObject(graph, object, count, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraph graph;
    conn->read(&graph, sizeof(graph));
    CUuserObject object;
    conn->read(&object, sizeof(object));
    unsigned int count;
    conn->read(&count, sizeof(count));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphReleaseUserObject(graph, object, count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *numBlocks;
    conn->read(&numBlocks, sizeof(numBlocks));
    CUfunction func;
    conn->read(&func, sizeof(func));
    int blockSize;
    conn->read(&blockSize, sizeof(blockSize));
    size_t dynamicSMemSize;
    conn->read(&dynamicSMemSize, sizeof(dynamicSMemSize));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *numBlocks;
    conn->read(&numBlocks, sizeof(numBlocks));
    CUfunction func;
    conn->read(&func, sizeof(func));
    int blockSize;
    conn->read(&blockSize, sizeof(blockSize));
    size_t dynamicSMemSize;
    conn->read(&dynamicSMemSize, sizeof(dynamicSMemSize));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *minGridSize;
    conn->read(&minGridSize, sizeof(minGridSize));
    int *blockSize;
    conn->read(&blockSize, sizeof(blockSize));
    CUfunction func;
    conn->read(&func, sizeof(func));
    CUoccupancyB2DSize blockSizeToDynamicSMemSize;
    conn->read(&blockSizeToDynamicSMemSize, sizeof(blockSizeToDynamicSMemSize));
    size_t dynamicSMemSize;
    conn->read(&dynamicSMemSize, sizeof(dynamicSMemSize));
    int blockSizeLimit;
    conn->read(&blockSizeLimit, sizeof(blockSizeLimit));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *minGridSize;
    conn->read(&minGridSize, sizeof(minGridSize));
    int *blockSize;
    conn->read(&blockSize, sizeof(blockSize));
    CUfunction func;
    conn->read(&func, sizeof(func));
    CUoccupancyB2DSize blockSizeToDynamicSMemSize;
    conn->read(&blockSizeToDynamicSMemSize, sizeof(blockSizeToDynamicSMemSize));
    size_t dynamicSMemSize;
    conn->read(&dynamicSMemSize, sizeof(dynamicSMemSize));
    int blockSizeLimit;
    conn->read(&blockSizeLimit, sizeof(blockSizeLimit));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    size_t *dynamicSmemSize;
    conn->read(&dynamicSmemSize, sizeof(dynamicSmemSize));
    CUfunction func;
    conn->read(&func, sizeof(func));
    int numBlocks;
    conn->read(&numBlocks, sizeof(numBlocks));
    int blockSize;
    conn->read(&blockSize, sizeof(blockSize));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUarray hArray;
    conn->read(&hArray, sizeof(hArray));
    unsigned int Flags;
    conn->read(&Flags, sizeof(Flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetArray(hTexRef, hArray, Flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUmipmappedArray hMipmappedArray;
    conn->read(&hMipmappedArray, sizeof(hMipmappedArray));
    unsigned int Flags;
    conn->read(&Flags, sizeof(Flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    size_t *ByteOffset;
    conn->read(&ByteOffset, sizeof(ByteOffset));
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUdeviceptr dptr;
    conn->read(&dptr, sizeof(dptr));
    size_t bytes;
    conn->read(&bytes, sizeof(bytes));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUDA_ARRAY_DESCRIPTOR *desc = nullptr;
    conn->read(&desc, sizeof(desc));
    CUdeviceptr dptr;
    conn->read(&dptr, sizeof(dptr));
    size_t Pitch;
    conn->read(&Pitch, sizeof(Pitch));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUarray_format fmt;
    conn->read(&fmt, sizeof(fmt));
    int NumPackedComponents;
    conn->read(&NumPackedComponents, sizeof(NumPackedComponents));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    int dim;
    conn->read(&dim, sizeof(dim));
    CUaddress_mode am;
    conn->read(&am, sizeof(am));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetAddressMode(hTexRef, dim, am);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUfilter_mode fm;
    conn->read(&fm, sizeof(fm));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetFilterMode(hTexRef, fm);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUfilter_mode fm;
    conn->read(&fm, sizeof(fm));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetMipmapFilterMode(hTexRef, fm);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    float bias;
    conn->read(&bias, sizeof(bias));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetMipmapLevelBias(hTexRef, bias);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    float minMipmapLevelClamp;
    conn->read(&minMipmapLevelClamp, sizeof(minMipmapLevelClamp));
    float maxMipmapLevelClamp;
    conn->read(&maxMipmapLevelClamp, sizeof(maxMipmapLevelClamp));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    unsigned int maxAniso;
    conn->read(&maxAniso, sizeof(maxAniso));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetMaxAnisotropy(hTexRef, maxAniso);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    float *pBorderColor;
    conn->read(&pBorderColor, sizeof(pBorderColor));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetBorderColor(hTexRef, pBorderColor);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    unsigned int Flags;
    conn->read(&Flags, sizeof(Flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefSetFlags(hTexRef, Flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarray *phArray;
    conn->read(&phArray, sizeof(phArray));
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetArray(phArray, hTexRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmipmappedArray *phMipmappedArray;
    conn->read(&phMipmappedArray, sizeof(phMipmappedArray));
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetMipmappedArray(phMipmappedArray, hTexRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUaddress_mode *pam;
    conn->read(&pam, sizeof(pam));
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    int dim;
    conn->read(&dim, sizeof(dim));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetAddressMode(pam, hTexRef, dim);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfilter_mode *pfm;
    conn->read(&pfm, sizeof(pfm));
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetFilterMode(pfm, hTexRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarray_format *pFormat;
    conn->read(&pFormat, sizeof(pFormat));
    int *pNumChannels;
    conn->read(&pNumChannels, sizeof(pNumChannels));
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetFormat(pFormat, pNumChannels, hTexRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUfilter_mode *pfm;
    conn->read(&pfm, sizeof(pfm));
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetMipmapFilterMode(pfm, hTexRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    float *pbias;
    conn->read(&pbias, sizeof(pbias));
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetMipmapLevelBias(pbias, hTexRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    float *pminMipmapLevelClamp;
    conn->read(&pminMipmapLevelClamp, sizeof(pminMipmapLevelClamp));
    float *pmaxMipmapLevelClamp;
    conn->read(&pmaxMipmapLevelClamp, sizeof(pmaxMipmapLevelClamp));
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *pmaxAniso;
    conn->read(&pmaxAniso, sizeof(pmaxAniso));
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    float *pBorderColor;
    conn->read(&pBorderColor, sizeof(pBorderColor));
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetBorderColor(pBorderColor, hTexRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int *pFlags;
    conn->read(&pFlags, sizeof(pFlags));
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefGetFlags(pFlags, hTexRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref *pTexRef;
    conn->read(&pTexRef, sizeof(pTexRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefCreate(pTexRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexref hTexRef;
    conn->read(&hTexRef, sizeof(hTexRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexRefDestroy(hTexRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUsurfref hSurfRef;
    conn->read(&hSurfRef, sizeof(hSurfRef));
    CUarray hArray;
    conn->read(&hArray, sizeof(hArray));
    unsigned int Flags;
    conn->read(&Flags, sizeof(Flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuSurfRefSetArray(hSurfRef, hArray, Flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarray *phArray;
    conn->read(&phArray, sizeof(phArray));
    CUsurfref hSurfRef;
    conn->read(&hSurfRef, sizeof(hSurfRef));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuSurfRefGetArray(phArray, hSurfRef);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexObject *pTexObject;
    conn->read(&pTexObject, sizeof(pTexObject));
    CUDA_RESOURCE_DESC *pResDesc = nullptr;
    conn->read(&pResDesc, sizeof(pResDesc));
    CUDA_TEXTURE_DESC *pTexDesc = nullptr;
    conn->read(&pTexDesc, sizeof(pTexDesc));
    CUDA_RESOURCE_VIEW_DESC *pResViewDesc = nullptr;
    conn->read(&pResViewDesc, sizeof(pResViewDesc));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUtexObject texObject;
    conn->read(&texObject, sizeof(texObject));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexObjectDestroy(texObject);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_RESOURCE_DESC *pResDesc;
    conn->read(&pResDesc, sizeof(pResDesc));
    CUtexObject texObject;
    conn->read(&texObject, sizeof(texObject));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexObjectGetResourceDesc(pResDesc, texObject);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_TEXTURE_DESC *pTexDesc;
    conn->read(&pTexDesc, sizeof(pTexDesc));
    CUtexObject texObject;
    conn->read(&texObject, sizeof(texObject));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexObjectGetTextureDesc(pTexDesc, texObject);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_RESOURCE_VIEW_DESC *pResViewDesc;
    conn->read(&pResViewDesc, sizeof(pResViewDesc));
    CUtexObject texObject;
    conn->read(&texObject, sizeof(texObject));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuTexObjectGetResourceViewDesc(pResViewDesc, texObject);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUsurfObject *pSurfObject;
    conn->read(&pSurfObject, sizeof(pSurfObject));
    CUDA_RESOURCE_DESC *pResDesc = nullptr;
    conn->read(&pResDesc, sizeof(pResDesc));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuSurfObjectCreate(pSurfObject, pResDesc);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUsurfObject surfObject;
    conn->read(&surfObject, sizeof(surfObject));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuSurfObjectDestroy(surfObject);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUDA_RESOURCE_DESC *pResDesc;
    conn->read(&pResDesc, sizeof(pResDesc));
    CUsurfObject surfObject;
    conn->read(&surfObject, sizeof(surfObject));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuSurfObjectGetResourceDesc(pResDesc, surfObject);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *canAccessPeer;
    conn->read(&canAccessPeer, sizeof(canAccessPeer));
    CUdevice dev;
    conn->read(&dev, sizeof(dev));
    CUdevice peerDev;
    conn->read(&peerDev, sizeof(peerDev));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceCanAccessPeer(canAccessPeer, dev, peerDev);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUcontext peerContext;
    conn->read(&peerContext, sizeof(peerContext));
    unsigned int Flags;
    conn->read(&Flags, sizeof(Flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxEnablePeerAccess(peerContext, Flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUcontext peerContext;
    conn->read(&peerContext, sizeof(peerContext));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuCtxDisablePeerAccess(peerContext);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *value;
    conn->read(&value, sizeof(value));
    CUdevice_P2PAttribute attrib;
    conn->read(&attrib, sizeof(attrib));
    CUdevice srcDevice;
    conn->read(&srcDevice, sizeof(srcDevice));
    CUdevice dstDevice;
    conn->read(&dstDevice, sizeof(dstDevice));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphicsResource resource;
    conn->read(&resource, sizeof(resource));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphicsUnregisterResource(resource);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUarray *pArray;
    conn->read(&pArray, sizeof(pArray));
    CUgraphicsResource resource;
    conn->read(&resource, sizeof(resource));
    unsigned int arrayIndex;
    conn->read(&arrayIndex, sizeof(arrayIndex));
    unsigned int mipLevel;
    conn->read(&mipLevel, sizeof(mipLevel));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUmipmappedArray *pMipmappedArray;
    conn->read(&pMipmappedArray, sizeof(pMipmappedArray));
    CUgraphicsResource resource;
    conn->read(&resource, sizeof(resource));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    CUgraphicsResource resource;
    conn->read(&resource, sizeof(resource));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphicsResourceSetMapFlags_v2(resource, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int count;
    conn->read(&count, sizeof(count));
    CUgraphicsResource *resources;
    conn->read(&resources, sizeof(resources));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphicsMapResources(count, resources, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int count;
    conn->read(&count, sizeof(count));
    CUgraphicsResource *resources;
    conn->read(&resources, sizeof(resources));
    CUstream hStream;
    conn->read(&hStream, sizeof(hStream));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGraphicsUnmapResources(count, resources, hStream);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    const void *ppExportTable;
    CUuuid *pExportTableId = nullptr;
    conn->read(&pExportTableId, sizeof(pExportTableId));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuGetExportTable(&ppExportTable, pExportTableId);
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

int handle_cuFlushGPUDirectRDMAWrites(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cuFlushGPUDirectRDMAWrites called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    CUflushGPUDirectRDMAWritesTarget target;
    conn->read(&target, sizeof(target));
    CUflushGPUDirectRDMAWritesScope scope;
    conn->read(&scope, sizeof(scope));
    CUresult _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cuFlushGPUDirectRDMAWrites(target, scope);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}
