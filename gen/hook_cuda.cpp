#include <iostream>
#include <map>
#include "cuda.h"

#include "hook_api.h"
#include "client.h"
extern "C" CUresult cuInit(unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuInit called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuInit);
    conn->write(&Flags, sizeof(Flags));
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

extern "C" CUresult cuDriverGetVersion(int *driverVersion) {
#ifdef DEBUG
    std::cout << "Hook: cuDriverGetVersion called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDriverGetVersion);
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

extern "C" CUresult cuDeviceGet(CUdevice *device, int ordinal) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGet called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGet);
    conn->write(&_0device, sizeof(_0device));
    updateTmpPtr((void *)device, _0device);
    conn->write(&ordinal, sizeof(ordinal));
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

extern "C" CUresult cuDeviceGetCount(int *count) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetCount called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetCount);
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

extern "C" CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetName called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetName);
    if(len > 0) {
        conn->read(name, len, true);
    }
    conn->write(&len, sizeof(len));
    conn->write(&dev, sizeof(dev));
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

extern "C" CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetUuid called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0uuid;
    mem2server(conn, &_0uuid, (void *)uuid, sizeof(*uuid));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetUuid);
    conn->write(&_0uuid, sizeof(_0uuid));
    updateTmpPtr((void *)uuid, _0uuid);
    conn->write(&dev, sizeof(dev));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)uuid, sizeof(*uuid), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetUuid_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0uuid;
    mem2server(conn, &_0uuid, (void *)uuid, sizeof(*uuid));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetUuid_v2);
    conn->write(&_0uuid, sizeof(_0uuid));
    updateTmpPtr((void *)uuid, _0uuid);
    conn->write(&dev, sizeof(dev));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)uuid, sizeof(*uuid), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetLuid called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0deviceNodeMask;
    mem2server(conn, &_0deviceNodeMask, (void *)deviceNodeMask, sizeof(*deviceNodeMask));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetLuid);
    if(32 > 0) {
        conn->read(luid, 32, true);
    }
    conn->write(&_0deviceNodeMask, sizeof(_0deviceNodeMask));
    updateTmpPtr((void *)deviceNodeMask, _0deviceNodeMask);
    conn->write(&dev, sizeof(dev));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)deviceNodeMask, sizeof(*deviceNodeMask), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceTotalMem_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0bytes;
    mem2server(conn, &_0bytes, (void *)bytes, sizeof(*bytes));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceTotalMem_v2);
    conn->write(&_0bytes, sizeof(_0bytes));
    updateTmpPtr((void *)bytes, _0bytes);
    conn->write(&dev, sizeof(dev));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)bytes, sizeof(*bytes), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetTexture1DLinearMaxWidth called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0maxWidthInElements;
    mem2server(conn, &_0maxWidthInElements, (void *)maxWidthInElements, sizeof(*maxWidthInElements));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetTexture1DLinearMaxWidth);
    conn->write(&_0maxWidthInElements, sizeof(_0maxWidthInElements));
    updateTmpPtr((void *)maxWidthInElements, _0maxWidthInElements);
    conn->write(&format, sizeof(format));
    conn->write(&numChannels, sizeof(numChannels));
    conn->write(&dev, sizeof(dev));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)maxWidthInElements, sizeof(*maxWidthInElements), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pi;
    mem2server(conn, &_0pi, (void *)pi, sizeof(*pi));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetAttribute);
    conn->write(&_0pi, sizeof(_0pi));
    updateTmpPtr((void *)pi, _0pi);
    conn->write(&attrib, sizeof(attrib));
    conn->write(&dev, sizeof(dev));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pi, sizeof(*pi), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, CUdevice dev, int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetNvSciSyncAttributes called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetNvSciSyncAttributes);
    conn->write(&_0nvSciSyncAttrList, sizeof(_0nvSciSyncAttrList));
    updateTmpPtr((void *)nvSciSyncAttrList, _0nvSciSyncAttrList);
    conn->write(&dev, sizeof(dev));
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

extern "C" CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceSetMemPool called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceSetMemPool);
    conn->write(&dev, sizeof(dev));
    conn->write(&pool, sizeof(pool));
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

extern "C" CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetMemPool called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pool;
    mem2server(conn, &_0pool, (void *)pool, sizeof(*pool));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetMemPool);
    conn->write(&_0pool, sizeof(_0pool));
    updateTmpPtr((void *)pool, _0pool);
    conn->write(&dev, sizeof(dev));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pool, sizeof(*pool), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetDefaultMemPool called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pool_out;
    mem2server(conn, &_0pool_out, (void *)pool_out, sizeof(*pool_out));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetDefaultMemPool);
    conn->write(&_0pool_out, sizeof(_0pool_out));
    updateTmpPtr((void *)pool_out, _0pool_out);
    conn->write(&dev, sizeof(dev));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pool_out, sizeof(*pool_out), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetProperties called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetProperties);
    conn->write(&_0prop, sizeof(_0prop));
    updateTmpPtr((void *)prop, _0prop);
    conn->write(&dev, sizeof(dev));
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

extern "C" CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceComputeCapability called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0major;
    mem2server(conn, &_0major, (void *)major, sizeof(*major));
    void *_0minor;
    mem2server(conn, &_0minor, (void *)minor, sizeof(*minor));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceComputeCapability);
    conn->write(&_0major, sizeof(_0major));
    updateTmpPtr((void *)major, _0major);
    conn->write(&_0minor, sizeof(_0minor));
    updateTmpPtr((void *)minor, _0minor);
    conn->write(&dev, sizeof(dev));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)major, sizeof(*major), true);
    mem2client(conn, (void *)minor, sizeof(*minor), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxRetain called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pctx;
    mem2server(conn, &_0pctx, (void *)pctx, sizeof(*pctx));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuDevicePrimaryCtxRetain);
    conn->write(&_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    conn->write(&dev, sizeof(dev));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pctx, sizeof(*pctx), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxRelease_v2 called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDevicePrimaryCtxRelease_v2);
    conn->write(&dev, sizeof(dev));
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

extern "C" CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxSetFlags_v2 called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDevicePrimaryCtxSetFlags_v2);
    conn->write(&dev, sizeof(dev));
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

extern "C" CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxGetState called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0flags;
    mem2server(conn, &_0flags, (void *)flags, sizeof(*flags));
    void *_0active;
    mem2server(conn, &_0active, (void *)active, sizeof(*active));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuDevicePrimaryCtxGetState);
    conn->write(&dev, sizeof(dev));
    conn->write(&_0flags, sizeof(_0flags));
    updateTmpPtr((void *)flags, _0flags);
    conn->write(&_0active, sizeof(_0active));
    updateTmpPtr((void *)active, _0active);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)flags, sizeof(*flags), true);
    mem2client(conn, (void *)active, sizeof(*active), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxReset_v2 called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDevicePrimaryCtxReset_v2);
    conn->write(&dev, sizeof(dev));
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

extern "C" CUresult cuDeviceGetExecAffinitySupport(int *pi, CUexecAffinityType type, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetExecAffinitySupport called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pi;
    mem2server(conn, &_0pi, (void *)pi, sizeof(*pi));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetExecAffinitySupport);
    conn->write(&_0pi, sizeof(_0pi));
    updateTmpPtr((void *)pi, _0pi);
    conn->write(&type, sizeof(type));
    conn->write(&dev, sizeof(dev));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pi, sizeof(*pi), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxCreate_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pctx;
    mem2server(conn, &_0pctx, (void *)pctx, sizeof(*pctx));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuCtxCreate_v2);
    conn->write(&_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    conn->write(&flags, sizeof(flags));
    conn->write(&dev, sizeof(dev));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pctx, sizeof(*pctx), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams, unsigned int flags, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxCreate_v3 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pctx;
    mem2server(conn, &_0pctx, (void *)pctx, sizeof(*pctx));
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxCreate_v3);
    conn->write(&_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    conn->write(&_0paramsArray, sizeof(_0paramsArray));
    updateTmpPtr((void *)paramsArray, _0paramsArray);
    conn->write(&numParams, sizeof(numParams));
    conn->write(&flags, sizeof(flags));
    conn->write(&dev, sizeof(dev));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pctx, sizeof(*pctx), true);
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

extern "C" CUresult cuCtxDestroy_v2(CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxDestroy_v2 called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxDestroy_v2);
    conn->write(&ctx, sizeof(ctx));
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

extern "C" CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxPushCurrent_v2 called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxPushCurrent_v2);
    conn->write(&ctx, sizeof(ctx));
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

extern "C" CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxPopCurrent_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pctx;
    mem2server(conn, &_0pctx, (void *)pctx, sizeof(*pctx));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuCtxPopCurrent_v2);
    conn->write(&_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pctx, sizeof(*pctx), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuCtxSetCurrent(CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSetCurrent called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxSetCurrent);
    conn->write(&ctx, sizeof(ctx));
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

extern "C" CUresult cuCtxGetCurrent(CUcontext *pctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetCurrent called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pctx;
    mem2server(conn, &_0pctx, (void *)pctx, sizeof(*pctx));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuCtxGetCurrent);
    conn->write(&_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pctx, sizeof(*pctx), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuCtxGetDevice(CUdevice *device) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetDevice called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxGetDevice);
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

extern "C" CUresult cuCtxGetFlags(unsigned int *flags) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetFlags called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxGetFlags);
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

extern "C" CUresult cuCtxSynchronize() {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSynchronize called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxSynchronize);
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

extern "C" CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSetLimit called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxSetLimit);
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

extern "C" CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetLimit called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pvalue;
    mem2server(conn, &_0pvalue, (void *)pvalue, sizeof(*pvalue));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuCtxGetLimit);
    conn->write(&_0pvalue, sizeof(_0pvalue));
    updateTmpPtr((void *)pvalue, _0pvalue);
    conn->write(&limit, sizeof(limit));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pvalue, sizeof(*pvalue), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetCacheConfig called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pconfig;
    mem2server(conn, &_0pconfig, (void *)pconfig, sizeof(*pconfig));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuCtxGetCacheConfig);
    conn->write(&_0pconfig, sizeof(_0pconfig));
    updateTmpPtr((void *)pconfig, _0pconfig);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pconfig, sizeof(*pconfig), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuCtxSetCacheConfig(CUfunc_cache config) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSetCacheConfig called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxSetCacheConfig);
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

extern "C" CUresult cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetSharedMemConfig called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxGetSharedMemConfig);
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

extern "C" CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSetSharedMemConfig called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxSetSharedMemConfig);
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

extern "C" CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetApiVersion called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0version;
    mem2server(conn, &_0version, (void *)version, sizeof(*version));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuCtxGetApiVersion);
    conn->write(&ctx, sizeof(ctx));
    conn->write(&_0version, sizeof(_0version));
    updateTmpPtr((void *)version, _0version);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)version, sizeof(*version), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetStreamPriorityRange called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxGetStreamPriorityRange);
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

extern "C" CUresult cuCtxResetPersistingL2Cache() {
#ifdef DEBUG
    std::cout << "Hook: cuCtxResetPersistingL2Cache called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxResetPersistingL2Cache);
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

extern "C" CUresult cuCtxGetExecAffinity(CUexecAffinityParam *pExecAffinity, CUexecAffinityType type) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetExecAffinity called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pExecAffinity;
    mem2server(conn, &_0pExecAffinity, (void *)pExecAffinity, sizeof(*pExecAffinity));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuCtxGetExecAffinity);
    conn->write(&_0pExecAffinity, sizeof(_0pExecAffinity));
    updateTmpPtr((void *)pExecAffinity, _0pExecAffinity);
    conn->write(&type, sizeof(type));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pExecAffinity, sizeof(*pExecAffinity), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxAttach called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pctx;
    mem2server(conn, &_0pctx, (void *)pctx, sizeof(*pctx));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuCtxAttach);
    conn->write(&_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pctx, sizeof(*pctx), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuCtxDetach(CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxDetach called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxDetach);
    conn->write(&ctx, sizeof(ctx));
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

extern "C" CUresult cuModuleLoad(CUmodule *module, const char *fname) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleLoad called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0module;
    mem2server(conn, &_0module, (void *)module, sizeof(*module));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuModuleLoad);
    conn->write(&_0module, sizeof(_0module));
    updateTmpPtr((void *)module, _0module);
    conn->write(fname, strlen(fname) + 1, true);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)module, sizeof(*module), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuModuleLoadData(CUmodule *module, const void *image) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleLoadData called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0module;
    mem2server(conn, &_0module, (void *)module, sizeof(*module));
    void *_0image;
    mem2server(conn, &_0image, (void *)image, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuModuleLoadData);
    conn->write(&_0module, sizeof(_0module));
    updateTmpPtr((void *)module, _0module);
    conn->write(&_0image, sizeof(_0image));
    updateTmpPtr((void *)image, _0image);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)module, sizeof(*module), true);
    mem2client(conn, (void *)image, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleLoadDataEx called" << std::endl;
#endif
    // PARAM void **optionValues
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0module;
    mem2server(conn, &_0module, (void *)module, sizeof(*module));
    void *_0image;
    mem2server(conn, &_0image, (void *)image, 0);
    void *_0options;
    mem2server(conn, &_0options, (void *)options, numOptions * sizeof(*options));
    // PARAM void **optionValues
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuModuleLoadDataEx);
    conn->write(&_0module, sizeof(_0module));
    updateTmpPtr((void *)module, _0module);
    conn->write(&_0image, sizeof(_0image));
    updateTmpPtr((void *)image, _0image);
    conn->write(&numOptions, sizeof(numOptions));
    conn->write(&_0options, sizeof(_0options));
    updateTmpPtr((void *)options, _0options);
    // PARAM void **optionValues
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    // PARAM void **optionValues
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)module, sizeof(*module), true);
    mem2client(conn, (void *)image, 0, true);
    mem2client(conn, (void *)options, numOptions * sizeof(*options), true);
    // PARAM void **optionValues
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    // PARAM void **optionValues
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleLoadFatBinary called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0module;
    mem2server(conn, &_0module, (void *)module, sizeof(*module));
    void *_0fatCubin;
    mem2server(conn, &_0fatCubin, (void *)fatCubin, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuModuleLoadFatBinary);
    conn->write(&_0module, sizeof(_0module));
    updateTmpPtr((void *)module, _0module);
    conn->write(&_0fatCubin, sizeof(_0fatCubin));
    updateTmpPtr((void *)fatCubin, _0fatCubin);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)module, sizeof(*module), true);
    mem2client(conn, (void *)fatCubin, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuModuleUnload(CUmodule hmod) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleUnload called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuModuleUnload);
    conn->write(&hmod, sizeof(hmod));
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

extern "C" CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleGetFunction called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0hfunc;
    mem2server(conn, &_0hfunc, (void *)hfunc, sizeof(*hfunc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuModuleGetFunction);
    conn->write(&_0hfunc, sizeof(_0hfunc));
    updateTmpPtr((void *)hfunc, _0hfunc);
    conn->write(&hmod, sizeof(hmod));
    conn->write(name, strlen(name) + 1, true);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)hfunc, sizeof(*hfunc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleGetTexRef called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pTexRef;
    mem2server(conn, &_0pTexRef, (void *)pTexRef, sizeof(*pTexRef));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuModuleGetTexRef);
    conn->write(&_0pTexRef, sizeof(_0pTexRef));
    updateTmpPtr((void *)pTexRef, _0pTexRef);
    conn->write(&hmod, sizeof(hmod));
    conn->write(name, strlen(name) + 1, true);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pTexRef, sizeof(*pTexRef), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleGetSurfRef called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pSurfRef;
    mem2server(conn, &_0pSurfRef, (void *)pSurfRef, sizeof(*pSurfRef));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuModuleGetSurfRef);
    conn->write(&_0pSurfRef, sizeof(_0pSurfRef));
    updateTmpPtr((void *)pSurfRef, _0pSurfRef);
    conn->write(&hmod, sizeof(hmod));
    conn->write(name, strlen(name) + 1, true);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pSurfRef, sizeof(*pSurfRef), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkCreate_v2 called" << std::endl;
#endif
    // PARAM void **optionValues
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0options;
    mem2server(conn, &_0options, (void *)options, numOptions * sizeof(*options));
    // PARAM void **optionValues
    void *_0stateOut;
    mem2server(conn, &_0stateOut, (void *)stateOut, sizeof(*stateOut));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuLinkCreate_v2);
    conn->write(&numOptions, sizeof(numOptions));
    conn->write(&_0options, sizeof(_0options));
    updateTmpPtr((void *)options, _0options);
    // PARAM void **optionValues
    conn->write(&_0stateOut, sizeof(_0stateOut));
    updateTmpPtr((void *)stateOut, _0stateOut);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    // PARAM void **optionValues
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)options, numOptions * sizeof(*options), true);
    // PARAM void **optionValues
    mem2client(conn, (void *)stateOut, sizeof(*stateOut), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    // PARAM void **optionValues
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkAddData_v2 called" << std::endl;
#endif
    // PARAM void **optionValues
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0data;
    mem2server(conn, &_0data, (void *)data, size);
    void *_0options;
    mem2server(conn, &_0options, (void *)options, numOptions * sizeof(*options));
    // PARAM void **optionValues
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuLinkAddData_v2);
    conn->write(&state, sizeof(state));
    conn->write(&type, sizeof(type));
    conn->write(&_0data, sizeof(_0data));
    updateTmpPtr((void *)data, _0data);
    conn->write(&size, sizeof(size));
    conn->write(name, strlen(name) + 1, true);
    conn->write(&numOptions, sizeof(numOptions));
    conn->write(&_0options, sizeof(_0options));
    updateTmpPtr((void *)options, _0options);
    // PARAM void **optionValues
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    // PARAM void **optionValues
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)data, size, true);
    mem2client(conn, (void *)options, numOptions * sizeof(*options), true);
    // PARAM void **optionValues
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    // PARAM void **optionValues
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkAddFile_v2 called" << std::endl;
#endif
    // PARAM void **optionValues
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0options;
    mem2server(conn, &_0options, (void *)options, numOptions * sizeof(*options));
    // PARAM void **optionValues
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuLinkAddFile_v2);
    conn->write(&state, sizeof(state));
    conn->write(&type, sizeof(type));
    conn->write(path, strlen(path) + 1, true);
    conn->write(&numOptions, sizeof(numOptions));
    conn->write(&_0options, sizeof(_0options));
    updateTmpPtr((void *)options, _0options);
    // PARAM void **optionValues
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    // PARAM void **optionValues
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)options, numOptions * sizeof(*options), true);
    // PARAM void **optionValues
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    // PARAM void **optionValues
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkComplete called" << std::endl;
#endif
    // PARAM void **cubinOut
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    // PARAM void **cubinOut
    void *_0sizeOut;
    mem2server(conn, &_0sizeOut, (void *)sizeOut, sizeof(*sizeOut));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuLinkComplete);
    conn->write(&state, sizeof(state));
    // PARAM void **cubinOut
    conn->write(&_0sizeOut, sizeof(_0sizeOut));
    updateTmpPtr((void *)sizeOut, _0sizeOut);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    // PARAM void **cubinOut
    conn->prepare_request(RPC_mem2client);
    // PARAM void **cubinOut
    mem2client(conn, (void *)sizeOut, sizeof(*sizeOut), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    // PARAM void **cubinOut
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuLinkDestroy(CUlinkState state) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkDestroy called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuLinkDestroy);
    conn->write(&state, sizeof(state));
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

extern "C" CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetInfo_v2 called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemGetInfo_v2);
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

extern "C" CUresult cuMemFree_v2(CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemFree_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dptr;
    mem2server(conn, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemFree_v2);
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
#ifdef DEBUG
    std::cout << "Hook: cuMemHostGetFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pFlags;
    mem2server(conn, &_0pFlags, (void *)pFlags, sizeof(*pFlags));
    void *_0p;
    mem2server(conn, &_0p, (void *)p, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemHostGetFlags);
    conn->write(&_0pFlags, sizeof(_0pFlags));
    updateTmpPtr((void *)pFlags, _0pFlags);
    conn->write(&_0p, sizeof(_0p));
    updateTmpPtr((void *)p, _0p);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pFlags, sizeof(*pFlags), true);
    mem2client(conn, (void *)p, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetByPCIBusId called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dev;
    mem2server(conn, &_0dev, (void *)dev, sizeof(*dev));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetByPCIBusId);
    conn->write(&_0dev, sizeof(_0dev));
    updateTmpPtr((void *)dev, _0dev);
    conn->write(pciBusId, strlen(pciBusId) + 1, true);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dev, sizeof(*dev), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetPCIBusId called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetPCIBusId);
    if(len > 0) {
        conn->read(pciBusId, len, true);
    }
    conn->write(&len, sizeof(len));
    conn->write(&dev, sizeof(dev));
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

extern "C" CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcGetEventHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pHandle;
    mem2server(conn, &_0pHandle, (void *)pHandle, sizeof(*pHandle));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuIpcGetEventHandle);
    conn->write(&_0pHandle, sizeof(_0pHandle));
    updateTmpPtr((void *)pHandle, _0pHandle);
    conn->write(&event, sizeof(event));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pHandle, sizeof(*pHandle), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcOpenEventHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phEvent;
    mem2server(conn, &_0phEvent, (void *)phEvent, sizeof(*phEvent));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuIpcOpenEventHandle);
    conn->write(&_0phEvent, sizeof(_0phEvent));
    updateTmpPtr((void *)phEvent, _0phEvent);
    conn->write(&handle, sizeof(handle));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phEvent, sizeof(*phEvent), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcGetMemHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pHandle;
    mem2server(conn, &_0pHandle, (void *)pHandle, sizeof(*pHandle));
    void *_0dptr;
    mem2server(conn, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuIpcGetMemHandle);
    conn->write(&_0pHandle, sizeof(_0pHandle));
    updateTmpPtr((void *)pHandle, _0pHandle);
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pHandle, sizeof(*pHandle), true);
    mem2client(conn, (void *)dptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcCloseMemHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dptr;
    mem2server(conn, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuIpcCloseMemHandle);
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemHostRegister_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0p;
    mem2server(conn, &_0p, (void *)p, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemHostRegister_v2);
    conn->write(&_0p, sizeof(_0p));
    updateTmpPtr((void *)p, _0p);
    conn->write(&bytesize, sizeof(bytesize));
    conn->write(&Flags, sizeof(Flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)p, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemHostUnregister(void *p) {
#ifdef DEBUG
    std::cout << "Hook: cuMemHostUnregister called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0p;
    mem2server(conn, &_0p, (void *)p, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemHostUnregister);
    conn->write(&_0p, sizeof(_0p));
    updateTmpPtr((void *)p, _0p);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)p, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, -1);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpy);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, -1, true);
    mem2client(conn, (void *)src, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyPeer called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *_0srcDevice;
    mem2server(conn, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyPeer);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&dstContext, sizeof(dstContext));
    conn->write(&_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    conn->write(&srcContext, sizeof(srcContext));
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    mem2client(conn, (void *)srcDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyHtoD_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *_0srcHost;
    mem2server(conn, &_0srcHost, (void *)srcHost, ByteCount);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyHtoD_v2);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&_0srcHost, sizeof(_0srcHost));
    updateTmpPtr((void *)srcHost, _0srcHost);
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    mem2client(conn, (void *)srcHost, ByteCount, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoH_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstHost;
    mem2server(conn, &_0dstHost, (void *)dstHost, ByteCount);
    void *_0srcDevice;
    mem2server(conn, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyDtoH_v2);
    conn->write(&_0dstHost, sizeof(_0dstHost));
    updateTmpPtr((void *)dstHost, _0dstHost);
    conn->write(&_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstHost, ByteCount, true);
    mem2client(conn, (void *)srcDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoD_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *_0srcDevice;
    mem2server(conn, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyDtoD_v2);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    mem2client(conn, (void *)srcDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoA_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0srcDevice;
    mem2server(conn, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyDtoA_v2);
    conn->write(&dstArray, sizeof(dstArray));
    conn->write(&dstOffset, sizeof(dstOffset));
    conn->write(&_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)srcDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAtoD_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyAtoD_v2);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&srcArray, sizeof(srcArray));
    conn->write(&srcOffset, sizeof(srcOffset));
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyHtoA_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0srcHost;
    mem2server(conn, &_0srcHost, (void *)srcHost, ByteCount);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyHtoA_v2);
    conn->write(&dstArray, sizeof(dstArray));
    conn->write(&dstOffset, sizeof(dstOffset));
    conn->write(&_0srcHost, sizeof(_0srcHost));
    updateTmpPtr((void *)srcHost, _0srcHost);
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)srcHost, ByteCount, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAtoH_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstHost;
    mem2server(conn, &_0dstHost, (void *)dstHost, ByteCount);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyAtoH_v2);
    conn->write(&_0dstHost, sizeof(_0dstHost));
    updateTmpPtr((void *)dstHost, _0dstHost);
    conn->write(&srcArray, sizeof(srcArray));
    conn->write(&srcOffset, sizeof(srcOffset));
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstHost, ByteCount, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAtoA_v2 called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyAtoA_v2);
    conn->write(&dstArray, sizeof(dstArray));
    conn->write(&dstOffset, sizeof(dstOffset));
    conn->write(&srcArray, sizeof(srcArray));
    conn->write(&srcOffset, sizeof(srcOffset));
    conn->write(&ByteCount, sizeof(ByteCount));
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

extern "C" CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy2D_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pCopy;
    mem2server(conn, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpy2D_v2);
    conn->write(&_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pCopy, sizeof(*pCopy), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy2DUnaligned_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pCopy;
    mem2server(conn, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpy2DUnaligned_v2);
    conn->write(&_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pCopy, sizeof(*pCopy), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy3D_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pCopy;
    mem2server(conn, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpy3D_v2);
    conn->write(&_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pCopy, sizeof(*pCopy), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy3DPeer called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pCopy;
    mem2server(conn, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpy3DPeer);
    conn->write(&_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pCopy, sizeof(*pCopy), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dst;
    mem2server(conn, &_0dst, (void *)dst, -1);
    void *_0src;
    mem2server(conn, &_0src, (void *)src, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyAsync);
    conn->write(&_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    conn->write(&_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dst, -1, true);
    mem2client(conn, (void *)src, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyPeerAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *_0srcDevice;
    mem2server(conn, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyPeerAsync);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&dstContext, sizeof(dstContext));
    conn->write(&_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    conn->write(&srcContext, sizeof(srcContext));
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    mem2client(conn, (void *)srcDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyHtoDAsync_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *_0srcHost;
    mem2server(conn, &_0srcHost, (void *)srcHost, ByteCount);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyHtoDAsync_v2);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&_0srcHost, sizeof(_0srcHost));
    updateTmpPtr((void *)srcHost, _0srcHost);
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    mem2client(conn, (void *)srcHost, ByteCount, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoHAsync_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstHost;
    mem2server(conn, &_0dstHost, (void *)dstHost, ByteCount);
    void *_0srcDevice;
    mem2server(conn, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyDtoHAsync_v2);
    conn->write(&_0dstHost, sizeof(_0dstHost));
    updateTmpPtr((void *)dstHost, _0dstHost);
    conn->write(&_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstHost, ByteCount, true);
    mem2client(conn, (void *)srcDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoDAsync_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *_0srcDevice;
    mem2server(conn, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyDtoDAsync_v2);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    mem2client(conn, (void *)srcDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyHtoAAsync_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0srcHost;
    mem2server(conn, &_0srcHost, (void *)srcHost, ByteCount);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyHtoAAsync_v2);
    conn->write(&dstArray, sizeof(dstArray));
    conn->write(&dstOffset, sizeof(dstOffset));
    conn->write(&_0srcHost, sizeof(_0srcHost));
    updateTmpPtr((void *)srcHost, _0srcHost);
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)srcHost, ByteCount, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAtoHAsync_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstHost;
    mem2server(conn, &_0dstHost, (void *)dstHost, ByteCount);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpyAtoHAsync_v2);
    conn->write(&_0dstHost, sizeof(_0dstHost));
    updateTmpPtr((void *)dstHost, _0dstHost);
    conn->write(&srcArray, sizeof(srcArray));
    conn->write(&srcOffset, sizeof(srcOffset));
    conn->write(&ByteCount, sizeof(ByteCount));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstHost, ByteCount, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy2DAsync_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pCopy;
    mem2server(conn, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpy2DAsync_v2);
    conn->write(&_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pCopy, sizeof(*pCopy), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy3DAsync_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pCopy;
    mem2server(conn, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpy3DAsync_v2);
    conn->write(&_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pCopy, sizeof(*pCopy), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy3DPeerAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pCopy;
    mem2server(conn, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemcpy3DPeerAsync);
    conn->write(&_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pCopy, sizeof(*pCopy), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD8_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemsetD8_v2);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&uc, sizeof(uc));
    conn->write(&N, sizeof(N));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD16_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemsetD16_v2);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&us, sizeof(us));
    conn->write(&N, sizeof(N));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD32_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemsetD32_v2);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&ui, sizeof(ui));
    conn->write(&N, sizeof(N));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D8_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemsetD2D8_v2);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&dstPitch, sizeof(dstPitch));
    conn->write(&uc, sizeof(uc));
    conn->write(&Width, sizeof(Width));
    conn->write(&Height, sizeof(Height));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D16_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemsetD2D16_v2);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&dstPitch, sizeof(dstPitch));
    conn->write(&us, sizeof(us));
    conn->write(&Width, sizeof(Width));
    conn->write(&Height, sizeof(Height));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D32_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemsetD2D32_v2);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&dstPitch, sizeof(dstPitch));
    conn->write(&ui, sizeof(ui));
    conn->write(&Width, sizeof(Width));
    conn->write(&Height, sizeof(Height));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD8Async called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemsetD8Async);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&uc, sizeof(uc));
    conn->write(&N, sizeof(N));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD16Async called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemsetD16Async);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&us, sizeof(us));
    conn->write(&N, sizeof(N));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD32Async called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemsetD32Async);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&ui, sizeof(ui));
    conn->write(&N, sizeof(N));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D8Async called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemsetD2D8Async);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&dstPitch, sizeof(dstPitch));
    conn->write(&uc, sizeof(uc));
    conn->write(&Width, sizeof(Width));
    conn->write(&Height, sizeof(Height));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D16Async called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemsetD2D16Async);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&dstPitch, sizeof(dstPitch));
    conn->write(&us, sizeof(us));
    conn->write(&Width, sizeof(Width));
    conn->write(&Height, sizeof(Height));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D32Async called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dstDevice;
    mem2server(conn, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemsetD2D32Async);
    conn->write(&_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    conn->write(&dstPitch, sizeof(dstPitch));
    conn->write(&ui, sizeof(ui));
    conn->write(&Width, sizeof(Width));
    conn->write(&Height, sizeof(Height));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dstDevice, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuArrayCreate_v2(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayCreate_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pHandle;
    mem2server(conn, &_0pHandle, (void *)pHandle, sizeof(*pHandle));
    void *_0pAllocateArray;
    mem2server(conn, &_0pAllocateArray, (void *)pAllocateArray, sizeof(*pAllocateArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuArrayCreate_v2);
    conn->write(&_0pHandle, sizeof(_0pHandle));
    updateTmpPtr((void *)pHandle, _0pHandle);
    conn->write(&_0pAllocateArray, sizeof(_0pAllocateArray));
    updateTmpPtr((void *)pAllocateArray, _0pAllocateArray);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pHandle, sizeof(*pHandle), true);
    mem2client(conn, (void *)pAllocateArray, sizeof(*pAllocateArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayGetDescriptor_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pArrayDescriptor;
    mem2server(conn, &_0pArrayDescriptor, (void *)pArrayDescriptor, sizeof(*pArrayDescriptor));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuArrayGetDescriptor_v2);
    conn->write(&_0pArrayDescriptor, sizeof(_0pArrayDescriptor));
    updateTmpPtr((void *)pArrayDescriptor, _0pArrayDescriptor);
    conn->write(&hArray, sizeof(hArray));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pArrayDescriptor, sizeof(*pArrayDescriptor), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUarray array) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayGetSparseProperties called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuArrayGetSparseProperties);
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

extern "C" CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUmipmappedArray mipmap) {
#ifdef DEBUG
    std::cout << "Hook: cuMipmappedArrayGetSparseProperties called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMipmappedArrayGetSparseProperties);
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

extern "C" CUresult cuArrayGetPlane(CUarray *pPlaneArray, CUarray hArray, unsigned int planeIdx) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayGetPlane called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuArrayGetPlane);
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

extern "C" CUresult cuArrayDestroy(CUarray hArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayDestroy called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuArrayDestroy);
    conn->write(&hArray, sizeof(hArray));
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

extern "C" CUresult cuArray3DCreate_v2(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArray3DCreate_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pHandle;
    mem2server(conn, &_0pHandle, (void *)pHandle, sizeof(*pHandle));
    void *_0pAllocateArray;
    mem2server(conn, &_0pAllocateArray, (void *)pAllocateArray, sizeof(*pAllocateArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuArray3DCreate_v2);
    conn->write(&_0pHandle, sizeof(_0pHandle));
    updateTmpPtr((void *)pHandle, _0pHandle);
    conn->write(&_0pAllocateArray, sizeof(_0pAllocateArray));
    updateTmpPtr((void *)pAllocateArray, _0pAllocateArray);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pHandle, sizeof(*pHandle), true);
    mem2client(conn, (void *)pAllocateArray, sizeof(*pAllocateArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArray3DGetDescriptor_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pArrayDescriptor;
    mem2server(conn, &_0pArrayDescriptor, (void *)pArrayDescriptor, sizeof(*pArrayDescriptor));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuArray3DGetDescriptor_v2);
    conn->write(&_0pArrayDescriptor, sizeof(_0pArrayDescriptor));
    updateTmpPtr((void *)pArrayDescriptor, _0pArrayDescriptor);
    conn->write(&hArray, sizeof(hArray));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pArrayDescriptor, sizeof(*pArrayDescriptor), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMipmappedArrayCreate(CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels) {
#ifdef DEBUG
    std::cout << "Hook: cuMipmappedArrayCreate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pHandle;
    mem2server(conn, &_0pHandle, (void *)pHandle, sizeof(*pHandle));
    void *_0pMipmappedArrayDesc;
    mem2server(conn, &_0pMipmappedArrayDesc, (void *)pMipmappedArrayDesc, sizeof(*pMipmappedArrayDesc));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMipmappedArrayCreate);
    conn->write(&_0pHandle, sizeof(_0pHandle));
    updateTmpPtr((void *)pHandle, _0pHandle);
    conn->write(&_0pMipmappedArrayDesc, sizeof(_0pMipmappedArrayDesc));
    updateTmpPtr((void *)pMipmappedArrayDesc, _0pMipmappedArrayDesc);
    conn->write(&numMipmapLevels, sizeof(numMipmapLevels));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pHandle, sizeof(*pHandle), true);
    mem2client(conn, (void *)pMipmappedArrayDesc, sizeof(*pMipmappedArrayDesc), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMipmappedArrayGetLevel(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) {
#ifdef DEBUG
    std::cout << "Hook: cuMipmappedArrayGetLevel called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pLevelArray;
    mem2server(conn, &_0pLevelArray, (void *)pLevelArray, sizeof(*pLevelArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMipmappedArrayGetLevel);
    conn->write(&_0pLevelArray, sizeof(_0pLevelArray));
    updateTmpPtr((void *)pLevelArray, _0pLevelArray);
    conn->write(&hMipmappedArray, sizeof(hMipmappedArray));
    conn->write(&level, sizeof(level));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pLevelArray, sizeof(*pLevelArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
#ifdef DEBUG
    std::cout << "Hook: cuMipmappedArrayDestroy called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMipmappedArrayDestroy);
    conn->write(&hMipmappedArray, sizeof(hMipmappedArray));
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

extern "C" CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAddressFree called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0ptr;
    mem2server(conn, &_0ptr, (void *)ptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemAddressFree);
    conn->write(&_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    conn->write(&size, sizeof(size));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)ptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemMapArrayAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0mapInfoList;
    mem2server(conn, &_0mapInfoList, (void *)mapInfoList, sizeof(*mapInfoList));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemMapArrayAsync);
    conn->write(&_0mapInfoList, sizeof(_0mapInfoList));
    updateTmpPtr((void *)mapInfoList, _0mapInfoList);
    conn->write(&count, sizeof(count));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)mapInfoList, sizeof(*mapInfoList), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
#ifdef DEBUG
    std::cout << "Hook: cuMemUnmap called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0ptr;
    mem2server(conn, &_0ptr, (void *)ptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemUnmap);
    conn->write(&_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    conn->write(&size, sizeof(size));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)ptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cuMemSetAccess called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0ptr;
    mem2server(conn, &_0ptr, (void *)ptr, -1);
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemSetAccess);
    conn->write(&_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    conn->write(&size, sizeof(size));
    conn->write(&_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    conn->write(&count, sizeof(count));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)ptr, -1, true);
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

extern "C" CUresult cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetAccess called" << std::endl;
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
    void *_0ptr;
    mem2server(conn, &_0ptr, (void *)ptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemGetAccess);
    conn->write(&_0flags, sizeof(_0flags));
    updateTmpPtr((void *)flags, _0flags);
    conn->write(&_0location, sizeof(_0location));
    updateTmpPtr((void *)location, _0location);
    conn->write(&_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)flags, sizeof(*flags), true);
    mem2client(conn, (void *)location, sizeof(*location), true);
    mem2client(conn, (void *)ptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemExportToShareableHandle called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemExportToShareableHandle);
    conn->write(&_0shareableHandle, sizeof(_0shareableHandle));
    updateTmpPtr((void *)shareableHandle, _0shareableHandle);
    conn->write(&handle, sizeof(handle));
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

extern "C" CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle, CUmemAllocationHandleType shHandleType) {
#ifdef DEBUG
    std::cout << "Hook: cuMemImportFromShareableHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0handle;
    mem2server(conn, &_0handle, (void *)handle, sizeof(*handle));
    void *_0osHandle;
    mem2server(conn, &_0osHandle, (void *)osHandle, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemImportFromShareableHandle);
    conn->write(&_0handle, sizeof(_0handle));
    updateTmpPtr((void *)handle, _0handle);
    conn->write(&_0osHandle, sizeof(_0osHandle));
    updateTmpPtr((void *)osHandle, _0osHandle);
    conn->write(&shHandleType, sizeof(shHandleType));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)handle, sizeof(*handle), true);
    mem2client(conn, (void *)osHandle, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetAllocationGranularity called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0granularity;
    mem2server(conn, &_0granularity, (void *)granularity, sizeof(*granularity));
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemGetAllocationGranularity);
    conn->write(&_0granularity, sizeof(_0granularity));
    updateTmpPtr((void *)granularity, _0granularity);
    conn->write(&_0prop, sizeof(_0prop));
    updateTmpPtr((void *)prop, _0prop);
    conn->write(&option, sizeof(option));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)granularity, sizeof(*granularity), true);
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

extern "C" CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetAllocationPropertiesFromHandle called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemGetAllocationPropertiesFromHandle);
    conn->write(&_0prop, sizeof(_0prop));
    updateTmpPtr((void *)prop, _0prop);
    conn->write(&handle, sizeof(handle));
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

extern "C" CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemRetainAllocationHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0handle;
    mem2server(conn, &_0handle, (void *)handle, sizeof(*handle));
    void *_0addr;
    mem2server(conn, &_0addr, (void *)addr, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemRetainAllocationHandle);
    conn->write(&_0handle, sizeof(_0handle));
    updateTmpPtr((void *)handle, _0handle);
    conn->write(&_0addr, sizeof(_0addr));
    updateTmpPtr((void *)addr, _0addr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)handle, sizeof(*handle), true);
    mem2client(conn, (void *)addr, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemFreeAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dptr;
    mem2server(conn, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemFreeAsync);
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAllocAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dptr;
    mem2server(conn, &_0dptr, (void *)dptr, sizeof(*dptr));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemAllocAsync);
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    conn->write(&bytesize, sizeof(bytesize));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dptr, sizeof(*dptr), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolTrimTo called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemPoolTrimTo);
    conn->write(&pool, sizeof(pool));
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

extern "C" CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolSetAttribute called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemPoolSetAttribute);
    conn->write(&pool, sizeof(pool));
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

extern "C" CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolGetAttribute called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemPoolGetAttribute);
    conn->write(&pool, sizeof(pool));
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

extern "C" CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolSetAccess called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0map;
    mem2server(conn, &_0map, (void *)map, sizeof(*map));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemPoolSetAccess);
    conn->write(&pool, sizeof(pool));
    conn->write(&_0map, sizeof(_0map));
    updateTmpPtr((void *)map, _0map);
    conn->write(&count, sizeof(count));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)map, sizeof(*map), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool, CUmemLocation *location) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolGetAccess called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemPoolGetAccess);
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

extern "C" CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolCreate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pool;
    mem2server(conn, &_0pool, (void *)pool, sizeof(*pool));
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemPoolCreate);
    conn->write(&_0pool, sizeof(_0pool));
    updateTmpPtr((void *)pool, _0pool);
    conn->write(&_0poolProps, sizeof(_0poolProps));
    updateTmpPtr((void *)poolProps, _0poolProps);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pool, sizeof(*pool), true);
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

extern "C" CUresult cuMemPoolDestroy(CUmemoryPool pool) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolDestroy called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemPoolDestroy);
    conn->write(&pool, sizeof(pool));
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

extern "C" CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAllocFromPoolAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dptr;
    mem2server(conn, &_0dptr, (void *)dptr, sizeof(*dptr));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemAllocFromPoolAsync);
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    conn->write(&bytesize, sizeof(bytesize));
    conn->write(&pool, sizeof(pool));
    conn->write(&hStream, sizeof(hStream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dptr, sizeof(*dptr), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolExportToShareableHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0handle_out;
    mem2server(conn, &_0handle_out, (void *)handle_out, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemPoolExportToShareableHandle);
    conn->write(&_0handle_out, sizeof(_0handle_out));
    updateTmpPtr((void *)handle_out, _0handle_out);
    conn->write(&pool, sizeof(pool));
    conn->write(&handleType, sizeof(handleType));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)handle_out, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool *pool_out, void *handle, CUmemAllocationHandleType handleType, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolImportFromShareableHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pool_out;
    mem2server(conn, &_0pool_out, (void *)pool_out, sizeof(*pool_out));
    void *_0handle;
    mem2server(conn, &_0handle, (void *)handle, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemPoolImportFromShareableHandle);
    conn->write(&_0pool_out, sizeof(_0pool_out));
    updateTmpPtr((void *)pool_out, _0pool_out);
    conn->write(&_0handle, sizeof(_0handle));
    updateTmpPtr((void *)handle, _0handle);
    conn->write(&handleType, sizeof(handleType));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pool_out, sizeof(*pool_out), true);
    mem2client(conn, (void *)handle, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolExportPointer called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0shareData_out;
    mem2server(conn, &_0shareData_out, (void *)shareData_out, sizeof(*shareData_out));
    void *_0ptr;
    mem2server(conn, &_0ptr, (void *)ptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuMemPoolExportPointer);
    conn->write(&_0shareData_out, sizeof(_0shareData_out));
    updateTmpPtr((void *)shareData_out, _0shareData_out);
    conn->write(&_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)shareData_out, sizeof(*shareData_out), true);
    mem2client(conn, (void *)ptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuPointerGetAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0data;
    mem2server(conn, &_0data, (void *)data, 0);
    void *_0ptr;
    mem2server(conn, &_0ptr, (void *)ptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuPointerGetAttribute);
    conn->write(&_0data, sizeof(_0data));
    updateTmpPtr((void *)data, _0data);
    conn->write(&attribute, sizeof(attribute));
    conn->write(&_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)data, 0, true);
    mem2client(conn, (void *)ptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPrefetchAsync called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemPrefetchAsync);
    conn->write(&_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    conn->write(&count, sizeof(count));
    conn->write(&dstDevice, sizeof(dstDevice));
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

extern "C" CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAdvise called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemAdvise);
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

extern "C" CUresult cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cuMemRangeGetAttribute called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuMemRangeGetAttribute);
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

extern "C" CUresult cuPointerSetAttribute(const void *value, CUpointer_attribute attribute, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuPointerSetAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0value;
    mem2server(conn, &_0value, (void *)value, 0);
    void *_0ptr;
    mem2server(conn, &_0ptr, (void *)ptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuPointerSetAttribute);
    conn->write(&_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    conn->write(&attribute, sizeof(attribute));
    conn->write(&_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)value, 0, true);
    mem2client(conn, (void *)ptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamCreate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phStream;
    mem2server(conn, &_0phStream, (void *)phStream, sizeof(*phStream));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuStreamCreate);
    conn->write(&_0phStream, sizeof(_0phStream));
    updateTmpPtr((void *)phStream, _0phStream);
    conn->write(&Flags, sizeof(Flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phStream, sizeof(*phStream), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamCreateWithPriority called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phStream;
    mem2server(conn, &_0phStream, (void *)phStream, sizeof(*phStream));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuStreamCreateWithPriority);
    conn->write(&_0phStream, sizeof(_0phStream));
    updateTmpPtr((void *)phStream, _0phStream);
    conn->write(&flags, sizeof(flags));
    conn->write(&priority, sizeof(priority));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phStream, sizeof(*phStream), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetPriority called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuStreamGetPriority);
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

extern "C" CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetFlags called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuStreamGetFlags);
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

extern "C" CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetCtx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pctx;
    mem2server(conn, &_0pctx, (void *)pctx, sizeof(*pctx));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuStreamGetCtx);
    conn->write(&hStream, sizeof(hStream));
    conn->write(&_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pctx, sizeof(*pctx), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWaitEvent called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuStreamWaitEvent);
    conn->write(&hStream, sizeof(hStream));
    conn->write(&hEvent, sizeof(hEvent));
    conn->write(&Flags, sizeof(Flags));
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

extern "C" CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamAddCallback called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuStreamAddCallback);
    conn->write(&hStream, sizeof(hStream));
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

extern "C" CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamBeginCapture_v2 called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuStreamBeginCapture_v2);
    conn->write(&hStream, sizeof(hStream));
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

extern "C" CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode) {
#ifdef DEBUG
    std::cout << "Hook: cuThreadExchangeStreamCaptureMode called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuThreadExchangeStreamCaptureMode);
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

extern "C" CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamEndCapture called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraph;
    mem2server(conn, &_0phGraph, (void *)phGraph, sizeof(*phGraph));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuStreamEndCapture);
    conn->write(&hStream, sizeof(hStream));
    conn->write(&_0phGraph, sizeof(_0phGraph));
    updateTmpPtr((void *)phGraph, _0phGraph);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phGraph, sizeof(*phGraph), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus *captureStatus) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamIsCapturing called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0captureStatus;
    mem2server(conn, &_0captureStatus, (void *)captureStatus, sizeof(*captureStatus));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuStreamIsCapturing);
    conn->write(&hStream, sizeof(hStream));
    conn->write(&_0captureStatus, sizeof(_0captureStatus));
    updateTmpPtr((void *)captureStatus, _0captureStatus);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)captureStatus, sizeof(*captureStatus), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetCaptureInfo called" << std::endl;
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
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuStreamGetCaptureInfo);
    conn->write(&hStream, sizeof(hStream));
    conn->write(&_0captureStatus_out, sizeof(_0captureStatus_out));
    updateTmpPtr((void *)captureStatus_out, _0captureStatus_out);
    conn->write(&_0id_out, sizeof(_0id_out));
    updateTmpPtr((void *)id_out, _0id_out);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)captureStatus_out, sizeof(*captureStatus_out), true);
    mem2client(conn, (void *)id_out, sizeof(*id_out), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetCaptureInfo_v2 called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuStreamGetCaptureInfo_v2);
    conn->write(&hStream, sizeof(hStream));
    conn->write(&_0captureStatus_out, sizeof(_0captureStatus_out));
    updateTmpPtr((void *)captureStatus_out, _0captureStatus_out);
    conn->write(&_0id_out, sizeof(_0id_out));
    updateTmpPtr((void *)id_out, _0id_out);
    conn->write(&_0graph_out, sizeof(_0graph_out));
    updateTmpPtr((void *)graph_out, _0graph_out);
    static CUgraphNode _cuStreamGetCaptureInfo_v2_dependencies_out;
    conn->read(&_cuStreamGetCaptureInfo_v2_dependencies_out, sizeof(CUgraphNode));
    conn->write(&_0numDependencies_out, sizeof(_0numDependencies_out));
    updateTmpPtr((void *)numDependencies_out, _0numDependencies_out);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    *dependencies_out = &_cuStreamGetCaptureInfo_v2_dependencies_out;
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

extern "C" CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode *dependencies, size_t numDependencies, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamUpdateCaptureDependencies called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuStreamUpdateCaptureDependencies);
    conn->write(&hStream, sizeof(hStream));
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

extern "C" CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamAttachMemAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dptr;
    mem2server(conn, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuStreamAttachMemAsync);
    conn->write(&hStream, sizeof(hStream));
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    conn->write(&length, sizeof(length));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuStreamQuery(CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamQuery called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuStreamQuery);
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

extern "C" CUresult cuStreamSynchronize(CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamSynchronize called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuStreamSynchronize);
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

extern "C" CUresult cuStreamDestroy_v2(CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamDestroy_v2 called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuStreamDestroy_v2);
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

extern "C" CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamCopyAttributes called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuStreamCopyAttributes);
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

extern "C" CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue *value_out) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetAttribute called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuStreamGetAttribute);
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

extern "C" CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue *value) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamSetAttribute called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuStreamSetAttribute);
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

extern "C" CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuEventCreate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phEvent;
    mem2server(conn, &_0phEvent, (void *)phEvent, sizeof(*phEvent));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuEventCreate);
    conn->write(&_0phEvent, sizeof(_0phEvent));
    updateTmpPtr((void *)phEvent, _0phEvent);
    conn->write(&Flags, sizeof(Flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phEvent, sizeof(*phEvent), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuEventRecord called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuEventRecord);
    conn->write(&hEvent, sizeof(hEvent));
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

extern "C" CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuEventRecordWithFlags called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuEventRecordWithFlags);
    conn->write(&hEvent, sizeof(hEvent));
    conn->write(&hStream, sizeof(hStream));
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

extern "C" CUresult cuEventQuery(CUevent hEvent) {
#ifdef DEBUG
    std::cout << "Hook: cuEventQuery called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuEventQuery);
    conn->write(&hEvent, sizeof(hEvent));
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

extern "C" CUresult cuEventSynchronize(CUevent hEvent) {
#ifdef DEBUG
    std::cout << "Hook: cuEventSynchronize called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuEventSynchronize);
    conn->write(&hEvent, sizeof(hEvent));
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

extern "C" CUresult cuEventDestroy_v2(CUevent hEvent) {
#ifdef DEBUG
    std::cout << "Hook: cuEventDestroy_v2 called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuEventDestroy_v2);
    conn->write(&hEvent, sizeof(hEvent));
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

extern "C" CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
#ifdef DEBUG
    std::cout << "Hook: cuEventElapsedTime called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pMilliseconds;
    mem2server(conn, &_0pMilliseconds, (void *)pMilliseconds, sizeof(*pMilliseconds));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuEventElapsedTime);
    conn->write(&_0pMilliseconds, sizeof(_0pMilliseconds));
    updateTmpPtr((void *)pMilliseconds, _0pMilliseconds);
    conn->write(&hStart, sizeof(hStart));
    conn->write(&hEnd, sizeof(hEnd));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pMilliseconds, sizeof(*pMilliseconds), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuExternalMemoryGetMappedMipmappedArray called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuExternalMemoryGetMappedMipmappedArray);
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

extern "C" CUresult cuDestroyExternalMemory(CUexternalMemory extMem) {
#ifdef DEBUG
    std::cout << "Hook: cuDestroyExternalMemory called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDestroyExternalMemory);
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

extern "C" CUresult cuImportExternalSemaphore(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuImportExternalSemaphore called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuImportExternalSemaphore);
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

extern "C" CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream) {
#ifdef DEBUG
    std::cout << "Hook: cuSignalExternalSemaphoresAsync called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuSignalExternalSemaphoresAsync);
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

extern "C" CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream) {
#ifdef DEBUG
    std::cout << "Hook: cuWaitExternalSemaphoresAsync called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuWaitExternalSemaphoresAsync);
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

extern "C" CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
#ifdef DEBUG
    std::cout << "Hook: cuDestroyExternalSemaphore called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDestroyExternalSemaphore);
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

extern "C" CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWaitValue32 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0addr;
    mem2server(conn, &_0addr, (void *)addr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuStreamWaitValue32);
    conn->write(&stream, sizeof(stream));
    conn->write(&_0addr, sizeof(_0addr));
    updateTmpPtr((void *)addr, _0addr);
    conn->write(&value, sizeof(value));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)addr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWaitValue64 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0addr;
    mem2server(conn, &_0addr, (void *)addr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuStreamWaitValue64);
    conn->write(&stream, sizeof(stream));
    conn->write(&_0addr, sizeof(_0addr));
    updateTmpPtr((void *)addr, _0addr);
    conn->write(&value, sizeof(value));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)addr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWriteValue32 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0addr;
    mem2server(conn, &_0addr, (void *)addr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuStreamWriteValue32);
    conn->write(&stream, sizeof(stream));
    conn->write(&_0addr, sizeof(_0addr));
    updateTmpPtr((void *)addr, _0addr);
    conn->write(&value, sizeof(value));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)addr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWriteValue64 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0addr;
    mem2server(conn, &_0addr, (void *)addr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuStreamWriteValue64);
    conn->write(&stream, sizeof(stream));
    conn->write(&_0addr, sizeof(_0addr));
    updateTmpPtr((void *)addr, _0addr);
    conn->write(&value, sizeof(value));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)addr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamBatchMemOp called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0paramArray;
    mem2server(conn, &_0paramArray, (void *)paramArray, sizeof(*paramArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuStreamBatchMemOp);
    conn->write(&stream, sizeof(stream));
    conn->write(&count, sizeof(count));
    conn->write(&_0paramArray, sizeof(_0paramArray));
    updateTmpPtr((void *)paramArray, _0paramArray);
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)paramArray, sizeof(*paramArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncGetAttribute called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pi;
    mem2server(conn, &_0pi, (void *)pi, sizeof(*pi));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuFuncGetAttribute);
    conn->write(&_0pi, sizeof(_0pi));
    updateTmpPtr((void *)pi, _0pi);
    conn->write(&attrib, sizeof(attrib));
    conn->write(&hfunc, sizeof(hfunc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pi, sizeof(*pi), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetAttribute called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuFuncSetAttribute);
    conn->write(&hfunc, sizeof(hfunc));
    conn->write(&attrib, sizeof(attrib));
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

extern "C" CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetCacheConfig called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuFuncSetCacheConfig);
    conn->write(&hfunc, sizeof(hfunc));
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

extern "C" CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetSharedMemConfig called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuFuncSetSharedMemConfig);
    conn->write(&hfunc, sizeof(hfunc));
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

extern "C" CUresult cuFuncGetModule(CUmodule *hmod, CUfunction hfunc) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncGetModule called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0hmod;
    mem2server(conn, &_0hmod, (void *)hmod, sizeof(*hmod));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuFuncGetModule);
    conn->write(&_0hmod, sizeof(_0hmod));
    updateTmpPtr((void *)hmod, _0hmod);
    conn->write(&hfunc, sizeof(hfunc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)hmod, sizeof(*hmod), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchKernel called" << std::endl;
#endif
    // PARAM void **kernelParams
    // PARAM void **extra
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    // PARAM void **kernelParams
    // PARAM void **extra
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuLaunchKernel);
    conn->write(&f, sizeof(f));
    conn->write(&gridDimX, sizeof(gridDimX));
    conn->write(&gridDimY, sizeof(gridDimY));
    conn->write(&gridDimZ, sizeof(gridDimZ));
    conn->write(&blockDimX, sizeof(blockDimX));
    conn->write(&blockDimY, sizeof(blockDimY));
    conn->write(&blockDimZ, sizeof(blockDimZ));
    conn->write(&sharedMemBytes, sizeof(sharedMemBytes));
    conn->write(&hStream, sizeof(hStream));
    // PARAM void **kernelParams
    // PARAM void **extra
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    // PARAM void **kernelParams
    // PARAM void **extra
    conn->prepare_request(RPC_mem2client);
    // PARAM void **kernelParams
    // PARAM void **extra
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    // PARAM void **kernelParams
    // PARAM void **extra
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchCooperativeKernelMultiDevice called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuLaunchCooperativeKernelMultiDevice);
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

extern "C" CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchHostFunc called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuLaunchHostFunc);
    conn->write(&hStream, sizeof(hStream));
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

extern "C" CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetBlockShape called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuFuncSetBlockShape);
    conn->write(&hfunc, sizeof(hfunc));
    conn->write(&x, sizeof(x));
    conn->write(&y, sizeof(y));
    conn->write(&z, sizeof(z));
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

extern "C" CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetSharedSize called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuFuncSetSharedSize);
    conn->write(&hfunc, sizeof(hfunc));
    conn->write(&bytes, sizeof(bytes));
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

extern "C" CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSetSize called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuParamSetSize);
    conn->write(&hfunc, sizeof(hfunc));
    conn->write(&numbytes, sizeof(numbytes));
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

extern "C" CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSeti called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuParamSeti);
    conn->write(&hfunc, sizeof(hfunc));
    conn->write(&offset, sizeof(offset));
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

extern "C" CUresult cuParamSetf(CUfunction hfunc, int offset, float value) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSetf called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuParamSetf);
    conn->write(&hfunc, sizeof(hfunc));
    conn->write(&offset, sizeof(offset));
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

extern "C" CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSetv called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0ptr;
    mem2server(conn, &_0ptr, (void *)ptr, numbytes);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuParamSetv);
    conn->write(&hfunc, sizeof(hfunc));
    conn->write(&offset, sizeof(offset));
    conn->write(&_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    conn->write(&numbytes, sizeof(numbytes));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)ptr, numbytes, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuLaunch(CUfunction f) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunch called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuLaunch);
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

extern "C" CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchGrid called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuLaunchGrid);
    conn->write(&f, sizeof(f));
    conn->write(&grid_width, sizeof(grid_width));
    conn->write(&grid_height, sizeof(grid_height));
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

extern "C" CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchGridAsync called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuLaunchGridAsync);
    conn->write(&f, sizeof(f));
    conn->write(&grid_width, sizeof(grid_width));
    conn->write(&grid_height, sizeof(grid_height));
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

extern "C" CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSetTexRef called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuParamSetTexRef);
    conn->write(&hfunc, sizeof(hfunc));
    conn->write(&texunit, sizeof(texunit));
    conn->write(&hTexRef, sizeof(hTexRef));
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

extern "C" CUresult cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphCreate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraph;
    mem2server(conn, &_0phGraph, (void *)phGraph, sizeof(*phGraph));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphCreate);
    conn->write(&_0phGraph, sizeof(_0phGraph));
    updateTmpPtr((void *)phGraph, _0phGraph);
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phGraph, sizeof(*phGraph), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddKernelNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphNode;
    mem2server(conn, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(conn, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphAddKernelNode);
    conn->write(&_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
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
    mem2client(conn, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(conn, (void *)dependencies, sizeof(*dependencies), true);
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

extern "C" CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeGetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphKernelNodeGetParams);
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

extern "C" CUresult cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeSetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphKernelNodeSetParams);
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

extern "C" CUresult cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddMemcpyNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphNode;
    mem2server(conn, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(conn, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
    void *_0copyParams;
    mem2server(conn, &_0copyParams, (void *)copyParams, sizeof(*copyParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphAddMemcpyNode);
    conn->write(&_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&_0copyParams, sizeof(_0copyParams));
    updateTmpPtr((void *)copyParams, _0copyParams);
    conn->write(&ctx, sizeof(ctx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(conn, (void *)dependencies, sizeof(*dependencies), true);
    mem2client(conn, (void *)copyParams, sizeof(*copyParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemcpyNodeGetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphMemcpyNodeGetParams);
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

extern "C" CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemcpyNodeSetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphMemcpyNodeSetParams);
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

extern "C" CUresult cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddMemsetNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphNode;
    mem2server(conn, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(conn, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
    void *_0memsetParams;
    mem2server(conn, &_0memsetParams, (void *)memsetParams, sizeof(*memsetParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphAddMemsetNode);
    conn->write(&_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&_0memsetParams, sizeof(_0memsetParams));
    updateTmpPtr((void *)memsetParams, _0memsetParams);
    conn->write(&ctx, sizeof(ctx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(conn, (void *)dependencies, sizeof(*dependencies), true);
    mem2client(conn, (void *)memsetParams, sizeof(*memsetParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemsetNodeGetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphMemsetNodeGetParams);
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

extern "C" CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemsetNodeSetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphMemsetNodeSetParams);
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

extern "C" CUresult cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddHostNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphNode;
    mem2server(conn, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(conn, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphAddHostNode);
    conn->write(&_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
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
    mem2client(conn, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(conn, (void *)dependencies, sizeof(*dependencies), true);
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

extern "C" CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphHostNodeGetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphHostNodeGetParams);
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

extern "C" CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphHostNodeSetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphHostNodeSetParams);
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

extern "C" CUresult cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddChildGraphNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphNode;
    mem2server(conn, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphAddChildGraphNode);
    conn->write(&_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&childGraph, sizeof(childGraph));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phGraphNode, sizeof(*phGraphNode), true);
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

extern "C" CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphChildGraphNodeGetGraph called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraph;
    mem2server(conn, &_0phGraph, (void *)phGraph, sizeof(*phGraph));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphChildGraphNodeGetGraph);
    conn->write(&hNode, sizeof(hNode));
    conn->write(&_0phGraph, sizeof(_0phGraph));
    updateTmpPtr((void *)phGraph, _0phGraph);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phGraph, sizeof(*phGraph), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddEmptyNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphNode;
    mem2server(conn, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphAddEmptyNode);
    conn->write(&_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phGraphNode, sizeof(*phGraphNode), true);
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

extern "C" CUresult cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddEventRecordNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphNode;
    mem2server(conn, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphAddEventRecordNode);
    conn->write(&_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&event, sizeof(event));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phGraphNode, sizeof(*phGraphNode), true);
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

extern "C" CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphEventRecordNodeGetEvent called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphEventRecordNodeGetEvent);
    conn->write(&hNode, sizeof(hNode));
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

extern "C" CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphEventRecordNodeSetEvent called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphEventRecordNodeSetEvent);
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

extern "C" CUresult cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddEventWaitNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphNode;
    mem2server(conn, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphAddEventWaitNode);
    conn->write(&_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    conn->write(&numDependencies, sizeof(numDependencies));
    conn->write(&event, sizeof(event));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phGraphNode, sizeof(*phGraphNode), true);
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

extern "C" CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphEventWaitNodeGetEvent called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphEventWaitNodeGetEvent);
    conn->write(&hNode, sizeof(hNode));
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

extern "C" CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphEventWaitNodeSetEvent called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphEventWaitNodeSetEvent);
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

extern "C" CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddExternalSemaphoresSignalNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphNode;
    mem2server(conn, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(conn, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphAddExternalSemaphoresSignalNode);
    conn->write(&_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
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
    mem2client(conn, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(conn, (void *)dependencies, sizeof(*dependencies), true);
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

extern "C" CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExternalSemaphoresSignalNodeGetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExternalSemaphoresSignalNodeGetParams);
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

extern "C" CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExternalSemaphoresSignalNodeSetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExternalSemaphoresSignalNodeSetParams);
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

extern "C" CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddExternalSemaphoresWaitNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphNode;
    mem2server(conn, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(conn, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphAddExternalSemaphoresWaitNode);
    conn->write(&_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
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
    mem2client(conn, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(conn, (void *)dependencies, sizeof(*dependencies), true);
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

extern "C" CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExternalSemaphoresWaitNodeGetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExternalSemaphoresWaitNodeGetParams);
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

extern "C" CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExternalSemaphoresWaitNodeSetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExternalSemaphoresWaitNodeSetParams);
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

extern "C" CUresult cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddMemAllocNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphNode;
    mem2server(conn, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(conn, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphAddMemAllocNode);
    conn->write(&_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
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
    mem2client(conn, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(conn, (void *)dependencies, sizeof(*dependencies), true);
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

extern "C" CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemAllocNodeGetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphMemAllocNodeGetParams);
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

extern "C" CUresult cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddMemFreeNode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphNode;
    mem2server(conn, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(conn, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
    void *_0dptr;
    mem2server(conn, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphAddMemFreeNode);
    conn->write(&_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
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
    mem2client(conn, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(conn, (void *)dependencies, sizeof(*dependencies), true);
    mem2client(conn, (void *)dptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuDeviceGraphMemTrim(CUdevice device) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGraphMemTrim called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGraphMemTrim);
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

extern "C" CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetGraphMemAttribute called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetGraphMemAttribute);
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

extern "C" CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceSetGraphMemAttribute called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceSetGraphMemAttribute);
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

extern "C" CUresult cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphClone called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphClone;
    mem2server(conn, &_0phGraphClone, (void *)phGraphClone, sizeof(*phGraphClone));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphClone);
    conn->write(&_0phGraphClone, sizeof(_0phGraphClone));
    updateTmpPtr((void *)phGraphClone, _0phGraphClone);
    conn->write(&originalGraph, sizeof(originalGraph));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phGraphClone, sizeof(*phGraphClone), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeFindInClone called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phNode;
    mem2server(conn, &_0phNode, (void *)phNode, sizeof(*phNode));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphNodeFindInClone);
    conn->write(&_0phNode, sizeof(_0phNode));
    updateTmpPtr((void *)phNode, _0phNode);
    conn->write(&hOriginalNode, sizeof(hOriginalNode));
    conn->write(&hClonedGraph, sizeof(hClonedGraph));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phNode, sizeof(*phNode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeGetType called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0type;
    mem2server(conn, &_0type, (void *)type, sizeof(*type));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphNodeGetType);
    conn->write(&hNode, sizeof(hNode));
    conn->write(&_0type, sizeof(_0type));
    updateTmpPtr((void *)type, _0type);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)type, sizeof(*type), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphGetNodes called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphGetNodes);
    conn->write(&hGraph, sizeof(hGraph));
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

extern "C" CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphGetRootNodes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0rootNodes;
    mem2server(conn, &_0rootNodes, (void *)rootNodes, sizeof(*rootNodes));
    void *_0numRootNodes;
    mem2server(conn, &_0numRootNodes, (void *)numRootNodes, sizeof(*numRootNodes));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphGetRootNodes);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0rootNodes, sizeof(_0rootNodes));
    updateTmpPtr((void *)rootNodes, _0rootNodes);
    conn->write(&_0numRootNodes, sizeof(_0numRootNodes));
    updateTmpPtr((void *)numRootNodes, _0numRootNodes);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)rootNodes, sizeof(*rootNodes), true);
    mem2client(conn, (void *)numRootNodes, sizeof(*numRootNodes), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphGetEdges called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphGetEdges);
    conn->write(&hGraph, sizeof(hGraph));
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

extern "C" CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeGetDependencies called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dependencies;
    mem2server(conn, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
    void *_0numDependencies;
    mem2server(conn, &_0numDependencies, (void *)numDependencies, sizeof(*numDependencies));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphNodeGetDependencies);
    conn->write(&hNode, sizeof(hNode));
    conn->write(&_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    conn->write(&_0numDependencies, sizeof(_0numDependencies));
    updateTmpPtr((void *)numDependencies, _0numDependencies);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dependencies, sizeof(*dependencies), true);
    mem2client(conn, (void *)numDependencies, sizeof(*numDependencies), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeGetDependentNodes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dependentNodes;
    mem2server(conn, &_0dependentNodes, (void *)dependentNodes, sizeof(*dependentNodes));
    void *_0numDependentNodes;
    mem2server(conn, &_0numDependentNodes, (void *)numDependentNodes, sizeof(*numDependentNodes));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphNodeGetDependentNodes);
    conn->write(&hNode, sizeof(hNode));
    conn->write(&_0dependentNodes, sizeof(_0dependentNodes));
    updateTmpPtr((void *)dependentNodes, _0dependentNodes);
    conn->write(&_0numDependentNodes, sizeof(_0numDependentNodes));
    updateTmpPtr((void *)numDependentNodes, _0numDependentNodes);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dependentNodes, sizeof(*dependentNodes), true);
    mem2client(conn, (void *)numDependentNodes, sizeof(*numDependentNodes), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddDependencies called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphAddDependencies);
    conn->write(&hGraph, sizeof(hGraph));
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

extern "C" CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphRemoveDependencies called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphRemoveDependencies);
    conn->write(&hGraph, sizeof(hGraph));
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

extern "C" CUresult cuGraphDestroyNode(CUgraphNode hNode) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphDestroyNode called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphDestroyNode);
    conn->write(&hNode, sizeof(hNode));
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

extern "C" CUresult cuGraphInstantiate_v2(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphInstantiate_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphExec;
    mem2server(conn, &_0phGraphExec, (void *)phGraphExec, sizeof(*phGraphExec));
    void *_0phErrorNode;
    mem2server(conn, &_0phErrorNode, (void *)phErrorNode, sizeof(*phErrorNode));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphInstantiate_v2);
    conn->write(&_0phGraphExec, sizeof(_0phGraphExec));
    updateTmpPtr((void *)phGraphExec, _0phGraphExec);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&_0phErrorNode, sizeof(_0phErrorNode));
    updateTmpPtr((void *)phErrorNode, _0phErrorNode);
    if(bufferSize > 0) {
        conn->read(logBuffer, bufferSize, true);
    }
    conn->write(&bufferSize, sizeof(bufferSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phGraphExec, sizeof(*phGraphExec), true);
    mem2client(conn, (void *)phErrorNode, sizeof(*phErrorNode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphInstantiateWithFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phGraphExec;
    mem2server(conn, &_0phGraphExec, (void *)phGraphExec, sizeof(*phGraphExec));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphInstantiateWithFlags);
    conn->write(&_0phGraphExec, sizeof(_0phGraphExec));
    updateTmpPtr((void *)phGraphExec, _0phGraphExec);
    conn->write(&hGraph, sizeof(hGraph));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phGraphExec, sizeof(*phGraphExec), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecKernelNodeSetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExecKernelNodeSetParams);
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

extern "C" CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecMemcpyNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0copyParams;
    mem2server(conn, &_0copyParams, (void *)copyParams, sizeof(*copyParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExecMemcpyNodeSetParams);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&hNode, sizeof(hNode));
    conn->write(&_0copyParams, sizeof(_0copyParams));
    updateTmpPtr((void *)copyParams, _0copyParams);
    conn->write(&ctx, sizeof(ctx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)copyParams, sizeof(*copyParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecMemsetNodeSetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0memsetParams;
    mem2server(conn, &_0memsetParams, (void *)memsetParams, sizeof(*memsetParams));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExecMemsetNodeSetParams);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&hNode, sizeof(hNode));
    conn->write(&_0memsetParams, sizeof(_0memsetParams));
    updateTmpPtr((void *)memsetParams, _0memsetParams);
    conn->write(&ctx, sizeof(ctx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)memsetParams, sizeof(*memsetParams), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecHostNodeSetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExecHostNodeSetParams);
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

extern "C" CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecChildGraphNodeSetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExecChildGraphNodeSetParams);
    conn->write(&hGraphExec, sizeof(hGraphExec));
    conn->write(&hNode, sizeof(hNode));
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

extern "C" CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecEventRecordNodeSetEvent called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExecEventRecordNodeSetEvent);
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

extern "C" CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecEventWaitNodeSetEvent called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExecEventWaitNodeSetEvent);
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

extern "C" CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecExternalSemaphoresSignalNodeSetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExecExternalSemaphoresSignalNodeSetParams);
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

extern "C" CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecExternalSemaphoresWaitNodeSetParams called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExecExternalSemaphoresWaitNodeSetParams);
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

extern "C" CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphUpload called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphUpload);
    conn->write(&hGraphExec, sizeof(hGraphExec));
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

extern "C" CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphLaunch called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphLaunch);
    conn->write(&hGraphExec, sizeof(hGraphExec));
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

extern "C" CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecDestroy called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExecDestroy);
    conn->write(&hGraphExec, sizeof(hGraphExec));
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

extern "C" CUresult cuGraphDestroy(CUgraph hGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphDestroy called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphDestroy);
    conn->write(&hGraph, sizeof(hGraph));
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

extern "C" CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode *hErrorNode_out, CUgraphExecUpdateResult *updateResult_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecUpdate called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphExecUpdate);
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

extern "C" CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeCopyAttributes called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphKernelNodeCopyAttributes);
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

extern "C" CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue *value_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeGetAttribute called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphKernelNodeGetAttribute);
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

extern "C" CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue *value) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeSetAttribute called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphKernelNodeSetAttribute);
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

extern "C" CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char *path, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphDebugDotPrint called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphDebugDotPrint);
    conn->write(&hGraph, sizeof(hGraph));
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

extern "C" CUresult cuUserObjectCreate(CUuserObject *object_out, void *ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuUserObjectCreate called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuUserObjectCreate);
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

extern "C" CUresult cuUserObjectRetain(CUuserObject object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cuUserObjectRetain called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuUserObjectRetain);
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

extern "C" CUresult cuUserObjectRelease(CUuserObject object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cuUserObjectRelease called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuUserObjectRelease);
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

extern "C" CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphRetainUserObject called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphRetainUserObject);
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

extern "C" CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphReleaseUserObject called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphReleaseUserObject);
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

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxActiveBlocksPerMultiprocessor called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0numBlocks;
    mem2server(conn, &_0numBlocks, (void *)numBlocks, sizeof(*numBlocks));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor);
    conn->write(&_0numBlocks, sizeof(_0numBlocks));
    updateTmpPtr((void *)numBlocks, _0numBlocks);
    conn->write(&func, sizeof(func));
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
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0numBlocks;
    mem2server(conn, &_0numBlocks, (void *)numBlocks, sizeof(*numBlocks));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    conn->write(&_0numBlocks, sizeof(_0numBlocks));
    updateTmpPtr((void *)numBlocks, _0numBlocks);
    conn->write(&func, sizeof(func));
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
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxPotentialBlockSize called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0minGridSize;
    mem2server(conn, &_0minGridSize, (void *)minGridSize, sizeof(*minGridSize));
    void *_0blockSize;
    mem2server(conn, &_0blockSize, (void *)blockSize, sizeof(*blockSize));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuOccupancyMaxPotentialBlockSize);
    conn->write(&_0minGridSize, sizeof(_0minGridSize));
    updateTmpPtr((void *)minGridSize, _0minGridSize);
    conn->write(&_0blockSize, sizeof(_0blockSize));
    updateTmpPtr((void *)blockSize, _0blockSize);
    conn->write(&func, sizeof(func));
    conn->write(&blockSizeToDynamicSMemSize, sizeof(blockSizeToDynamicSMemSize));
    conn->write(&dynamicSMemSize, sizeof(dynamicSMemSize));
    conn->write(&blockSizeLimit, sizeof(blockSizeLimit));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)minGridSize, sizeof(*minGridSize), true);
    mem2client(conn, (void *)blockSize, sizeof(*blockSize), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxPotentialBlockSizeWithFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0minGridSize;
    mem2server(conn, &_0minGridSize, (void *)minGridSize, sizeof(*minGridSize));
    void *_0blockSize;
    mem2server(conn, &_0blockSize, (void *)blockSize, sizeof(*blockSize));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuOccupancyMaxPotentialBlockSizeWithFlags);
    conn->write(&_0minGridSize, sizeof(_0minGridSize));
    updateTmpPtr((void *)minGridSize, _0minGridSize);
    conn->write(&_0blockSize, sizeof(_0blockSize));
    updateTmpPtr((void *)blockSize, _0blockSize);
    conn->write(&func, sizeof(func));
    conn->write(&blockSizeToDynamicSMemSize, sizeof(blockSizeToDynamicSMemSize));
    conn->write(&dynamicSMemSize, sizeof(dynamicSMemSize));
    conn->write(&blockSizeLimit, sizeof(blockSizeLimit));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)minGridSize, sizeof(*minGridSize), true);
    mem2client(conn, (void *)blockSize, sizeof(*blockSize), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyAvailableDynamicSMemPerBlock called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dynamicSmemSize;
    mem2server(conn, &_0dynamicSmemSize, (void *)dynamicSmemSize, sizeof(*dynamicSmemSize));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuOccupancyAvailableDynamicSMemPerBlock);
    conn->write(&_0dynamicSmemSize, sizeof(_0dynamicSmemSize));
    updateTmpPtr((void *)dynamicSmemSize, _0dynamicSmemSize);
    conn->write(&func, sizeof(func));
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
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetArray called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefSetArray);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&hArray, sizeof(hArray));
    conn->write(&Flags, sizeof(Flags));
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

extern "C" CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMipmappedArray called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefSetMipmappedArray);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&hMipmappedArray, sizeof(hMipmappedArray));
    conn->write(&Flags, sizeof(Flags));
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

extern "C" CUresult cuTexRefSetAddress_v2(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetAddress_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0ByteOffset;
    mem2server(conn, &_0ByteOffset, (void *)ByteOffset, sizeof(*ByteOffset));
    void *_0dptr;
    mem2server(conn, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefSetAddress_v2);
    conn->write(&_0ByteOffset, sizeof(_0ByteOffset));
    updateTmpPtr((void *)ByteOffset, _0ByteOffset);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    conn->write(&bytes, sizeof(bytes));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)ByteOffset, sizeof(*ByteOffset), true);
    mem2client(conn, (void *)dptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetAddress2D_v3 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0desc;
    mem2server(conn, &_0desc, (void *)desc, sizeof(*desc));
    void *_0dptr;
    mem2server(conn, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefSetAddress2D_v3);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    conn->write(&_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    conn->write(&Pitch, sizeof(Pitch));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)desc, sizeof(*desc), true);
    mem2client(conn, (void *)dptr, -1, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetFormat called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefSetFormat);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&fmt, sizeof(fmt));
    conn->write(&NumPackedComponents, sizeof(NumPackedComponents));
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

extern "C" CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetAddressMode called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefSetAddressMode);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&dim, sizeof(dim));
    conn->write(&am, sizeof(am));
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

extern "C" CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetFilterMode called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefSetFilterMode);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&fm, sizeof(fm));
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

extern "C" CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMipmapFilterMode called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefSetMipmapFilterMode);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&fm, sizeof(fm));
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

extern "C" CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMipmapLevelBias called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefSetMipmapLevelBias);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&bias, sizeof(bias));
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

extern "C" CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMipmapLevelClamp called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefSetMipmapLevelClamp);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&minMipmapLevelClamp, sizeof(minMipmapLevelClamp));
    conn->write(&maxMipmapLevelClamp, sizeof(maxMipmapLevelClamp));
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

extern "C" CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMaxAnisotropy called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefSetMaxAnisotropy);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&maxAniso, sizeof(maxAniso));
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

extern "C" CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetBorderColor called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pBorderColor;
    mem2server(conn, &_0pBorderColor, (void *)pBorderColor, sizeof(*pBorderColor));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefSetBorderColor);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&_0pBorderColor, sizeof(_0pBorderColor));
    updateTmpPtr((void *)pBorderColor, _0pBorderColor);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pBorderColor, sizeof(*pBorderColor), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetFlags called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefSetFlags);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&Flags, sizeof(Flags));
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

extern "C" CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phArray;
    mem2server(conn, &_0phArray, (void *)phArray, sizeof(*phArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefGetArray);
    conn->write(&_0phArray, sizeof(_0phArray));
    updateTmpPtr((void *)phArray, _0phArray);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phArray, sizeof(*phArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMipmappedArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phMipmappedArray;
    mem2server(conn, &_0phMipmappedArray, (void *)phMipmappedArray, sizeof(*phMipmappedArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefGetMipmappedArray);
    conn->write(&_0phMipmappedArray, sizeof(_0phMipmappedArray));
    updateTmpPtr((void *)phMipmappedArray, _0phMipmappedArray);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phMipmappedArray, sizeof(*phMipmappedArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetAddressMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pam;
    mem2server(conn, &_0pam, (void *)pam, sizeof(*pam));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefGetAddressMode);
    conn->write(&_0pam, sizeof(_0pam));
    updateTmpPtr((void *)pam, _0pam);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->write(&dim, sizeof(dim));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pam, sizeof(*pam), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetFilterMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pfm;
    mem2server(conn, &_0pfm, (void *)pfm, sizeof(*pfm));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefGetFilterMode);
    conn->write(&_0pfm, sizeof(_0pfm));
    updateTmpPtr((void *)pfm, _0pfm);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pfm, sizeof(*pfm), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetFormat called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pFormat;
    mem2server(conn, &_0pFormat, (void *)pFormat, sizeof(*pFormat));
    void *_0pNumChannels;
    mem2server(conn, &_0pNumChannels, (void *)pNumChannels, sizeof(*pNumChannels));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefGetFormat);
    conn->write(&_0pFormat, sizeof(_0pFormat));
    updateTmpPtr((void *)pFormat, _0pFormat);
    conn->write(&_0pNumChannels, sizeof(_0pNumChannels));
    updateTmpPtr((void *)pNumChannels, _0pNumChannels);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pFormat, sizeof(*pFormat), true);
    mem2client(conn, (void *)pNumChannels, sizeof(*pNumChannels), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMipmapFilterMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pfm;
    mem2server(conn, &_0pfm, (void *)pfm, sizeof(*pfm));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefGetMipmapFilterMode);
    conn->write(&_0pfm, sizeof(_0pfm));
    updateTmpPtr((void *)pfm, _0pfm);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pfm, sizeof(*pfm), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMipmapLevelBias called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pbias;
    mem2server(conn, &_0pbias, (void *)pbias, sizeof(*pbias));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefGetMipmapLevelBias);
    conn->write(&_0pbias, sizeof(_0pbias));
    updateTmpPtr((void *)pbias, _0pbias);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pbias, sizeof(*pbias), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMipmapLevelClamp called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pminMipmapLevelClamp;
    mem2server(conn, &_0pminMipmapLevelClamp, (void *)pminMipmapLevelClamp, sizeof(*pminMipmapLevelClamp));
    void *_0pmaxMipmapLevelClamp;
    mem2server(conn, &_0pmaxMipmapLevelClamp, (void *)pmaxMipmapLevelClamp, sizeof(*pmaxMipmapLevelClamp));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefGetMipmapLevelClamp);
    conn->write(&_0pminMipmapLevelClamp, sizeof(_0pminMipmapLevelClamp));
    updateTmpPtr((void *)pminMipmapLevelClamp, _0pminMipmapLevelClamp);
    conn->write(&_0pmaxMipmapLevelClamp, sizeof(_0pmaxMipmapLevelClamp));
    updateTmpPtr((void *)pmaxMipmapLevelClamp, _0pmaxMipmapLevelClamp);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pminMipmapLevelClamp, sizeof(*pminMipmapLevelClamp), true);
    mem2client(conn, (void *)pmaxMipmapLevelClamp, sizeof(*pmaxMipmapLevelClamp), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMaxAnisotropy called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pmaxAniso;
    mem2server(conn, &_0pmaxAniso, (void *)pmaxAniso, sizeof(*pmaxAniso));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefGetMaxAnisotropy);
    conn->write(&_0pmaxAniso, sizeof(_0pmaxAniso));
    updateTmpPtr((void *)pmaxAniso, _0pmaxAniso);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pmaxAniso, sizeof(*pmaxAniso), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetBorderColor called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pBorderColor;
    mem2server(conn, &_0pBorderColor, (void *)pBorderColor, sizeof(*pBorderColor));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefGetBorderColor);
    conn->write(&_0pBorderColor, sizeof(_0pBorderColor));
    updateTmpPtr((void *)pBorderColor, _0pBorderColor);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pBorderColor, sizeof(*pBorderColor), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetFlags called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pFlags;
    mem2server(conn, &_0pFlags, (void *)pFlags, sizeof(*pFlags));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefGetFlags);
    conn->write(&_0pFlags, sizeof(_0pFlags));
    updateTmpPtr((void *)pFlags, _0pFlags);
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pFlags, sizeof(*pFlags), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefCreate(CUtexref *pTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefCreate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pTexRef;
    mem2server(conn, &_0pTexRef, (void *)pTexRef, sizeof(*pTexRef));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefCreate);
    conn->write(&_0pTexRef, sizeof(_0pTexRef));
    updateTmpPtr((void *)pTexRef, _0pTexRef);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pTexRef, sizeof(*pTexRef), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefDestroy(CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefDestroy called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexRefDestroy);
    conn->write(&hTexRef, sizeof(hTexRef));
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

extern "C" CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfRefSetArray called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuSurfRefSetArray);
    conn->write(&hSurfRef, sizeof(hSurfRef));
    conn->write(&hArray, sizeof(hArray));
    conn->write(&Flags, sizeof(Flags));
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

extern "C" CUresult cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfRefGetArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0phArray;
    mem2server(conn, &_0phArray, (void *)phArray, sizeof(*phArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuSurfRefGetArray);
    conn->write(&_0phArray, sizeof(_0phArray));
    updateTmpPtr((void *)phArray, _0phArray);
    conn->write(&hSurfRef, sizeof(hSurfRef));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)phArray, sizeof(*phArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc, const CUDA_TEXTURE_DESC *pTexDesc, const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectCreate called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexObjectCreate);
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

extern "C" CUresult cuTexObjectDestroy(CUtexObject texObject) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectDestroy called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexObjectDestroy);
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

extern "C" CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectGetResourceDesc called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexObjectGetResourceDesc);
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

extern "C" CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectGetTextureDesc called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexObjectGetTextureDesc);
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

extern "C" CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectGetResourceViewDesc called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuTexObjectGetResourceViewDesc);
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

extern "C" CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfObjectCreate called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuSurfObjectCreate);
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

extern "C" CUresult cuSurfObjectDestroy(CUsurfObject surfObject) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfObjectDestroy called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuSurfObjectDestroy);
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

extern "C" CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUsurfObject surfObject) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfObjectGetResourceDesc called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuSurfObjectGetResourceDesc);
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

extern "C" CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceCanAccessPeer called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceCanAccessPeer);
    conn->write(&_0canAccessPeer, sizeof(_0canAccessPeer));
    updateTmpPtr((void *)canAccessPeer, _0canAccessPeer);
    conn->write(&dev, sizeof(dev));
    conn->write(&peerDev, sizeof(peerDev));
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

extern "C" CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxEnablePeerAccess called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxEnablePeerAccess);
    conn->write(&peerContext, sizeof(peerContext));
    conn->write(&Flags, sizeof(Flags));
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

extern "C" CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxDisablePeerAccess called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuCtxDisablePeerAccess);
    conn->write(&peerContext, sizeof(peerContext));
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

extern "C" CUresult cuDeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetP2PAttribute called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuDeviceGetP2PAttribute);
    conn->write(&_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    conn->write(&attrib, sizeof(attrib));
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

extern "C" CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsUnregisterResource called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphicsUnregisterResource);
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

extern "C" CUresult cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsSubResourceGetMappedArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pArray;
    mem2server(conn, &_0pArray, (void *)pArray, sizeof(*pArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphicsSubResourceGetMappedArray);
    conn->write(&_0pArray, sizeof(_0pArray));
    updateTmpPtr((void *)pArray, _0pArray);
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
    mem2client(conn, (void *)pArray, sizeof(*pArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsResourceGetMappedMipmappedArray called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pMipmappedArray;
    mem2server(conn, &_0pMipmappedArray, (void *)pMipmappedArray, sizeof(*pMipmappedArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    CUresult _result;
    conn->prepare_request(RPC_cuGraphicsResourceGetMappedMipmappedArray);
    conn->write(&_0pMipmappedArray, sizeof(_0pMipmappedArray));
    updateTmpPtr((void *)pMipmappedArray, _0pMipmappedArray);
    conn->write(&resource, sizeof(resource));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pMipmappedArray, sizeof(*pMipmappedArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsResourceSetMapFlags_v2 called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphicsResourceSetMapFlags_v2);
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

extern "C" CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsMapResources called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphicsMapResources);
    conn->write(&count, sizeof(count));
    conn->write(&_0resources, sizeof(_0resources));
    updateTmpPtr((void *)resources, _0resources);
    conn->write(&hStream, sizeof(hStream));
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

extern "C" CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsUnmapResources called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGraphicsUnmapResources);
    conn->write(&count, sizeof(count));
    conn->write(&_0resources, sizeof(_0resources));
    updateTmpPtr((void *)resources, _0resources);
    conn->write(&hStream, sizeof(hStream));
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

extern "C" CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
#ifdef DEBUG
    std::cout << "Hook: cuGetExportTable called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuGetExportTable);
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

extern "C" CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) {
#ifdef DEBUG
    std::cout << "Hook: cuFlushGPUDirectRDMAWrites called" << std::endl;
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
    CUresult _result;
    conn->prepare_request(RPC_cuFlushGPUDirectRDMAWrites);
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
