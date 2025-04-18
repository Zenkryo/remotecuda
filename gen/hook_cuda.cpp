#include <iostream>
#include <unordered_map>
#include "cuda.h"

#include "hook_api.h"
#include "../rpc.h"
extern void *(*real_dlsym)(void *, const char *);

extern "C" void mem2server(RpcClient *client, void **serverPtr, void *clientPtr, ssize_t size);
extern "C" void mem2client(RpcClient *client, void *clientPtr, ssize_t size, bool del_tmp_ptr);
extern "C" void updateTmpPtr(void *clientPtr, void *serverPtr);
void *get_so_handle(const std::string &so_file);
extern "C" CUresult cuInit(unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuInit called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuInit);
    rpc_write(client, &Flags, sizeof(Flags));
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

extern "C" CUresult cuDriverGetVersion(int *driverVersion) {
#ifdef DEBUG
    std::cout << "Hook: cuDriverGetVersion called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDriverGetVersion);
    rpc_write(client, &_0driverVersion, sizeof(_0driverVersion));
    updateTmpPtr((void *)driverVersion, _0driverVersion);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)driverVersion, sizeof(*driverVersion), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGet(CUdevice *device, int ordinal) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGet called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGet);
    rpc_write(client, &_0device, sizeof(_0device));
    updateTmpPtr((void *)device, _0device);
    rpc_write(client, &ordinal, sizeof(ordinal));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)device, sizeof(*device), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetCount(int *count) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetCount called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetCount);
    rpc_write(client, &_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)count, sizeof(*count), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetName called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetName);
    if(len > 0) {
        rpc_read(client, name, len, true);
    }
    rpc_write(client, &len, sizeof(len));
    rpc_write(client, &dev, sizeof(dev));
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

extern "C" CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetUuid called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0uuid;
    mem2server(client, &_0uuid, (void *)uuid, sizeof(*uuid));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetUuid);
    rpc_write(client, &_0uuid, sizeof(_0uuid));
    updateTmpPtr((void *)uuid, _0uuid);
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)uuid, sizeof(*uuid), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetUuid_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0uuid;
    mem2server(client, &_0uuid, (void *)uuid, sizeof(*uuid));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetUuid_v2);
    rpc_write(client, &_0uuid, sizeof(_0uuid));
    updateTmpPtr((void *)uuid, _0uuid);
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)uuid, sizeof(*uuid), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetLuid called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0deviceNodeMask;
    mem2server(client, &_0deviceNodeMask, (void *)deviceNodeMask, sizeof(*deviceNodeMask));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetLuid);
    if(32 > 0) {
        rpc_read(client, luid, 32, true);
    }
    rpc_write(client, &_0deviceNodeMask, sizeof(_0deviceNodeMask));
    updateTmpPtr((void *)deviceNodeMask, _0deviceNodeMask);
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)deviceNodeMask, sizeof(*deviceNodeMask), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceTotalMem_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0bytes;
    mem2server(client, &_0bytes, (void *)bytes, sizeof(*bytes));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceTotalMem_v2);
    rpc_write(client, &_0bytes, sizeof(_0bytes));
    updateTmpPtr((void *)bytes, _0bytes);
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)bytes, sizeof(*bytes), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetTexture1DLinearMaxWidth called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0maxWidthInElements;
    mem2server(client, &_0maxWidthInElements, (void *)maxWidthInElements, sizeof(*maxWidthInElements));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetTexture1DLinearMaxWidth);
    rpc_write(client, &_0maxWidthInElements, sizeof(_0maxWidthInElements));
    updateTmpPtr((void *)maxWidthInElements, _0maxWidthInElements);
    rpc_write(client, &format, sizeof(format));
    rpc_write(client, &numChannels, sizeof(numChannels));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)maxWidthInElements, sizeof(*maxWidthInElements), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pi;
    mem2server(client, &_0pi, (void *)pi, sizeof(*pi));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetAttribute);
    rpc_write(client, &_0pi, sizeof(_0pi));
    updateTmpPtr((void *)pi, _0pi);
    rpc_write(client, &attrib, sizeof(attrib));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pi, sizeof(*pi), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, CUdevice dev, int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetNvSciSyncAttributes called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetNvSciSyncAttributes);
    rpc_write(client, &_0nvSciSyncAttrList, sizeof(_0nvSciSyncAttrList));
    updateTmpPtr((void *)nvSciSyncAttrList, _0nvSciSyncAttrList);
    rpc_write(client, &dev, sizeof(dev));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nvSciSyncAttrList, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceSetMemPool called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceSetMemPool);
    rpc_write(client, &dev, sizeof(dev));
    rpc_write(client, &pool, sizeof(pool));
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

extern "C" CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetMemPool called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pool;
    mem2server(client, &_0pool, (void *)pool, sizeof(*pool));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetMemPool);
    rpc_write(client, &_0pool, sizeof(_0pool));
    updateTmpPtr((void *)pool, _0pool);
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pool, sizeof(*pool), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetDefaultMemPool called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pool_out;
    mem2server(client, &_0pool_out, (void *)pool_out, sizeof(*pool_out));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetDefaultMemPool);
    rpc_write(client, &_0pool_out, sizeof(_0pool_out));
    updateTmpPtr((void *)pool_out, _0pool_out);
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pool_out, sizeof(*pool_out), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetProperties called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetProperties);
    rpc_write(client, &_0prop, sizeof(_0prop));
    updateTmpPtr((void *)prop, _0prop);
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)prop, sizeof(*prop), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceComputeCapability called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0major;
    mem2server(client, &_0major, (void *)major, sizeof(*major));
    void *_0minor;
    mem2server(client, &_0minor, (void *)minor, sizeof(*minor));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceComputeCapability);
    rpc_write(client, &_0major, sizeof(_0major));
    updateTmpPtr((void *)major, _0major);
    rpc_write(client, &_0minor, sizeof(_0minor));
    updateTmpPtr((void *)minor, _0minor);
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)major, sizeof(*major), true);
    mem2client(client, (void *)minor, sizeof(*minor), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxRetain called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pctx;
    mem2server(client, &_0pctx, (void *)pctx, sizeof(*pctx));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDevicePrimaryCtxRetain);
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pctx, sizeof(*pctx), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxRelease_v2 called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDevicePrimaryCtxRelease_v2);
    rpc_write(client, &dev, sizeof(dev));
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

extern "C" CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxSetFlags_v2 called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDevicePrimaryCtxSetFlags_v2);
    rpc_write(client, &dev, sizeof(dev));
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

extern "C" CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxGetState called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0flags;
    mem2server(client, &_0flags, (void *)flags, sizeof(*flags));
    void *_0active;
    mem2server(client, &_0active, (void *)active, sizeof(*active));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDevicePrimaryCtxGetState);
    rpc_write(client, &dev, sizeof(dev));
    rpc_write(client, &_0flags, sizeof(_0flags));
    updateTmpPtr((void *)flags, _0flags);
    rpc_write(client, &_0active, sizeof(_0active));
    updateTmpPtr((void *)active, _0active);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)flags, sizeof(*flags), true);
    mem2client(client, (void *)active, sizeof(*active), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDevicePrimaryCtxReset_v2 called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDevicePrimaryCtxReset_v2);
    rpc_write(client, &dev, sizeof(dev));
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

extern "C" CUresult cuDeviceGetExecAffinitySupport(int *pi, CUexecAffinityType type, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetExecAffinitySupport called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pi;
    mem2server(client, &_0pi, (void *)pi, sizeof(*pi));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetExecAffinitySupport);
    rpc_write(client, &_0pi, sizeof(_0pi));
    updateTmpPtr((void *)pi, _0pi);
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pi, sizeof(*pi), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxCreate_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pctx;
    mem2server(client, &_0pctx, (void *)pctx, sizeof(*pctx));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxCreate_v2);
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    rpc_write(client, &flags, sizeof(flags));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pctx, sizeof(*pctx), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams, unsigned int flags, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxCreate_v3 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pctx;
    mem2server(client, &_0pctx, (void *)pctx, sizeof(*pctx));
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxCreate_v3);
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    rpc_write(client, &_0paramsArray, sizeof(_0paramsArray));
    updateTmpPtr((void *)paramsArray, _0paramsArray);
    rpc_write(client, &numParams, sizeof(numParams));
    rpc_write(client, &flags, sizeof(flags));
    rpc_write(client, &dev, sizeof(dev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pctx, sizeof(*pctx), true);
    mem2client(client, (void *)paramsArray, sizeof(*paramsArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxDestroy_v2(CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxDestroy_v2 called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxDestroy_v2);
    rpc_write(client, &ctx, sizeof(ctx));
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

extern "C" CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxPushCurrent_v2 called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxPushCurrent_v2);
    rpc_write(client, &ctx, sizeof(ctx));
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

extern "C" CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxPopCurrent_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pctx;
    mem2server(client, &_0pctx, (void *)pctx, sizeof(*pctx));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxPopCurrent_v2);
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pctx, sizeof(*pctx), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxSetCurrent(CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSetCurrent called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxSetCurrent);
    rpc_write(client, &ctx, sizeof(ctx));
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

extern "C" CUresult cuCtxGetCurrent(CUcontext *pctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetCurrent called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pctx;
    mem2server(client, &_0pctx, (void *)pctx, sizeof(*pctx));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxGetCurrent);
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pctx, sizeof(*pctx), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxGetDevice(CUdevice *device) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetDevice called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxGetDevice);
    rpc_write(client, &_0device, sizeof(_0device));
    updateTmpPtr((void *)device, _0device);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)device, sizeof(*device), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxGetFlags(unsigned int *flags) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetFlags called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxGetFlags);
    rpc_write(client, &_0flags, sizeof(_0flags));
    updateTmpPtr((void *)flags, _0flags);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)flags, sizeof(*flags), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxSynchronize() {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSynchronize called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxSynchronize);
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

extern "C" CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSetLimit called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxSetLimit);
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

extern "C" CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetLimit called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pvalue;
    mem2server(client, &_0pvalue, (void *)pvalue, sizeof(*pvalue));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxGetLimit);
    rpc_write(client, &_0pvalue, sizeof(_0pvalue));
    updateTmpPtr((void *)pvalue, _0pvalue);
    rpc_write(client, &limit, sizeof(limit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pvalue, sizeof(*pvalue), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetCacheConfig called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pconfig;
    mem2server(client, &_0pconfig, (void *)pconfig, sizeof(*pconfig));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxGetCacheConfig);
    rpc_write(client, &_0pconfig, sizeof(_0pconfig));
    updateTmpPtr((void *)pconfig, _0pconfig);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pconfig, sizeof(*pconfig), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxSetCacheConfig(CUfunc_cache config) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSetCacheConfig called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxSetCacheConfig);
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

extern "C" CUresult cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetSharedMemConfig called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxGetSharedMemConfig);
    rpc_write(client, &_0pConfig, sizeof(_0pConfig));
    updateTmpPtr((void *)pConfig, _0pConfig);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pConfig, sizeof(*pConfig), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxSetSharedMemConfig called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxSetSharedMemConfig);
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

extern "C" CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetApiVersion called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0version;
    mem2server(client, &_0version, (void *)version, sizeof(*version));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxGetApiVersion);
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_write(client, &_0version, sizeof(_0version));
    updateTmpPtr((void *)version, _0version);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)version, sizeof(*version), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetStreamPriorityRange called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxGetStreamPriorityRange);
    rpc_write(client, &_0leastPriority, sizeof(_0leastPriority));
    updateTmpPtr((void *)leastPriority, _0leastPriority);
    rpc_write(client, &_0greatestPriority, sizeof(_0greatestPriority));
    updateTmpPtr((void *)greatestPriority, _0greatestPriority);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)leastPriority, sizeof(*leastPriority), true);
    mem2client(client, (void *)greatestPriority, sizeof(*greatestPriority), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxResetPersistingL2Cache() {
#ifdef DEBUG
    std::cout << "Hook: cuCtxResetPersistingL2Cache called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxResetPersistingL2Cache);
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

extern "C" CUresult cuCtxGetExecAffinity(CUexecAffinityParam *pExecAffinity, CUexecAffinityType type) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxGetExecAffinity called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pExecAffinity;
    mem2server(client, &_0pExecAffinity, (void *)pExecAffinity, sizeof(*pExecAffinity));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxGetExecAffinity);
    rpc_write(client, &_0pExecAffinity, sizeof(_0pExecAffinity));
    updateTmpPtr((void *)pExecAffinity, _0pExecAffinity);
    rpc_write(client, &type, sizeof(type));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pExecAffinity, sizeof(*pExecAffinity), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxAttach called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pctx;
    mem2server(client, &_0pctx, (void *)pctx, sizeof(*pctx));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxAttach);
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pctx, sizeof(*pctx), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxDetach(CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxDetach called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxDetach);
    rpc_write(client, &ctx, sizeof(ctx));
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

extern "C" CUresult cuModuleLoad(CUmodule *module, const char *fname) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleLoad called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0module;
    mem2server(client, &_0module, (void *)module, sizeof(*module));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuModuleLoad);
    rpc_write(client, &_0module, sizeof(_0module));
    updateTmpPtr((void *)module, _0module);
    rpc_write(client, fname, strlen(fname) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)module, sizeof(*module), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuModuleLoadData(CUmodule *module, const void *image) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleLoadData called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0module;
    mem2server(client, &_0module, (void *)module, sizeof(*module));
    void *_0image;
    mem2server(client, &_0image, (void *)image, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuModuleLoadData);
    rpc_write(client, &_0module, sizeof(_0module));
    updateTmpPtr((void *)module, _0module);
    rpc_write(client, &_0image, sizeof(_0image));
    updateTmpPtr((void *)image, _0image);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)module, sizeof(*module), true);
    mem2client(client, (void *)image, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleLoadDataEx called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0module;
    mem2server(client, &_0module, (void *)module, sizeof(*module));
    void *_0image;
    mem2server(client, &_0image, (void *)image, 0);
    void *_0options;
    mem2server(client, &_0options, (void *)options, numOptions * sizeof(*options));
    // PARAM void **optionValues
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuModuleLoadDataEx);
    rpc_write(client, &_0module, sizeof(_0module));
    updateTmpPtr((void *)module, _0module);
    rpc_write(client, &_0image, sizeof(_0image));
    updateTmpPtr((void *)image, _0image);
    rpc_write(client, &numOptions, sizeof(numOptions));
    rpc_write(client, &_0options, sizeof(_0options));
    updateTmpPtr((void *)options, _0options);
    // PARAM void **optionValues
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **optionValues
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)module, sizeof(*module), true);
    mem2client(client, (void *)image, 0, true);
    mem2client(client, (void *)options, numOptions * sizeof(*options), true);
    // PARAM void **optionValues
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleLoadFatBinary called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0module;
    mem2server(client, &_0module, (void *)module, sizeof(*module));
    void *_0fatCubin;
    mem2server(client, &_0fatCubin, (void *)fatCubin, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuModuleLoadFatBinary);
    rpc_write(client, &_0module, sizeof(_0module));
    updateTmpPtr((void *)module, _0module);
    rpc_write(client, &_0fatCubin, sizeof(_0fatCubin));
    updateTmpPtr((void *)fatCubin, _0fatCubin);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)module, sizeof(*module), true);
    mem2client(client, (void *)fatCubin, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuModuleUnload(CUmodule hmod) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleUnload called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuModuleUnload);
    rpc_write(client, &hmod, sizeof(hmod));
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

extern "C" CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleGetFunction called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0hfunc;
    mem2server(client, &_0hfunc, (void *)hfunc, sizeof(*hfunc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuModuleGetFunction);
    rpc_write(client, &_0hfunc, sizeof(_0hfunc));
    updateTmpPtr((void *)hfunc, _0hfunc);
    rpc_write(client, &hmod, sizeof(hmod));
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)hfunc, sizeof(*hfunc), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleGetTexRef called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pTexRef;
    mem2server(client, &_0pTexRef, (void *)pTexRef, sizeof(*pTexRef));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuModuleGetTexRef);
    rpc_write(client, &_0pTexRef, sizeof(_0pTexRef));
    updateTmpPtr((void *)pTexRef, _0pTexRef);
    rpc_write(client, &hmod, sizeof(hmod));
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pTexRef, sizeof(*pTexRef), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleGetSurfRef called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pSurfRef;
    mem2server(client, &_0pSurfRef, (void *)pSurfRef, sizeof(*pSurfRef));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuModuleGetSurfRef);
    rpc_write(client, &_0pSurfRef, sizeof(_0pSurfRef));
    updateTmpPtr((void *)pSurfRef, _0pSurfRef);
    rpc_write(client, &hmod, sizeof(hmod));
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pSurfRef, sizeof(*pSurfRef), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkCreate_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0options;
    mem2server(client, &_0options, (void *)options, numOptions * sizeof(*options));
    // PARAM void **optionValues
    void *_0stateOut;
    mem2server(client, &_0stateOut, (void *)stateOut, sizeof(*stateOut));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuLinkCreate_v2);
    rpc_write(client, &numOptions, sizeof(numOptions));
    rpc_write(client, &_0options, sizeof(_0options));
    updateTmpPtr((void *)options, _0options);
    // PARAM void **optionValues
    rpc_write(client, &_0stateOut, sizeof(_0stateOut));
    updateTmpPtr((void *)stateOut, _0stateOut);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **optionValues
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)options, numOptions * sizeof(*options), true);
    // PARAM void **optionValues
    mem2client(client, (void *)stateOut, sizeof(*stateOut), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkAddData_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0data;
    mem2server(client, &_0data, (void *)data, size);
    void *_0options;
    mem2server(client, &_0options, (void *)options, numOptions * sizeof(*options));
    // PARAM void **optionValues
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuLinkAddData_v2);
    rpc_write(client, &state, sizeof(state));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &_0data, sizeof(_0data));
    updateTmpPtr((void *)data, _0data);
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_write(client, &numOptions, sizeof(numOptions));
    rpc_write(client, &_0options, sizeof(_0options));
    updateTmpPtr((void *)options, _0options);
    // PARAM void **optionValues
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **optionValues
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)data, size, true);
    mem2client(client, (void *)options, numOptions * sizeof(*options), true);
    // PARAM void **optionValues
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkAddFile_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0options;
    mem2server(client, &_0options, (void *)options, numOptions * sizeof(*options));
    // PARAM void **optionValues
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuLinkAddFile_v2);
    rpc_write(client, &state, sizeof(state));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, path, strlen(path) + 1, true);
    rpc_write(client, &numOptions, sizeof(numOptions));
    rpc_write(client, &_0options, sizeof(_0options));
    updateTmpPtr((void *)options, _0options);
    // PARAM void **optionValues
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **optionValues
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)options, numOptions * sizeof(*options), true);
    // PARAM void **optionValues
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkComplete called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    // PARAM void **cubinOut
    void *_0sizeOut;
    mem2server(client, &_0sizeOut, (void *)sizeOut, sizeof(*sizeOut));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuLinkComplete);
    rpc_write(client, &state, sizeof(state));
    // PARAM void **cubinOut
    rpc_write(client, &_0sizeOut, sizeof(_0sizeOut));
    updateTmpPtr((void *)sizeOut, _0sizeOut);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **cubinOut
    rpc_prepare_request(client, RPC_mem2client);
    // PARAM void **cubinOut
    mem2client(client, (void *)sizeOut, sizeof(*sizeOut), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLinkDestroy(CUlinkState state) {
#ifdef DEBUG
    std::cout << "Hook: cuLinkDestroy called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuLinkDestroy);
    rpc_write(client, &state, sizeof(state));
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

extern "C" CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetInfo_v2 called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemGetInfo_v2);
    rpc_write(client, &_0free, sizeof(_0free));
    updateTmpPtr((void *)free, _0free);
    rpc_write(client, &_0total, sizeof(_0total));
    updateTmpPtr((void *)total, _0total);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)free, sizeof(*free), true);
    mem2client(client, (void *)total, sizeof(*total), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemFree_v2(CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemFree_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dptr;
    mem2server(client, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemFree_v2);
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
#ifdef DEBUG
    std::cout << "Hook: cuMemHostGetFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pFlags;
    mem2server(client, &_0pFlags, (void *)pFlags, sizeof(*pFlags));
    void *_0p;
    mem2server(client, &_0p, (void *)p, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemHostGetFlags);
    rpc_write(client, &_0pFlags, sizeof(_0pFlags));
    updateTmpPtr((void *)pFlags, _0pFlags);
    rpc_write(client, &_0p, sizeof(_0p));
    updateTmpPtr((void *)p, _0p);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pFlags, sizeof(*pFlags), true);
    mem2client(client, (void *)p, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetByPCIBusId called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dev;
    mem2server(client, &_0dev, (void *)dev, sizeof(*dev));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetByPCIBusId);
    rpc_write(client, &_0dev, sizeof(_0dev));
    updateTmpPtr((void *)dev, _0dev);
    rpc_write(client, pciBusId, strlen(pciBusId) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dev, sizeof(*dev), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetPCIBusId called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetPCIBusId);
    if(len > 0) {
        rpc_read(client, pciBusId, len, true);
    }
    rpc_write(client, &len, sizeof(len));
    rpc_write(client, &dev, sizeof(dev));
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

extern "C" CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcGetEventHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pHandle;
    mem2server(client, &_0pHandle, (void *)pHandle, sizeof(*pHandle));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuIpcGetEventHandle);
    rpc_write(client, &_0pHandle, sizeof(_0pHandle));
    updateTmpPtr((void *)pHandle, _0pHandle);
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pHandle, sizeof(*pHandle), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcOpenEventHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phEvent;
    mem2server(client, &_0phEvent, (void *)phEvent, sizeof(*phEvent));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuIpcOpenEventHandle);
    rpc_write(client, &_0phEvent, sizeof(_0phEvent));
    updateTmpPtr((void *)phEvent, _0phEvent);
    rpc_write(client, &handle, sizeof(handle));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phEvent, sizeof(*phEvent), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcGetMemHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pHandle;
    mem2server(client, &_0pHandle, (void *)pHandle, sizeof(*pHandle));
    void *_0dptr;
    mem2server(client, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuIpcGetMemHandle);
    rpc_write(client, &_0pHandle, sizeof(_0pHandle));
    updateTmpPtr((void *)pHandle, _0pHandle);
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pHandle, sizeof(*pHandle), true);
    mem2client(client, (void *)dptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcCloseMemHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dptr;
    mem2server(client, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuIpcCloseMemHandle);
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemHostRegister_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0p;
    mem2server(client, &_0p, (void *)p, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemHostRegister_v2);
    rpc_write(client, &_0p, sizeof(_0p));
    updateTmpPtr((void *)p, _0p);
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)p, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemHostUnregister(void *p) {
#ifdef DEBUG
    std::cout << "Hook: cuMemHostUnregister called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0p;
    mem2server(client, &_0p, (void *)p, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemHostUnregister);
    rpc_write(client, &_0p, sizeof(_0p));
    updateTmpPtr((void *)p, _0p);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)p, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, -1);
    void *_0src;
    mem2server(client, &_0src, (void *)src, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpy);
    rpc_write(client, &_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    rpc_write(client, &_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, -1, true);
    mem2client(client, (void *)src, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyPeer called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *_0srcDevice;
    mem2server(client, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyPeer);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    rpc_write(client, &dstContext, sizeof(dstContext));
    rpc_write(client, &_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    rpc_write(client, &srcContext, sizeof(srcContext));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    mem2client(client, (void *)srcDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyHtoD_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *_0srcHost;
    mem2server(client, &_0srcHost, (void *)srcHost, ByteCount);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyHtoD_v2);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    rpc_write(client, &_0srcHost, sizeof(_0srcHost));
    updateTmpPtr((void *)srcHost, _0srcHost);
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    mem2client(client, (void *)srcHost, ByteCount, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoH_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstHost;
    mem2server(client, &_0dstHost, (void *)dstHost, ByteCount);
    void *_0srcDevice;
    mem2server(client, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyDtoH_v2);
    rpc_write(client, &_0dstHost, sizeof(_0dstHost));
    updateTmpPtr((void *)dstHost, _0dstHost);
    rpc_write(client, &_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstHost, ByteCount, true);
    mem2client(client, (void *)srcDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoD_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *_0srcDevice;
    mem2server(client, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyDtoD_v2);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    rpc_write(client, &_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    mem2client(client, (void *)srcDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoA_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0srcDevice;
    mem2server(client, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyDtoA_v2);
    rpc_write(client, &dstArray, sizeof(dstArray));
    rpc_write(client, &dstOffset, sizeof(dstOffset));
    rpc_write(client, &_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)srcDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAtoD_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyAtoD_v2);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    rpc_write(client, &srcArray, sizeof(srcArray));
    rpc_write(client, &srcOffset, sizeof(srcOffset));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyHtoA_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0srcHost;
    mem2server(client, &_0srcHost, (void *)srcHost, ByteCount);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyHtoA_v2);
    rpc_write(client, &dstArray, sizeof(dstArray));
    rpc_write(client, &dstOffset, sizeof(dstOffset));
    rpc_write(client, &_0srcHost, sizeof(_0srcHost));
    updateTmpPtr((void *)srcHost, _0srcHost);
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)srcHost, ByteCount, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAtoH_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstHost;
    mem2server(client, &_0dstHost, (void *)dstHost, ByteCount);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyAtoH_v2);
    rpc_write(client, &_0dstHost, sizeof(_0dstHost));
    updateTmpPtr((void *)dstHost, _0dstHost);
    rpc_write(client, &srcArray, sizeof(srcArray));
    rpc_write(client, &srcOffset, sizeof(srcOffset));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstHost, ByteCount, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAtoA_v2 called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy2D_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pCopy;
    mem2server(client, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpy2D_v2);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pCopy, sizeof(*pCopy), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy2DUnaligned_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pCopy;
    mem2server(client, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpy2DUnaligned_v2);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pCopy, sizeof(*pCopy), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy3D_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pCopy;
    mem2server(client, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpy3D_v2);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pCopy, sizeof(*pCopy), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy3DPeer called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pCopy;
    mem2server(client, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpy3DPeer);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pCopy, sizeof(*pCopy), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dst;
    mem2server(client, &_0dst, (void *)dst, -1);
    void *_0src;
    mem2server(client, &_0src, (void *)src, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyAsync);
    rpc_write(client, &_0dst, sizeof(_0dst));
    updateTmpPtr((void *)dst, _0dst);
    rpc_write(client, &_0src, sizeof(_0src));
    updateTmpPtr((void *)src, _0src);
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dst, -1, true);
    mem2client(client, (void *)src, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyPeerAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *_0srcDevice;
    mem2server(client, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyPeerAsync);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    rpc_write(client, &dstContext, sizeof(dstContext));
    rpc_write(client, &_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    rpc_write(client, &srcContext, sizeof(srcContext));
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    mem2client(client, (void *)srcDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyHtoDAsync_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *_0srcHost;
    mem2server(client, &_0srcHost, (void *)srcHost, ByteCount);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyHtoDAsync_v2);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    rpc_write(client, &_0srcHost, sizeof(_0srcHost));
    updateTmpPtr((void *)srcHost, _0srcHost);
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    mem2client(client, (void *)srcHost, ByteCount, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoHAsync_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstHost;
    mem2server(client, &_0dstHost, (void *)dstHost, ByteCount);
    void *_0srcDevice;
    mem2server(client, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyDtoHAsync_v2);
    rpc_write(client, &_0dstHost, sizeof(_0dstHost));
    updateTmpPtr((void *)dstHost, _0dstHost);
    rpc_write(client, &_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstHost, ByteCount, true);
    mem2client(client, (void *)srcDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyDtoDAsync_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *_0srcDevice;
    mem2server(client, &_0srcDevice, (void *)srcDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyDtoDAsync_v2);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    rpc_write(client, &_0srcDevice, sizeof(_0srcDevice));
    updateTmpPtr((void *)srcDevice, _0srcDevice);
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    mem2client(client, (void *)srcDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyHtoAAsync_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0srcHost;
    mem2server(client, &_0srcHost, (void *)srcHost, ByteCount);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyHtoAAsync_v2);
    rpc_write(client, &dstArray, sizeof(dstArray));
    rpc_write(client, &dstOffset, sizeof(dstOffset));
    rpc_write(client, &_0srcHost, sizeof(_0srcHost));
    updateTmpPtr((void *)srcHost, _0srcHost);
    rpc_write(client, &ByteCount, sizeof(ByteCount));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)srcHost, ByteCount, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpyAtoHAsync_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstHost;
    mem2server(client, &_0dstHost, (void *)dstHost, ByteCount);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpyAtoHAsync_v2);
    rpc_write(client, &_0dstHost, sizeof(_0dstHost));
    updateTmpPtr((void *)dstHost, _0dstHost);
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
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstHost, ByteCount, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy2DAsync_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pCopy;
    mem2server(client, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpy2DAsync_v2);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pCopy, sizeof(*pCopy), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy3DAsync_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pCopy;
    mem2server(client, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpy3DAsync_v2);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pCopy, sizeof(*pCopy), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemcpy3DPeerAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pCopy;
    mem2server(client, &_0pCopy, (void *)pCopy, sizeof(*pCopy));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemcpy3DPeerAsync);
    rpc_write(client, &_0pCopy, sizeof(_0pCopy));
    updateTmpPtr((void *)pCopy, _0pCopy);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pCopy, sizeof(*pCopy), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD8_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemsetD8_v2);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    rpc_write(client, &uc, sizeof(uc));
    rpc_write(client, &N, sizeof(N));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD16_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemsetD16_v2);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    rpc_write(client, &us, sizeof(us));
    rpc_write(client, &N, sizeof(N));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD32_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemsetD32_v2);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    rpc_write(client, &ui, sizeof(ui));
    rpc_write(client, &N, sizeof(N));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D8_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemsetD2D8_v2);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
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
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D16_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemsetD2D16_v2);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
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
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D32_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemsetD2D32_v2);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
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
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD8Async called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemsetD8Async);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    rpc_write(client, &uc, sizeof(uc));
    rpc_write(client, &N, sizeof(N));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD16Async called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemsetD16Async);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    rpc_write(client, &us, sizeof(us));
    rpc_write(client, &N, sizeof(N));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD32Async called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemsetD32Async);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
    rpc_write(client, &ui, sizeof(ui));
    rpc_write(client, &N, sizeof(N));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D8Async called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemsetD2D8Async);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
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
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D16Async called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemsetD2D16Async);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
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
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemsetD2D32Async called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dstDevice;
    mem2server(client, &_0dstDevice, (void *)dstDevice, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemsetD2D32Async);
    rpc_write(client, &_0dstDevice, sizeof(_0dstDevice));
    updateTmpPtr((void *)dstDevice, _0dstDevice);
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
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dstDevice, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuArrayCreate_v2(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayCreate_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pHandle;
    mem2server(client, &_0pHandle, (void *)pHandle, sizeof(*pHandle));
    void *_0pAllocateArray;
    mem2server(client, &_0pAllocateArray, (void *)pAllocateArray, sizeof(*pAllocateArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuArrayCreate_v2);
    rpc_write(client, &_0pHandle, sizeof(_0pHandle));
    updateTmpPtr((void *)pHandle, _0pHandle);
    rpc_write(client, &_0pAllocateArray, sizeof(_0pAllocateArray));
    updateTmpPtr((void *)pAllocateArray, _0pAllocateArray);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pHandle, sizeof(*pHandle), true);
    mem2client(client, (void *)pAllocateArray, sizeof(*pAllocateArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayGetDescriptor_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pArrayDescriptor;
    mem2server(client, &_0pArrayDescriptor, (void *)pArrayDescriptor, sizeof(*pArrayDescriptor));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuArrayGetDescriptor_v2);
    rpc_write(client, &_0pArrayDescriptor, sizeof(_0pArrayDescriptor));
    updateTmpPtr((void *)pArrayDescriptor, _0pArrayDescriptor);
    rpc_write(client, &hArray, sizeof(hArray));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pArrayDescriptor, sizeof(*pArrayDescriptor), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUarray array) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayGetSparseProperties called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuArrayGetSparseProperties);
    rpc_write(client, &_0sparseProperties, sizeof(_0sparseProperties));
    updateTmpPtr((void *)sparseProperties, _0sparseProperties);
    rpc_write(client, &array, sizeof(array));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)sparseProperties, sizeof(*sparseProperties), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUmipmappedArray mipmap) {
#ifdef DEBUG
    std::cout << "Hook: cuMipmappedArrayGetSparseProperties called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMipmappedArrayGetSparseProperties);
    rpc_write(client, &_0sparseProperties, sizeof(_0sparseProperties));
    updateTmpPtr((void *)sparseProperties, _0sparseProperties);
    rpc_write(client, &mipmap, sizeof(mipmap));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)sparseProperties, sizeof(*sparseProperties), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuArrayGetPlane(CUarray *pPlaneArray, CUarray hArray, unsigned int planeIdx) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayGetPlane called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuArrayGetPlane);
    rpc_write(client, &_0pPlaneArray, sizeof(_0pPlaneArray));
    updateTmpPtr((void *)pPlaneArray, _0pPlaneArray);
    rpc_write(client, &hArray, sizeof(hArray));
    rpc_write(client, &planeIdx, sizeof(planeIdx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pPlaneArray, sizeof(*pPlaneArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuArrayDestroy(CUarray hArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArrayDestroy called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuArrayDestroy);
    rpc_write(client, &hArray, sizeof(hArray));
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

extern "C" CUresult cuArray3DCreate_v2(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArray3DCreate_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pHandle;
    mem2server(client, &_0pHandle, (void *)pHandle, sizeof(*pHandle));
    void *_0pAllocateArray;
    mem2server(client, &_0pAllocateArray, (void *)pAllocateArray, sizeof(*pAllocateArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuArray3DCreate_v2);
    rpc_write(client, &_0pHandle, sizeof(_0pHandle));
    updateTmpPtr((void *)pHandle, _0pHandle);
    rpc_write(client, &_0pAllocateArray, sizeof(_0pAllocateArray));
    updateTmpPtr((void *)pAllocateArray, _0pAllocateArray);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pHandle, sizeof(*pHandle), true);
    mem2client(client, (void *)pAllocateArray, sizeof(*pAllocateArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
#ifdef DEBUG
    std::cout << "Hook: cuArray3DGetDescriptor_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pArrayDescriptor;
    mem2server(client, &_0pArrayDescriptor, (void *)pArrayDescriptor, sizeof(*pArrayDescriptor));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuArray3DGetDescriptor_v2);
    rpc_write(client, &_0pArrayDescriptor, sizeof(_0pArrayDescriptor));
    updateTmpPtr((void *)pArrayDescriptor, _0pArrayDescriptor);
    rpc_write(client, &hArray, sizeof(hArray));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pArrayDescriptor, sizeof(*pArrayDescriptor), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMipmappedArrayCreate(CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels) {
#ifdef DEBUG
    std::cout << "Hook: cuMipmappedArrayCreate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pHandle;
    mem2server(client, &_0pHandle, (void *)pHandle, sizeof(*pHandle));
    void *_0pMipmappedArrayDesc;
    mem2server(client, &_0pMipmappedArrayDesc, (void *)pMipmappedArrayDesc, sizeof(*pMipmappedArrayDesc));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMipmappedArrayCreate);
    rpc_write(client, &_0pHandle, sizeof(_0pHandle));
    updateTmpPtr((void *)pHandle, _0pHandle);
    rpc_write(client, &_0pMipmappedArrayDesc, sizeof(_0pMipmappedArrayDesc));
    updateTmpPtr((void *)pMipmappedArrayDesc, _0pMipmappedArrayDesc);
    rpc_write(client, &numMipmapLevels, sizeof(numMipmapLevels));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pHandle, sizeof(*pHandle), true);
    mem2client(client, (void *)pMipmappedArrayDesc, sizeof(*pMipmappedArrayDesc), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMipmappedArrayGetLevel(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) {
#ifdef DEBUG
    std::cout << "Hook: cuMipmappedArrayGetLevel called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pLevelArray;
    mem2server(client, &_0pLevelArray, (void *)pLevelArray, sizeof(*pLevelArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMipmappedArrayGetLevel);
    rpc_write(client, &_0pLevelArray, sizeof(_0pLevelArray));
    updateTmpPtr((void *)pLevelArray, _0pLevelArray);
    rpc_write(client, &hMipmappedArray, sizeof(hMipmappedArray));
    rpc_write(client, &level, sizeof(level));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pLevelArray, sizeof(*pLevelArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
#ifdef DEBUG
    std::cout << "Hook: cuMipmappedArrayDestroy called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMipmappedArrayDestroy);
    rpc_write(client, &hMipmappedArray, sizeof(hMipmappedArray));
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

extern "C" CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAddressFree called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0ptr;
    mem2server(client, &_0ptr, (void *)ptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemAddressFree);
    rpc_write(client, &_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    rpc_write(client, &size, sizeof(size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)ptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemMapArrayAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0mapInfoList;
    mem2server(client, &_0mapInfoList, (void *)mapInfoList, sizeof(*mapInfoList));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemMapArrayAsync);
    rpc_write(client, &_0mapInfoList, sizeof(_0mapInfoList));
    updateTmpPtr((void *)mapInfoList, _0mapInfoList);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)mapInfoList, sizeof(*mapInfoList), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
#ifdef DEBUG
    std::cout << "Hook: cuMemUnmap called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0ptr;
    mem2server(client, &_0ptr, (void *)ptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemUnmap);
    rpc_write(client, &_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    rpc_write(client, &size, sizeof(size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)ptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cuMemSetAccess called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0ptr;
    mem2server(client, &_0ptr, (void *)ptr, -1);
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemSetAccess);
    rpc_write(client, &_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)ptr, -1, true);
    mem2client(client, (void *)desc, sizeof(*desc), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetAccess called" << std::endl;
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
    void *_0ptr;
    mem2server(client, &_0ptr, (void *)ptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemGetAccess);
    rpc_write(client, &_0flags, sizeof(_0flags));
    updateTmpPtr((void *)flags, _0flags);
    rpc_write(client, &_0location, sizeof(_0location));
    updateTmpPtr((void *)location, _0location);
    rpc_write(client, &_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)flags, sizeof(*flags), true);
    mem2client(client, (void *)location, sizeof(*location), true);
    mem2client(client, (void *)ptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemExportToShareableHandle called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemExportToShareableHandle);
    rpc_write(client, &_0shareableHandle, sizeof(_0shareableHandle));
    updateTmpPtr((void *)shareableHandle, _0shareableHandle);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &handleType, sizeof(handleType));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)shareableHandle, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle, CUmemAllocationHandleType shHandleType) {
#ifdef DEBUG
    std::cout << "Hook: cuMemImportFromShareableHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0handle;
    mem2server(client, &_0handle, (void *)handle, sizeof(*handle));
    void *_0osHandle;
    mem2server(client, &_0osHandle, (void *)osHandle, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemImportFromShareableHandle);
    rpc_write(client, &_0handle, sizeof(_0handle));
    updateTmpPtr((void *)handle, _0handle);
    rpc_write(client, &_0osHandle, sizeof(_0osHandle));
    updateTmpPtr((void *)osHandle, _0osHandle);
    rpc_write(client, &shHandleType, sizeof(shHandleType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)handle, sizeof(*handle), true);
    mem2client(client, (void *)osHandle, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetAllocationGranularity called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0granularity;
    mem2server(client, &_0granularity, (void *)granularity, sizeof(*granularity));
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemGetAllocationGranularity);
    rpc_write(client, &_0granularity, sizeof(_0granularity));
    updateTmpPtr((void *)granularity, _0granularity);
    rpc_write(client, &_0prop, sizeof(_0prop));
    updateTmpPtr((void *)prop, _0prop);
    rpc_write(client, &option, sizeof(option));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)granularity, sizeof(*granularity), true);
    mem2client(client, (void *)prop, sizeof(*prop), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetAllocationPropertiesFromHandle called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemGetAllocationPropertiesFromHandle);
    rpc_write(client, &_0prop, sizeof(_0prop));
    updateTmpPtr((void *)prop, _0prop);
    rpc_write(client, &handle, sizeof(handle));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)prop, sizeof(*prop), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemRetainAllocationHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0handle;
    mem2server(client, &_0handle, (void *)handle, sizeof(*handle));
    void *_0addr;
    mem2server(client, &_0addr, (void *)addr, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemRetainAllocationHandle);
    rpc_write(client, &_0handle, sizeof(_0handle));
    updateTmpPtr((void *)handle, _0handle);
    rpc_write(client, &_0addr, sizeof(_0addr));
    updateTmpPtr((void *)addr, _0addr);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)handle, sizeof(*handle), true);
    mem2client(client, (void *)addr, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemFreeAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dptr;
    mem2server(client, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemFreeAsync);
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAllocAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dptr;
    mem2server(client, &_0dptr, (void *)dptr, sizeof(*dptr));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemAllocAsync);
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dptr, sizeof(*dptr), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolTrimTo called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemPoolTrimTo);
    rpc_write(client, &pool, sizeof(pool));
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

extern "C" CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolSetAttribute called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemPoolSetAttribute);
    rpc_write(client, &pool, sizeof(pool));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolGetAttribute called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemPoolGetAttribute);
    rpc_write(client, &pool, sizeof(pool));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolSetAccess called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0map;
    mem2server(client, &_0map, (void *)map, sizeof(*map));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemPoolSetAccess);
    rpc_write(client, &pool, sizeof(pool));
    rpc_write(client, &_0map, sizeof(_0map));
    updateTmpPtr((void *)map, _0map);
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)map, sizeof(*map), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool, CUmemLocation *location) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolGetAccess called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemPoolGetAccess);
    rpc_write(client, &_0flags, sizeof(_0flags));
    updateTmpPtr((void *)flags, _0flags);
    rpc_write(client, &memPool, sizeof(memPool));
    rpc_write(client, &_0location, sizeof(_0location));
    updateTmpPtr((void *)location, _0location);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)flags, sizeof(*flags), true);
    mem2client(client, (void *)location, sizeof(*location), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolCreate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pool;
    mem2server(client, &_0pool, (void *)pool, sizeof(*pool));
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemPoolCreate);
    rpc_write(client, &_0pool, sizeof(_0pool));
    updateTmpPtr((void *)pool, _0pool);
    rpc_write(client, &_0poolProps, sizeof(_0poolProps));
    updateTmpPtr((void *)poolProps, _0poolProps);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pool, sizeof(*pool), true);
    mem2client(client, (void *)poolProps, sizeof(*poolProps), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPoolDestroy(CUmemoryPool pool) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolDestroy called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemPoolDestroy);
    rpc_write(client, &pool, sizeof(pool));
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

extern "C" CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAllocFromPoolAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dptr;
    mem2server(client, &_0dptr, (void *)dptr, sizeof(*dptr));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemAllocFromPoolAsync);
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_write(client, &pool, sizeof(pool));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dptr, sizeof(*dptr), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolExportToShareableHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0handle_out;
    mem2server(client, &_0handle_out, (void *)handle_out, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemPoolExportToShareableHandle);
    rpc_write(client, &_0handle_out, sizeof(_0handle_out));
    updateTmpPtr((void *)handle_out, _0handle_out);
    rpc_write(client, &pool, sizeof(pool));
    rpc_write(client, &handleType, sizeof(handleType));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)handle_out, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool *pool_out, void *handle, CUmemAllocationHandleType handleType, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolImportFromShareableHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pool_out;
    mem2server(client, &_0pool_out, (void *)pool_out, sizeof(*pool_out));
    void *_0handle;
    mem2server(client, &_0handle, (void *)handle, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemPoolImportFromShareableHandle);
    rpc_write(client, &_0pool_out, sizeof(_0pool_out));
    updateTmpPtr((void *)pool_out, _0pool_out);
    rpc_write(client, &_0handle, sizeof(_0handle));
    updateTmpPtr((void *)handle, _0handle);
    rpc_write(client, &handleType, sizeof(handleType));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pool_out, sizeof(*pool_out), true);
    mem2client(client, (void *)handle, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolExportPointer called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0shareData_out;
    mem2server(client, &_0shareData_out, (void *)shareData_out, sizeof(*shareData_out));
    void *_0ptr;
    mem2server(client, &_0ptr, (void *)ptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemPoolExportPointer);
    rpc_write(client, &_0shareData_out, sizeof(_0shareData_out));
    updateTmpPtr((void *)shareData_out, _0shareData_out);
    rpc_write(client, &_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)shareData_out, sizeof(*shareData_out), true);
    mem2client(client, (void *)ptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuPointerGetAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0data;
    mem2server(client, &_0data, (void *)data, 0);
    void *_0ptr;
    mem2server(client, &_0ptr, (void *)ptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuPointerGetAttribute);
    rpc_write(client, &_0data, sizeof(_0data));
    updateTmpPtr((void *)data, _0data);
    rpc_write(client, &attribute, sizeof(attribute));
    rpc_write(client, &_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)data, 0, true);
    mem2client(client, (void *)ptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPrefetchAsync called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemPrefetchAsync);
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)devPtr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAdvise called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemAdvise);
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
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
    mem2client(client, (void *)devPtr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cuMemRangeGetAttribute called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuMemRangeGetAttribute);
    rpc_write(client, &_0data, sizeof(_0data));
    updateTmpPtr((void *)data, _0data);
    rpc_write(client, &dataSize, sizeof(dataSize));
    rpc_write(client, &attribute, sizeof(attribute));
    rpc_write(client, &_0devPtr, sizeof(_0devPtr));
    updateTmpPtr((void *)devPtr, _0devPtr);
    rpc_write(client, &count, sizeof(count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)data, dataSize, true);
    mem2client(client, (void *)devPtr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuPointerSetAttribute(const void *value, CUpointer_attribute attribute, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuPointerSetAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0value;
    mem2server(client, &_0value, (void *)value, 0);
    void *_0ptr;
    mem2server(client, &_0ptr, (void *)ptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuPointerSetAttribute);
    rpc_write(client, &_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    rpc_write(client, &attribute, sizeof(attribute));
    rpc_write(client, &_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, 0, true);
    mem2client(client, (void *)ptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamCreate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phStream;
    mem2server(client, &_0phStream, (void *)phStream, sizeof(*phStream));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamCreate);
    rpc_write(client, &_0phStream, sizeof(_0phStream));
    updateTmpPtr((void *)phStream, _0phStream);
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phStream, sizeof(*phStream), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamCreateWithPriority called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phStream;
    mem2server(client, &_0phStream, (void *)phStream, sizeof(*phStream));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamCreateWithPriority);
    rpc_write(client, &_0phStream, sizeof(_0phStream));
    updateTmpPtr((void *)phStream, _0phStream);
    rpc_write(client, &flags, sizeof(flags));
    rpc_write(client, &priority, sizeof(priority));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phStream, sizeof(*phStream), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetPriority called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamGetPriority);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0priority, sizeof(_0priority));
    updateTmpPtr((void *)priority, _0priority);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)priority, sizeof(*priority), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetFlags called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamGetFlags);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0flags, sizeof(_0flags));
    updateTmpPtr((void *)flags, _0flags);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)flags, sizeof(*flags), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetCtx called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pctx;
    mem2server(client, &_0pctx, (void *)pctx, sizeof(*pctx));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamGetCtx);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0pctx, sizeof(_0pctx));
    updateTmpPtr((void *)pctx, _0pctx);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pctx, sizeof(*pctx), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWaitEvent called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamAddCallback called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamAddCallback);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &callback, sizeof(callback));
    rpc_write(client, &_0userData, sizeof(_0userData));
    updateTmpPtr((void *)userData, _0userData);
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)userData, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamBeginCapture_v2 called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamBeginCapture_v2);
    rpc_write(client, &hStream, sizeof(hStream));
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

extern "C" CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode) {
#ifdef DEBUG
    std::cout << "Hook: cuThreadExchangeStreamCaptureMode called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuThreadExchangeStreamCaptureMode);
    rpc_write(client, &_0mode, sizeof(_0mode));
    updateTmpPtr((void *)mode, _0mode);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)mode, sizeof(*mode), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamEndCapture called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraph;
    mem2server(client, &_0phGraph, (void *)phGraph, sizeof(*phGraph));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamEndCapture);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0phGraph, sizeof(_0phGraph));
    updateTmpPtr((void *)phGraph, _0phGraph);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraph, sizeof(*phGraph), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus *captureStatus) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamIsCapturing called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0captureStatus;
    mem2server(client, &_0captureStatus, (void *)captureStatus, sizeof(*captureStatus));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamIsCapturing);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0captureStatus, sizeof(_0captureStatus));
    updateTmpPtr((void *)captureStatus, _0captureStatus);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)captureStatus, sizeof(*captureStatus), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetCaptureInfo called" << std::endl;
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
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamGetCaptureInfo);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0captureStatus_out, sizeof(_0captureStatus_out));
    updateTmpPtr((void *)captureStatus_out, _0captureStatus_out);
    rpc_write(client, &_0id_out, sizeof(_0id_out));
    updateTmpPtr((void *)id_out, _0id_out);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)captureStatus_out, sizeof(*captureStatus_out), true);
    mem2client(client, (void *)id_out, sizeof(*id_out), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetCaptureInfo_v2 called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamGetCaptureInfo_v2);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0captureStatus_out, sizeof(_0captureStatus_out));
    updateTmpPtr((void *)captureStatus_out, _0captureStatus_out);
    rpc_write(client, &_0id_out, sizeof(_0id_out));
    updateTmpPtr((void *)id_out, _0id_out);
    rpc_write(client, &_0graph_out, sizeof(_0graph_out));
    updateTmpPtr((void *)graph_out, _0graph_out);
    static CUgraphNode _cuStreamGetCaptureInfo_v2_dependencies_out;
    rpc_read(client, &_cuStreamGetCaptureInfo_v2_dependencies_out, sizeof(CUgraphNode));
    rpc_write(client, &_0numDependencies_out, sizeof(_0numDependencies_out));
    updateTmpPtr((void *)numDependencies_out, _0numDependencies_out);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    *dependencies_out = &_cuStreamGetCaptureInfo_v2_dependencies_out;
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)captureStatus_out, sizeof(*captureStatus_out), true);
    mem2client(client, (void *)id_out, sizeof(*id_out), true);
    mem2client(client, (void *)graph_out, sizeof(*graph_out), true);
    mem2client(client, (void *)numDependencies_out, sizeof(*numDependencies_out), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode *dependencies, size_t numDependencies, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamUpdateCaptureDependencies called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamUpdateCaptureDependencies);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamAttachMemAsync called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dptr;
    mem2server(client, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamAttachMemAsync);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    rpc_write(client, &length, sizeof(length));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamQuery(CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamQuery called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamQuery);
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

extern "C" CUresult cuStreamSynchronize(CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamSynchronize called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamSynchronize);
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

extern "C" CUresult cuStreamDestroy_v2(CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamDestroy_v2 called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamDestroy_v2);
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

extern "C" CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamCopyAttributes called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamCopyAttributes);
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

extern "C" CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue *value_out) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamGetAttribute called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamGetAttribute);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value_out, sizeof(_0value_out));
    updateTmpPtr((void *)value_out, _0value_out);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value_out, sizeof(*value_out), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue *value) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamSetAttribute called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamSetAttribute);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, sizeof(*value), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuEventCreate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phEvent;
    mem2server(client, &_0phEvent, (void *)phEvent, sizeof(*phEvent));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuEventCreate);
    rpc_write(client, &_0phEvent, sizeof(_0phEvent));
    updateTmpPtr((void *)phEvent, _0phEvent);
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phEvent, sizeof(*phEvent), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuEventRecord called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuEventRecord);
    rpc_write(client, &hEvent, sizeof(hEvent));
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

extern "C" CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuEventRecordWithFlags called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuEventQuery(CUevent hEvent) {
#ifdef DEBUG
    std::cout << "Hook: cuEventQuery called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuEventQuery);
    rpc_write(client, &hEvent, sizeof(hEvent));
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

extern "C" CUresult cuEventSynchronize(CUevent hEvent) {
#ifdef DEBUG
    std::cout << "Hook: cuEventSynchronize called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuEventSynchronize);
    rpc_write(client, &hEvent, sizeof(hEvent));
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

extern "C" CUresult cuEventDestroy_v2(CUevent hEvent) {
#ifdef DEBUG
    std::cout << "Hook: cuEventDestroy_v2 called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuEventDestroy_v2);
    rpc_write(client, &hEvent, sizeof(hEvent));
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

extern "C" CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
#ifdef DEBUG
    std::cout << "Hook: cuEventElapsedTime called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pMilliseconds;
    mem2server(client, &_0pMilliseconds, (void *)pMilliseconds, sizeof(*pMilliseconds));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuEventElapsedTime);
    rpc_write(client, &_0pMilliseconds, sizeof(_0pMilliseconds));
    updateTmpPtr((void *)pMilliseconds, _0pMilliseconds);
    rpc_write(client, &hStart, sizeof(hStart));
    rpc_write(client, &hEnd, sizeof(hEnd));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pMilliseconds, sizeof(*pMilliseconds), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuExternalMemoryGetMappedMipmappedArray called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuExternalMemoryGetMappedMipmappedArray);
    rpc_write(client, &_0mipmap, sizeof(_0mipmap));
    updateTmpPtr((void *)mipmap, _0mipmap);
    rpc_write(client, &extMem, sizeof(extMem));
    rpc_write(client, &_0mipmapDesc, sizeof(_0mipmapDesc));
    updateTmpPtr((void *)mipmapDesc, _0mipmapDesc);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)mipmap, sizeof(*mipmap), true);
    mem2client(client, (void *)mipmapDesc, sizeof(*mipmapDesc), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDestroyExternalMemory(CUexternalMemory extMem) {
#ifdef DEBUG
    std::cout << "Hook: cuDestroyExternalMemory called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDestroyExternalMemory);
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

extern "C" CUresult cuImportExternalSemaphore(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuImportExternalSemaphore called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuImportExternalSemaphore);
    rpc_write(client, &_0extSem_out, sizeof(_0extSem_out));
    updateTmpPtr((void *)extSem_out, _0extSem_out);
    rpc_write(client, &_0semHandleDesc, sizeof(_0semHandleDesc));
    updateTmpPtr((void *)semHandleDesc, _0semHandleDesc);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)extSem_out, sizeof(*extSem_out), true);
    mem2client(client, (void *)semHandleDesc, sizeof(*semHandleDesc), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream) {
#ifdef DEBUG
    std::cout << "Hook: cuSignalExternalSemaphoresAsync called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuSignalExternalSemaphoresAsync);
    rpc_write(client, &_0extSemArray, sizeof(_0extSemArray));
    updateTmpPtr((void *)extSemArray, _0extSemArray);
    rpc_write(client, &_0paramsArray, sizeof(_0paramsArray));
    updateTmpPtr((void *)paramsArray, _0paramsArray);
    rpc_write(client, &numExtSems, sizeof(numExtSems));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)extSemArray, sizeof(*extSemArray), true);
    mem2client(client, (void *)paramsArray, sizeof(*paramsArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream) {
#ifdef DEBUG
    std::cout << "Hook: cuWaitExternalSemaphoresAsync called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuWaitExternalSemaphoresAsync);
    rpc_write(client, &_0extSemArray, sizeof(_0extSemArray));
    updateTmpPtr((void *)extSemArray, _0extSemArray);
    rpc_write(client, &_0paramsArray, sizeof(_0paramsArray));
    updateTmpPtr((void *)paramsArray, _0paramsArray);
    rpc_write(client, &numExtSems, sizeof(numExtSems));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)extSemArray, sizeof(*extSemArray), true);
    mem2client(client, (void *)paramsArray, sizeof(*paramsArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
#ifdef DEBUG
    std::cout << "Hook: cuDestroyExternalSemaphore called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDestroyExternalSemaphore);
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

extern "C" CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWaitValue32 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0addr;
    mem2server(client, &_0addr, (void *)addr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamWaitValue32);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &_0addr, sizeof(_0addr));
    updateTmpPtr((void *)addr, _0addr);
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)addr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWaitValue64 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0addr;
    mem2server(client, &_0addr, (void *)addr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamWaitValue64);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &_0addr, sizeof(_0addr));
    updateTmpPtr((void *)addr, _0addr);
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)addr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWriteValue32 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0addr;
    mem2server(client, &_0addr, (void *)addr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamWriteValue32);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &_0addr, sizeof(_0addr));
    updateTmpPtr((void *)addr, _0addr);
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)addr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamWriteValue64 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0addr;
    mem2server(client, &_0addr, (void *)addr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamWriteValue64);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &_0addr, sizeof(_0addr));
    updateTmpPtr((void *)addr, _0addr);
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)addr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuStreamBatchMemOp called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0paramArray;
    mem2server(client, &_0paramArray, (void *)paramArray, sizeof(*paramArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuStreamBatchMemOp);
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_0paramArray, sizeof(_0paramArray));
    updateTmpPtr((void *)paramArray, _0paramArray);
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)paramArray, sizeof(*paramArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncGetAttribute called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pi;
    mem2server(client, &_0pi, (void *)pi, sizeof(*pi));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuFuncGetAttribute);
    rpc_write(client, &_0pi, sizeof(_0pi));
    updateTmpPtr((void *)pi, _0pi);
    rpc_write(client, &attrib, sizeof(attrib));
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pi, sizeof(*pi), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetAttribute called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetCacheConfig called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuFuncSetCacheConfig);
    rpc_write(client, &hfunc, sizeof(hfunc));
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

extern "C" CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetSharedMemConfig called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuFuncSetSharedMemConfig);
    rpc_write(client, &hfunc, sizeof(hfunc));
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

extern "C" CUresult cuFuncGetModule(CUmodule *hmod, CUfunction hfunc) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncGetModule called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0hmod;
    mem2server(client, &_0hmod, (void *)hmod, sizeof(*hmod));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuFuncGetModule);
    rpc_write(client, &_0hmod, sizeof(_0hmod));
    updateTmpPtr((void *)hmod, _0hmod);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)hmod, sizeof(*hmod), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchKernel called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    // PARAM void **kernelParams
    // PARAM void **extra
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
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
    rpc_prepare_request(client, RPC_mem2client);
    // PARAM void **kernelParams
    // PARAM void **extra
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchCooperativeKernelMultiDevice called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuLaunchCooperativeKernelMultiDevice);
    rpc_write(client, &_0launchParamsList, sizeof(_0launchParamsList));
    updateTmpPtr((void *)launchParamsList, _0launchParamsList);
    rpc_write(client, &numDevices, sizeof(numDevices));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)launchParamsList, sizeof(*launchParamsList), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchHostFunc called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuLaunchHostFunc);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_write(client, &fn, sizeof(fn));
    rpc_write(client, &_0userData, sizeof(_0userData));
    updateTmpPtr((void *)userData, _0userData);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)userData, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetBlockShape called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
#ifdef DEBUG
    std::cout << "Hook: cuFuncSetSharedSize called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuFuncSetSharedSize);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_write(client, &bytes, sizeof(bytes));
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

extern "C" CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSetSize called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuParamSetSize);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_write(client, &numbytes, sizeof(numbytes));
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

extern "C" CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSeti called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuParamSetf(CUfunction hfunc, int offset, float value) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSetf called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSetv called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0ptr;
    mem2server(client, &_0ptr, (void *)ptr, numbytes);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuParamSetv);
    rpc_write(client, &hfunc, sizeof(hfunc));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
    rpc_write(client, &numbytes, sizeof(numbytes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)ptr, numbytes, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLaunch(CUfunction f) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunch called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuLaunch);
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

extern "C" CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchGrid called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchGridAsync called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuParamSetTexRef called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphCreate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraph;
    mem2server(client, &_0phGraph, (void *)phGraph, sizeof(*phGraph));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphCreate);
    rpc_write(client, &_0phGraph, sizeof(_0phGraph));
    updateTmpPtr((void *)phGraph, _0phGraph);
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraph, sizeof(*phGraph), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddKernelNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphNode;
    mem2server(client, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(client, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphAddKernelNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeGetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphKernelNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeSetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphKernelNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddMemcpyNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphNode;
    mem2server(client, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(client, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
    void *_0copyParams;
    mem2server(client, &_0copyParams, (void *)copyParams, sizeof(*copyParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphAddMemcpyNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0copyParams, sizeof(_0copyParams));
    updateTmpPtr((void *)copyParams, _0copyParams);
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    mem2client(client, (void *)copyParams, sizeof(*copyParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemcpyNodeGetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphMemcpyNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemcpyNodeSetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphMemcpyNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddMemsetNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphNode;
    mem2server(client, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(client, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
    void *_0memsetParams;
    mem2server(client, &_0memsetParams, (void *)memsetParams, sizeof(*memsetParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphAddMemsetNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0memsetParams, sizeof(_0memsetParams));
    updateTmpPtr((void *)memsetParams, _0memsetParams);
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    mem2client(client, (void *)memsetParams, sizeof(*memsetParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemsetNodeGetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphMemsetNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemsetNodeSetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphMemsetNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddHostNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphNode;
    mem2server(client, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(client, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphAddHostNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphHostNodeGetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphHostNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphHostNodeSetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphHostNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddChildGraphNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphNode;
    mem2server(client, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphAddChildGraphNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &childGraph, sizeof(childGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphChildGraphNodeGetGraph called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraph;
    mem2server(client, &_0phGraph, (void *)phGraph, sizeof(*phGraph));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphChildGraphNodeGetGraph);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0phGraph, sizeof(_0phGraph));
    updateTmpPtr((void *)phGraph, _0phGraph);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraph, sizeof(*phGraph), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddEmptyNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphNode;
    mem2server(client, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphAddEmptyNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddEventRecordNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphNode;
    mem2server(client, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphAddEventRecordNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphEventRecordNodeGetEvent called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphEventRecordNodeGetEvent);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0event_out, sizeof(_0event_out));
    updateTmpPtr((void *)event_out, _0event_out);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)event_out, sizeof(*event_out), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphEventRecordNodeSetEvent called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphEventRecordNodeSetEvent);
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

extern "C" CUresult cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddEventWaitNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphNode;
    mem2server(client, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphAddEventWaitNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &event, sizeof(event));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphEventWaitNodeGetEvent called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphEventWaitNodeGetEvent);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0event_out, sizeof(_0event_out));
    updateTmpPtr((void *)event_out, _0event_out);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)event_out, sizeof(*event_out), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphEventWaitNodeSetEvent called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphEventWaitNodeSetEvent);
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

extern "C" CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddExternalSemaphoresSignalNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphNode;
    mem2server(client, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(client, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphAddExternalSemaphoresSignalNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExternalSemaphoresSignalNodeGetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphExternalSemaphoresSignalNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0params_out, sizeof(_0params_out));
    updateTmpPtr((void *)params_out, _0params_out);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)params_out, sizeof(*params_out), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExternalSemaphoresSignalNodeSetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphExternalSemaphoresSignalNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddExternalSemaphoresWaitNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphNode;
    mem2server(client, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(client, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphAddExternalSemaphoresWaitNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExternalSemaphoresWaitNodeGetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphExternalSemaphoresWaitNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0params_out, sizeof(_0params_out));
    updateTmpPtr((void *)params_out, _0params_out);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)params_out, sizeof(*params_out), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExternalSemaphoresWaitNodeSetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphExternalSemaphoresWaitNodeSetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddMemAllocNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphNode;
    mem2server(client, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(client, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphAddMemAllocNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS *params_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemAllocNodeGetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphMemAllocNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0params_out, sizeof(_0params_out));
    updateTmpPtr((void *)params_out, _0params_out);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)params_out, sizeof(*params_out), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddMemFreeNode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphNode;
    mem2server(client, &_0phGraphNode, (void *)phGraphNode, sizeof(*phGraphNode));
    void *_0dependencies;
    mem2server(client, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
    void *_0dptr;
    mem2server(client, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphAddMemFreeNode);
    rpc_write(client, &_0phGraphNode, sizeof(_0phGraphNode));
    updateTmpPtr((void *)phGraphNode, _0phGraphNode);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphNode, sizeof(*phGraphNode), true);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    mem2client(client, (void *)dptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceGraphMemTrim(CUdevice device) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGraphMemTrim called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGraphMemTrim);
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

extern "C" CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetGraphMemAttribute called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetGraphMemAttribute);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void *value) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceSetGraphMemAttribute called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceSetGraphMemAttribute);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphClone called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphClone;
    mem2server(client, &_0phGraphClone, (void *)phGraphClone, sizeof(*phGraphClone));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphClone);
    rpc_write(client, &_0phGraphClone, sizeof(_0phGraphClone));
    updateTmpPtr((void *)phGraphClone, _0phGraphClone);
    rpc_write(client, &originalGraph, sizeof(originalGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphClone, sizeof(*phGraphClone), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeFindInClone called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phNode;
    mem2server(client, &_0phNode, (void *)phNode, sizeof(*phNode));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphNodeFindInClone);
    rpc_write(client, &_0phNode, sizeof(_0phNode));
    updateTmpPtr((void *)phNode, _0phNode);
    rpc_write(client, &hOriginalNode, sizeof(hOriginalNode));
    rpc_write(client, &hClonedGraph, sizeof(hClonedGraph));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phNode, sizeof(*phNode), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeGetType called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0type;
    mem2server(client, &_0type, (void *)type, sizeof(*type));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphNodeGetType);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0type, sizeof(_0type));
    updateTmpPtr((void *)type, _0type);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)type, sizeof(*type), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphGetNodes called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphGetNodes);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0nodes, sizeof(_0nodes));
    updateTmpPtr((void *)nodes, _0nodes);
    rpc_write(client, &_0numNodes, sizeof(_0numNodes));
    updateTmpPtr((void *)numNodes, _0numNodes);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodes, sizeof(*nodes), true);
    mem2client(client, (void *)numNodes, sizeof(*numNodes), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphGetRootNodes called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0rootNodes;
    mem2server(client, &_0rootNodes, (void *)rootNodes, sizeof(*rootNodes));
    void *_0numRootNodes;
    mem2server(client, &_0numRootNodes, (void *)numRootNodes, sizeof(*numRootNodes));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphGetRootNodes);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0rootNodes, sizeof(_0rootNodes));
    updateTmpPtr((void *)rootNodes, _0rootNodes);
    rpc_write(client, &_0numRootNodes, sizeof(_0numRootNodes));
    updateTmpPtr((void *)numRootNodes, _0numRootNodes);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)rootNodes, sizeof(*rootNodes), true);
    mem2client(client, (void *)numRootNodes, sizeof(*numRootNodes), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphGetEdges called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphGetEdges);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0from, sizeof(_0from));
    updateTmpPtr((void *)from, _0from);
    rpc_write(client, &_0to, sizeof(_0to));
    updateTmpPtr((void *)to, _0to);
    rpc_write(client, &_0numEdges, sizeof(_0numEdges));
    updateTmpPtr((void *)numEdges, _0numEdges);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)from, sizeof(*from), true);
    mem2client(client, (void *)to, sizeof(*to), true);
    mem2client(client, (void *)numEdges, sizeof(*numEdges), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeGetDependencies called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dependencies;
    mem2server(client, &_0dependencies, (void *)dependencies, sizeof(*dependencies));
    void *_0numDependencies;
    mem2server(client, &_0numDependencies, (void *)numDependencies, sizeof(*numDependencies));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphNodeGetDependencies);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0dependencies, sizeof(_0dependencies));
    updateTmpPtr((void *)dependencies, _0dependencies);
    rpc_write(client, &_0numDependencies, sizeof(_0numDependencies));
    updateTmpPtr((void *)numDependencies, _0numDependencies);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dependencies, sizeof(*dependencies), true);
    mem2client(client, (void *)numDependencies, sizeof(*numDependencies), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphNodeGetDependentNodes called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dependentNodes;
    mem2server(client, &_0dependentNodes, (void *)dependentNodes, sizeof(*dependentNodes));
    void *_0numDependentNodes;
    mem2server(client, &_0numDependentNodes, (void *)numDependentNodes, sizeof(*numDependentNodes));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphNodeGetDependentNodes);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0dependentNodes, sizeof(_0dependentNodes));
    updateTmpPtr((void *)dependentNodes, _0dependentNodes);
    rpc_write(client, &_0numDependentNodes, sizeof(_0numDependentNodes));
    updateTmpPtr((void *)numDependentNodes, _0numDependentNodes);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dependentNodes, sizeof(*dependentNodes), true);
    mem2client(client, (void *)numDependentNodes, sizeof(*numDependentNodes), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphAddDependencies called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphAddDependencies);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0from, sizeof(_0from));
    updateTmpPtr((void *)from, _0from);
    rpc_write(client, &_0to, sizeof(_0to));
    updateTmpPtr((void *)to, _0to);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)from, sizeof(*from), true);
    mem2client(client, (void *)to, sizeof(*to), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphRemoveDependencies called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphRemoveDependencies);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0from, sizeof(_0from));
    updateTmpPtr((void *)from, _0from);
    rpc_write(client, &_0to, sizeof(_0to));
    updateTmpPtr((void *)to, _0to);
    rpc_write(client, &numDependencies, sizeof(numDependencies));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)from, sizeof(*from), true);
    mem2client(client, (void *)to, sizeof(*to), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphDestroyNode(CUgraphNode hNode) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphDestroyNode called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphDestroyNode);
    rpc_write(client, &hNode, sizeof(hNode));
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

extern "C" CUresult cuGraphInstantiate_v2(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphInstantiate_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphExec;
    mem2server(client, &_0phGraphExec, (void *)phGraphExec, sizeof(*phGraphExec));
    void *_0phErrorNode;
    mem2server(client, &_0phErrorNode, (void *)phErrorNode, sizeof(*phErrorNode));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphInstantiate_v2);
    rpc_write(client, &_0phGraphExec, sizeof(_0phGraphExec));
    updateTmpPtr((void *)phGraphExec, _0phGraphExec);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0phErrorNode, sizeof(_0phErrorNode));
    updateTmpPtr((void *)phErrorNode, _0phErrorNode);
    if(bufferSize > 0) {
        rpc_read(client, logBuffer, bufferSize, true);
    }
    rpc_write(client, &bufferSize, sizeof(bufferSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphExec, sizeof(*phGraphExec), true);
    mem2client(client, (void *)phErrorNode, sizeof(*phErrorNode), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphInstantiateWithFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phGraphExec;
    mem2server(client, &_0phGraphExec, (void *)phGraphExec, sizeof(*phGraphExec));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphInstantiateWithFlags);
    rpc_write(client, &_0phGraphExec, sizeof(_0phGraphExec));
    updateTmpPtr((void *)phGraphExec, _0phGraphExec);
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phGraphExec, sizeof(*phGraphExec), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecKernelNodeSetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphExecKernelNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecMemcpyNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0copyParams;
    mem2server(client, &_0copyParams, (void *)copyParams, sizeof(*copyParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphExecMemcpyNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0copyParams, sizeof(_0copyParams));
    updateTmpPtr((void *)copyParams, _0copyParams);
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)copyParams, sizeof(*copyParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecMemsetNodeSetParams called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0memsetParams;
    mem2server(client, &_0memsetParams, (void *)memsetParams, sizeof(*memsetParams));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphExecMemsetNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0memsetParams, sizeof(_0memsetParams));
    updateTmpPtr((void *)memsetParams, _0memsetParams);
    rpc_write(client, &ctx, sizeof(ctx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)memsetParams, sizeof(*memsetParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecHostNodeSetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphExecHostNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecChildGraphNodeSetParams called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecEventRecordNodeSetEvent called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecEventWaitNodeSetEvent called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecExternalSemaphoresSignalNodeSetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphExecExternalSemaphoresSignalNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecExternalSemaphoresWaitNodeSetParams called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphExecExternalSemaphoresWaitNodeSetParams);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &_0nodeParams, sizeof(_0nodeParams));
    updateTmpPtr((void *)nodeParams, _0nodeParams);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeParams, sizeof(*nodeParams), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphUpload called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphUpload);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
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

extern "C" CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphLaunch called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphLaunch);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
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

extern "C" CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecDestroy called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphExecDestroy);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
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

extern "C" CUresult cuGraphDestroy(CUgraph hGraph) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphDestroy called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphDestroy);
    rpc_write(client, &hGraph, sizeof(hGraph));
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

extern "C" CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode *hErrorNode_out, CUgraphExecUpdateResult *updateResult_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphExecUpdate called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphExecUpdate);
    rpc_write(client, &hGraphExec, sizeof(hGraphExec));
    rpc_write(client, &hGraph, sizeof(hGraph));
    rpc_write(client, &_0hErrorNode_out, sizeof(_0hErrorNode_out));
    updateTmpPtr((void *)hErrorNode_out, _0hErrorNode_out);
    rpc_write(client, &_0updateResult_out, sizeof(_0updateResult_out));
    updateTmpPtr((void *)updateResult_out, _0updateResult_out);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)hErrorNode_out, sizeof(*hErrorNode_out), true);
    mem2client(client, (void *)updateResult_out, sizeof(*updateResult_out), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeCopyAttributes called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphKernelNodeCopyAttributes);
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

extern "C" CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue *value_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeGetAttribute called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphKernelNodeGetAttribute);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value_out, sizeof(_0value_out));
    updateTmpPtr((void *)value_out, _0value_out);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value_out, sizeof(*value_out), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue *value) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphKernelNodeSetAttribute called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphKernelNodeSetAttribute);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_write(client, &attr, sizeof(attr));
    rpc_write(client, &_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, sizeof(*value), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char *path, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphDebugDotPrint called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuUserObjectCreate(CUuserObject *object_out, void *ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuUserObjectCreate called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuUserObjectCreate);
    rpc_write(client, &_0object_out, sizeof(_0object_out));
    updateTmpPtr((void *)object_out, _0object_out);
    rpc_write(client, &_0ptr, sizeof(_0ptr));
    updateTmpPtr((void *)ptr, _0ptr);
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
    mem2client(client, (void *)object_out, sizeof(*object_out), true);
    mem2client(client, (void *)ptr, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuUserObjectRetain(CUuserObject object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cuUserObjectRetain called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuUserObjectRetain);
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

extern "C" CUresult cuUserObjectRelease(CUuserObject object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cuUserObjectRelease called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuUserObjectRelease);
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

extern "C" CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphRetainUserObject called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphReleaseUserObject called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxActiveBlocksPerMultiprocessor called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0numBlocks;
    mem2server(client, &_0numBlocks, (void *)numBlocks, sizeof(*numBlocks));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor);
    rpc_write(client, &_0numBlocks, sizeof(_0numBlocks));
    updateTmpPtr((void *)numBlocks, _0numBlocks);
    rpc_write(client, &func, sizeof(func));
    rpc_write(client, &blockSize, sizeof(blockSize));
    rpc_write(client, &dynamicSMemSize, sizeof(dynamicSMemSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)numBlocks, sizeof(*numBlocks), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0numBlocks;
    mem2server(client, &_0numBlocks, (void *)numBlocks, sizeof(*numBlocks));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    rpc_write(client, &_0numBlocks, sizeof(_0numBlocks));
    updateTmpPtr((void *)numBlocks, _0numBlocks);
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
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)numBlocks, sizeof(*numBlocks), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxPotentialBlockSize called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0minGridSize;
    mem2server(client, &_0minGridSize, (void *)minGridSize, sizeof(*minGridSize));
    void *_0blockSize;
    mem2server(client, &_0blockSize, (void *)blockSize, sizeof(*blockSize));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuOccupancyMaxPotentialBlockSize);
    rpc_write(client, &_0minGridSize, sizeof(_0minGridSize));
    updateTmpPtr((void *)minGridSize, _0minGridSize);
    rpc_write(client, &_0blockSize, sizeof(_0blockSize));
    updateTmpPtr((void *)blockSize, _0blockSize);
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
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)minGridSize, sizeof(*minGridSize), true);
    mem2client(client, (void *)blockSize, sizeof(*blockSize), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyMaxPotentialBlockSizeWithFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0minGridSize;
    mem2server(client, &_0minGridSize, (void *)minGridSize, sizeof(*minGridSize));
    void *_0blockSize;
    mem2server(client, &_0blockSize, (void *)blockSize, sizeof(*blockSize));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuOccupancyMaxPotentialBlockSizeWithFlags);
    rpc_write(client, &_0minGridSize, sizeof(_0minGridSize));
    updateTmpPtr((void *)minGridSize, _0minGridSize);
    rpc_write(client, &_0blockSize, sizeof(_0blockSize));
    updateTmpPtr((void *)blockSize, _0blockSize);
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
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)minGridSize, sizeof(*minGridSize), true);
    mem2client(client, (void *)blockSize, sizeof(*blockSize), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) {
#ifdef DEBUG
    std::cout << "Hook: cuOccupancyAvailableDynamicSMemPerBlock called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0dynamicSmemSize;
    mem2server(client, &_0dynamicSmemSize, (void *)dynamicSmemSize, sizeof(*dynamicSmemSize));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuOccupancyAvailableDynamicSMemPerBlock);
    rpc_write(client, &_0dynamicSmemSize, sizeof(_0dynamicSmemSize));
    updateTmpPtr((void *)dynamicSmemSize, _0dynamicSmemSize);
    rpc_write(client, &func, sizeof(func));
    rpc_write(client, &numBlocks, sizeof(numBlocks));
    rpc_write(client, &blockSize, sizeof(blockSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)dynamicSmemSize, sizeof(*dynamicSmemSize), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetArray called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMipmappedArray called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuTexRefSetAddress_v2(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetAddress_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0ByteOffset;
    mem2server(client, &_0ByteOffset, (void *)ByteOffset, sizeof(*ByteOffset));
    void *_0dptr;
    mem2server(client, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefSetAddress_v2);
    rpc_write(client, &_0ByteOffset, sizeof(_0ByteOffset));
    updateTmpPtr((void *)ByteOffset, _0ByteOffset);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    rpc_write(client, &bytes, sizeof(bytes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)ByteOffset, sizeof(*ByteOffset), true);
    mem2client(client, (void *)dptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetAddress2D_v3 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0desc;
    mem2server(client, &_0desc, (void *)desc, sizeof(*desc));
    void *_0dptr;
    mem2server(client, &_0dptr, (void *)dptr, -1);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefSetAddress2D_v3);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &_0desc, sizeof(_0desc));
    updateTmpPtr((void *)desc, _0desc);
    rpc_write(client, &_0dptr, sizeof(_0dptr));
    updateTmpPtr((void *)dptr, _0dptr);
    rpc_write(client, &Pitch, sizeof(Pitch));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)desc, sizeof(*desc), true);
    mem2client(client, (void *)dptr, -1, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetFormat called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetAddressMode called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetFilterMode called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefSetFilterMode);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &fm, sizeof(fm));
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

extern "C" CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMipmapFilterMode called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefSetMipmapFilterMode);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &fm, sizeof(fm));
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

extern "C" CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMipmapLevelBias called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefSetMipmapLevelBias);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &bias, sizeof(bias));
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

extern "C" CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMipmapLevelClamp called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetMaxAnisotropy called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefSetMaxAnisotropy);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &maxAniso, sizeof(maxAniso));
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

extern "C" CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetBorderColor called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pBorderColor;
    mem2server(client, &_0pBorderColor, (void *)pBorderColor, sizeof(*pBorderColor));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefSetBorderColor);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &_0pBorderColor, sizeof(_0pBorderColor));
    updateTmpPtr((void *)pBorderColor, _0pBorderColor);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pBorderColor, sizeof(*pBorderColor), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefSetFlags called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefSetFlags);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &Flags, sizeof(Flags));
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

extern "C" CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phArray;
    mem2server(client, &_0phArray, (void *)phArray, sizeof(*phArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefGetArray);
    rpc_write(client, &_0phArray, sizeof(_0phArray));
    updateTmpPtr((void *)phArray, _0phArray);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phArray, sizeof(*phArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMipmappedArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phMipmappedArray;
    mem2server(client, &_0phMipmappedArray, (void *)phMipmappedArray, sizeof(*phMipmappedArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefGetMipmappedArray);
    rpc_write(client, &_0phMipmappedArray, sizeof(_0phMipmappedArray));
    updateTmpPtr((void *)phMipmappedArray, _0phMipmappedArray);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phMipmappedArray, sizeof(*phMipmappedArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetAddressMode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pam;
    mem2server(client, &_0pam, (void *)pam, sizeof(*pam));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefGetAddressMode);
    rpc_write(client, &_0pam, sizeof(_0pam));
    updateTmpPtr((void *)pam, _0pam);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_write(client, &dim, sizeof(dim));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pam, sizeof(*pam), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetFilterMode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pfm;
    mem2server(client, &_0pfm, (void *)pfm, sizeof(*pfm));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefGetFilterMode);
    rpc_write(client, &_0pfm, sizeof(_0pfm));
    updateTmpPtr((void *)pfm, _0pfm);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pfm, sizeof(*pfm), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetFormat called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pFormat;
    mem2server(client, &_0pFormat, (void *)pFormat, sizeof(*pFormat));
    void *_0pNumChannels;
    mem2server(client, &_0pNumChannels, (void *)pNumChannels, sizeof(*pNumChannels));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefGetFormat);
    rpc_write(client, &_0pFormat, sizeof(_0pFormat));
    updateTmpPtr((void *)pFormat, _0pFormat);
    rpc_write(client, &_0pNumChannels, sizeof(_0pNumChannels));
    updateTmpPtr((void *)pNumChannels, _0pNumChannels);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pFormat, sizeof(*pFormat), true);
    mem2client(client, (void *)pNumChannels, sizeof(*pNumChannels), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMipmapFilterMode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pfm;
    mem2server(client, &_0pfm, (void *)pfm, sizeof(*pfm));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefGetMipmapFilterMode);
    rpc_write(client, &_0pfm, sizeof(_0pfm));
    updateTmpPtr((void *)pfm, _0pfm);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pfm, sizeof(*pfm), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMipmapLevelBias called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pbias;
    mem2server(client, &_0pbias, (void *)pbias, sizeof(*pbias));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefGetMipmapLevelBias);
    rpc_write(client, &_0pbias, sizeof(_0pbias));
    updateTmpPtr((void *)pbias, _0pbias);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pbias, sizeof(*pbias), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMipmapLevelClamp called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pminMipmapLevelClamp;
    mem2server(client, &_0pminMipmapLevelClamp, (void *)pminMipmapLevelClamp, sizeof(*pminMipmapLevelClamp));
    void *_0pmaxMipmapLevelClamp;
    mem2server(client, &_0pmaxMipmapLevelClamp, (void *)pmaxMipmapLevelClamp, sizeof(*pmaxMipmapLevelClamp));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefGetMipmapLevelClamp);
    rpc_write(client, &_0pminMipmapLevelClamp, sizeof(_0pminMipmapLevelClamp));
    updateTmpPtr((void *)pminMipmapLevelClamp, _0pminMipmapLevelClamp);
    rpc_write(client, &_0pmaxMipmapLevelClamp, sizeof(_0pmaxMipmapLevelClamp));
    updateTmpPtr((void *)pmaxMipmapLevelClamp, _0pmaxMipmapLevelClamp);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pminMipmapLevelClamp, sizeof(*pminMipmapLevelClamp), true);
    mem2client(client, (void *)pmaxMipmapLevelClamp, sizeof(*pmaxMipmapLevelClamp), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetMaxAnisotropy called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pmaxAniso;
    mem2server(client, &_0pmaxAniso, (void *)pmaxAniso, sizeof(*pmaxAniso));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefGetMaxAnisotropy);
    rpc_write(client, &_0pmaxAniso, sizeof(_0pmaxAniso));
    updateTmpPtr((void *)pmaxAniso, _0pmaxAniso);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pmaxAniso, sizeof(*pmaxAniso), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetBorderColor called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pBorderColor;
    mem2server(client, &_0pBorderColor, (void *)pBorderColor, sizeof(*pBorderColor));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefGetBorderColor);
    rpc_write(client, &_0pBorderColor, sizeof(_0pBorderColor));
    updateTmpPtr((void *)pBorderColor, _0pBorderColor);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pBorderColor, sizeof(*pBorderColor), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetFlags called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pFlags;
    mem2server(client, &_0pFlags, (void *)pFlags, sizeof(*pFlags));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefGetFlags);
    rpc_write(client, &_0pFlags, sizeof(_0pFlags));
    updateTmpPtr((void *)pFlags, _0pFlags);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pFlags, sizeof(*pFlags), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefCreate(CUtexref *pTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefCreate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pTexRef;
    mem2server(client, &_0pTexRef, (void *)pTexRef, sizeof(*pTexRef));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefCreate);
    rpc_write(client, &_0pTexRef, sizeof(_0pTexRef));
    updateTmpPtr((void *)pTexRef, _0pTexRef);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pTexRef, sizeof(*pTexRef), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefDestroy(CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefDestroy called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexRefDestroy);
    rpc_write(client, &hTexRef, sizeof(hTexRef));
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

extern "C" CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfRefSetArray called" << std::endl;
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
    CUresult _result;
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

extern "C" CUresult cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfRefGetArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0phArray;
    mem2server(client, &_0phArray, (void *)phArray, sizeof(*phArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuSurfRefGetArray);
    rpc_write(client, &_0phArray, sizeof(_0phArray));
    updateTmpPtr((void *)phArray, _0phArray);
    rpc_write(client, &hSurfRef, sizeof(hSurfRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)phArray, sizeof(*phArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc, const CUDA_TEXTURE_DESC *pTexDesc, const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectCreate called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexObjectCreate);
    rpc_write(client, &_0pTexObject, sizeof(_0pTexObject));
    updateTmpPtr((void *)pTexObject, _0pTexObject);
    rpc_write(client, &_0pResDesc, sizeof(_0pResDesc));
    updateTmpPtr((void *)pResDesc, _0pResDesc);
    rpc_write(client, &_0pTexDesc, sizeof(_0pTexDesc));
    updateTmpPtr((void *)pTexDesc, _0pTexDesc);
    rpc_write(client, &_0pResViewDesc, sizeof(_0pResViewDesc));
    updateTmpPtr((void *)pResViewDesc, _0pResViewDesc);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pTexObject, sizeof(*pTexObject), true);
    mem2client(client, (void *)pResDesc, sizeof(*pResDesc), true);
    mem2client(client, (void *)pTexDesc, sizeof(*pTexDesc), true);
    mem2client(client, (void *)pResViewDesc, sizeof(*pResViewDesc), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexObjectDestroy(CUtexObject texObject) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectDestroy called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexObjectDestroy);
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

extern "C" CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectGetResourceDesc called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexObjectGetResourceDesc);
    rpc_write(client, &_0pResDesc, sizeof(_0pResDesc));
    updateTmpPtr((void *)pResDesc, _0pResDesc);
    rpc_write(client, &texObject, sizeof(texObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pResDesc, sizeof(*pResDesc), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectGetTextureDesc called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexObjectGetTextureDesc);
    rpc_write(client, &_0pTexDesc, sizeof(_0pTexDesc));
    updateTmpPtr((void *)pTexDesc, _0pTexDesc);
    rpc_write(client, &texObject, sizeof(texObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pTexDesc, sizeof(*pTexDesc), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject) {
#ifdef DEBUG
    std::cout << "Hook: cuTexObjectGetResourceViewDesc called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuTexObjectGetResourceViewDesc);
    rpc_write(client, &_0pResViewDesc, sizeof(_0pResViewDesc));
    updateTmpPtr((void *)pResViewDesc, _0pResViewDesc);
    rpc_write(client, &texObject, sizeof(texObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pResViewDesc, sizeof(*pResViewDesc), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfObjectCreate called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuSurfObjectCreate);
    rpc_write(client, &_0pSurfObject, sizeof(_0pSurfObject));
    updateTmpPtr((void *)pSurfObject, _0pSurfObject);
    rpc_write(client, &_0pResDesc, sizeof(_0pResDesc));
    updateTmpPtr((void *)pResDesc, _0pResDesc);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pSurfObject, sizeof(*pSurfObject), true);
    mem2client(client, (void *)pResDesc, sizeof(*pResDesc), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuSurfObjectDestroy(CUsurfObject surfObject) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfObjectDestroy called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuSurfObjectDestroy);
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

extern "C" CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUsurfObject surfObject) {
#ifdef DEBUG
    std::cout << "Hook: cuSurfObjectGetResourceDesc called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuSurfObjectGetResourceDesc);
    rpc_write(client, &_0pResDesc, sizeof(_0pResDesc));
    updateTmpPtr((void *)pResDesc, _0pResDesc);
    rpc_write(client, &surfObject, sizeof(surfObject));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pResDesc, sizeof(*pResDesc), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceCanAccessPeer called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceCanAccessPeer);
    rpc_write(client, &_0canAccessPeer, sizeof(_0canAccessPeer));
    updateTmpPtr((void *)canAccessPeer, _0canAccessPeer);
    rpc_write(client, &dev, sizeof(dev));
    rpc_write(client, &peerDev, sizeof(peerDev));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)canAccessPeer, sizeof(*canAccessPeer), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxEnablePeerAccess called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxEnablePeerAccess);
    rpc_write(client, &peerContext, sizeof(peerContext));
    rpc_write(client, &Flags, sizeof(Flags));
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

extern "C" CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
#ifdef DEBUG
    std::cout << "Hook: cuCtxDisablePeerAccess called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuCtxDisablePeerAccess);
    rpc_write(client, &peerContext, sizeof(peerContext));
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

extern "C" CUresult cuDeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) {
#ifdef DEBUG
    std::cout << "Hook: cuDeviceGetP2PAttribute called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuDeviceGetP2PAttribute);
    rpc_write(client, &_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    rpc_write(client, &attrib, sizeof(attrib));
    rpc_write(client, &srcDevice, sizeof(srcDevice));
    rpc_write(client, &dstDevice, sizeof(dstDevice));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)value, sizeof(*value), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsUnregisterResource called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphicsUnregisterResource);
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

extern "C" CUresult cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsSubResourceGetMappedArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pArray;
    mem2server(client, &_0pArray, (void *)pArray, sizeof(*pArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphicsSubResourceGetMappedArray);
    rpc_write(client, &_0pArray, sizeof(_0pArray));
    updateTmpPtr((void *)pArray, _0pArray);
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
    mem2client(client, (void *)pArray, sizeof(*pArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsResourceGetMappedMipmappedArray called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pMipmappedArray;
    mem2server(client, &_0pMipmappedArray, (void *)pMipmappedArray, sizeof(*pMipmappedArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphicsResourceGetMappedMipmappedArray);
    rpc_write(client, &_0pMipmappedArray, sizeof(_0pMipmappedArray));
    updateTmpPtr((void *)pMipmappedArray, _0pMipmappedArray);
    rpc_write(client, &resource, sizeof(resource));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pMipmappedArray, sizeof(*pMipmappedArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsResourceSetMapFlags_v2 called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphicsResourceSetMapFlags_v2);
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

extern "C" CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsMapResources called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphicsMapResources);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_0resources, sizeof(_0resources));
    updateTmpPtr((void *)resources, _0resources);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)resources, sizeof(*resources), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsUnmapResources called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGraphicsUnmapResources);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_0resources, sizeof(_0resources));
    updateTmpPtr((void *)resources, _0resources);
    rpc_write(client, &hStream, sizeof(hStream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)resources, sizeof(*resources), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
#ifdef DEBUG
    std::cout << "Hook: cuGetExportTable called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuGetExportTable);
    rpc_read(client, ppExportTable, sizeof(*ppExportTable));
    rpc_write(client, &_0pExportTableId, sizeof(_0pExportTableId));
    updateTmpPtr((void *)pExportTableId, _0pExportTableId);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pExportTableId, sizeof(*pExportTableId), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) {
#ifdef DEBUG
    std::cout << "Hook: cuFlushGPUDirectRDMAWrites called" << std::endl;
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
    CUresult _result;
    rpc_prepare_request(client, RPC_cuFlushGPUDirectRDMAWrites);
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
