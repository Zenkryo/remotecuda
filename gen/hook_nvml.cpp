#include <iostream>
#include <unordered_map>
#include "nvml.h"

#include "hook_api.h"
#include "../rpc.h"
extern void *(*real_dlsym)(void *, const char *);

extern "C" void mem2server(RpcClient *client, void **serverPtr, void *clientPtr, ssize_t size);
extern "C" void mem2client(RpcClient *client, void *clientPtr, ssize_t size, bool del_tmp_ptr);
extern "C" void updateTmpPtr(void *clientPtr, void *serverPtr);
void *get_so_handle(const std::string &so_file);
extern "C" nvmlReturn_t nvmlInit_v2() {
#ifdef DEBUG
    std::cout << "Hook: nvmlInit_v2 called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlInit_v2);
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

extern "C" nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: nvmlInitWithFlags called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlInitWithFlags);
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

extern "C" nvmlReturn_t nvmlShutdown() {
#ifdef DEBUG
    std::cout << "Hook: nvmlShutdown called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlShutdown);
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

extern "C" nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetDriverVersion called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlSystemGetDriverVersion);
    if(length > 0) {
        rpc_read(client, version, length, true);
    }
    rpc_write(client, &length, sizeof(length));
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

extern "C" nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetNVMLVersion called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlSystemGetNVMLVersion);
    if(length > 0) {
        rpc_read(client, version, length, true);
    }
    rpc_write(client, &length, sizeof(length));
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

extern "C" nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetCudaDriverVersion called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0cudaDriverVersion;
    mem2server(client, &_0cudaDriverVersion, (void *)cudaDriverVersion, sizeof(*cudaDriverVersion));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlSystemGetCudaDriverVersion);
    rpc_write(client, &_0cudaDriverVersion, sizeof(_0cudaDriverVersion));
    updateTmpPtr((void *)cudaDriverVersion, _0cudaDriverVersion);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)cudaDriverVersion, sizeof(*cudaDriverVersion), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetCudaDriverVersion_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0cudaDriverVersion;
    mem2server(client, &_0cudaDriverVersion, (void *)cudaDriverVersion, sizeof(*cudaDriverVersion));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlSystemGetCudaDriverVersion_v2);
    rpc_write(client, &_0cudaDriverVersion, sizeof(_0cudaDriverVersion));
    updateTmpPtr((void *)cudaDriverVersion, _0cudaDriverVersion);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)cudaDriverVersion, sizeof(*cudaDriverVersion), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char *name, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetProcessName called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlSystemGetProcessName);
    rpc_write(client, &pid, sizeof(pid));
    if(length > 0) {
        rpc_read(client, name, length, true);
    }
    rpc_write(client, &length, sizeof(length));
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

extern "C" nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetCount called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0unitCount;
    mem2server(client, &_0unitCount, (void *)unitCount, sizeof(*unitCount));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlUnitGetCount);
    rpc_write(client, &_0unitCount, sizeof(_0unitCount));
    updateTmpPtr((void *)unitCount, _0unitCount);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)unitCount, sizeof(*unitCount), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t *unit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetHandleByIndex called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0unit;
    mem2server(client, &_0unit, (void *)unit, sizeof(*unit));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlUnitGetHandleByIndex);
    rpc_write(client, &index, sizeof(index));
    rpc_write(client, &_0unit, sizeof(_0unit));
    updateTmpPtr((void *)unit, _0unit);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)unit, sizeof(*unit), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetUnitInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0info;
    mem2server(client, &_0info, (void *)info, sizeof(*info));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlUnitGetUnitInfo);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)info, sizeof(*info), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t *state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetLedState called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0state;
    mem2server(client, &_0state, (void *)state, sizeof(*state));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlUnitGetLedState);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &_0state, sizeof(_0state));
    updateTmpPtr((void *)state, _0state);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)state, sizeof(*state), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t *psu) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetPsuInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0psu;
    mem2server(client, &_0psu, (void *)psu, sizeof(*psu));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlUnitGetPsuInfo);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &_0psu, sizeof(_0psu));
    updateTmpPtr((void *)psu, _0psu);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)psu, sizeof(*psu), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetTemperature called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0temp;
    mem2server(client, &_0temp, (void *)temp, sizeof(*temp));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlUnitGetTemperature);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &_0temp, sizeof(_0temp));
    updateTmpPtr((void *)temp, _0temp);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)temp, sizeof(*temp), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t *fanSpeeds) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetFanSpeedInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0fanSpeeds;
    mem2server(client, &_0fanSpeeds, (void *)fanSpeeds, sizeof(*fanSpeeds));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlUnitGetFanSpeedInfo);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &_0fanSpeeds, sizeof(_0fanSpeeds));
    updateTmpPtr((void *)fanSpeeds, _0fanSpeeds);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)fanSpeeds, sizeof(*fanSpeeds), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int *deviceCount, nvmlDevice_t *devices) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetDevices called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0deviceCount;
    mem2server(client, &_0deviceCount, (void *)deviceCount, sizeof(*deviceCount));
    void *_0devices;
    mem2server(client, &_0devices, (void *)devices, sizeof(*devices));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlUnitGetDevices);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &_0deviceCount, sizeof(_0deviceCount));
    updateTmpPtr((void *)deviceCount, _0deviceCount);
    rpc_write(client, &_0devices, sizeof(_0devices));
    updateTmpPtr((void *)devices, _0devices);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)deviceCount, sizeof(*deviceCount), true);
    mem2client(client, (void *)devices, sizeof(*devices), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetHicVersion(unsigned int *hwbcCount, nvmlHwbcEntry_t *hwbcEntries) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetHicVersion called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0hwbcCount;
    mem2server(client, &_0hwbcCount, (void *)hwbcCount, sizeof(*hwbcCount));
    void *_0hwbcEntries;
    mem2server(client, &_0hwbcEntries, (void *)hwbcEntries, sizeof(*hwbcEntries));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlSystemGetHicVersion);
    rpc_write(client, &_0hwbcCount, sizeof(_0hwbcCount));
    updateTmpPtr((void *)hwbcCount, _0hwbcCount);
    rpc_write(client, &_0hwbcEntries, sizeof(_0hwbcEntries));
    updateTmpPtr((void *)hwbcEntries, _0hwbcEntries);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)hwbcCount, sizeof(*hwbcCount), true);
    mem2client(client, (void *)hwbcEntries, sizeof(*hwbcEntries), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCount_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0deviceCount;
    mem2server(client, &_0deviceCount, (void *)deviceCount, sizeof(*deviceCount));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetCount_v2);
    rpc_write(client, &_0deviceCount, sizeof(_0deviceCount));
    updateTmpPtr((void *)deviceCount, _0deviceCount);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)deviceCount, sizeof(*deviceCount), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t *attributes) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAttributes_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0attributes;
    mem2server(client, &_0attributes, (void *)attributes, sizeof(*attributes));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetAttributes_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0attributes, sizeof(_0attributes));
    updateTmpPtr((void *)attributes, _0attributes);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)attributes, sizeof(*attributes), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleByIndex_v2 called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetHandleByIndex_v2);
    rpc_write(client, &index, sizeof(index));
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

extern "C" nvmlReturn_t nvmlDeviceGetHandleBySerial(const char *serial, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleBySerial called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetHandleBySerial);
    rpc_write(client, serial, strlen(serial) + 1, true);
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

extern "C" nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleByUUID called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetHandleByUUID);
    rpc_write(client, uuid, strlen(uuid) + 1, true);
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

extern "C" nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleByPciBusId_v2 called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetHandleByPciBusId_v2);
    rpc_write(client, pciBusId, strlen(pciBusId) + 1, true);
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

extern "C" nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetName called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetName);
    rpc_write(client, &device, sizeof(device));
    if(length > 0) {
        rpc_read(client, name, length, true);
    }
    rpc_write(client, &length, sizeof(length));
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

extern "C" nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t *type) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBrand called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetBrand);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetIndex called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0index;
    mem2server(client, &_0index, (void *)index, sizeof(*index));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetIndex);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0index, sizeof(_0index));
    updateTmpPtr((void *)index, _0index);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)index, sizeof(*index), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char *serial, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSerial called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetSerial);
    rpc_write(client, &device, sizeof(device));
    if(length > 0) {
        rpc_read(client, serial, length, true);
    }
    rpc_write(client, &length, sizeof(length));
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

extern "C" nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long *nodeSet, nvmlAffinityScope_t scope) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryAffinity called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0nodeSet;
    mem2server(client, &_0nodeSet, (void *)nodeSet, sizeof(*nodeSet));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetMemoryAffinity);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &nodeSetSize, sizeof(nodeSetSize));
    rpc_write(client, &_0nodeSet, sizeof(_0nodeSet));
    updateTmpPtr((void *)nodeSet, _0nodeSet);
    rpc_write(client, &scope, sizeof(scope));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)nodeSet, sizeof(*nodeSet), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet, nvmlAffinityScope_t scope) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCpuAffinityWithinScope called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0cpuSet;
    mem2server(client, &_0cpuSet, (void *)cpuSet, sizeof(*cpuSet));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetCpuAffinityWithinScope);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &cpuSetSize, sizeof(cpuSetSize));
    rpc_write(client, &_0cpuSet, sizeof(_0cpuSet));
    updateTmpPtr((void *)cpuSet, _0cpuSet);
    rpc_write(client, &scope, sizeof(scope));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)cpuSet, sizeof(*cpuSet), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCpuAffinity called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0cpuSet;
    mem2server(client, &_0cpuSet, (void *)cpuSet, sizeof(*cpuSet));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetCpuAffinity);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &cpuSetSize, sizeof(cpuSetSize));
    rpc_write(client, &_0cpuSet, sizeof(_0cpuSet));
    updateTmpPtr((void *)cpuSet, _0cpuSet);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)cpuSet, sizeof(*cpuSet), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetCpuAffinity called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetCpuAffinity);
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

extern "C" nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceClearCpuAffinity called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceClearCpuAffinity);
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

extern "C" nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t *pathInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTopologyCommonAncestor called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pathInfo;
    mem2server(client, &_0pathInfo, (void *)pathInfo, sizeof(*pathInfo));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetTopologyCommonAncestor);
    rpc_write(client, &device1, sizeof(device1));
    rpc_write(client, &device2, sizeof(device2));
    rpc_write(client, &_0pathInfo, sizeof(_0pathInfo));
    updateTmpPtr((void *)pathInfo, _0pathInfo);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pathInfo, sizeof(*pathInfo), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int *count, nvmlDevice_t *deviceArray) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTopologyNearestGpus called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0count;
    mem2server(client, &_0count, (void *)count, sizeof(*count));
    void *_0deviceArray;
    mem2server(client, &_0deviceArray, (void *)deviceArray, sizeof(*deviceArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetTopologyNearestGpus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &level, sizeof(level));
    rpc_write(client, &_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    rpc_write(client, &_0deviceArray, sizeof(_0deviceArray));
    updateTmpPtr((void *)deviceArray, _0deviceArray);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)count, sizeof(*count), true);
    mem2client(client, (void *)deviceArray, sizeof(*deviceArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int *count, nvmlDevice_t *deviceArray) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetTopologyGpuSet called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0count;
    mem2server(client, &_0count, (void *)count, sizeof(*count));
    void *_0deviceArray;
    mem2server(client, &_0deviceArray, (void *)deviceArray, sizeof(*deviceArray));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlSystemGetTopologyGpuSet);
    rpc_write(client, &cpuNumber, sizeof(cpuNumber));
    rpc_write(client, &_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    rpc_write(client, &_0deviceArray, sizeof(_0deviceArray));
    updateTmpPtr((void *)deviceArray, _0deviceArray);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)count, sizeof(*count), true);
    mem2client(client, (void *)deviceArray, sizeof(*deviceArray), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t *p2pStatus) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetP2PStatus called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0p2pStatus;
    mem2server(client, &_0p2pStatus, (void *)p2pStatus, sizeof(*p2pStatus));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetP2PStatus);
    rpc_write(client, &device1, sizeof(device1));
    rpc_write(client, &device2, sizeof(device2));
    rpc_write(client, &p2pIndex, sizeof(p2pIndex));
    rpc_write(client, &_0p2pStatus, sizeof(_0p2pStatus));
    updateTmpPtr((void *)p2pStatus, _0p2pStatus);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)p2pStatus, sizeof(*p2pStatus), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetUUID called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetUUID);
    rpc_write(client, &device, sizeof(device));
    if(length > 0) {
        rpc_read(client, uuid, length, true);
    }
    rpc_write(client, &length, sizeof(length));
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

extern "C" nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char *mdevUuid, unsigned int size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetMdevUUID called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetMdevUUID);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    if(size > 0) {
        rpc_read(client, mdevUuid, size, true);
    }
    rpc_write(client, &size, sizeof(size));
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

extern "C" nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minorNumber) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMinorNumber called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0minorNumber;
    mem2server(client, &_0minorNumber, (void *)minorNumber, sizeof(*minorNumber));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetMinorNumber);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0minorNumber, sizeof(_0minorNumber));
    updateTmpPtr((void *)minorNumber, _0minorNumber);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)minorNumber, sizeof(*minorNumber), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char *partNumber, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBoardPartNumber called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetBoardPartNumber);
    rpc_write(client, &device, sizeof(device));
    if(length > 0) {
        rpc_read(client, partNumber, length, true);
    }
    rpc_write(client, &length, sizeof(length));
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

extern "C" nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetInforomVersion called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetInforomVersion);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &object, sizeof(object));
    if(length > 0) {
        rpc_read(client, version, length, true);
    }
    rpc_write(client, &length, sizeof(length));
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

extern "C" nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetInforomImageVersion called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetInforomImageVersion);
    rpc_write(client, &device, sizeof(device));
    if(length > 0) {
        rpc_read(client, version, length, true);
    }
    rpc_write(client, &length, sizeof(length));
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

extern "C" nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int *checksum) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetInforomConfigurationChecksum called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0checksum;
    mem2server(client, &_0checksum, (void *)checksum, sizeof(*checksum));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetInforomConfigurationChecksum);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0checksum, sizeof(_0checksum));
    updateTmpPtr((void *)checksum, _0checksum);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)checksum, sizeof(*checksum), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceValidateInforom called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceValidateInforom);
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

extern "C" nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t *display) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDisplayMode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0display;
    mem2server(client, &_0display, (void *)display, sizeof(*display));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetDisplayMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0display, sizeof(_0display));
    updateTmpPtr((void *)display, _0display);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)display, sizeof(*display), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t *isActive) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDisplayActive called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0isActive;
    mem2server(client, &_0isActive, (void *)isActive, sizeof(*isActive));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetDisplayActive);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0isActive, sizeof(_0isActive));
    updateTmpPtr((void *)isActive, _0isActive);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)isActive, sizeof(*isActive), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPersistenceMode called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetPersistenceMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPciInfo_v3 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pci;
    mem2server(client, &_0pci, (void *)pci, sizeof(*pci));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetPciInfo_v3);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pci, sizeof(_0pci));
    updateTmpPtr((void *)pci, _0pci);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pci, sizeof(*pci), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGen) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxPcieLinkGeneration called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0maxLinkGen;
    mem2server(client, &_0maxLinkGen, (void *)maxLinkGen, sizeof(*maxLinkGen));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxPcieLinkGeneration);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0maxLinkGen, sizeof(_0maxLinkGen));
    updateTmpPtr((void *)maxLinkGen, _0maxLinkGen);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)maxLinkGen, sizeof(*maxLinkGen), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int *maxLinkWidth) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxPcieLinkWidth called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0maxLinkWidth;
    mem2server(client, &_0maxLinkWidth, (void *)maxLinkWidth, sizeof(*maxLinkWidth));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxPcieLinkWidth);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0maxLinkWidth, sizeof(_0maxLinkWidth));
    updateTmpPtr((void *)maxLinkWidth, _0maxLinkWidth);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)maxLinkWidth, sizeof(*maxLinkWidth), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int *currLinkGen) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrPcieLinkGeneration called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0currLinkGen;
    mem2server(client, &_0currLinkGen, (void *)currLinkGen, sizeof(*currLinkGen));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetCurrPcieLinkGeneration);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0currLinkGen, sizeof(_0currLinkGen));
    updateTmpPtr((void *)currLinkGen, _0currLinkGen);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)currLinkGen, sizeof(*currLinkGen), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int *currLinkWidth) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrPcieLinkWidth called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0currLinkWidth;
    mem2server(client, &_0currLinkWidth, (void *)currLinkWidth, sizeof(*currLinkWidth));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetCurrPcieLinkWidth);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0currLinkWidth, sizeof(_0currLinkWidth));
    updateTmpPtr((void *)currLinkWidth, _0currLinkWidth);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)currLinkWidth, sizeof(*currLinkWidth), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int *value) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPcieThroughput called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetPcieThroughput);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &counter, sizeof(counter));
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

extern "C" nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int *value) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPcieReplayCounter called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetPcieReplayCounter);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClockInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0clock;
    mem2server(client, &_0clock, (void *)clock, sizeof(*clock));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetClockInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &_0clock, sizeof(_0clock));
    updateTmpPtr((void *)clock, _0clock);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)clock, sizeof(*clock), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxClockInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0clock;
    mem2server(client, &_0clock, (void *)clock, sizeof(*clock));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxClockInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &_0clock, sizeof(_0clock));
    updateTmpPtr((void *)clock, _0clock);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)clock, sizeof(*clock), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetApplicationsClock called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0clockMHz;
    mem2server(client, &_0clockMHz, (void *)clockMHz, sizeof(*clockMHz));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetApplicationsClock);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &clockType, sizeof(clockType));
    rpc_write(client, &_0clockMHz, sizeof(_0clockMHz));
    updateTmpPtr((void *)clockMHz, _0clockMHz);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)clockMHz, sizeof(*clockMHz), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDefaultApplicationsClock called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0clockMHz;
    mem2server(client, &_0clockMHz, (void *)clockMHz, sizeof(*clockMHz));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetDefaultApplicationsClock);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &clockType, sizeof(clockType));
    rpc_write(client, &_0clockMHz, sizeof(_0clockMHz));
    updateTmpPtr((void *)clockMHz, _0clockMHz);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)clockMHz, sizeof(*clockMHz), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceResetApplicationsClocks(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetApplicationsClocks called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceResetApplicationsClocks);
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

extern "C" nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClock called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0clockMHz;
    mem2server(client, &_0clockMHz, (void *)clockMHz, sizeof(*clockMHz));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetClock);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &clockType, sizeof(clockType));
    rpc_write(client, &clockId, sizeof(clockId));
    rpc_write(client, &_0clockMHz, sizeof(_0clockMHz));
    updateTmpPtr((void *)clockMHz, _0clockMHz);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)clockMHz, sizeof(*clockMHz), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxCustomerBoostClock called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0clockMHz;
    mem2server(client, &_0clockMHz, (void *)clockMHz, sizeof(*clockMHz));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxCustomerBoostClock);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &clockType, sizeof(clockType));
    rpc_write(client, &_0clockMHz, sizeof(_0clockMHz));
    updateTmpPtr((void *)clockMHz, _0clockMHz);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)clockMHz, sizeof(*clockMHz), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int *count, unsigned int *clocksMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedMemoryClocks called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0count;
    mem2server(client, &_0count, (void *)count, sizeof(*count));
    void *_0clocksMHz;
    mem2server(client, &_0clocksMHz, (void *)clocksMHz, sizeof(*clocksMHz));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedMemoryClocks);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    rpc_write(client, &_0clocksMHz, sizeof(_0clocksMHz));
    updateTmpPtr((void *)clocksMHz, _0clocksMHz);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)count, sizeof(*count), true);
    mem2client(client, (void *)clocksMHz, sizeof(*clocksMHz), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int *count, unsigned int *clocksMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedGraphicsClocks called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0count;
    mem2server(client, &_0count, (void *)count, sizeof(*count));
    void *_0clocksMHz;
    mem2server(client, &_0clocksMHz, (void *)clocksMHz, sizeof(*clocksMHz));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedGraphicsClocks);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &memoryClockMHz, sizeof(memoryClockMHz));
    rpc_write(client, &_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    rpc_write(client, &_0clocksMHz, sizeof(_0clocksMHz));
    updateTmpPtr((void *)clocksMHz, _0clocksMHz);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)count, sizeof(*count), true);
    mem2client(client, (void *)clocksMHz, sizeof(*clocksMHz), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t *isEnabled, nvmlEnableState_t *defaultIsEnabled) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAutoBoostedClocksEnabled called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0isEnabled;
    mem2server(client, &_0isEnabled, (void *)isEnabled, sizeof(*isEnabled));
    void *_0defaultIsEnabled;
    mem2server(client, &_0defaultIsEnabled, (void *)defaultIsEnabled, sizeof(*defaultIsEnabled));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetAutoBoostedClocksEnabled);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0isEnabled, sizeof(_0isEnabled));
    updateTmpPtr((void *)isEnabled, _0isEnabled);
    rpc_write(client, &_0defaultIsEnabled, sizeof(_0defaultIsEnabled));
    updateTmpPtr((void *)defaultIsEnabled, _0defaultIsEnabled);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)isEnabled, sizeof(*isEnabled), true);
    mem2client(client, (void *)defaultIsEnabled, sizeof(*defaultIsEnabled), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetAutoBoostedClocksEnabled called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetAutoBoostedClocksEnabled);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &enabled, sizeof(enabled));
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

extern "C" nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetDefaultAutoBoostedClocksEnabled called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetDefaultAutoBoostedClocksEnabled);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &enabled, sizeof(enabled));
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

extern "C" nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFanSpeed called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0speed;
    mem2server(client, &_0speed, (void *)speed, sizeof(*speed));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetFanSpeed);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0speed, sizeof(_0speed));
    updateTmpPtr((void *)speed, _0speed);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)speed, sizeof(*speed), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int *speed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFanSpeed_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0speed;
    mem2server(client, &_0speed, (void *)speed, sizeof(*speed));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetFanSpeed_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &fan, sizeof(fan));
    rpc_write(client, &_0speed, sizeof(_0speed));
    updateTmpPtr((void *)speed, _0speed);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)speed, sizeof(*speed), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTemperature called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0temp;
    mem2server(client, &_0temp, (void *)temp, sizeof(*temp));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetTemperature);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &sensorType, sizeof(sensorType));
    rpc_write(client, &_0temp, sizeof(_0temp));
    updateTmpPtr((void *)temp, _0temp);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)temp, sizeof(*temp), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTemperatureThreshold called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0temp;
    mem2server(client, &_0temp, (void *)temp, sizeof(*temp));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetTemperatureThreshold);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &thresholdType, sizeof(thresholdType));
    rpc_write(client, &_0temp, sizeof(_0temp));
    updateTmpPtr((void *)temp, _0temp);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)temp, sizeof(*temp), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetTemperatureThreshold called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0temp;
    mem2server(client, &_0temp, (void *)temp, sizeof(*temp));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetTemperatureThreshold);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &thresholdType, sizeof(thresholdType));
    rpc_write(client, &_0temp, sizeof(_0temp));
    updateTmpPtr((void *)temp, _0temp);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)temp, sizeof(*temp), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t *pState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPerformanceState called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pState;
    mem2server(client, &_0pState, (void *)pState, sizeof(*pState));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetPerformanceState);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pState, sizeof(_0pState));
    updateTmpPtr((void *)pState, _0pState);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pState, sizeof(*pState), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long *clocksThrottleReasons) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrentClocksThrottleReasons called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0clocksThrottleReasons;
    mem2server(client, &_0clocksThrottleReasons, (void *)clocksThrottleReasons, sizeof(*clocksThrottleReasons));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetCurrentClocksThrottleReasons);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0clocksThrottleReasons, sizeof(_0clocksThrottleReasons));
    updateTmpPtr((void *)clocksThrottleReasons, _0clocksThrottleReasons);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)clocksThrottleReasons, sizeof(*clocksThrottleReasons), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long *supportedClocksThrottleReasons) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedClocksThrottleReasons called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0supportedClocksThrottleReasons;
    mem2server(client, &_0supportedClocksThrottleReasons, (void *)supportedClocksThrottleReasons, sizeof(*supportedClocksThrottleReasons));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedClocksThrottleReasons);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0supportedClocksThrottleReasons, sizeof(_0supportedClocksThrottleReasons));
    updateTmpPtr((void *)supportedClocksThrottleReasons, _0supportedClocksThrottleReasons);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)supportedClocksThrottleReasons, sizeof(*supportedClocksThrottleReasons), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t *pState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerState called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pState;
    mem2server(client, &_0pState, (void *)pState, sizeof(*pState));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerState);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pState, sizeof(_0pState));
    updateTmpPtr((void *)pState, _0pState);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pState, sizeof(*pState), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementMode called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerManagementMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int *limit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementLimit called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0limit;
    mem2server(client, &_0limit, (void *)limit, sizeof(*limit));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerManagementLimit);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0limit, sizeof(_0limit));
    updateTmpPtr((void *)limit, _0limit);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)limit, sizeof(*limit), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementLimitConstraints called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0minLimit;
    mem2server(client, &_0minLimit, (void *)minLimit, sizeof(*minLimit));
    void *_0maxLimit;
    mem2server(client, &_0maxLimit, (void *)maxLimit, sizeof(*maxLimit));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerManagementLimitConstraints);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0minLimit, sizeof(_0minLimit));
    updateTmpPtr((void *)minLimit, _0minLimit);
    rpc_write(client, &_0maxLimit, sizeof(_0maxLimit));
    updateTmpPtr((void *)maxLimit, _0maxLimit);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)minLimit, sizeof(*minLimit), true);
    mem2client(client, (void *)maxLimit, sizeof(*maxLimit), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int *defaultLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementDefaultLimit called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0defaultLimit;
    mem2server(client, &_0defaultLimit, (void *)defaultLimit, sizeof(*defaultLimit));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerManagementDefaultLimit);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0defaultLimit, sizeof(_0defaultLimit));
    updateTmpPtr((void *)defaultLimit, _0defaultLimit);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)defaultLimit, sizeof(*defaultLimit), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerUsage called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0power;
    mem2server(client, &_0power, (void *)power, sizeof(*power));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerUsage);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0power, sizeof(_0power));
    updateTmpPtr((void *)power, _0power);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)power, sizeof(*power), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long *energy) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTotalEnergyConsumption called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0energy;
    mem2server(client, &_0energy, (void *)energy, sizeof(*energy));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetTotalEnergyConsumption);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0energy, sizeof(_0energy));
    updateTmpPtr((void *)energy, _0energy);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)energy, sizeof(*energy), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEnforcedPowerLimit called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0limit;
    mem2server(client, &_0limit, (void *)limit, sizeof(*limit));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetEnforcedPowerLimit);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0limit, sizeof(_0limit));
    updateTmpPtr((void *)limit, _0limit);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)limit, sizeof(*limit), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t *current, nvmlGpuOperationMode_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuOperationMode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0current;
    mem2server(client, &_0current, (void *)current, sizeof(*current));
    void *_0pending;
    mem2server(client, &_0pending, (void *)pending, sizeof(*pending));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuOperationMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0current, sizeof(_0current));
    updateTmpPtr((void *)current, _0current);
    rpc_write(client, &_0pending, sizeof(_0pending));
    updateTmpPtr((void *)pending, _0pending);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)current, sizeof(*current), true);
    mem2client(client, (void *)pending, sizeof(*pending), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0memory;
    mem2server(client, &_0memory, (void *)memory, sizeof(*memory));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetMemoryInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0memory, sizeof(_0memory));
    updateTmpPtr((void *)memory, _0memory);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)memory, sizeof(*memory), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetComputeMode called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetComputeMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCudaComputeCapability called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetCudaComputeCapability);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0major, sizeof(_0major));
    updateTmpPtr((void *)major, _0major);
    rpc_write(client, &_0minor, sizeof(_0minor));
    updateTmpPtr((void *)minor, _0minor);
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

extern "C" nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t *current, nvmlEnableState_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEccMode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0current;
    mem2server(client, &_0current, (void *)current, sizeof(*current));
    void *_0pending;
    mem2server(client, &_0pending, (void *)pending, sizeof(*pending));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetEccMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0current, sizeof(_0current));
    updateTmpPtr((void *)current, _0current);
    rpc_write(client, &_0pending, sizeof(_0pending));
    updateTmpPtr((void *)pending, _0pending);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)current, sizeof(*current), true);
    mem2client(client, (void *)pending, sizeof(*pending), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int *boardId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBoardId called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0boardId;
    mem2server(client, &_0boardId, (void *)boardId, sizeof(*boardId));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetBoardId);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0boardId, sizeof(_0boardId));
    updateTmpPtr((void *)boardId, _0boardId);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)boardId, sizeof(*boardId), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int *multiGpuBool) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMultiGpuBoard called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0multiGpuBool;
    mem2server(client, &_0multiGpuBool, (void *)multiGpuBool, sizeof(*multiGpuBool));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetMultiGpuBoard);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0multiGpuBool, sizeof(_0multiGpuBool));
    updateTmpPtr((void *)multiGpuBool, _0multiGpuBool);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)multiGpuBool, sizeof(*multiGpuBool), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long *eccCounts) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTotalEccErrors called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0eccCounts;
    mem2server(client, &_0eccCounts, (void *)eccCounts, sizeof(*eccCounts));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetTotalEccErrors);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &errorType, sizeof(errorType));
    rpc_write(client, &counterType, sizeof(counterType));
    rpc_write(client, &_0eccCounts, sizeof(_0eccCounts));
    updateTmpPtr((void *)eccCounts, _0eccCounts);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)eccCounts, sizeof(*eccCounts), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t *eccCounts) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDetailedEccErrors called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0eccCounts;
    mem2server(client, &_0eccCounts, (void *)eccCounts, sizeof(*eccCounts));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetDetailedEccErrors);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &errorType, sizeof(errorType));
    rpc_write(client, &counterType, sizeof(counterType));
    rpc_write(client, &_0eccCounts, sizeof(_0eccCounts));
    updateTmpPtr((void *)eccCounts, _0eccCounts);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)eccCounts, sizeof(*eccCounts), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryErrorCounter called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetMemoryErrorCounter);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &errorType, sizeof(errorType));
    rpc_write(client, &counterType, sizeof(counterType));
    rpc_write(client, &locationType, sizeof(locationType));
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

extern "C" nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetUtilizationRates called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0utilization;
    mem2server(client, &_0utilization, (void *)utilization, sizeof(*utilization));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetUtilizationRates);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0utilization, sizeof(_0utilization));
    updateTmpPtr((void *)utilization, _0utilization);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)utilization, sizeof(*utilization), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderUtilization called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0utilization;
    mem2server(client, &_0utilization, (void *)utilization, sizeof(*utilization));
    void *_0samplingPeriodUs;
    mem2server(client, &_0samplingPeriodUs, (void *)samplingPeriodUs, sizeof(*samplingPeriodUs));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetEncoderUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0utilization, sizeof(_0utilization));
    updateTmpPtr((void *)utilization, _0utilization);
    rpc_write(client, &_0samplingPeriodUs, sizeof(_0samplingPeriodUs));
    updateTmpPtr((void *)samplingPeriodUs, _0samplingPeriodUs);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)utilization, sizeof(*utilization), true);
    mem2client(client, (void *)samplingPeriodUs, sizeof(*samplingPeriodUs), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int *encoderCapacity) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderCapacity called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0encoderCapacity;
    mem2server(client, &_0encoderCapacity, (void *)encoderCapacity, sizeof(*encoderCapacity));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetEncoderCapacity);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &encoderQueryType, sizeof(encoderQueryType));
    rpc_write(client, &_0encoderCapacity, sizeof(_0encoderCapacity));
    updateTmpPtr((void *)encoderCapacity, _0encoderCapacity);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)encoderCapacity, sizeof(*encoderCapacity), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderStats called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0sessionCount;
    mem2server(client, &_0sessionCount, (void *)sessionCount, sizeof(*sessionCount));
    void *_0averageFps;
    mem2server(client, &_0averageFps, (void *)averageFps, sizeof(*averageFps));
    void *_0averageLatency;
    mem2server(client, &_0averageLatency, (void *)averageLatency, sizeof(*averageLatency));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetEncoderStats);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0sessionCount, sizeof(_0sessionCount));
    updateTmpPtr((void *)sessionCount, _0sessionCount);
    rpc_write(client, &_0averageFps, sizeof(_0averageFps));
    updateTmpPtr((void *)averageFps, _0averageFps);
    rpc_write(client, &_0averageLatency, sizeof(_0averageLatency));
    updateTmpPtr((void *)averageLatency, _0averageLatency);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)sessionCount, sizeof(*sessionCount), true);
    mem2client(client, (void *)averageFps, sizeof(*averageFps), true);
    mem2client(client, (void *)averageLatency, sizeof(*averageLatency), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderSessions called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0sessionCount;
    mem2server(client, &_0sessionCount, (void *)sessionCount, sizeof(*sessionCount));
    void *_0sessionInfos;
    mem2server(client, &_0sessionInfos, (void *)sessionInfos, sizeof(*sessionInfos));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetEncoderSessions);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0sessionCount, sizeof(_0sessionCount));
    updateTmpPtr((void *)sessionCount, _0sessionCount);
    rpc_write(client, &_0sessionInfos, sizeof(_0sessionInfos));
    updateTmpPtr((void *)sessionInfos, _0sessionInfos);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)sessionCount, sizeof(*sessionCount), true);
    mem2client(client, (void *)sessionInfos, sizeof(*sessionInfos), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDecoderUtilization called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0utilization;
    mem2server(client, &_0utilization, (void *)utilization, sizeof(*utilization));
    void *_0samplingPeriodUs;
    mem2server(client, &_0samplingPeriodUs, (void *)samplingPeriodUs, sizeof(*samplingPeriodUs));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetDecoderUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0utilization, sizeof(_0utilization));
    updateTmpPtr((void *)utilization, _0utilization);
    rpc_write(client, &_0samplingPeriodUs, sizeof(_0samplingPeriodUs));
    updateTmpPtr((void *)samplingPeriodUs, _0samplingPeriodUs);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)utilization, sizeof(*utilization), true);
    mem2client(client, (void *)samplingPeriodUs, sizeof(*samplingPeriodUs), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t *fbcStats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFBCStats called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0fbcStats;
    mem2server(client, &_0fbcStats, (void *)fbcStats, sizeof(*fbcStats));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetFBCStats);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0fbcStats, sizeof(_0fbcStats));
    updateTmpPtr((void *)fbcStats, _0fbcStats);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)fbcStats, sizeof(*fbcStats), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFBCSessions called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0sessionCount;
    mem2server(client, &_0sessionCount, (void *)sessionCount, sizeof(*sessionCount));
    void *_0sessionInfo;
    mem2server(client, &_0sessionInfo, (void *)sessionInfo, sizeof(*sessionInfo));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetFBCSessions);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0sessionCount, sizeof(_0sessionCount));
    updateTmpPtr((void *)sessionCount, _0sessionCount);
    rpc_write(client, &_0sessionInfo, sizeof(_0sessionInfo));
    updateTmpPtr((void *)sessionInfo, _0sessionInfo);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)sessionCount, sizeof(*sessionCount), true);
    mem2client(client, (void *)sessionInfo, sizeof(*sessionInfo), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t device, nvmlDriverModel_t *current, nvmlDriverModel_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDriverModel called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0current;
    mem2server(client, &_0current, (void *)current, sizeof(*current));
    void *_0pending;
    mem2server(client, &_0pending, (void *)pending, sizeof(*pending));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetDriverModel);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0current, sizeof(_0current));
    updateTmpPtr((void *)current, _0current);
    rpc_write(client, &_0pending, sizeof(_0pending));
    updateTmpPtr((void *)pending, _0pending);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)current, sizeof(*current), true);
    mem2client(client, (void *)pending, sizeof(*pending), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVbiosVersion called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetVbiosVersion);
    rpc_write(client, &device, sizeof(device));
    if(length > 0) {
        rpc_read(client, version, length, true);
    }
    rpc_write(client, &length, sizeof(length));
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

extern "C" nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t *bridgeHierarchy) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBridgeChipInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0bridgeHierarchy;
    mem2server(client, &_0bridgeHierarchy, (void *)bridgeHierarchy, sizeof(*bridgeHierarchy));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetBridgeChipInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0bridgeHierarchy, sizeof(_0bridgeHierarchy));
    updateTmpPtr((void *)bridgeHierarchy, _0bridgeHierarchy);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)bridgeHierarchy, sizeof(*bridgeHierarchy), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetComputeRunningProcesses_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0infoCount;
    mem2server(client, &_0infoCount, (void *)infoCount, sizeof(*infoCount));
    void *_0infos;
    mem2server(client, &_0infos, (void *)infos, sizeof(*infos));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetComputeRunningProcesses_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0infoCount, sizeof(_0infoCount));
    updateTmpPtr((void *)infoCount, _0infoCount);
    rpc_write(client, &_0infos, sizeof(_0infos));
    updateTmpPtr((void *)infos, _0infos);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)infoCount, sizeof(*infoCount), true);
    mem2client(client, (void *)infos, sizeof(*infos), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGraphicsRunningProcesses_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0infoCount;
    mem2server(client, &_0infoCount, (void *)infoCount, sizeof(*infoCount));
    void *_0infos;
    mem2server(client, &_0infos, (void *)infos, sizeof(*infos));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetGraphicsRunningProcesses_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0infoCount, sizeof(_0infoCount));
    updateTmpPtr((void *)infoCount, _0infoCount);
    rpc_write(client, &_0infos, sizeof(_0infos));
    updateTmpPtr((void *)infos, _0infos);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)infoCount, sizeof(*infoCount), true);
    mem2client(client, (void *)infos, sizeof(*infos), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMPSComputeRunningProcesses_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0infoCount;
    mem2server(client, &_0infoCount, (void *)infoCount, sizeof(*infoCount));
    void *_0infos;
    mem2server(client, &_0infos, (void *)infos, sizeof(*infos));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetMPSComputeRunningProcesses_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0infoCount, sizeof(_0infoCount));
    updateTmpPtr((void *)infoCount, _0infoCount);
    rpc_write(client, &_0infos, sizeof(_0infos));
    updateTmpPtr((void *)infos, _0infos);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)infoCount, sizeof(*infoCount), true);
    mem2client(client, (void *)infos, sizeof(*infos), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int *onSameBoard) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceOnSameBoard called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0onSameBoard;
    mem2server(client, &_0onSameBoard, (void *)onSameBoard, sizeof(*onSameBoard));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceOnSameBoard);
    rpc_write(client, &device1, sizeof(device1));
    rpc_write(client, &device2, sizeof(device2));
    rpc_write(client, &_0onSameBoard, sizeof(_0onSameBoard));
    updateTmpPtr((void *)onSameBoard, _0onSameBoard);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)onSameBoard, sizeof(*onSameBoard), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t *isRestricted) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAPIRestriction called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0isRestricted;
    mem2server(client, &_0isRestricted, (void *)isRestricted, sizeof(*isRestricted));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetAPIRestriction);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &apiType, sizeof(apiType));
    rpc_write(client, &_0isRestricted, sizeof(_0isRestricted));
    updateTmpPtr((void *)isRestricted, _0isRestricted);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)isRestricted, sizeof(*isRestricted), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *sampleCount, nvmlSample_t *samples) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSamples called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0sampleValType;
    mem2server(client, &_0sampleValType, (void *)sampleValType, sizeof(*sampleValType));
    void *_0sampleCount;
    mem2server(client, &_0sampleCount, (void *)sampleCount, sizeof(*sampleCount));
    void *_0samples;
    mem2server(client, &_0samples, (void *)samples, sizeof(*samples));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetSamples);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    rpc_write(client, &_0sampleValType, sizeof(_0sampleValType));
    updateTmpPtr((void *)sampleValType, _0sampleValType);
    rpc_write(client, &_0sampleCount, sizeof(_0sampleCount));
    updateTmpPtr((void *)sampleCount, _0sampleCount);
    rpc_write(client, &_0samples, sizeof(_0samples));
    updateTmpPtr((void *)samples, _0samples);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)sampleValType, sizeof(*sampleValType), true);
    mem2client(client, (void *)sampleCount, sizeof(*sampleCount), true);
    mem2client(client, (void *)samples, sizeof(*samples), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t *bar1Memory) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBAR1MemoryInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0bar1Memory;
    mem2server(client, &_0bar1Memory, (void *)bar1Memory, sizeof(*bar1Memory));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetBAR1MemoryInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0bar1Memory, sizeof(_0bar1Memory));
    updateTmpPtr((void *)bar1Memory, _0bar1Memory);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)bar1Memory, sizeof(*bar1Memory), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t *violTime) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetViolationStatus called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0violTime;
    mem2server(client, &_0violTime, (void *)violTime, sizeof(*violTime));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetViolationStatus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &perfPolicyType, sizeof(perfPolicyType));
    rpc_write(client, &_0violTime, sizeof(_0violTime));
    updateTmpPtr((void *)violTime, _0violTime);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)violTime, sizeof(*violTime), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingMode called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetAccountingMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t *stats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingStats called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0stats;
    mem2server(client, &_0stats, (void *)stats, sizeof(*stats));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetAccountingStats);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &pid, sizeof(pid));
    rpc_write(client, &_0stats, sizeof(_0stats));
    updateTmpPtr((void *)stats, _0stats);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)stats, sizeof(*stats), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int *count, unsigned int *pids) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingPids called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0count;
    mem2server(client, &_0count, (void *)count, sizeof(*count));
    void *_0pids;
    mem2server(client, &_0pids, (void *)pids, sizeof(*pids));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetAccountingPids);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    rpc_write(client, &_0pids, sizeof(_0pids));
    updateTmpPtr((void *)pids, _0pids);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)count, sizeof(*count), true);
    mem2client(client, (void *)pids, sizeof(*pids), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingBufferSize called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0bufferSize;
    mem2server(client, &_0bufferSize, (void *)bufferSize, sizeof(*bufferSize));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetAccountingBufferSize);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0bufferSize, sizeof(_0bufferSize));
    updateTmpPtr((void *)bufferSize, _0bufferSize);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)bufferSize, sizeof(*bufferSize), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRetiredPages called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pageCount;
    mem2server(client, &_0pageCount, (void *)pageCount, sizeof(*pageCount));
    void *_0addresses;
    mem2server(client, &_0addresses, (void *)addresses, sizeof(*addresses));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetRetiredPages);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &cause, sizeof(cause));
    rpc_write(client, &_0pageCount, sizeof(_0pageCount));
    updateTmpPtr((void *)pageCount, _0pageCount);
    rpc_write(client, &_0addresses, sizeof(_0addresses));
    updateTmpPtr((void *)addresses, _0addresses);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pageCount, sizeof(*pageCount), true);
    mem2client(client, (void *)addresses, sizeof(*addresses), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses, unsigned long long *timestamps) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRetiredPages_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pageCount;
    mem2server(client, &_0pageCount, (void *)pageCount, sizeof(*pageCount));
    void *_0addresses;
    mem2server(client, &_0addresses, (void *)addresses, sizeof(*addresses));
    void *_0timestamps;
    mem2server(client, &_0timestamps, (void *)timestamps, sizeof(*timestamps));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetRetiredPages_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &cause, sizeof(cause));
    rpc_write(client, &_0pageCount, sizeof(_0pageCount));
    updateTmpPtr((void *)pageCount, _0pageCount);
    rpc_write(client, &_0addresses, sizeof(_0addresses));
    updateTmpPtr((void *)addresses, _0addresses);
    rpc_write(client, &_0timestamps, sizeof(_0timestamps));
    updateTmpPtr((void *)timestamps, _0timestamps);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pageCount, sizeof(*pageCount), true);
    mem2client(client, (void *)addresses, sizeof(*addresses), true);
    mem2client(client, (void *)timestamps, sizeof(*timestamps), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t *isPending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRetiredPagesPendingStatus called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0isPending;
    mem2server(client, &_0isPending, (void *)isPending, sizeof(*isPending));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetRetiredPagesPendingStatus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0isPending, sizeof(_0isPending));
    updateTmpPtr((void *)isPending, _0isPending);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)isPending, sizeof(*isPending), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int *corrRows, unsigned int *uncRows, unsigned int *isPending, unsigned int *failureOccurred) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRemappedRows called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0corrRows;
    mem2server(client, &_0corrRows, (void *)corrRows, sizeof(*corrRows));
    void *_0uncRows;
    mem2server(client, &_0uncRows, (void *)uncRows, sizeof(*uncRows));
    void *_0isPending;
    mem2server(client, &_0isPending, (void *)isPending, sizeof(*isPending));
    void *_0failureOccurred;
    mem2server(client, &_0failureOccurred, (void *)failureOccurred, sizeof(*failureOccurred));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetRemappedRows);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0corrRows, sizeof(_0corrRows));
    updateTmpPtr((void *)corrRows, _0corrRows);
    rpc_write(client, &_0uncRows, sizeof(_0uncRows));
    updateTmpPtr((void *)uncRows, _0uncRows);
    rpc_write(client, &_0isPending, sizeof(_0isPending));
    updateTmpPtr((void *)isPending, _0isPending);
    rpc_write(client, &_0failureOccurred, sizeof(_0failureOccurred));
    updateTmpPtr((void *)failureOccurred, _0failureOccurred);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)corrRows, sizeof(*corrRows), true);
    mem2client(client, (void *)uncRows, sizeof(*uncRows), true);
    mem2client(client, (void *)isPending, sizeof(*isPending), true);
    mem2client(client, (void *)failureOccurred, sizeof(*failureOccurred), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t *values) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRowRemapperHistogram called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0values;
    mem2server(client, &_0values, (void *)values, sizeof(*values));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetRowRemapperHistogram);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0values, sizeof(_0values));
    updateTmpPtr((void *)values, _0values);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)values, sizeof(*values), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t *arch) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetArchitecture called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0arch;
    mem2server(client, &_0arch, (void *)arch, sizeof(*arch));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetArchitecture);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0arch, sizeof(_0arch));
    updateTmpPtr((void *)arch, _0arch);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)arch, sizeof(*arch), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitSetLedState called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlUnitSetLedState);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &color, sizeof(color));
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

extern "C" nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetPersistenceMode called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetPersistenceMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetComputeMode called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetComputeMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetEccMode called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetEccMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &ecc, sizeof(ecc));
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

extern "C" nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceClearEccErrorCounts called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceClearEccErrorCounts);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &counterType, sizeof(counterType));
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

extern "C" nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetDriverModel called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetDriverModel);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &driverModel, sizeof(driverModel));
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

extern "C" nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetGpuLockedClocks called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetGpuLockedClocks);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &minGpuClockMHz, sizeof(minGpuClockMHz));
    rpc_write(client, &maxGpuClockMHz, sizeof(maxGpuClockMHz));
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

extern "C" nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetGpuLockedClocks called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceResetGpuLockedClocks);
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

extern "C" nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetMemoryLockedClocks called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetMemoryLockedClocks);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &minMemClockMHz, sizeof(minMemClockMHz));
    rpc_write(client, &maxMemClockMHz, sizeof(maxMemClockMHz));
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

extern "C" nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetMemoryLockedClocks called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceResetMemoryLockedClocks);
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

extern "C" nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetApplicationsClocks called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetApplicationsClocks);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &memClockMHz, sizeof(memClockMHz));
    rpc_write(client, &graphicsClockMHz, sizeof(graphicsClockMHz));
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

extern "C" nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t *status) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClkMonStatus called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0status;
    mem2server(client, &_0status, (void *)status, sizeof(*status));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetClkMonStatus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0status, sizeof(_0status));
    updateTmpPtr((void *)status, _0status);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)status, sizeof(*status), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int limit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetPowerManagementLimit called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetPowerManagementLimit);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &limit, sizeof(limit));
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

extern "C" nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetGpuOperationMode called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetGpuOperationMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetAPIRestriction called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetAPIRestriction);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &apiType, sizeof(apiType));
    rpc_write(client, &isRestricted, sizeof(isRestricted));
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

extern "C" nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetAccountingMode called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetAccountingMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceClearAccountingPids called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceClearAccountingPids);
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

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkState called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0isActive;
    mem2server(client, &_0isActive, (void *)isActive, sizeof(*isActive));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkState);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &_0isActive, sizeof(_0isActive));
    updateTmpPtr((void *)isActive, _0isActive);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)isActive, sizeof(*isActive), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int *version) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkVersion called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkVersion);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
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

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int *capResult) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkCapability called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0capResult;
    mem2server(client, &_0capResult, (void *)capResult, sizeof(*capResult));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkCapability);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &capability, sizeof(capability));
    rpc_write(client, &_0capResult, sizeof(_0capResult));
    updateTmpPtr((void *)capResult, _0capResult);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)capResult, sizeof(*capResult), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkRemotePciInfo_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pci;
    mem2server(client, &_0pci, (void *)pci, sizeof(*pci));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &_0pci, sizeof(_0pci));
    updateTmpPtr((void *)pci, _0pci);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pci, sizeof(*pci), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long *counterValue) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkErrorCounter called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0counterValue;
    mem2server(client, &_0counterValue, (void *)counterValue, sizeof(*counterValue));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkErrorCounter);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &counter, sizeof(counter));
    rpc_write(client, &_0counterValue, sizeof(_0counterValue));
    updateTmpPtr((void *)counterValue, _0counterValue);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)counterValue, sizeof(*counterValue), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetNvLinkErrorCounters called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceResetNvLinkErrorCounters);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
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

extern "C" nvmlReturn_t nvmlDeviceSetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control, unsigned int reset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetNvLinkUtilizationControl called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0control;
    mem2server(client, &_0control, (void *)control, sizeof(*control));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetNvLinkUtilizationControl);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &counter, sizeof(counter));
    rpc_write(client, &_0control, sizeof(_0control));
    updateTmpPtr((void *)control, _0control);
    rpc_write(client, &reset, sizeof(reset));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)control, sizeof(*control), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkUtilizationControl called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0control;
    mem2server(client, &_0control, (void *)control, sizeof(*control));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkUtilizationControl);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &counter, sizeof(counter));
    rpc_write(client, &_0control, sizeof(_0control));
    updateTmpPtr((void *)control, _0control);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)control, sizeof(*control), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, unsigned long long *rxcounter, unsigned long long *txcounter) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkUtilizationCounter called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0rxcounter;
    mem2server(client, &_0rxcounter, (void *)rxcounter, sizeof(*rxcounter));
    void *_0txcounter;
    mem2server(client, &_0txcounter, (void *)txcounter, sizeof(*txcounter));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkUtilizationCounter);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &counter, sizeof(counter));
    rpc_write(client, &_0rxcounter, sizeof(_0rxcounter));
    updateTmpPtr((void *)rxcounter, _0rxcounter);
    rpc_write(client, &_0txcounter, sizeof(_0txcounter));
    updateTmpPtr((void *)txcounter, _0txcounter);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)rxcounter, sizeof(*rxcounter), true);
    mem2client(client, (void *)txcounter, sizeof(*txcounter), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceFreezeNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlEnableState_t freeze) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceFreezeNvLinkUtilizationCounter called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceFreezeNvLinkUtilizationCounter);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &counter, sizeof(counter));
    rpc_write(client, &freeze, sizeof(freeze));
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

extern "C" nvmlReturn_t nvmlDeviceResetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetNvLinkUtilizationCounter called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceResetNvLinkUtilizationCounter);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &counter, sizeof(counter));
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

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkRemoteDeviceType called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pNvLinkDeviceType;
    mem2server(client, &_0pNvLinkDeviceType, (void *)pNvLinkDeviceType, sizeof(*pNvLinkDeviceType));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkRemoteDeviceType);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &_0pNvLinkDeviceType, sizeof(_0pNvLinkDeviceType));
    updateTmpPtr((void *)pNvLinkDeviceType, _0pNvLinkDeviceType);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pNvLinkDeviceType, sizeof(*pNvLinkDeviceType), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t *set) {
#ifdef DEBUG
    std::cout << "Hook: nvmlEventSetCreate called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0set;
    mem2server(client, &_0set, (void *)set, sizeof(*set));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlEventSetCreate);
    rpc_write(client, &_0set, sizeof(_0set));
    updateTmpPtr((void *)set, _0set);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)set, sizeof(*set), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceRegisterEvents called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceRegisterEvents);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &eventTypes, sizeof(eventTypes));
    rpc_write(client, &set, sizeof(set));
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

extern "C" nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long *eventTypes) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedEventTypes called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0eventTypes;
    mem2server(client, &_0eventTypes, (void *)eventTypes, sizeof(*eventTypes));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedEventTypes);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0eventTypes, sizeof(_0eventTypes));
    updateTmpPtr((void *)eventTypes, _0eventTypes);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)eventTypes, sizeof(*eventTypes), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t *data, unsigned int timeoutms) {
#ifdef DEBUG
    std::cout << "Hook: nvmlEventSetWait_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0data;
    mem2server(client, &_0data, (void *)data, 0);
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlEventSetWait_v2);
    rpc_write(client, &set, sizeof(set));
    rpc_write(client, &_0data, sizeof(_0data));
    updateTmpPtr((void *)data, _0data);
    rpc_write(client, &timeoutms, sizeof(timeoutms));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)data, 0, true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set) {
#ifdef DEBUG
    std::cout << "Hook: nvmlEventSetFree called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlEventSetFree);
    rpc_write(client, &set, sizeof(set));
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

extern "C" nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t *pciInfo, nvmlEnableState_t newState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceModifyDrainState called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pciInfo;
    mem2server(client, &_0pciInfo, (void *)pciInfo, sizeof(*pciInfo));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceModifyDrainState);
    rpc_write(client, &_0pciInfo, sizeof(_0pciInfo));
    updateTmpPtr((void *)pciInfo, _0pciInfo);
    rpc_write(client, &newState, sizeof(newState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pciInfo, sizeof(*pciInfo), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t *pciInfo, nvmlEnableState_t *currentState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceQueryDrainState called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pciInfo;
    mem2server(client, &_0pciInfo, (void *)pciInfo, sizeof(*pciInfo));
    void *_0currentState;
    mem2server(client, &_0currentState, (void *)currentState, sizeof(*currentState));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceQueryDrainState);
    rpc_write(client, &_0pciInfo, sizeof(_0pciInfo));
    updateTmpPtr((void *)pciInfo, _0pciInfo);
    rpc_write(client, &_0currentState, sizeof(_0currentState));
    updateTmpPtr((void *)currentState, _0currentState);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pciInfo, sizeof(*pciInfo), true);
    mem2client(client, (void *)currentState, sizeof(*currentState), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t *pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceRemoveGpu_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pciInfo;
    mem2server(client, &_0pciInfo, (void *)pciInfo, sizeof(*pciInfo));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceRemoveGpu_v2);
    rpc_write(client, &_0pciInfo, sizeof(_0pciInfo));
    updateTmpPtr((void *)pciInfo, _0pciInfo);
    rpc_write(client, &gpuState, sizeof(gpuState));
    rpc_write(client, &linkState, sizeof(linkState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pciInfo, sizeof(*pciInfo), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t *pciInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceDiscoverGpus called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pciInfo;
    mem2server(client, &_0pciInfo, (void *)pciInfo, sizeof(*pciInfo));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceDiscoverGpus);
    rpc_write(client, &_0pciInfo, sizeof(_0pciInfo));
    updateTmpPtr((void *)pciInfo, _0pciInfo);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pciInfo, sizeof(*pciInfo), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFieldValues called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0values;
    mem2server(client, &_0values, (void *)values, sizeof(*values));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetFieldValues);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &valuesCount, sizeof(valuesCount));
    rpc_write(client, &_0values, sizeof(_0values));
    updateTmpPtr((void *)values, _0values);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)values, sizeof(*values), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t *pVirtualMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVirtualizationMode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pVirtualMode;
    mem2server(client, &_0pVirtualMode, (void *)pVirtualMode, sizeof(*pVirtualMode));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetVirtualizationMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pVirtualMode, sizeof(_0pVirtualMode));
    updateTmpPtr((void *)pVirtualMode, _0pVirtualMode);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pVirtualMode, sizeof(*pVirtualMode), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t *pHostVgpuMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHostVgpuMode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pHostVgpuMode;
    mem2server(client, &_0pHostVgpuMode, (void *)pHostVgpuMode, sizeof(*pHostVgpuMode));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetHostVgpuMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pHostVgpuMode, sizeof(_0pHostVgpuMode));
    updateTmpPtr((void *)pHostVgpuMode, _0pHostVgpuMode);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pHostVgpuMode, sizeof(*pHostVgpuMode), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetVirtualizationMode called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetVirtualizationMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &virtualMode, sizeof(virtualMode));
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

extern "C" nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v3(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGridLicensableFeatures_v3 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pGridLicensableFeatures;
    mem2server(client, &_0pGridLicensableFeatures, (void *)pGridLicensableFeatures, sizeof(*pGridLicensableFeatures));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetGridLicensableFeatures_v3);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pGridLicensableFeatures, sizeof(_0pGridLicensableFeatures));
    updateTmpPtr((void *)pGridLicensableFeatures, _0pGridLicensableFeatures);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pGridLicensableFeatures, sizeof(*pGridLicensableFeatures), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization, unsigned int *processSamplesCount, unsigned long long lastSeenTimeStamp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetProcessUtilization called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0utilization;
    mem2server(client, &_0utilization, (void *)utilization, sizeof(*utilization));
    void *_0processSamplesCount;
    mem2server(client, &_0processSamplesCount, (void *)processSamplesCount, sizeof(*processSamplesCount));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetProcessUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0utilization, sizeof(_0utilization));
    updateTmpPtr((void *)utilization, _0utilization);
    rpc_write(client, &_0processSamplesCount, sizeof(_0processSamplesCount));
    updateTmpPtr((void *)processSamplesCount, _0processSamplesCount);
    rpc_write(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)utilization, sizeof(*utilization), true);
    mem2client(client, (void *)processSamplesCount, sizeof(*processSamplesCount), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedVgpus called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0vgpuCount;
    mem2server(client, &_0vgpuCount, (void *)vgpuCount, sizeof(*vgpuCount));
    void *_0vgpuTypeIds;
    mem2server(client, &_0vgpuTypeIds, (void *)vgpuTypeIds, sizeof(*vgpuTypeIds));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedVgpus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0vgpuCount, sizeof(_0vgpuCount));
    updateTmpPtr((void *)vgpuCount, _0vgpuCount);
    rpc_write(client, &_0vgpuTypeIds, sizeof(_0vgpuTypeIds));
    updateTmpPtr((void *)vgpuTypeIds, _0vgpuTypeIds);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)vgpuCount, sizeof(*vgpuCount), true);
    mem2client(client, (void *)vgpuTypeIds, sizeof(*vgpuTypeIds), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCreatableVgpus called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0vgpuCount;
    mem2server(client, &_0vgpuCount, (void *)vgpuCount, sizeof(*vgpuCount));
    void *_0vgpuTypeIds;
    mem2server(client, &_0vgpuTypeIds, (void *)vgpuTypeIds, sizeof(*vgpuTypeIds));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetCreatableVgpus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0vgpuCount, sizeof(_0vgpuCount));
    updateTmpPtr((void *)vgpuCount, _0vgpuCount);
    rpc_write(client, &_0vgpuTypeIds, sizeof(_0vgpuTypeIds));
    updateTmpPtr((void *)vgpuTypeIds, _0vgpuTypeIds);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)vgpuCount, sizeof(*vgpuCount), true);
    mem2client(client, (void *)vgpuTypeIds, sizeof(*vgpuTypeIds), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeClass, unsigned int *size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetClass called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetClass);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    if(*size > 0) {
        rpc_read(client, vgpuTypeClass, *size, true);
    }
    rpc_write(client, &_0size, sizeof(_0size));
    updateTmpPtr((void *)size, _0size);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)size, sizeof(*size), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeName, unsigned int *size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetName called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetName);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    if(*size > 0) {
        rpc_read(client, vgpuTypeName, *size, true);
    }
    rpc_write(client, &_0size, sizeof(_0size));
    updateTmpPtr((void *)size, _0size);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)size, sizeof(*size), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *gpuInstanceProfileId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetGpuInstanceProfileId called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0gpuInstanceProfileId;
    mem2server(client, &_0gpuInstanceProfileId, (void *)gpuInstanceProfileId, sizeof(*gpuInstanceProfileId));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetGpuInstanceProfileId);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0gpuInstanceProfileId, sizeof(_0gpuInstanceProfileId));
    updateTmpPtr((void *)gpuInstanceProfileId, _0gpuInstanceProfileId);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)gpuInstanceProfileId, sizeof(*gpuInstanceProfileId), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *deviceID, unsigned long long *subsystemID) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetDeviceID called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0deviceID;
    mem2server(client, &_0deviceID, (void *)deviceID, sizeof(*deviceID));
    void *_0subsystemID;
    mem2server(client, &_0subsystemID, (void *)subsystemID, sizeof(*subsystemID));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetDeviceID);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0deviceID, sizeof(_0deviceID));
    updateTmpPtr((void *)deviceID, _0deviceID);
    rpc_write(client, &_0subsystemID, sizeof(_0subsystemID));
    updateTmpPtr((void *)subsystemID, _0subsystemID);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)deviceID, sizeof(*deviceID), true);
    mem2client(client, (void *)subsystemID, sizeof(*subsystemID), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *fbSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetFramebufferSize called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0fbSize;
    mem2server(client, &_0fbSize, (void *)fbSize, sizeof(*fbSize));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetFramebufferSize);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0fbSize, sizeof(_0fbSize));
    updateTmpPtr((void *)fbSize, _0fbSize);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)fbSize, sizeof(*fbSize), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *numDisplayHeads) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetNumDisplayHeads called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0numDisplayHeads;
    mem2server(client, &_0numDisplayHeads, (void *)numDisplayHeads, sizeof(*numDisplayHeads));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetNumDisplayHeads);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0numDisplayHeads, sizeof(_0numDisplayHeads));
    updateTmpPtr((void *)numDisplayHeads, _0numDisplayHeads);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)numDisplayHeads, sizeof(*numDisplayHeads), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int *xdim, unsigned int *ydim) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetResolution called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0xdim;
    mem2server(client, &_0xdim, (void *)xdim, sizeof(*xdim));
    void *_0ydim;
    mem2server(client, &_0ydim, (void *)ydim, sizeof(*ydim));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetResolution);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &displayIndex, sizeof(displayIndex));
    rpc_write(client, &_0xdim, sizeof(_0xdim));
    updateTmpPtr((void *)xdim, _0xdim);
    rpc_write(client, &_0ydim, sizeof(_0ydim));
    updateTmpPtr((void *)ydim, _0ydim);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)xdim, sizeof(*xdim), true);
    mem2client(client, (void *)ydim, sizeof(*ydim), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeLicenseString, unsigned int size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetLicense called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetLicense);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    if(size > 0) {
        rpc_read(client, vgpuTypeLicenseString, size, true);
    }
    rpc_write(client, &size, sizeof(size));
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

extern "C" nvmlReturn_t nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *frameRateLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetFrameRateLimit called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0frameRateLimit;
    mem2server(client, &_0frameRateLimit, (void *)frameRateLimit, sizeof(*frameRateLimit));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetFrameRateLimit);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0frameRateLimit, sizeof(_0frameRateLimit));
    updateTmpPtr((void *)frameRateLimit, _0frameRateLimit);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)frameRateLimit, sizeof(*frameRateLimit), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetMaxInstances called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0vgpuInstanceCount;
    mem2server(client, &_0vgpuInstanceCount, (void *)vgpuInstanceCount, sizeof(*vgpuInstanceCount));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetMaxInstances);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0vgpuInstanceCount, sizeof(_0vgpuInstanceCount));
    updateTmpPtr((void *)vgpuInstanceCount, _0vgpuInstanceCount);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)vgpuInstanceCount, sizeof(*vgpuInstanceCount), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCountPerVm) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetMaxInstancesPerVm called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0vgpuInstanceCountPerVm;
    mem2server(client, &_0vgpuInstanceCountPerVm, (void *)vgpuInstanceCountPerVm, sizeof(*vgpuInstanceCountPerVm));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetMaxInstancesPerVm);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0vgpuInstanceCountPerVm, sizeof(_0vgpuInstanceCountPerVm));
    updateTmpPtr((void *)vgpuInstanceCountPerVm, _0vgpuInstanceCountPerVm);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)vgpuInstanceCountPerVm, sizeof(*vgpuInstanceCountPerVm), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuInstance_t *vgpuInstances) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetActiveVgpus called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0vgpuCount;
    mem2server(client, &_0vgpuCount, (void *)vgpuCount, sizeof(*vgpuCount));
    void *_0vgpuInstances;
    mem2server(client, &_0vgpuInstances, (void *)vgpuInstances, sizeof(*vgpuInstances));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetActiveVgpus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0vgpuCount, sizeof(_0vgpuCount));
    updateTmpPtr((void *)vgpuCount, _0vgpuCount);
    rpc_write(client, &_0vgpuInstances, sizeof(_0vgpuInstances));
    updateTmpPtr((void *)vgpuInstances, _0vgpuInstances);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)vgpuCount, sizeof(*vgpuCount), true);
    mem2client(client, (void *)vgpuInstances, sizeof(*vgpuInstances), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char *vmId, unsigned int size, nvmlVgpuVmIdType_t *vmIdType) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetVmID called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0vmIdType;
    mem2server(client, &_0vmIdType, (void *)vmIdType, sizeof(*vmIdType));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetVmID);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    if(size > 0) {
        rpc_read(client, vmId, size, true);
    }
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &_0vmIdType, sizeof(_0vmIdType));
    updateTmpPtr((void *)vmIdType, _0vmIdType);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)vmIdType, sizeof(*vmIdType), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char *uuid, unsigned int size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetUUID called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetUUID);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    if(size > 0) {
        rpc_read(client, uuid, size, true);
    }
    rpc_write(client, &size, sizeof(size));
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

extern "C" nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetVmDriverVersion called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetVmDriverVersion);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    if(length > 0) {
        rpc_read(client, version, length, true);
    }
    rpc_write(client, &length, sizeof(length));
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

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long *fbUsage) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFbUsage called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0fbUsage;
    mem2server(client, &_0fbUsage, (void *)fbUsage, sizeof(*fbUsage));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetFbUsage);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0fbUsage, sizeof(_0fbUsage));
    updateTmpPtr((void *)fbUsage, _0fbUsage);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)fbUsage, sizeof(*fbUsage), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int *licensed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetLicenseStatus called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0licensed;
    mem2server(client, &_0licensed, (void *)licensed, sizeof(*licensed));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetLicenseStatus);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0licensed, sizeof(_0licensed));
    updateTmpPtr((void *)licensed, _0licensed);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)licensed, sizeof(*licensed), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t *vgpuTypeId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetType called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0vgpuTypeId;
    mem2server(client, &_0vgpuTypeId, (void *)vgpuTypeId, sizeof(*vgpuTypeId));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetType);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0vgpuTypeId, sizeof(_0vgpuTypeId));
    updateTmpPtr((void *)vgpuTypeId, _0vgpuTypeId);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)vgpuTypeId, sizeof(*vgpuTypeId), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int *frameRateLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFrameRateLimit called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0frameRateLimit;
    mem2server(client, &_0frameRateLimit, (void *)frameRateLimit, sizeof(*frameRateLimit));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetFrameRateLimit);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0frameRateLimit, sizeof(_0frameRateLimit));
    updateTmpPtr((void *)frameRateLimit, _0frameRateLimit);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)frameRateLimit, sizeof(*frameRateLimit), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *eccMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEccMode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0eccMode;
    mem2server(client, &_0eccMode, (void *)eccMode, sizeof(*eccMode));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetEccMode);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0eccMode, sizeof(_0eccMode));
    updateTmpPtr((void *)eccMode, _0eccMode);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)eccMode, sizeof(*eccMode), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int *encoderCapacity) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEncoderCapacity called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0encoderCapacity;
    mem2server(client, &_0encoderCapacity, (void *)encoderCapacity, sizeof(*encoderCapacity));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetEncoderCapacity);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0encoderCapacity, sizeof(_0encoderCapacity));
    updateTmpPtr((void *)encoderCapacity, _0encoderCapacity);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)encoderCapacity, sizeof(*encoderCapacity), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceSetEncoderCapacity called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceSetEncoderCapacity);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &encoderCapacity, sizeof(encoderCapacity));
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

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEncoderStats called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0sessionCount;
    mem2server(client, &_0sessionCount, (void *)sessionCount, sizeof(*sessionCount));
    void *_0averageFps;
    mem2server(client, &_0averageFps, (void *)averageFps, sizeof(*averageFps));
    void *_0averageLatency;
    mem2server(client, &_0averageLatency, (void *)averageLatency, sizeof(*averageLatency));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetEncoderStats);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0sessionCount, sizeof(_0sessionCount));
    updateTmpPtr((void *)sessionCount, _0sessionCount);
    rpc_write(client, &_0averageFps, sizeof(_0averageFps));
    updateTmpPtr((void *)averageFps, _0averageFps);
    rpc_write(client, &_0averageLatency, sizeof(_0averageLatency));
    updateTmpPtr((void *)averageLatency, _0averageLatency);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)sessionCount, sizeof(*sessionCount), true);
    mem2client(client, (void *)averageFps, sizeof(*averageFps), true);
    mem2client(client, (void *)averageLatency, sizeof(*averageLatency), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEncoderSessions called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0sessionCount;
    mem2server(client, &_0sessionCount, (void *)sessionCount, sizeof(*sessionCount));
    void *_0sessionInfo;
    mem2server(client, &_0sessionInfo, (void *)sessionInfo, sizeof(*sessionInfo));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetEncoderSessions);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0sessionCount, sizeof(_0sessionCount));
    updateTmpPtr((void *)sessionCount, _0sessionCount);
    rpc_write(client, &_0sessionInfo, sizeof(_0sessionInfo));
    updateTmpPtr((void *)sessionInfo, _0sessionInfo);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)sessionCount, sizeof(*sessionCount), true);
    mem2client(client, (void *)sessionInfo, sizeof(*sessionInfo), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t *fbcStats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFBCStats called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0fbcStats;
    mem2server(client, &_0fbcStats, (void *)fbcStats, sizeof(*fbcStats));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetFBCStats);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0fbcStats, sizeof(_0fbcStats));
    updateTmpPtr((void *)fbcStats, _0fbcStats);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)fbcStats, sizeof(*fbcStats), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFBCSessions called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0sessionCount;
    mem2server(client, &_0sessionCount, (void *)sessionCount, sizeof(*sessionCount));
    void *_0sessionInfo;
    mem2server(client, &_0sessionInfo, (void *)sessionInfo, sizeof(*sessionInfo));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetFBCSessions);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0sessionCount, sizeof(_0sessionCount));
    updateTmpPtr((void *)sessionCount, _0sessionCount);
    rpc_write(client, &_0sessionInfo, sizeof(_0sessionInfo));
    updateTmpPtr((void *)sessionInfo, _0sessionInfo);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)sessionCount, sizeof(*sessionCount), true);
    mem2client(client, (void *)sessionInfo, sizeof(*sessionInfo), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int *gpuInstanceId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetGpuInstanceId called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0gpuInstanceId;
    mem2server(client, &_0gpuInstanceId, (void *)gpuInstanceId, sizeof(*gpuInstanceId));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetGpuInstanceId);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0gpuInstanceId, sizeof(_0gpuInstanceId));
    updateTmpPtr((void *)gpuInstanceId, _0gpuInstanceId);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)gpuInstanceId, sizeof(*gpuInstanceId), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t *vgpuMetadata, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetMetadata called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0vgpuMetadata;
    mem2server(client, &_0vgpuMetadata, (void *)vgpuMetadata, sizeof(*vgpuMetadata));
    void *_0bufferSize;
    mem2server(client, &_0bufferSize, (void *)bufferSize, sizeof(*bufferSize));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetMetadata);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0vgpuMetadata, sizeof(_0vgpuMetadata));
    updateTmpPtr((void *)vgpuMetadata, _0vgpuMetadata);
    rpc_write(client, &_0bufferSize, sizeof(_0bufferSize));
    updateTmpPtr((void *)bufferSize, _0bufferSize);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)vgpuMetadata, sizeof(*vgpuMetadata), true);
    mem2client(client, (void *)bufferSize, sizeof(*bufferSize), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t *pgpuMetadata, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuMetadata called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0pgpuMetadata;
    mem2server(client, &_0pgpuMetadata, (void *)pgpuMetadata, sizeof(*pgpuMetadata));
    void *_0bufferSize;
    mem2server(client, &_0bufferSize, (void *)bufferSize, sizeof(*bufferSize));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuMetadata);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pgpuMetadata, sizeof(_0pgpuMetadata));
    updateTmpPtr((void *)pgpuMetadata, _0pgpuMetadata);
    rpc_write(client, &_0bufferSize, sizeof(_0bufferSize));
    updateTmpPtr((void *)bufferSize, _0bufferSize);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)pgpuMetadata, sizeof(*pgpuMetadata), true);
    mem2client(client, (void *)bufferSize, sizeof(*bufferSize), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t *vgpuMetadata, nvmlVgpuPgpuMetadata_t *pgpuMetadata, nvmlVgpuPgpuCompatibility_t *compatibilityInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetVgpuCompatibility called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0vgpuMetadata;
    mem2server(client, &_0vgpuMetadata, (void *)vgpuMetadata, sizeof(*vgpuMetadata));
    void *_0pgpuMetadata;
    mem2server(client, &_0pgpuMetadata, (void *)pgpuMetadata, sizeof(*pgpuMetadata));
    void *_0compatibilityInfo;
    mem2server(client, &_0compatibilityInfo, (void *)compatibilityInfo, sizeof(*compatibilityInfo));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlGetVgpuCompatibility);
    rpc_write(client, &_0vgpuMetadata, sizeof(_0vgpuMetadata));
    updateTmpPtr((void *)vgpuMetadata, _0vgpuMetadata);
    rpc_write(client, &_0pgpuMetadata, sizeof(_0pgpuMetadata));
    updateTmpPtr((void *)pgpuMetadata, _0pgpuMetadata);
    rpc_write(client, &_0compatibilityInfo, sizeof(_0compatibilityInfo));
    updateTmpPtr((void *)compatibilityInfo, _0compatibilityInfo);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)vgpuMetadata, sizeof(*vgpuMetadata), true);
    mem2client(client, (void *)pgpuMetadata, sizeof(*pgpuMetadata), true);
    mem2client(client, (void *)compatibilityInfo, sizeof(*compatibilityInfo), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char *pgpuMetadata, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPgpuMetadataString called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0bufferSize;
    mem2server(client, &_0bufferSize, (void *)bufferSize, sizeof(*bufferSize));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetPgpuMetadataString);
    rpc_write(client, &device, sizeof(device));
    if(*bufferSize > 0) {
        rpc_read(client, pgpuMetadata, *bufferSize, true);
    }
    rpc_write(client, &_0bufferSize, sizeof(_0bufferSize));
    updateTmpPtr((void *)bufferSize, _0bufferSize);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)bufferSize, sizeof(*bufferSize), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t *supported, nvmlVgpuVersion_t *current) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetVgpuVersion called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0supported;
    mem2server(client, &_0supported, (void *)supported, sizeof(*supported));
    void *_0current;
    mem2server(client, &_0current, (void *)current, sizeof(*current));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlGetVgpuVersion);
    rpc_write(client, &_0supported, sizeof(_0supported));
    updateTmpPtr((void *)supported, _0supported);
    rpc_write(client, &_0current, sizeof(_0current));
    updateTmpPtr((void *)current, _0current);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)supported, sizeof(*supported), true);
    mem2client(client, (void *)current, sizeof(*current), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t *vgpuVersion) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSetVgpuVersion called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0vgpuVersion;
    mem2server(client, &_0vgpuVersion, (void *)vgpuVersion, sizeof(*vgpuVersion));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlSetVgpuVersion);
    rpc_write(client, &_0vgpuVersion, sizeof(_0vgpuVersion));
    updateTmpPtr((void *)vgpuVersion, _0vgpuVersion);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)vgpuVersion, sizeof(*vgpuVersion), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t *utilizationSamples) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuUtilization called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0sampleValType;
    mem2server(client, &_0sampleValType, (void *)sampleValType, sizeof(*sampleValType));
    void *_0vgpuInstanceSamplesCount;
    mem2server(client, &_0vgpuInstanceSamplesCount, (void *)vgpuInstanceSamplesCount, sizeof(*vgpuInstanceSamplesCount));
    void *_0utilizationSamples;
    mem2server(client, &_0utilizationSamples, (void *)utilizationSamples, sizeof(*utilizationSamples));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    rpc_write(client, &_0sampleValType, sizeof(_0sampleValType));
    updateTmpPtr((void *)sampleValType, _0sampleValType);
    rpc_write(client, &_0vgpuInstanceSamplesCount, sizeof(_0vgpuInstanceSamplesCount));
    updateTmpPtr((void *)vgpuInstanceSamplesCount, _0vgpuInstanceSamplesCount);
    rpc_write(client, &_0utilizationSamples, sizeof(_0utilizationSamples));
    updateTmpPtr((void *)utilizationSamples, _0utilizationSamples);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)sampleValType, sizeof(*sampleValType), true);
    mem2client(client, (void *)vgpuInstanceSamplesCount, sizeof(*vgpuInstanceSamplesCount), true);
    mem2client(client, (void *)utilizationSamples, sizeof(*utilizationSamples), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int *vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t *utilizationSamples) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuProcessUtilization called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0vgpuProcessSamplesCount;
    mem2server(client, &_0vgpuProcessSamplesCount, (void *)vgpuProcessSamplesCount, sizeof(*vgpuProcessSamplesCount));
    void *_0utilizationSamples;
    mem2server(client, &_0utilizationSamples, (void *)utilizationSamples, sizeof(*utilizationSamples));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuProcessUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    rpc_write(client, &_0vgpuProcessSamplesCount, sizeof(_0vgpuProcessSamplesCount));
    updateTmpPtr((void *)vgpuProcessSamplesCount, _0vgpuProcessSamplesCount);
    rpc_write(client, &_0utilizationSamples, sizeof(_0utilizationSamples));
    updateTmpPtr((void *)utilizationSamples, _0utilizationSamples);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)vgpuProcessSamplesCount, sizeof(*vgpuProcessSamplesCount), true);
    mem2client(client, (void *)utilizationSamples, sizeof(*utilizationSamples), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetAccountingMode called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetAccountingMode);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
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

extern "C" nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int *count, unsigned int *pids) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetAccountingPids called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0count;
    mem2server(client, &_0count, (void *)count, sizeof(*count));
    void *_0pids;
    mem2server(client, &_0pids, (void *)pids, sizeof(*pids));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetAccountingPids);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    rpc_write(client, &_0pids, sizeof(_0pids));
    updateTmpPtr((void *)pids, _0pids);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)count, sizeof(*count), true);
    mem2client(client, (void *)pids, sizeof(*pids), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t *stats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetAccountingStats called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0stats;
    mem2server(client, &_0stats, (void *)stats, sizeof(*stats));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetAccountingStats);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &pid, sizeof(pid));
    rpc_write(client, &_0stats, sizeof(_0stats));
    updateTmpPtr((void *)stats, _0stats);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)stats, sizeof(*stats), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceClearAccountingPids called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceClearAccountingPids);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
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

extern "C" nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int *deviceCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetExcludedDeviceCount called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0deviceCount;
    mem2server(client, &_0deviceCount, (void *)deviceCount, sizeof(*deviceCount));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlGetExcludedDeviceCount);
    rpc_write(client, &_0deviceCount, sizeof(_0deviceCount));
    updateTmpPtr((void *)deviceCount, _0deviceCount);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)deviceCount, sizeof(*deviceCount), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetExcludedDeviceInfoByIndex called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0info;
    mem2server(client, &_0info, (void *)info, sizeof(*info));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlGetExcludedDeviceInfoByIndex);
    rpc_write(client, &index, sizeof(index));
    rpc_write(client, &_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)info, sizeof(*info), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t *activationStatus) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetMigMode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0activationStatus;
    mem2server(client, &_0activationStatus, (void *)activationStatus, sizeof(*activationStatus));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceSetMigMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &_0activationStatus, sizeof(_0activationStatus));
    updateTmpPtr((void *)activationStatus, _0activationStatus);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)activationStatus, sizeof(*activationStatus), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *currentMode, unsigned int *pendingMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMigMode called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0currentMode;
    mem2server(client, &_0currentMode, (void *)currentMode, sizeof(*currentMode));
    void *_0pendingMode;
    mem2server(client, &_0pendingMode, (void *)pendingMode, sizeof(*pendingMode));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetMigMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0currentMode, sizeof(_0currentMode));
    updateTmpPtr((void *)currentMode, _0currentMode);
    rpc_write(client, &_0pendingMode, sizeof(_0pendingMode));
    updateTmpPtr((void *)pendingMode, _0pendingMode);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)currentMode, sizeof(*currentMode), true);
    mem2client(client, (void *)pendingMode, sizeof(*pendingMode), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceProfileInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0info;
    mem2server(client, &_0info, (void *)info, sizeof(*info));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstanceProfileInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profile, sizeof(profile));
    rpc_write(client, &_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)info, sizeof(*info), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t *placements, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstancePossiblePlacements_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0placements;
    mem2server(client, &_0placements, (void *)placements, sizeof(*placements));
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstancePossiblePlacements_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0placements, sizeof(_0placements));
    updateTmpPtr((void *)placements, _0placements);
    rpc_write(client, &_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)placements, sizeof(*placements), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceRemainingCapacity called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstanceRemainingCapacity);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
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

extern "C" nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceCreateGpuInstance called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0gpuInstance;
    mem2server(client, &_0gpuInstance, (void *)gpuInstance, sizeof(*gpuInstance));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceCreateGpuInstance);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0gpuInstance, sizeof(_0gpuInstance));
    updateTmpPtr((void *)gpuInstance, _0gpuInstance);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)gpuInstance, sizeof(*gpuInstance), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t *placement, nvmlGpuInstance_t *gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceCreateGpuInstanceWithPlacement called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0placement;
    mem2server(client, &_0placement, (void *)placement, sizeof(*placement));
    void *_0gpuInstance;
    mem2server(client, &_0gpuInstance, (void *)gpuInstance, sizeof(*gpuInstance));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceCreateGpuInstanceWithPlacement);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0placement, sizeof(_0placement));
    updateTmpPtr((void *)placement, _0placement);
    rpc_write(client, &_0gpuInstance, sizeof(_0gpuInstance));
    updateTmpPtr((void *)gpuInstance, _0gpuInstance);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)placement, sizeof(*placement), true);
    mem2client(client, (void *)gpuInstance, sizeof(*gpuInstance), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceDestroy called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlGpuInstanceDestroy);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
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

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstances, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstances called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0gpuInstances;
    mem2server(client, &_0gpuInstances, (void *)gpuInstances, sizeof(*gpuInstances));
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstances);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0gpuInstances, sizeof(_0gpuInstances));
    updateTmpPtr((void *)gpuInstances, _0gpuInstances);
    rpc_write(client, &_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)gpuInstances, sizeof(*gpuInstances), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t *gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceById called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0gpuInstance;
    mem2server(client, &_0gpuInstance, (void *)gpuInstance, sizeof(*gpuInstance));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstanceById);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &id, sizeof(id));
    rpc_write(client, &_0gpuInstance, sizeof(_0gpuInstance));
    updateTmpPtr((void *)gpuInstance, _0gpuInstance);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)gpuInstance, sizeof(*gpuInstance), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0info;
    mem2server(client, &_0info, (void *)info, sizeof(*info));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetInfo);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)info, sizeof(*info), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceProfileInfo called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0info;
    mem2server(client, &_0info, (void *)info, sizeof(*info));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetComputeInstanceProfileInfo);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profile, sizeof(profile));
    rpc_write(client, &engProfile, sizeof(engProfile));
    rpc_write(client, &_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)info, sizeof(*info), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceRemainingCapacity called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetComputeInstanceRemainingCapacity);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profileId, sizeof(profileId));
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

extern "C" nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceCreateComputeInstance called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0computeInstance;
    mem2server(client, &_0computeInstance, (void *)computeInstance, sizeof(*computeInstance));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlGpuInstanceCreateComputeInstance);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0computeInstance, sizeof(_0computeInstance));
    updateTmpPtr((void *)computeInstance, _0computeInstance);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)computeInstance, sizeof(*computeInstance), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlComputeInstanceDestroy called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlComputeInstanceDestroy);
    rpc_write(client, &computeInstance, sizeof(computeInstance));
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

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstances, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstances called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0computeInstances;
    mem2server(client, &_0computeInstances, (void *)computeInstances, sizeof(*computeInstances));
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetComputeInstances);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0computeInstances, sizeof(_0computeInstances));
    updateTmpPtr((void *)computeInstances, _0computeInstances);
    rpc_write(client, &_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)computeInstances, sizeof(*computeInstances), true);
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

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t *computeInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceById called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0computeInstance;
    mem2server(client, &_0computeInstance, (void *)computeInstance, sizeof(*computeInstance));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetComputeInstanceById);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &id, sizeof(id));
    rpc_write(client, &_0computeInstance, sizeof(_0computeInstance));
    updateTmpPtr((void *)computeInstance, _0computeInstance);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)computeInstance, sizeof(*computeInstance), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlComputeInstanceGetInfo_v2 called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0info;
    mem2server(client, &_0info, (void *)info, sizeof(*info));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlComputeInstanceGetInfo_v2);
    rpc_write(client, &computeInstance, sizeof(computeInstance));
    rpc_write(client, &_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)info, sizeof(*info), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int *isMigDevice) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceIsMigDeviceHandle called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0isMigDevice;
    mem2server(client, &_0isMigDevice, (void *)isMigDevice, sizeof(*isMigDevice));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceIsMigDeviceHandle);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0isMigDevice, sizeof(_0isMigDevice));
    updateTmpPtr((void *)isMigDevice, _0isMigDevice);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)isMigDevice, sizeof(*isMigDevice), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int *id) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceId called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0id;
    mem2server(client, &_0id, (void *)id, sizeof(*id));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstanceId);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0id, sizeof(_0id));
    updateTmpPtr((void *)id, _0id);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)id, sizeof(*id), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int *id) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetComputeInstanceId called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0id;
    mem2server(client, &_0id, (void *)id, sizeof(*id));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetComputeInstanceId);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0id, sizeof(_0id));
    updateTmpPtr((void *)id, _0id);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)id, sizeof(*id), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxMigDeviceCount called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxMigDeviceCount);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t *migDevice) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMigDeviceHandleByIndex called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    void *_0migDevice;
    mem2server(client, &_0migDevice, (void *)migDevice, sizeof(*migDevice));
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetMigDeviceHandleByIndex);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &index, sizeof(index));
    rpc_write(client, &_0migDevice, sizeof(_0migDevice));
    updateTmpPtr((void *)migDevice, _0migDevice);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2client);
    mem2client(client, (void *)migDevice, sizeof(*migDevice), true);
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDeviceHandleFromMigDeviceHandle called" << std::endl;
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
    nvmlReturn_t _result;
    rpc_prepare_request(client, RPC_nvmlDeviceGetDeviceHandleFromMigDeviceHandle);
    rpc_write(client, &migDevice, sizeof(migDevice));
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
