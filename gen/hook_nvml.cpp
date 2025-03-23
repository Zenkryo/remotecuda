#include <iostream>
#include <unordered_map>
#include "nvml.h"

#include "hook_api.h"
#include "../rpc.h"
extern void *(*real_dlsym)(void *, const char *);

extern "C" void *mem2server(void *clientPtr, size_t size);
extern "C" void mem2client(void *clientPtr, size_t size);
void *get_so_handle(const std::string &so_file);
extern "C" nvmlReturn_t nvmlInit_v2() {
#ifdef DEBUG
    std::cout << "Hook: nvmlInit_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlInit_v2);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: nvmlInitWithFlags called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlInitWithFlags);
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

extern "C" nvmlReturn_t nvmlShutdown() {
#ifdef DEBUG
    std::cout << "Hook: nvmlShutdown called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlShutdown);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetDriverVersion called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetDriverVersion);
    rpc_read(client, version, length, true);
    rpc_write(client, &length, sizeof(length));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetNVMLVersion called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetNVMLVersion);
    rpc_read(client, version, length, true);
    rpc_write(client, &length, sizeof(length));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetCudaDriverVersion called" << std::endl;
#endif
    void *_0cudaDriverVersion = mem2server((void *)cudaDriverVersion, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetCudaDriverVersion);
    rpc_write(client, &_0cudaDriverVersion, sizeof(_0cudaDriverVersion));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)cudaDriverVersion, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetCudaDriverVersion_v2 called" << std::endl;
#endif
    void *_0cudaDriverVersion = mem2server((void *)cudaDriverVersion, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetCudaDriverVersion_v2);
    rpc_write(client, &_0cudaDriverVersion, sizeof(_0cudaDriverVersion));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)cudaDriverVersion, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char *name, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetProcessName called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetProcessName);
    rpc_write(client, &pid, sizeof(pid));
    rpc_read(client, name, length, true);
    rpc_write(client, &length, sizeof(length));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetHicVersion(unsigned int *hwbcCount, nvmlHwbcEntry_t *hwbcEntries) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetHicVersion called" << std::endl;
#endif
    void *_0hwbcCount = mem2server((void *)hwbcCount, 0);
    void *_0hwbcEntries = mem2server((void *)hwbcEntries, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetHicVersion);
    rpc_write(client, &_0hwbcCount, sizeof(_0hwbcCount));
    rpc_write(client, &_0hwbcEntries, sizeof(_0hwbcEntries));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)hwbcCount, 0);
    mem2client((void *)hwbcEntries, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int *count, nvmlDevice_t *deviceArray) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetTopologyGpuSet called" << std::endl;
#endif
    void *_0count = mem2server((void *)count, 0);
    void *_0deviceArray = mem2server((void *)deviceArray, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetTopologyGpuSet);
    rpc_write(client, &cpuNumber, sizeof(cpuNumber));
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_write(client, &_0deviceArray, sizeof(_0deviceArray));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)count, 0);
    mem2client((void *)deviceArray, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetDriverBranch(nvmlSystemDriverBranchInfo_t *branchInfo, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetDriverBranch called" << std::endl;
#endif
    void *_0branchInfo = mem2server((void *)branchInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetDriverBranch);
    rpc_write(client, &_0branchInfo, sizeof(_0branchInfo));
    rpc_write(client, &length, sizeof(length));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)branchInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetCount called" << std::endl;
#endif
    void *_0unitCount = mem2server((void *)unitCount, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetCount);
    rpc_write(client, &_0unitCount, sizeof(_0unitCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)unitCount, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t *unit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetHandleByIndex called" << std::endl;
#endif
    void *_0unit = mem2server((void *)unit, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetHandleByIndex);
    rpc_write(client, &index, sizeof(index));
    rpc_write(client, &_0unit, sizeof(_0unit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)unit, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetUnitInfo called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetUnitInfo);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t *state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetLedState called" << std::endl;
#endif
    void *_0state = mem2server((void *)state, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetLedState);
    rpc_write(client, &unit, sizeof(unit));
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

extern "C" nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t *psu) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetPsuInfo called" << std::endl;
#endif
    void *_0psu = mem2server((void *)psu, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetPsuInfo);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &_0psu, sizeof(_0psu));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)psu, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetTemperature called" << std::endl;
#endif
    void *_0temp = mem2server((void *)temp, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetTemperature);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &_0temp, sizeof(_0temp));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)temp, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t *fanSpeeds) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetFanSpeedInfo called" << std::endl;
#endif
    void *_0fanSpeeds = mem2server((void *)fanSpeeds, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetFanSpeedInfo);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &_0fanSpeeds, sizeof(_0fanSpeeds));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)fanSpeeds, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int *deviceCount, nvmlDevice_t *devices) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetDevices called" << std::endl;
#endif
    void *_0deviceCount = mem2server((void *)deviceCount, 0);
    void *_0devices = mem2server((void *)devices, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetDevices);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &_0deviceCount, sizeof(_0deviceCount));
    rpc_write(client, &_0devices, sizeof(_0devices));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)deviceCount, 0);
    mem2client((void *)devices, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCount_v2 called" << std::endl;
#endif
    void *_0deviceCount = mem2server((void *)deviceCount, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCount_v2);
    rpc_write(client, &_0deviceCount, sizeof(_0deviceCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)deviceCount, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t *attributes) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAttributes_v2 called" << std::endl;
#endif
    void *_0attributes = mem2server((void *)attributes, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAttributes_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0attributes, sizeof(_0attributes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)attributes, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleByIndex_v2 called" << std::endl;
#endif
    void *_0device = mem2server((void *)device, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetHandleByIndex_v2);
    rpc_write(client, &index, sizeof(index));
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

extern "C" nvmlReturn_t nvmlDeviceGetHandleBySerial(const char *serial, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleBySerial called" << std::endl;
#endif
    void *_0device = mem2server((void *)device, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetHandleBySerial);
    rpc_write(client, serial, strlen(serial) + 1, true);
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

extern "C" nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleByUUID called" << std::endl;
#endif
    void *_0device = mem2server((void *)device, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetHandleByUUID);
    rpc_write(client, uuid, strlen(uuid) + 1, true);
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

extern "C" nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleByPciBusId_v2 called" << std::endl;
#endif
    void *_0device = mem2server((void *)device, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetHandleByPciBusId_v2);
    rpc_write(client, pciBusId, strlen(pciBusId) + 1, true);
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

extern "C" nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetName called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetName);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, name, length, true);
    rpc_write(client, &length, sizeof(length));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t *type) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBrand called" << std::endl;
#endif
    void *_0type = mem2server((void *)type, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetBrand);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetIndex called" << std::endl;
#endif
    void *_0index = mem2server((void *)index, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetIndex);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0index, sizeof(_0index));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)index, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char *serial, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSerial called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSerial);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, serial, length, true);
    rpc_write(client, &length, sizeof(length));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetModuleId(nvmlDevice_t device, unsigned int *moduleId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetModuleId called" << std::endl;
#endif
    void *_0moduleId = mem2server((void *)moduleId, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetModuleId);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0moduleId, sizeof(_0moduleId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)moduleId, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetC2cModeInfoV(nvmlDevice_t device, nvmlC2cModeInfo_v1_t *c2cModeInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetC2cModeInfoV called" << std::endl;
#endif
    void *_0c2cModeInfo = mem2server((void *)c2cModeInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetC2cModeInfoV);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0c2cModeInfo, sizeof(_0c2cModeInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)c2cModeInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long *nodeSet, nvmlAffinityScope_t scope) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryAffinity called" << std::endl;
#endif
    void *_0nodeSet = mem2server((void *)nodeSet, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMemoryAffinity);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &nodeSetSize, sizeof(nodeSetSize));
    rpc_write(client, &_0nodeSet, sizeof(_0nodeSet));
    rpc_write(client, &scope, sizeof(scope));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nodeSet, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet, nvmlAffinityScope_t scope) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCpuAffinityWithinScope called" << std::endl;
#endif
    void *_0cpuSet = mem2server((void *)cpuSet, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCpuAffinityWithinScope);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &cpuSetSize, sizeof(cpuSetSize));
    rpc_write(client, &_0cpuSet, sizeof(_0cpuSet));
    rpc_write(client, &scope, sizeof(scope));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)cpuSet, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCpuAffinity called" << std::endl;
#endif
    void *_0cpuSet = mem2server((void *)cpuSet, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCpuAffinity);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &cpuSetSize, sizeof(cpuSetSize));
    rpc_write(client, &_0cpuSet, sizeof(_0cpuSet));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)cpuSet, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetCpuAffinity called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetCpuAffinity);
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

extern "C" nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceClearCpuAffinity called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceClearCpuAffinity);
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

extern "C" nvmlReturn_t nvmlDeviceGetNumaNodeId(nvmlDevice_t device, unsigned int *node) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNumaNodeId called" << std::endl;
#endif
    void *_0node = mem2server((void *)node, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNumaNodeId);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0node, sizeof(_0node));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)node, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t *pathInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTopologyCommonAncestor called" << std::endl;
#endif
    void *_0pathInfo = mem2server((void *)pathInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetTopologyCommonAncestor);
    rpc_write(client, &device1, sizeof(device1));
    rpc_write(client, &device2, sizeof(device2));
    rpc_write(client, &_0pathInfo, sizeof(_0pathInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pathInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int *count, nvmlDevice_t *deviceArray) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTopologyNearestGpus called" << std::endl;
#endif
    void *_0count = mem2server((void *)count, 0);
    void *_0deviceArray = mem2server((void *)deviceArray, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetTopologyNearestGpus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &level, sizeof(level));
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_write(client, &_0deviceArray, sizeof(_0deviceArray));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)count, 0);
    mem2client((void *)deviceArray, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t *p2pStatus) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetP2PStatus called" << std::endl;
#endif
    void *_0p2pStatus = mem2server((void *)p2pStatus, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetP2PStatus);
    rpc_write(client, &device1, sizeof(device1));
    rpc_write(client, &device2, sizeof(device2));
    rpc_write(client, &p2pIndex, sizeof(p2pIndex));
    rpc_write(client, &_0p2pStatus, sizeof(_0p2pStatus));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)p2pStatus, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetUUID called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetUUID);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, uuid, length, true);
    rpc_write(client, &length, sizeof(length));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minorNumber) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMinorNumber called" << std::endl;
#endif
    void *_0minorNumber = mem2server((void *)minorNumber, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMinorNumber);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0minorNumber, sizeof(_0minorNumber));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)minorNumber, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char *partNumber, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBoardPartNumber called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetBoardPartNumber);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, partNumber, length, true);
    rpc_write(client, &length, sizeof(length));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetInforomVersion called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetInforomVersion);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &object, sizeof(object));
    rpc_read(client, version, length, true);
    rpc_write(client, &length, sizeof(length));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetInforomImageVersion called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetInforomImageVersion);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, version, length, true);
    rpc_write(client, &length, sizeof(length));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int *checksum) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetInforomConfigurationChecksum called" << std::endl;
#endif
    void *_0checksum = mem2server((void *)checksum, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetInforomConfigurationChecksum);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0checksum, sizeof(_0checksum));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)checksum, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceValidateInforom called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceValidateInforom);
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

extern "C" nvmlReturn_t nvmlDeviceGetLastBBXFlushTime(nvmlDevice_t device, unsigned long long *timestamp, unsigned long *durationUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetLastBBXFlushTime called" << std::endl;
#endif
    void *_0timestamp = mem2server((void *)timestamp, 0);
    void *_0durationUs = mem2server((void *)durationUs, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetLastBBXFlushTime);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0timestamp, sizeof(_0timestamp));
    rpc_write(client, &_0durationUs, sizeof(_0durationUs));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)timestamp, 0);
    mem2client((void *)durationUs, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t *display) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDisplayMode called" << std::endl;
#endif
    void *_0display = mem2server((void *)display, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDisplayMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0display, sizeof(_0display));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)display, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t *isActive) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDisplayActive called" << std::endl;
#endif
    void *_0isActive = mem2server((void *)isActive, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDisplayActive);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0isActive, sizeof(_0isActive));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)isActive, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPersistenceMode called" << std::endl;
#endif
    void *_0mode = mem2server((void *)mode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPersistenceMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetPciInfoExt(nvmlDevice_t device, nvmlPciInfoExt_t *pci) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPciInfoExt called" << std::endl;
#endif
    void *_0pci = mem2server((void *)pci, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPciInfoExt);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pci, sizeof(_0pci));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pci, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPciInfo_v3 called" << std::endl;
#endif
    void *_0pci = mem2server((void *)pci, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPciInfo_v3);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pci, sizeof(_0pci));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pci, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGen) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxPcieLinkGeneration called" << std::endl;
#endif
    void *_0maxLinkGen = mem2server((void *)maxLinkGen, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxPcieLinkGeneration);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0maxLinkGen, sizeof(_0maxLinkGen));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)maxLinkGen, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGenDevice) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuMaxPcieLinkGeneration called" << std::endl;
#endif
    void *_0maxLinkGenDevice = mem2server((void *)maxLinkGenDevice, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuMaxPcieLinkGeneration);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0maxLinkGenDevice, sizeof(_0maxLinkGenDevice));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)maxLinkGenDevice, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int *maxLinkWidth) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxPcieLinkWidth called" << std::endl;
#endif
    void *_0maxLinkWidth = mem2server((void *)maxLinkWidth, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxPcieLinkWidth);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0maxLinkWidth, sizeof(_0maxLinkWidth));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)maxLinkWidth, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int *currLinkGen) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrPcieLinkGeneration called" << std::endl;
#endif
    void *_0currLinkGen = mem2server((void *)currLinkGen, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCurrPcieLinkGeneration);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0currLinkGen, sizeof(_0currLinkGen));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)currLinkGen, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int *currLinkWidth) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrPcieLinkWidth called" << std::endl;
#endif
    void *_0currLinkWidth = mem2server((void *)currLinkWidth, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCurrPcieLinkWidth);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0currLinkWidth, sizeof(_0currLinkWidth));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)currLinkWidth, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int *value) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPcieThroughput called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPcieThroughput);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &counter, sizeof(counter));
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

extern "C" nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int *value) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPcieReplayCounter called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPcieReplayCounter);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClockInfo called" << std::endl;
#endif
    void *_0clock = mem2server((void *)clock, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetClockInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &_0clock, sizeof(_0clock));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)clock, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxClockInfo called" << std::endl;
#endif
    void *_0clock = mem2server((void *)clock, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxClockInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &_0clock, sizeof(_0clock));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)clock, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int *offset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpcClkVfOffset called" << std::endl;
#endif
    void *_0offset = mem2server((void *)offset, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpcClkVfOffset);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0offset, sizeof(_0offset));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)offset, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetApplicationsClock called" << std::endl;
#endif
    void *_0clockMHz = mem2server((void *)clockMHz, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetApplicationsClock);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &clockType, sizeof(clockType));
    rpc_write(client, &_0clockMHz, sizeof(_0clockMHz));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)clockMHz, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDefaultApplicationsClock called" << std::endl;
#endif
    void *_0clockMHz = mem2server((void *)clockMHz, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDefaultApplicationsClock);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &clockType, sizeof(clockType));
    rpc_write(client, &_0clockMHz, sizeof(_0clockMHz));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)clockMHz, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClock called" << std::endl;
#endif
    void *_0clockMHz = mem2server((void *)clockMHz, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetClock);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &clockType, sizeof(clockType));
    rpc_write(client, &clockId, sizeof(clockId));
    rpc_write(client, &_0clockMHz, sizeof(_0clockMHz));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)clockMHz, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxCustomerBoostClock called" << std::endl;
#endif
    void *_0clockMHz = mem2server((void *)clockMHz, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxCustomerBoostClock);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &clockType, sizeof(clockType));
    rpc_write(client, &_0clockMHz, sizeof(_0clockMHz));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)clockMHz, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int *count, unsigned int *clocksMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedMemoryClocks called" << std::endl;
#endif
    void *_0count = mem2server((void *)count, 0);
    void *_0clocksMHz = mem2server((void *)clocksMHz, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedMemoryClocks);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_write(client, &_0clocksMHz, sizeof(_0clocksMHz));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)count, 0);
    mem2client((void *)clocksMHz, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int *count, unsigned int *clocksMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedGraphicsClocks called" << std::endl;
#endif
    void *_0count = mem2server((void *)count, 0);
    void *_0clocksMHz = mem2server((void *)clocksMHz, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedGraphicsClocks);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &memoryClockMHz, sizeof(memoryClockMHz));
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_write(client, &_0clocksMHz, sizeof(_0clocksMHz));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)count, 0);
    mem2client((void *)clocksMHz, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t *isEnabled, nvmlEnableState_t *defaultIsEnabled) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAutoBoostedClocksEnabled called" << std::endl;
#endif
    void *_0isEnabled = mem2server((void *)isEnabled, 0);
    void *_0defaultIsEnabled = mem2server((void *)defaultIsEnabled, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAutoBoostedClocksEnabled);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0isEnabled, sizeof(_0isEnabled));
    rpc_write(client, &_0defaultIsEnabled, sizeof(_0defaultIsEnabled));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)isEnabled, 0);
    mem2client((void *)defaultIsEnabled, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFanSpeed called" << std::endl;
#endif
    void *_0speed = mem2server((void *)speed, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetFanSpeed);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0speed, sizeof(_0speed));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)speed, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int *speed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFanSpeed_v2 called" << std::endl;
#endif
    void *_0speed = mem2server((void *)speed, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetFanSpeed_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &fan, sizeof(fan));
    rpc_write(client, &_0speed, sizeof(_0speed));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)speed, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFanSpeedRPM(nvmlDevice_t device, nvmlFanSpeedInfo_t *fanSpeed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFanSpeedRPM called" << std::endl;
#endif
    void *_0fanSpeed = mem2server((void *)fanSpeed, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetFanSpeedRPM);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0fanSpeed, sizeof(_0fanSpeed));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)fanSpeed, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int *targetSpeed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTargetFanSpeed called" << std::endl;
#endif
    void *_0targetSpeed = mem2server((void *)targetSpeed, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetTargetFanSpeed);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &fan, sizeof(fan));
    rpc_write(client, &_0targetSpeed, sizeof(_0targetSpeed));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)targetSpeed, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int *minSpeed, unsigned int *maxSpeed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMinMaxFanSpeed called" << std::endl;
#endif
    void *_0minSpeed = mem2server((void *)minSpeed, 0);
    void *_0maxSpeed = mem2server((void *)maxSpeed, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMinMaxFanSpeed);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0minSpeed, sizeof(_0minSpeed));
    rpc_write(client, &_0maxSpeed, sizeof(_0maxSpeed));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)minSpeed, 0);
    mem2client((void *)maxSpeed, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t *policy) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFanControlPolicy_v2 called" << std::endl;
#endif
    void *_0policy = mem2server((void *)policy, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetFanControlPolicy_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &fan, sizeof(fan));
    rpc_write(client, &_0policy, sizeof(_0policy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)policy, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int *numFans) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNumFans called" << std::endl;
#endif
    void *_0numFans = mem2server((void *)numFans, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNumFans);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0numFans, sizeof(_0numFans));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)numFans, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTemperature called" << std::endl;
#endif
    void *_0temp = mem2server((void *)temp, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetTemperature);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &sensorType, sizeof(sensorType));
    rpc_write(client, &_0temp, sizeof(_0temp));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)temp, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCoolerInfo(nvmlDevice_t device, nvmlCoolerInfo_t *coolerInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCoolerInfo called" << std::endl;
#endif
    void *_0coolerInfo = mem2server((void *)coolerInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCoolerInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0coolerInfo, sizeof(_0coolerInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)coolerInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTemperatureV(nvmlDevice_t device, nvmlTemperature_t *temperature) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTemperatureV called" << std::endl;
#endif
    void *_0temperature = mem2server((void *)temperature, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetTemperatureV);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0temperature, sizeof(_0temperature));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)temperature, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTemperatureThreshold called" << std::endl;
#endif
    void *_0temp = mem2server((void *)temp, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetTemperatureThreshold);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &thresholdType, sizeof(thresholdType));
    rpc_write(client, &_0temp, sizeof(_0temp));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)temp, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMarginTemperature(nvmlDevice_t device, nvmlMarginTemperature_t *marginTempInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMarginTemperature called" << std::endl;
#endif
    void *_0marginTempInfo = mem2server((void *)marginTempInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMarginTemperature);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0marginTempInfo, sizeof(_0marginTempInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)marginTempInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t *pThermalSettings) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetThermalSettings called" << std::endl;
#endif
    void *_0pThermalSettings = mem2server((void *)pThermalSettings, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetThermalSettings);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &sensorIndex, sizeof(sensorIndex));
    rpc_write(client, &_0pThermalSettings, sizeof(_0pThermalSettings));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pThermalSettings, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t *pState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPerformanceState called" << std::endl;
#endif
    void *_0pState = mem2server((void *)pState, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPerformanceState);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pState, sizeof(_0pState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pState, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t device, unsigned long long *clocksEventReasons) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrentClocksEventReasons called" << std::endl;
#endif
    void *_0clocksEventReasons = mem2server((void *)clocksEventReasons, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCurrentClocksEventReasons);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0clocksEventReasons, sizeof(_0clocksEventReasons));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)clocksEventReasons, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long *clocksThrottleReasons) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrentClocksThrottleReasons called" << std::endl;
#endif
    void *_0clocksThrottleReasons = mem2server((void *)clocksThrottleReasons, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCurrentClocksThrottleReasons);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0clocksThrottleReasons, sizeof(_0clocksThrottleReasons));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)clocksThrottleReasons, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedClocksEventReasons(nvmlDevice_t device, unsigned long long *supportedClocksEventReasons) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedClocksEventReasons called" << std::endl;
#endif
    void *_0supportedClocksEventReasons = mem2server((void *)supportedClocksEventReasons, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedClocksEventReasons);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0supportedClocksEventReasons, sizeof(_0supportedClocksEventReasons));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)supportedClocksEventReasons, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long *supportedClocksThrottleReasons) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedClocksThrottleReasons called" << std::endl;
#endif
    void *_0supportedClocksThrottleReasons = mem2server((void *)supportedClocksThrottleReasons, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedClocksThrottleReasons);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0supportedClocksThrottleReasons, sizeof(_0supportedClocksThrottleReasons));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)supportedClocksThrottleReasons, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t *pState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerState called" << std::endl;
#endif
    void *_0pState = mem2server((void *)pState, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerState);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pState, sizeof(_0pState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pState, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t *pDynamicPstatesInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDynamicPstatesInfo called" << std::endl;
#endif
    void *_0pDynamicPstatesInfo = mem2server((void *)pDynamicPstatesInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDynamicPstatesInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pDynamicPstatesInfo, sizeof(_0pDynamicPstatesInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pDynamicPstatesInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int *offset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemClkVfOffset called" << std::endl;
#endif
    void *_0offset = mem2server((void *)offset, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMemClkVfOffset);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0offset, sizeof(_0offset));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)offset, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int *minClockMHz, unsigned int *maxClockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMinMaxClockOfPState called" << std::endl;
#endif
    void *_0minClockMHz = mem2server((void *)minClockMHz, 0);
    void *_0maxClockMHz = mem2server((void *)maxClockMHz, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMinMaxClockOfPState);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &pstate, sizeof(pstate));
    rpc_write(client, &_0minClockMHz, sizeof(_0minClockMHz));
    rpc_write(client, &_0maxClockMHz, sizeof(_0maxClockMHz));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)minClockMHz, 0);
    mem2client((void *)maxClockMHz, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t *pstates, unsigned int size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedPerformanceStates called" << std::endl;
#endif
    void *_0pstates = mem2server((void *)pstates, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedPerformanceStates);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pstates, sizeof(_0pstates));
    rpc_write(client, &size, sizeof(size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pstates, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device, int *minOffset, int *maxOffset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpcClkMinMaxVfOffset called" << std::endl;
#endif
    void *_0minOffset = mem2server((void *)minOffset, 0);
    void *_0maxOffset = mem2server((void *)maxOffset, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpcClkMinMaxVfOffset);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0minOffset, sizeof(_0minOffset));
    rpc_write(client, &_0maxOffset, sizeof(_0maxOffset));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)minOffset, 0);
    mem2client((void *)maxOffset, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device, int *minOffset, int *maxOffset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemClkMinMaxVfOffset called" << std::endl;
#endif
    void *_0minOffset = mem2server((void *)minOffset, 0);
    void *_0maxOffset = mem2server((void *)maxOffset, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMemClkMinMaxVfOffset);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0minOffset, sizeof(_0minOffset));
    rpc_write(client, &_0maxOffset, sizeof(_0maxOffset));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)minOffset, 0);
    mem2client((void *)maxOffset, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClockOffsets called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetClockOffsets);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetClockOffsets called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetClockOffsets);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPerformanceModes(nvmlDevice_t device, nvmlDevicePerfModes_t *perfModes) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPerformanceModes called" << std::endl;
#endif
    void *_0perfModes = mem2server((void *)perfModes, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPerformanceModes);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0perfModes, sizeof(_0perfModes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)perfModes, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrentClockFreqs(nvmlDevice_t device, nvmlDeviceCurrentClockFreqs_t *currentClockFreqs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrentClockFreqs called" << std::endl;
#endif
    void *_0currentClockFreqs = mem2server((void *)currentClockFreqs, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCurrentClockFreqs);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0currentClockFreqs, sizeof(_0currentClockFreqs));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)currentClockFreqs, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementMode called" << std::endl;
#endif
    void *_0mode = mem2server((void *)mode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerManagementMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int *limit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementLimit called" << std::endl;
#endif
    void *_0limit = mem2server((void *)limit, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerManagementLimit);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0limit, sizeof(_0limit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)limit, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementLimitConstraints called" << std::endl;
#endif
    void *_0minLimit = mem2server((void *)minLimit, 0);
    void *_0maxLimit = mem2server((void *)maxLimit, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerManagementLimitConstraints);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0minLimit, sizeof(_0minLimit));
    rpc_write(client, &_0maxLimit, sizeof(_0maxLimit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)minLimit, 0);
    mem2client((void *)maxLimit, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int *defaultLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementDefaultLimit called" << std::endl;
#endif
    void *_0defaultLimit = mem2server((void *)defaultLimit, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerManagementDefaultLimit);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0defaultLimit, sizeof(_0defaultLimit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)defaultLimit, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerUsage called" << std::endl;
#endif
    void *_0power = mem2server((void *)power, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerUsage);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0power, sizeof(_0power));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)power, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long *energy) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTotalEnergyConsumption called" << std::endl;
#endif
    void *_0energy = mem2server((void *)energy, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetTotalEnergyConsumption);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0energy, sizeof(_0energy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)energy, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEnforcedPowerLimit called" << std::endl;
#endif
    void *_0limit = mem2server((void *)limit, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetEnforcedPowerLimit);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0limit, sizeof(_0limit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)limit, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t *current, nvmlGpuOperationMode_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuOperationMode called" << std::endl;
#endif
    void *_0current = mem2server((void *)current, 0);
    void *_0pending = mem2server((void *)pending, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuOperationMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0current, sizeof(_0current));
    rpc_write(client, &_0pending, sizeof(_0pending));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)current, 0);
    mem2client((void *)pending, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryInfo called" << std::endl;
#endif
    void *_0memory = mem2server((void *)memory, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMemoryInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0memory, sizeof(_0memory));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)memory, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryInfo_v2 called" << std::endl;
#endif
    void *_0memory = mem2server((void *)memory, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMemoryInfo_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0memory, sizeof(_0memory));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)memory, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetComputeMode called" << std::endl;
#endif
    void *_0mode = mem2server((void *)mode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetComputeMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCudaComputeCapability called" << std::endl;
#endif
    void *_0major = mem2server((void *)major, 0);
    void *_0minor = mem2server((void *)minor, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCudaComputeCapability);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0major, sizeof(_0major));
    rpc_write(client, &_0minor, sizeof(_0minor));
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

extern "C" nvmlReturn_t nvmlDeviceGetDramEncryptionMode(nvmlDevice_t device, nvmlDramEncryptionInfo_t *current, nvmlDramEncryptionInfo_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDramEncryptionMode called" << std::endl;
#endif
    void *_0current = mem2server((void *)current, 0);
    void *_0pending = mem2server((void *)pending, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDramEncryptionMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0current, sizeof(_0current));
    rpc_write(client, &_0pending, sizeof(_0pending));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)current, 0);
    mem2client((void *)pending, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetDramEncryptionMode(nvmlDevice_t device, const nvmlDramEncryptionInfo_t *dramEncryption) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetDramEncryptionMode called" << std::endl;
#endif
    void *_0dramEncryption = mem2server((void *)dramEncryption, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetDramEncryptionMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0dramEncryption, sizeof(_0dramEncryption));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)dramEncryption, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t *current, nvmlEnableState_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEccMode called" << std::endl;
#endif
    void *_0current = mem2server((void *)current, 0);
    void *_0pending = mem2server((void *)pending, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetEccMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0current, sizeof(_0current));
    rpc_write(client, &_0pending, sizeof(_0pending));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)current, 0);
    mem2client((void *)pending, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t *defaultMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDefaultEccMode called" << std::endl;
#endif
    void *_0defaultMode = mem2server((void *)defaultMode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDefaultEccMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0defaultMode, sizeof(_0defaultMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)defaultMode, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int *boardId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBoardId called" << std::endl;
#endif
    void *_0boardId = mem2server((void *)boardId, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetBoardId);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0boardId, sizeof(_0boardId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)boardId, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int *multiGpuBool) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMultiGpuBoard called" << std::endl;
#endif
    void *_0multiGpuBool = mem2server((void *)multiGpuBool, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMultiGpuBoard);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0multiGpuBool, sizeof(_0multiGpuBool));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)multiGpuBool, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long *eccCounts) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTotalEccErrors called" << std::endl;
#endif
    void *_0eccCounts = mem2server((void *)eccCounts, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetTotalEccErrors);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &errorType, sizeof(errorType));
    rpc_write(client, &counterType, sizeof(counterType));
    rpc_write(client, &_0eccCounts, sizeof(_0eccCounts));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)eccCounts, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t *eccCounts) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDetailedEccErrors called" << std::endl;
#endif
    void *_0eccCounts = mem2server((void *)eccCounts, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDetailedEccErrors);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &errorType, sizeof(errorType));
    rpc_write(client, &counterType, sizeof(counterType));
    rpc_write(client, &_0eccCounts, sizeof(_0eccCounts));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)eccCounts, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryErrorCounter called" << std::endl;
#endif
    void *_0count = mem2server((void *)count, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMemoryErrorCounter);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &errorType, sizeof(errorType));
    rpc_write(client, &counterType, sizeof(counterType));
    rpc_write(client, &locationType, sizeof(locationType));
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

extern "C" nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetUtilizationRates called" << std::endl;
#endif
    void *_0utilization = mem2server((void *)utilization, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetUtilizationRates);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0utilization, sizeof(_0utilization));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)utilization, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderUtilization called" << std::endl;
#endif
    void *_0utilization = mem2server((void *)utilization, 0);
    void *_0samplingPeriodUs = mem2server((void *)samplingPeriodUs, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetEncoderUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0utilization, sizeof(_0utilization));
    rpc_write(client, &_0samplingPeriodUs, sizeof(_0samplingPeriodUs));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)utilization, 0);
    mem2client((void *)samplingPeriodUs, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int *encoderCapacity) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderCapacity called" << std::endl;
#endif
    void *_0encoderCapacity = mem2server((void *)encoderCapacity, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetEncoderCapacity);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &encoderQueryType, sizeof(encoderQueryType));
    rpc_write(client, &_0encoderCapacity, sizeof(_0encoderCapacity));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)encoderCapacity, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderStats called" << std::endl;
#endif
    void *_0sessionCount = mem2server((void *)sessionCount, 0);
    void *_0averageFps = mem2server((void *)averageFps, 0);
    void *_0averageLatency = mem2server((void *)averageLatency, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetEncoderStats);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0sessionCount, sizeof(_0sessionCount));
    rpc_write(client, &_0averageFps, sizeof(_0averageFps));
    rpc_write(client, &_0averageLatency, sizeof(_0averageLatency));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)sessionCount, 0);
    mem2client((void *)averageFps, 0);
    mem2client((void *)averageLatency, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderSessions called" << std::endl;
#endif
    void *_0sessionCount = mem2server((void *)sessionCount, 0);
    void *_0sessionInfos = mem2server((void *)sessionInfos, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetEncoderSessions);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0sessionCount, sizeof(_0sessionCount));
    rpc_write(client, &_0sessionInfos, sizeof(_0sessionInfos));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)sessionCount, 0);
    mem2client((void *)sessionInfos, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDecoderUtilization called" << std::endl;
#endif
    void *_0utilization = mem2server((void *)utilization, 0);
    void *_0samplingPeriodUs = mem2server((void *)samplingPeriodUs, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDecoderUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0utilization, sizeof(_0utilization));
    rpc_write(client, &_0samplingPeriodUs, sizeof(_0samplingPeriodUs));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)utilization, 0);
    mem2client((void *)samplingPeriodUs, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetJpgUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetJpgUtilization called" << std::endl;
#endif
    void *_0utilization = mem2server((void *)utilization, 0);
    void *_0samplingPeriodUs = mem2server((void *)samplingPeriodUs, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetJpgUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0utilization, sizeof(_0utilization));
    rpc_write(client, &_0samplingPeriodUs, sizeof(_0samplingPeriodUs));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)utilization, 0);
    mem2client((void *)samplingPeriodUs, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetOfaUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetOfaUtilization called" << std::endl;
#endif
    void *_0utilization = mem2server((void *)utilization, 0);
    void *_0samplingPeriodUs = mem2server((void *)samplingPeriodUs, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetOfaUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0utilization, sizeof(_0utilization));
    rpc_write(client, &_0samplingPeriodUs, sizeof(_0samplingPeriodUs));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)utilization, 0);
    mem2client((void *)samplingPeriodUs, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t *fbcStats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFBCStats called" << std::endl;
#endif
    void *_0fbcStats = mem2server((void *)fbcStats, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetFBCStats);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0fbcStats, sizeof(_0fbcStats));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)fbcStats, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFBCSessions called" << std::endl;
#endif
    void *_0sessionCount = mem2server((void *)sessionCount, 0);
    void *_0sessionInfo = mem2server((void *)sessionInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetFBCSessions);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0sessionCount, sizeof(_0sessionCount));
    rpc_write(client, &_0sessionInfo, sizeof(_0sessionInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)sessionCount, 0);
    mem2client((void *)sessionInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDriverModel_v2(nvmlDevice_t device, nvmlDriverModel_t *current, nvmlDriverModel_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDriverModel_v2 called" << std::endl;
#endif
    void *_0current = mem2server((void *)current, 0);
    void *_0pending = mem2server((void *)pending, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDriverModel_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0current, sizeof(_0current));
    rpc_write(client, &_0pending, sizeof(_0pending));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)current, 0);
    mem2client((void *)pending, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVbiosVersion called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVbiosVersion);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, version, length, true);
    rpc_write(client, &length, sizeof(length));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t *bridgeHierarchy) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBridgeChipInfo called" << std::endl;
#endif
    void *_0bridgeHierarchy = mem2server((void *)bridgeHierarchy, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetBridgeChipInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0bridgeHierarchy, sizeof(_0bridgeHierarchy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)bridgeHierarchy, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetComputeRunningProcesses_v3 called" << std::endl;
#endif
    void *_0infoCount = mem2server((void *)infoCount, 0);
    void *_0infos = mem2server((void *)infos, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetComputeRunningProcesses_v3);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0infoCount, sizeof(_0infoCount));
    rpc_write(client, &_0infos, sizeof(_0infos));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)infoCount, 0);
    mem2client((void *)infos, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGraphicsRunningProcesses_v3 called" << std::endl;
#endif
    void *_0infoCount = mem2server((void *)infoCount, 0);
    void *_0infos = mem2server((void *)infos, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGraphicsRunningProcesses_v3);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0infoCount, sizeof(_0infoCount));
    rpc_write(client, &_0infos, sizeof(_0infos));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)infoCount, 0);
    mem2client((void *)infos, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMPSComputeRunningProcesses_v3 called" << std::endl;
#endif
    void *_0infoCount = mem2server((void *)infoCount, 0);
    void *_0infos = mem2server((void *)infos, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMPSComputeRunningProcesses_v3);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0infoCount, sizeof(_0infoCount));
    rpc_write(client, &_0infos, sizeof(_0infos));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)infoCount, 0);
    mem2client((void *)infos, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRunningProcessDetailList(nvmlDevice_t device, nvmlProcessDetailList_t *plist) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRunningProcessDetailList called" << std::endl;
#endif
    void *_0plist = mem2server((void *)plist, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetRunningProcessDetailList);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0plist, sizeof(_0plist));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)plist, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int *onSameBoard) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceOnSameBoard called" << std::endl;
#endif
    void *_0onSameBoard = mem2server((void *)onSameBoard, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceOnSameBoard);
    rpc_write(client, &device1, sizeof(device1));
    rpc_write(client, &device2, sizeof(device2));
    rpc_write(client, &_0onSameBoard, sizeof(_0onSameBoard));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)onSameBoard, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t *isRestricted) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAPIRestriction called" << std::endl;
#endif
    void *_0isRestricted = mem2server((void *)isRestricted, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAPIRestriction);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &apiType, sizeof(apiType));
    rpc_write(client, &_0isRestricted, sizeof(_0isRestricted));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)isRestricted, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *sampleCount, nvmlSample_t *samples) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSamples called" << std::endl;
#endif
    void *_0sampleValType = mem2server((void *)sampleValType, 0);
    void *_0sampleCount = mem2server((void *)sampleCount, 0);
    void *_0samples = mem2server((void *)samples, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSamples);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    rpc_write(client, &_0sampleValType, sizeof(_0sampleValType));
    rpc_write(client, &_0sampleCount, sizeof(_0sampleCount));
    rpc_write(client, &_0samples, sizeof(_0samples));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)sampleValType, 0);
    mem2client((void *)sampleCount, 0);
    mem2client((void *)samples, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t *bar1Memory) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBAR1MemoryInfo called" << std::endl;
#endif
    void *_0bar1Memory = mem2server((void *)bar1Memory, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetBAR1MemoryInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0bar1Memory, sizeof(_0bar1Memory));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)bar1Memory, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t *violTime) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetViolationStatus called" << std::endl;
#endif
    void *_0violTime = mem2server((void *)violTime, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetViolationStatus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &perfPolicyType, sizeof(perfPolicyType));
    rpc_write(client, &_0violTime, sizeof(_0violTime));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)violTime, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int *irqNum) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetIrqNum called" << std::endl;
#endif
    void *_0irqNum = mem2server((void *)irqNum, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetIrqNum);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0irqNum, sizeof(_0irqNum));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)irqNum, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int *numCores) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNumGpuCores called" << std::endl;
#endif
    void *_0numCores = mem2server((void *)numCores, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNumGpuCores);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0numCores, sizeof(_0numCores));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)numCores, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t *powerSource) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerSource called" << std::endl;
#endif
    void *_0powerSource = mem2server((void *)powerSource, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerSource);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0powerSource, sizeof(_0powerSource));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)powerSource, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int *busWidth) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryBusWidth called" << std::endl;
#endif
    void *_0busWidth = mem2server((void *)busWidth, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMemoryBusWidth);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0busWidth, sizeof(_0busWidth));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)busWidth, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int *maxSpeed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPcieLinkMaxSpeed called" << std::endl;
#endif
    void *_0maxSpeed = mem2server((void *)maxSpeed, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPcieLinkMaxSpeed);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0maxSpeed, sizeof(_0maxSpeed));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)maxSpeed, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int *pcieSpeed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPcieSpeed called" << std::endl;
#endif
    void *_0pcieSpeed = mem2server((void *)pcieSpeed, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPcieSpeed);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pcieSpeed, sizeof(_0pcieSpeed));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pcieSpeed, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int *adaptiveClockStatus) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAdaptiveClockInfoStatus called" << std::endl;
#endif
    void *_0adaptiveClockStatus = mem2server((void *)adaptiveClockStatus, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAdaptiveClockInfoStatus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0adaptiveClockStatus, sizeof(_0adaptiveClockStatus));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)adaptiveClockStatus, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t *type) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBusType called" << std::endl;
#endif
    void *_0type = mem2server((void *)type, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetBusType);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetGpuFabricInfo(nvmlDevice_t device, nvmlGpuFabricInfo_t *gpuFabricInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuFabricInfo called" << std::endl;
#endif
    void *_0gpuFabricInfo = mem2server((void *)gpuFabricInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuFabricInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0gpuFabricInfo, sizeof(_0gpuFabricInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)gpuFabricInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t *gpuFabricInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuFabricInfoV called" << std::endl;
#endif
    void *_0gpuFabricInfo = mem2server((void *)gpuFabricInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuFabricInfoV);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0gpuFabricInfo, sizeof(_0gpuFabricInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)gpuFabricInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetConfComputeCapabilities(nvmlConfComputeSystemCaps_t *capabilities) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetConfComputeCapabilities called" << std::endl;
#endif
    void *_0capabilities = mem2server((void *)capabilities, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetConfComputeCapabilities);
    rpc_write(client, &_0capabilities, sizeof(_0capabilities));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)capabilities, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetConfComputeState(nvmlConfComputeSystemState_t *state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetConfComputeState called" << std::endl;
#endif
    void *_0state = mem2server((void *)state, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetConfComputeState);
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

extern "C" nvmlReturn_t nvmlDeviceGetConfComputeMemSizeInfo(nvmlDevice_t device, nvmlConfComputeMemSizeInfo_t *memInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetConfComputeMemSizeInfo called" << std::endl;
#endif
    void *_0memInfo = mem2server((void *)memInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetConfComputeMemSizeInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0memInfo, sizeof(_0memInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)memInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetConfComputeGpusReadyState(unsigned int *isAcceptingWork) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetConfComputeGpusReadyState called" << std::endl;
#endif
    void *_0isAcceptingWork = mem2server((void *)isAcceptingWork, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetConfComputeGpusReadyState);
    rpc_write(client, &_0isAcceptingWork, sizeof(_0isAcceptingWork));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)isAcceptingWork, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetConfComputeProtectedMemoryUsage(nvmlDevice_t device, nvmlMemory_t *memory) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetConfComputeProtectedMemoryUsage called" << std::endl;
#endif
    void *_0memory = mem2server((void *)memory, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetConfComputeProtectedMemoryUsage);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0memory, sizeof(_0memory));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)memory, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetConfComputeGpuCertificate(nvmlDevice_t device, nvmlConfComputeGpuCertificate_t *gpuCert) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetConfComputeGpuCertificate called" << std::endl;
#endif
    void *_0gpuCert = mem2server((void *)gpuCert, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetConfComputeGpuCertificate);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0gpuCert, sizeof(_0gpuCert));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)gpuCert, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetConfComputeGpuAttestationReport(nvmlDevice_t device, nvmlConfComputeGpuAttestationReport_t *gpuAtstReport) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetConfComputeGpuAttestationReport called" << std::endl;
#endif
    void *_0gpuAtstReport = mem2server((void *)gpuAtstReport, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetConfComputeGpuAttestationReport);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0gpuAtstReport, sizeof(_0gpuAtstReport));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)gpuAtstReport, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetConfComputeKeyRotationThresholdInfo(nvmlConfComputeGetKeyRotationThresholdInfo_t *pKeyRotationThrInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetConfComputeKeyRotationThresholdInfo called" << std::endl;
#endif
    void *_0pKeyRotationThrInfo = mem2server((void *)pKeyRotationThrInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetConfComputeKeyRotationThresholdInfo);
    rpc_write(client, &_0pKeyRotationThrInfo, sizeof(_0pKeyRotationThrInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pKeyRotationThrInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetConfComputeUnprotectedMemSize(nvmlDevice_t device, unsigned long long sizeKiB) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetConfComputeUnprotectedMemSize called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetConfComputeUnprotectedMemSize);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &sizeKiB, sizeof(sizeKiB));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemSetConfComputeGpusReadyState(unsigned int isAcceptingWork) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemSetConfComputeGpusReadyState called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemSetConfComputeGpusReadyState);
    rpc_write(client, &isAcceptingWork, sizeof(isAcceptingWork));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemSetConfComputeKeyRotationThresholdInfo(nvmlConfComputeSetKeyRotationThresholdInfo_t *pKeyRotationThrInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemSetConfComputeKeyRotationThresholdInfo called" << std::endl;
#endif
    void *_0pKeyRotationThrInfo = mem2server((void *)pKeyRotationThrInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemSetConfComputeKeyRotationThresholdInfo);
    rpc_write(client, &_0pKeyRotationThrInfo, sizeof(_0pKeyRotationThrInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pKeyRotationThrInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetConfComputeSettings(nvmlSystemConfComputeSettings_t *settings) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetConfComputeSettings called" << std::endl;
#endif
    void *_0settings = mem2server((void *)settings, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetConfComputeSettings);
    rpc_write(client, &_0settings, sizeof(_0settings));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)settings, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char *version) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGspFirmwareVersion called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGspFirmwareVersion);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, version, 32, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGspFirmwareMode(nvmlDevice_t device, unsigned int *isEnabled, unsigned int *defaultMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGspFirmwareMode called" << std::endl;
#endif
    void *_0isEnabled = mem2server((void *)isEnabled, 0);
    void *_0defaultMode = mem2server((void *)defaultMode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGspFirmwareMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0isEnabled, sizeof(_0isEnabled));
    rpc_write(client, &_0defaultMode, sizeof(_0defaultMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)isEnabled, 0);
    mem2client((void *)defaultMode, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSramEccErrorStatus(nvmlDevice_t device, nvmlEccSramErrorStatus_t *status) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSramEccErrorStatus called" << std::endl;
#endif
    void *_0status = mem2server((void *)status, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSramEccErrorStatus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0status, sizeof(_0status));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)status, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingMode called" << std::endl;
#endif
    void *_0mode = mem2server((void *)mode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAccountingMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t *stats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingStats called" << std::endl;
#endif
    void *_0stats = mem2server((void *)stats, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAccountingStats);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &pid, sizeof(pid));
    rpc_write(client, &_0stats, sizeof(_0stats));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)stats, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int *count, unsigned int *pids) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingPids called" << std::endl;
#endif
    void *_0count = mem2server((void *)count, 0);
    void *_0pids = mem2server((void *)pids, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAccountingPids);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_write(client, &_0pids, sizeof(_0pids));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)count, 0);
    mem2client((void *)pids, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingBufferSize called" << std::endl;
#endif
    void *_0bufferSize = mem2server((void *)bufferSize, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAccountingBufferSize);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0bufferSize, sizeof(_0bufferSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)bufferSize, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRetiredPages called" << std::endl;
#endif
    void *_0pageCount = mem2server((void *)pageCount, 0);
    void *_0addresses = mem2server((void *)addresses, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetRetiredPages);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &cause, sizeof(cause));
    rpc_write(client, &_0pageCount, sizeof(_0pageCount));
    rpc_write(client, &_0addresses, sizeof(_0addresses));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pageCount, 0);
    mem2client((void *)addresses, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses, unsigned long long *timestamps) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRetiredPages_v2 called" << std::endl;
#endif
    void *_0pageCount = mem2server((void *)pageCount, 0);
    void *_0addresses = mem2server((void *)addresses, 0);
    void *_0timestamps = mem2server((void *)timestamps, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetRetiredPages_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &cause, sizeof(cause));
    rpc_write(client, &_0pageCount, sizeof(_0pageCount));
    rpc_write(client, &_0addresses, sizeof(_0addresses));
    rpc_write(client, &_0timestamps, sizeof(_0timestamps));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pageCount, 0);
    mem2client((void *)addresses, 0);
    mem2client((void *)timestamps, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t *isPending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRetiredPagesPendingStatus called" << std::endl;
#endif
    void *_0isPending = mem2server((void *)isPending, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetRetiredPagesPendingStatus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0isPending, sizeof(_0isPending));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)isPending, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int *corrRows, unsigned int *uncRows, unsigned int *isPending, unsigned int *failureOccurred) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRemappedRows called" << std::endl;
#endif
    void *_0corrRows = mem2server((void *)corrRows, 0);
    void *_0uncRows = mem2server((void *)uncRows, 0);
    void *_0isPending = mem2server((void *)isPending, 0);
    void *_0failureOccurred = mem2server((void *)failureOccurred, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetRemappedRows);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0corrRows, sizeof(_0corrRows));
    rpc_write(client, &_0uncRows, sizeof(_0uncRows));
    rpc_write(client, &_0isPending, sizeof(_0isPending));
    rpc_write(client, &_0failureOccurred, sizeof(_0failureOccurred));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)corrRows, 0);
    mem2client((void *)uncRows, 0);
    mem2client((void *)isPending, 0);
    mem2client((void *)failureOccurred, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t *values) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRowRemapperHistogram called" << std::endl;
#endif
    void *_0values = mem2server((void *)values, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetRowRemapperHistogram);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0values, sizeof(_0values));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)values, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t *arch) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetArchitecture called" << std::endl;
#endif
    void *_0arch = mem2server((void *)arch, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetArchitecture);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0arch, sizeof(_0arch));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)arch, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t *status) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClkMonStatus called" << std::endl;
#endif
    void *_0status = mem2server((void *)status, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetClkMonStatus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0status, sizeof(_0status));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)status, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization, unsigned int *processSamplesCount, unsigned long long lastSeenTimeStamp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetProcessUtilization called" << std::endl;
#endif
    void *_0utilization = mem2server((void *)utilization, 0);
    void *_0processSamplesCount = mem2server((void *)processSamplesCount, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetProcessUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0utilization, sizeof(_0utilization));
    rpc_write(client, &_0processSamplesCount, sizeof(_0processSamplesCount));
    rpc_write(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)utilization, 0);
    mem2client((void *)processSamplesCount, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetProcessesUtilizationInfo(nvmlDevice_t device, nvmlProcessesUtilizationInfo_t *procesesUtilInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetProcessesUtilizationInfo called" << std::endl;
#endif
    void *_0procesesUtilInfo = mem2server((void *)procesesUtilInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetProcessesUtilizationInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0procesesUtilInfo, sizeof(_0procesesUtilInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)procesesUtilInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPlatformInfo(nvmlDevice_t device, nvmlPlatformInfo_t *platformInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPlatformInfo called" << std::endl;
#endif
    void *_0platformInfo = mem2server((void *)platformInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPlatformInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0platformInfo, sizeof(_0platformInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)platformInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitSetLedState called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitSetLedState);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &color, sizeof(color));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetPersistenceMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetPersistenceMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetComputeMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetComputeMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetEccMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetEccMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &ecc, sizeof(ecc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceClearEccErrorCounts called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceClearEccErrorCounts);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &counterType, sizeof(counterType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetDriverModel called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
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
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetGpuLockedClocks called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
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
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetGpuLockedClocks called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceResetGpuLockedClocks);
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

extern "C" nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetMemoryLockedClocks called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
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
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetMemoryLockedClocks called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceResetMemoryLockedClocks);
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

extern "C" nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetApplicationsClocks called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
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
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceResetApplicationsClocks(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetApplicationsClocks called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceResetApplicationsClocks);
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

extern "C" nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetAutoBoostedClocksEnabled called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetAutoBoostedClocksEnabled);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &enabled, sizeof(enabled));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetDefaultAutoBoostedClocksEnabled called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
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
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device, unsigned int fan) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetDefaultFanSpeed_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetDefaultFanSpeed_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &fan, sizeof(fan));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetFanControlPolicy(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t policy) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetFanControlPolicy called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetFanControlPolicy);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &fan, sizeof(fan));
    rpc_write(client, &policy, sizeof(policy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetTemperatureThreshold called" << std::endl;
#endif
    void *_0temp = mem2server((void *)temp, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetTemperatureThreshold);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &thresholdType, sizeof(thresholdType));
    rpc_write(client, &_0temp, sizeof(_0temp));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)temp, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int limit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetPowerManagementLimit called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetPowerManagementLimit);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &limit, sizeof(limit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetGpuOperationMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetGpuOperationMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetAPIRestriction called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
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
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int speed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetFanSpeed_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetFanSpeed_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &fan, sizeof(fan));
    rpc_write(client, &speed, sizeof(speed));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetGpcClkVfOffset(nvmlDevice_t device, int offset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetGpcClkVfOffset called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetGpcClkVfOffset);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &offset, sizeof(offset));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetMemClkVfOffset(nvmlDevice_t device, int offset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetMemClkVfOffset called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetMemClkVfOffset);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &offset, sizeof(offset));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetAccountingMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetAccountingMode);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceClearAccountingPids called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceClearAccountingPids);
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

extern "C" nvmlReturn_t nvmlDeviceSetPowerManagementLimit_v2(nvmlDevice_t device, nvmlPowerValue_v2_t *powerValue) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetPowerManagementLimit_v2 called" << std::endl;
#endif
    void *_0powerValue = mem2server((void *)powerValue, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetPowerManagementLimit_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0powerValue, sizeof(_0powerValue));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)powerValue, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkState called" << std::endl;
#endif
    void *_0isActive = mem2server((void *)isActive, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkState);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &_0isActive, sizeof(_0isActive));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)isActive, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int *version) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkVersion called" << std::endl;
#endif
    void *_0version = mem2server((void *)version, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkVersion);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
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

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int *capResult) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkCapability called" << std::endl;
#endif
    void *_0capResult = mem2server((void *)capResult, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkCapability);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &capability, sizeof(capability));
    rpc_write(client, &_0capResult, sizeof(_0capResult));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)capResult, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkRemotePciInfo_v2 called" << std::endl;
#endif
    void *_0pci = mem2server((void *)pci, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &_0pci, sizeof(_0pci));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pci, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long *counterValue) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkErrorCounter called" << std::endl;
#endif
    void *_0counterValue = mem2server((void *)counterValue, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkErrorCounter);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &counter, sizeof(counter));
    rpc_write(client, &_0counterValue, sizeof(_0counterValue));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)counterValue, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetNvLinkErrorCounters called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceResetNvLinkErrorCounters);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control, unsigned int reset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetNvLinkUtilizationControl called" << std::endl;
#endif
    void *_0control = mem2server((void *)control, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetNvLinkUtilizationControl);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &counter, sizeof(counter));
    rpc_write(client, &_0control, sizeof(_0control));
    rpc_write(client, &reset, sizeof(reset));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)control, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkUtilizationControl called" << std::endl;
#endif
    void *_0control = mem2server((void *)control, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkUtilizationControl);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &counter, sizeof(counter));
    rpc_write(client, &_0control, sizeof(_0control));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)control, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, unsigned long long *rxcounter, unsigned long long *txcounter) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkUtilizationCounter called" << std::endl;
#endif
    void *_0rxcounter = mem2server((void *)rxcounter, 0);
    void *_0txcounter = mem2server((void *)txcounter, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkUtilizationCounter);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &counter, sizeof(counter));
    rpc_write(client, &_0rxcounter, sizeof(_0rxcounter));
    rpc_write(client, &_0txcounter, sizeof(_0txcounter));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)rxcounter, 0);
    mem2client((void *)txcounter, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceFreezeNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlEnableState_t freeze) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceFreezeNvLinkUtilizationCounter called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
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
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceResetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetNvLinkUtilizationCounter called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
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
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkRemoteDeviceType called" << std::endl;
#endif
    void *_0pNvLinkDeviceType = mem2server((void *)pNvLinkDeviceType, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkRemoteDeviceType);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_write(client, &_0pNvLinkDeviceType, sizeof(_0pNvLinkDeviceType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pNvLinkDeviceType, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetNvLinkDeviceLowPowerThreshold called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetNvLinkDeviceLowPowerThreshold);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemSetNvlinkBwMode(unsigned int nvlinkBwMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemSetNvlinkBwMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemSetNvlinkBwMode);
    rpc_write(client, &nvlinkBwMode, sizeof(nvlinkBwMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetNvlinkBwMode(unsigned int *nvlinkBwMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetNvlinkBwMode called" << std::endl;
#endif
    void *_0nvlinkBwMode = mem2server((void *)nvlinkBwMode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetNvlinkBwMode);
    rpc_write(client, &_0nvlinkBwMode, sizeof(_0nvlinkBwMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)nvlinkBwMode, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvlinkSupportedBwModes(nvmlDevice_t device, nvmlNvlinkSupportedBwModes_t *supportedBwMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvlinkSupportedBwModes called" << std::endl;
#endif
    void *_0supportedBwMode = mem2server((void *)supportedBwMode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvlinkSupportedBwModes);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0supportedBwMode, sizeof(_0supportedBwMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)supportedBwMode, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkGetBwMode_t *getBwMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvlinkBwMode called" << std::endl;
#endif
    void *_0getBwMode = mem2server((void *)getBwMode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvlinkBwMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0getBwMode, sizeof(_0getBwMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)getBwMode, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkSetBwMode_t *setBwMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetNvlinkBwMode called" << std::endl;
#endif
    void *_0setBwMode = mem2server((void *)setBwMode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetNvlinkBwMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0setBwMode, sizeof(_0setBwMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)setBwMode, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t *set) {
#ifdef DEBUG
    std::cout << "Hook: nvmlEventSetCreate called" << std::endl;
#endif
    void *_0set = mem2server((void *)set, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlEventSetCreate);
    rpc_write(client, &_0set, sizeof(_0set));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)set, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceRegisterEvents called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
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
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long *eventTypes) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedEventTypes called" << std::endl;
#endif
    void *_0eventTypes = mem2server((void *)eventTypes, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedEventTypes);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0eventTypes, sizeof(_0eventTypes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)eventTypes, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t *data, unsigned int timeoutms) {
#ifdef DEBUG
    std::cout << "Hook: nvmlEventSetWait_v2 called" << std::endl;
#endif
    void *_0data = mem2server((void *)data, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlEventSetWait_v2);
    rpc_write(client, &set, sizeof(set));
    rpc_write(client, &_0data, sizeof(_0data));
    rpc_write(client, &timeoutms, sizeof(timeoutms));
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

extern "C" nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set) {
#ifdef DEBUG
    std::cout << "Hook: nvmlEventSetFree called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlEventSetFree);
    rpc_write(client, &set, sizeof(set));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t *pciInfo, nvmlEnableState_t newState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceModifyDrainState called" << std::endl;
#endif
    void *_0pciInfo = mem2server((void *)pciInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceModifyDrainState);
    rpc_write(client, &_0pciInfo, sizeof(_0pciInfo));
    rpc_write(client, &newState, sizeof(newState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pciInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t *pciInfo, nvmlEnableState_t *currentState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceQueryDrainState called" << std::endl;
#endif
    void *_0pciInfo = mem2server((void *)pciInfo, 0);
    void *_0currentState = mem2server((void *)currentState, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceQueryDrainState);
    rpc_write(client, &_0pciInfo, sizeof(_0pciInfo));
    rpc_write(client, &_0currentState, sizeof(_0currentState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pciInfo, 0);
    mem2client((void *)currentState, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t *pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceRemoveGpu_v2 called" << std::endl;
#endif
    void *_0pciInfo = mem2server((void *)pciInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceRemoveGpu_v2);
    rpc_write(client, &_0pciInfo, sizeof(_0pciInfo));
    rpc_write(client, &gpuState, sizeof(gpuState));
    rpc_write(client, &linkState, sizeof(linkState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pciInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t *pciInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceDiscoverGpus called" << std::endl;
#endif
    void *_0pciInfo = mem2server((void *)pciInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceDiscoverGpus);
    rpc_write(client, &_0pciInfo, sizeof(_0pciInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pciInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFieldValues called" << std::endl;
#endif
    void *_0values = mem2server((void *)values, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetFieldValues);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &valuesCount, sizeof(valuesCount));
    rpc_write(client, &_0values, sizeof(_0values));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)values, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceClearFieldValues called" << std::endl;
#endif
    void *_0values = mem2server((void *)values, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceClearFieldValues);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &valuesCount, sizeof(valuesCount));
    rpc_write(client, &_0values, sizeof(_0values));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)values, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t *pVirtualMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVirtualizationMode called" << std::endl;
#endif
    void *_0pVirtualMode = mem2server((void *)pVirtualMode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVirtualizationMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pVirtualMode, sizeof(_0pVirtualMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pVirtualMode, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t *pHostVgpuMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHostVgpuMode called" << std::endl;
#endif
    void *_0pHostVgpuMode = mem2server((void *)pHostVgpuMode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetHostVgpuMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pHostVgpuMode, sizeof(_0pHostVgpuMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pHostVgpuMode, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetVirtualizationMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetVirtualizationMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &virtualMode, sizeof(virtualMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuHeterogeneousMode(nvmlDevice_t device, nvmlVgpuHeterogeneousMode_t *pHeterogeneousMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuHeterogeneousMode called" << std::endl;
#endif
    void *_0pHeterogeneousMode = mem2server((void *)pHeterogeneousMode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuHeterogeneousMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pHeterogeneousMode, sizeof(_0pHeterogeneousMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pHeterogeneousMode, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetVgpuHeterogeneousMode(nvmlDevice_t device, const nvmlVgpuHeterogeneousMode_t *pHeterogeneousMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetVgpuHeterogeneousMode called" << std::endl;
#endif
    void *_0pHeterogeneousMode = mem2server((void *)pHeterogeneousMode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetVgpuHeterogeneousMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pHeterogeneousMode, sizeof(_0pHeterogeneousMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pHeterogeneousMode, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetPlacementId(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuPlacementId_t *pPlacement) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetPlacementId called" << std::endl;
#endif
    void *_0pPlacement = mem2server((void *)pPlacement, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetPlacementId);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0pPlacement, sizeof(_0pPlacement));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pPlacement, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuTypeSupportedPlacements(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuPlacementList_t *pPlacementList) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuTypeSupportedPlacements called" << std::endl;
#endif
    void *_0pPlacementList = mem2server((void *)pPlacementList, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuTypeSupportedPlacements);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0pPlacementList, sizeof(_0pPlacementList));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pPlacementList, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuTypeCreatablePlacements(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuPlacementList_t *pPlacementList) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuTypeCreatablePlacements called" << std::endl;
#endif
    void *_0pPlacementList = mem2server((void *)pPlacementList, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuTypeCreatablePlacements);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0pPlacementList, sizeof(_0pPlacementList));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pPlacementList, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetGspHeapSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *gspHeapSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetGspHeapSize called" << std::endl;
#endif
    void *_0gspHeapSize = mem2server((void *)gspHeapSize, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetGspHeapSize);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0gspHeapSize, sizeof(_0gspHeapSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)gspHeapSize, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetFbReservation(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *fbReservation) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetFbReservation called" << std::endl;
#endif
    void *_0fbReservation = mem2server((void *)fbReservation, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetFbReservation);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0fbReservation, sizeof(_0fbReservation));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)fbReservation, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetRuntimeStateSize(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuRuntimeState_t *pState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetRuntimeStateSize called" << std::endl;
#endif
    void *_0pState = mem2server((void *)pState, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetRuntimeStateSize);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0pState, sizeof(_0pState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pState, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, nvmlEnableState_t state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetVgpuCapabilities called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetVgpuCapabilities);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &capability, sizeof(capability));
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

extern "C" nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGridLicensableFeatures_v4 called" << std::endl;
#endif
    void *_0pGridLicensableFeatures = mem2server((void *)pGridLicensableFeatures, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGridLicensableFeatures_v4);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pGridLicensableFeatures, sizeof(_0pGridLicensableFeatures));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pGridLicensableFeatures, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t capability, unsigned int *capResult) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetVgpuDriverCapabilities called" << std::endl;
#endif
    void *_0capResult = mem2server((void *)capResult, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGetVgpuDriverCapabilities);
    rpc_write(client, &capability, sizeof(capability));
    rpc_write(client, &_0capResult, sizeof(_0capResult));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)capResult, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, unsigned int *capResult) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuCapabilities called" << std::endl;
#endif
    void *_0capResult = mem2server((void *)capResult, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuCapabilities);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &capability, sizeof(capability));
    rpc_write(client, &_0capResult, sizeof(_0capResult));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)capResult, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedVgpus called" << std::endl;
#endif
    void *_0vgpuCount = mem2server((void *)vgpuCount, 0);
    void *_0vgpuTypeIds = mem2server((void *)vgpuTypeIds, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedVgpus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0vgpuCount, sizeof(_0vgpuCount));
    rpc_write(client, &_0vgpuTypeIds, sizeof(_0vgpuTypeIds));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)vgpuCount, 0);
    mem2client((void *)vgpuTypeIds, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCreatableVgpus called" << std::endl;
#endif
    void *_0vgpuCount = mem2server((void *)vgpuCount, 0);
    void *_0vgpuTypeIds = mem2server((void *)vgpuTypeIds, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCreatableVgpus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0vgpuCount, sizeof(_0vgpuCount));
    rpc_write(client, &_0vgpuTypeIds, sizeof(_0vgpuTypeIds));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)vgpuCount, 0);
    mem2client((void *)vgpuTypeIds, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeClass, unsigned int *size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetClass called" << std::endl;
#endif
    void *_0size = mem2server((void *)size, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetClass);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_read(client, vgpuTypeClass, *size, true);
    rpc_write(client, &_0size, sizeof(_0size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)size, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeName, unsigned int *size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetName called" << std::endl;
#endif
    void *_0size = mem2server((void *)size, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetName);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_read(client, vgpuTypeName, *size, true);
    rpc_write(client, &_0size, sizeof(_0size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)size, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *gpuInstanceProfileId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetGpuInstanceProfileId called" << std::endl;
#endif
    void *_0gpuInstanceProfileId = mem2server((void *)gpuInstanceProfileId, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetGpuInstanceProfileId);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0gpuInstanceProfileId, sizeof(_0gpuInstanceProfileId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)gpuInstanceProfileId, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *deviceID, unsigned long long *subsystemID) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetDeviceID called" << std::endl;
#endif
    void *_0deviceID = mem2server((void *)deviceID, 0);
    void *_0subsystemID = mem2server((void *)subsystemID, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetDeviceID);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0deviceID, sizeof(_0deviceID));
    rpc_write(client, &_0subsystemID, sizeof(_0subsystemID));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)deviceID, 0);
    mem2client((void *)subsystemID, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *fbSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetFramebufferSize called" << std::endl;
#endif
    void *_0fbSize = mem2server((void *)fbSize, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetFramebufferSize);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0fbSize, sizeof(_0fbSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)fbSize, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *numDisplayHeads) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetNumDisplayHeads called" << std::endl;
#endif
    void *_0numDisplayHeads = mem2server((void *)numDisplayHeads, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetNumDisplayHeads);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0numDisplayHeads, sizeof(_0numDisplayHeads));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)numDisplayHeads, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int *xdim, unsigned int *ydim) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetResolution called" << std::endl;
#endif
    void *_0xdim = mem2server((void *)xdim, 0);
    void *_0ydim = mem2server((void *)ydim, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetResolution);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &displayIndex, sizeof(displayIndex));
    rpc_write(client, &_0xdim, sizeof(_0xdim));
    rpc_write(client, &_0ydim, sizeof(_0ydim));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)xdim, 0);
    mem2client((void *)ydim, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeLicenseString, unsigned int size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetLicense called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetLicense);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_read(client, vgpuTypeLicenseString, size, true);
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

extern "C" nvmlReturn_t nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *frameRateLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetFrameRateLimit called" << std::endl;
#endif
    void *_0frameRateLimit = mem2server((void *)frameRateLimit, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetFrameRateLimit);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0frameRateLimit, sizeof(_0frameRateLimit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)frameRateLimit, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetMaxInstances called" << std::endl;
#endif
    void *_0vgpuInstanceCount = mem2server((void *)vgpuInstanceCount, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetMaxInstances);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0vgpuInstanceCount, sizeof(_0vgpuInstanceCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)vgpuInstanceCount, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCountPerVm) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetMaxInstancesPerVm called" << std::endl;
#endif
    void *_0vgpuInstanceCountPerVm = mem2server((void *)vgpuInstanceCountPerVm, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetMaxInstancesPerVm);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0vgpuInstanceCountPerVm, sizeof(_0vgpuInstanceCountPerVm));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)vgpuInstanceCountPerVm, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetBAR1Info(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuTypeBar1Info_t *bar1Info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetBAR1Info called" << std::endl;
#endif
    void *_0bar1Info = mem2server((void *)bar1Info, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetBAR1Info);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_0bar1Info, sizeof(_0bar1Info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)bar1Info, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuInstance_t *vgpuInstances) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetActiveVgpus called" << std::endl;
#endif
    void *_0vgpuCount = mem2server((void *)vgpuCount, 0);
    void *_0vgpuInstances = mem2server((void *)vgpuInstances, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetActiveVgpus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0vgpuCount, sizeof(_0vgpuCount));
    rpc_write(client, &_0vgpuInstances, sizeof(_0vgpuInstances));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)vgpuCount, 0);
    mem2client((void *)vgpuInstances, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char *vmId, unsigned int size, nvmlVgpuVmIdType_t *vmIdType) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetVmID called" << std::endl;
#endif
    void *_0vmIdType = mem2server((void *)vmIdType, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetVmID);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, vmId, size, true);
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &_0vmIdType, sizeof(_0vmIdType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)vmIdType, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char *uuid, unsigned int size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetUUID called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetUUID);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, uuid, size, true);
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

extern "C" nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetVmDriverVersion called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetVmDriverVersion);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, version, length, true);
    rpc_write(client, &length, sizeof(length));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long *fbUsage) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFbUsage called" << std::endl;
#endif
    void *_0fbUsage = mem2server((void *)fbUsage, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetFbUsage);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0fbUsage, sizeof(_0fbUsage));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)fbUsage, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int *licensed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetLicenseStatus called" << std::endl;
#endif
    void *_0licensed = mem2server((void *)licensed, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetLicenseStatus);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0licensed, sizeof(_0licensed));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)licensed, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t *vgpuTypeId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetType called" << std::endl;
#endif
    void *_0vgpuTypeId = mem2server((void *)vgpuTypeId, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetType);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0vgpuTypeId, sizeof(_0vgpuTypeId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)vgpuTypeId, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int *frameRateLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFrameRateLimit called" << std::endl;
#endif
    void *_0frameRateLimit = mem2server((void *)frameRateLimit, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetFrameRateLimit);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0frameRateLimit, sizeof(_0frameRateLimit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)frameRateLimit, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *eccMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEccMode called" << std::endl;
#endif
    void *_0eccMode = mem2server((void *)eccMode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetEccMode);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0eccMode, sizeof(_0eccMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)eccMode, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int *encoderCapacity) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEncoderCapacity called" << std::endl;
#endif
    void *_0encoderCapacity = mem2server((void *)encoderCapacity, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetEncoderCapacity);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0encoderCapacity, sizeof(_0encoderCapacity));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)encoderCapacity, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceSetEncoderCapacity called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceSetEncoderCapacity);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &encoderCapacity, sizeof(encoderCapacity));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEncoderStats called" << std::endl;
#endif
    void *_0sessionCount = mem2server((void *)sessionCount, 0);
    void *_0averageFps = mem2server((void *)averageFps, 0);
    void *_0averageLatency = mem2server((void *)averageLatency, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetEncoderStats);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0sessionCount, sizeof(_0sessionCount));
    rpc_write(client, &_0averageFps, sizeof(_0averageFps));
    rpc_write(client, &_0averageLatency, sizeof(_0averageLatency));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)sessionCount, 0);
    mem2client((void *)averageFps, 0);
    mem2client((void *)averageLatency, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEncoderSessions called" << std::endl;
#endif
    void *_0sessionCount = mem2server((void *)sessionCount, 0);
    void *_0sessionInfo = mem2server((void *)sessionInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetEncoderSessions);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0sessionCount, sizeof(_0sessionCount));
    rpc_write(client, &_0sessionInfo, sizeof(_0sessionInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)sessionCount, 0);
    mem2client((void *)sessionInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t *fbcStats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFBCStats called" << std::endl;
#endif
    void *_0fbcStats = mem2server((void *)fbcStats, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetFBCStats);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0fbcStats, sizeof(_0fbcStats));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)fbcStats, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFBCSessions called" << std::endl;
#endif
    void *_0sessionCount = mem2server((void *)sessionCount, 0);
    void *_0sessionInfo = mem2server((void *)sessionInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetFBCSessions);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0sessionCount, sizeof(_0sessionCount));
    rpc_write(client, &_0sessionInfo, sizeof(_0sessionInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)sessionCount, 0);
    mem2client((void *)sessionInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int *gpuInstanceId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetGpuInstanceId called" << std::endl;
#endif
    void *_0gpuInstanceId = mem2server((void *)gpuInstanceId, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetGpuInstanceId);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0gpuInstanceId, sizeof(_0gpuInstanceId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)gpuInstanceId, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char *vgpuPciId, unsigned int *length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetGpuPciId called" << std::endl;
#endif
    void *_0length = mem2server((void *)length, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetGpuPciId);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, vgpuPciId, *length, true);
    rpc_write(client, &_0length, sizeof(_0length));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)length, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetCapabilities(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int *capResult) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetCapabilities called" << std::endl;
#endif
    void *_0capResult = mem2server((void *)capResult, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetCapabilities);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &capability, sizeof(capability));
    rpc_write(client, &_0capResult, sizeof(_0capResult));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)capResult, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char *mdevUuid, unsigned int size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetMdevUUID called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetMdevUUID);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, mdevUuid, size, true);
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

extern "C" nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t *vgpuMetadata, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetMetadata called" << std::endl;
#endif
    void *_0vgpuMetadata = mem2server((void *)vgpuMetadata, 0);
    void *_0bufferSize = mem2server((void *)bufferSize, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetMetadata);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0vgpuMetadata, sizeof(_0vgpuMetadata));
    rpc_write(client, &_0bufferSize, sizeof(_0bufferSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)vgpuMetadata, 0);
    mem2client((void *)bufferSize, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t *pgpuMetadata, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuMetadata called" << std::endl;
#endif
    void *_0pgpuMetadata = mem2server((void *)pgpuMetadata, 0);
    void *_0bufferSize = mem2server((void *)bufferSize, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuMetadata);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pgpuMetadata, sizeof(_0pgpuMetadata));
    rpc_write(client, &_0bufferSize, sizeof(_0bufferSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pgpuMetadata, 0);
    mem2client((void *)bufferSize, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t *vgpuMetadata, nvmlVgpuPgpuMetadata_t *pgpuMetadata, nvmlVgpuPgpuCompatibility_t *compatibilityInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetVgpuCompatibility called" << std::endl;
#endif
    void *_0vgpuMetadata = mem2server((void *)vgpuMetadata, 0);
    void *_0pgpuMetadata = mem2server((void *)pgpuMetadata, 0);
    void *_0compatibilityInfo = mem2server((void *)compatibilityInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGetVgpuCompatibility);
    rpc_write(client, &_0vgpuMetadata, sizeof(_0vgpuMetadata));
    rpc_write(client, &_0pgpuMetadata, sizeof(_0pgpuMetadata));
    rpc_write(client, &_0compatibilityInfo, sizeof(_0compatibilityInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)vgpuMetadata, 0);
    mem2client((void *)pgpuMetadata, 0);
    mem2client((void *)compatibilityInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char *pgpuMetadata, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPgpuMetadataString called" << std::endl;
#endif
    void *_0bufferSize = mem2server((void *)bufferSize, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPgpuMetadataString);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, pgpuMetadata, *bufferSize, true);
    rpc_write(client, &_0bufferSize, sizeof(_0bufferSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)bufferSize, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuSchedulerLog(nvmlDevice_t device, nvmlVgpuSchedulerLog_t *pSchedulerLog) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuSchedulerLog called" << std::endl;
#endif
    void *_0pSchedulerLog = mem2server((void *)pSchedulerLog, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuSchedulerLog);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pSchedulerLog, sizeof(_0pSchedulerLog));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pSchedulerLog, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerGetState_t *pSchedulerState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuSchedulerState called" << std::endl;
#endif
    void *_0pSchedulerState = mem2server((void *)pSchedulerState, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuSchedulerState);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pSchedulerState, sizeof(_0pSchedulerState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pSchedulerState, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuSchedulerCapabilities(nvmlDevice_t device, nvmlVgpuSchedulerCapabilities_t *pCapabilities) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuSchedulerCapabilities called" << std::endl;
#endif
    void *_0pCapabilities = mem2server((void *)pCapabilities, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuSchedulerCapabilities);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pCapabilities, sizeof(_0pCapabilities));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pCapabilities, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerSetState_t *pSchedulerState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetVgpuSchedulerState called" << std::endl;
#endif
    void *_0pSchedulerState = mem2server((void *)pSchedulerState, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetVgpuSchedulerState);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0pSchedulerState, sizeof(_0pSchedulerState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)pSchedulerState, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t *supported, nvmlVgpuVersion_t *current) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetVgpuVersion called" << std::endl;
#endif
    void *_0supported = mem2server((void *)supported, 0);
    void *_0current = mem2server((void *)current, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGetVgpuVersion);
    rpc_write(client, &_0supported, sizeof(_0supported));
    rpc_write(client, &_0current, sizeof(_0current));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)supported, 0);
    mem2client((void *)current, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t *vgpuVersion) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSetVgpuVersion called" << std::endl;
#endif
    void *_0vgpuVersion = mem2server((void *)vgpuVersion, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSetVgpuVersion);
    rpc_write(client, &_0vgpuVersion, sizeof(_0vgpuVersion));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)vgpuVersion, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t *utilizationSamples) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuUtilization called" << std::endl;
#endif
    void *_0sampleValType = mem2server((void *)sampleValType, 0);
    void *_0vgpuInstanceSamplesCount = mem2server((void *)vgpuInstanceSamplesCount, 0);
    void *_0utilizationSamples = mem2server((void *)utilizationSamples, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    rpc_write(client, &_0sampleValType, sizeof(_0sampleValType));
    rpc_write(client, &_0vgpuInstanceSamplesCount, sizeof(_0vgpuInstanceSamplesCount));
    rpc_write(client, &_0utilizationSamples, sizeof(_0utilizationSamples));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)sampleValType, 0);
    mem2client((void *)vgpuInstanceSamplesCount, 0);
    mem2client((void *)utilizationSamples, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuInstancesUtilizationInfo(nvmlDevice_t device, nvmlVgpuInstancesUtilizationInfo_t *vgpuUtilInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuInstancesUtilizationInfo called" << std::endl;
#endif
    void *_0vgpuUtilInfo = mem2server((void *)vgpuUtilInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuInstancesUtilizationInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0vgpuUtilInfo, sizeof(_0vgpuUtilInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)vgpuUtilInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int *vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t *utilizationSamples) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuProcessUtilization called" << std::endl;
#endif
    void *_0vgpuProcessSamplesCount = mem2server((void *)vgpuProcessSamplesCount, 0);
    void *_0utilizationSamples = mem2server((void *)utilizationSamples, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuProcessUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    rpc_write(client, &_0vgpuProcessSamplesCount, sizeof(_0vgpuProcessSamplesCount));
    rpc_write(client, &_0utilizationSamples, sizeof(_0utilizationSamples));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)vgpuProcessSamplesCount, 0);
    mem2client((void *)utilizationSamples, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuProcessesUtilizationInfo(nvmlDevice_t device, nvmlVgpuProcessesUtilizationInfo_t *vgpuProcUtilInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuProcessesUtilizationInfo called" << std::endl;
#endif
    void *_0vgpuProcUtilInfo = mem2server((void *)vgpuProcUtilInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuProcessesUtilizationInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0vgpuProcUtilInfo, sizeof(_0vgpuProcUtilInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)vgpuProcUtilInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetAccountingMode called" << std::endl;
#endif
    void *_0mode = mem2server((void *)mode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetAccountingMode);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
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

extern "C" nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int *count, unsigned int *pids) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetAccountingPids called" << std::endl;
#endif
    void *_0count = mem2server((void *)count, 0);
    void *_0pids = mem2server((void *)pids, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetAccountingPids);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_write(client, &_0pids, sizeof(_0pids));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)count, 0);
    mem2client((void *)pids, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t *stats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetAccountingStats called" << std::endl;
#endif
    void *_0stats = mem2server((void *)stats, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetAccountingStats);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &pid, sizeof(pid));
    rpc_write(client, &_0stats, sizeof(_0stats));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)stats, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceClearAccountingPids called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceClearAccountingPids);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t *licenseInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetLicenseInfo_v2 called" << std::endl;
#endif
    void *_0licenseInfo = mem2server((void *)licenseInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetLicenseInfo_v2);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &_0licenseInfo, sizeof(_0licenseInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)licenseInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int *deviceCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetExcludedDeviceCount called" << std::endl;
#endif
    void *_0deviceCount = mem2server((void *)deviceCount, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGetExcludedDeviceCount);
    rpc_write(client, &_0deviceCount, sizeof(_0deviceCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)deviceCount, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetExcludedDeviceInfoByIndex called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGetExcludedDeviceInfoByIndex);
    rpc_write(client, &index, sizeof(index));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t *activationStatus) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetMigMode called" << std::endl;
#endif
    void *_0activationStatus = mem2server((void *)activationStatus, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetMigMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &_0activationStatus, sizeof(_0activationStatus));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)activationStatus, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *currentMode, unsigned int *pendingMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMigMode called" << std::endl;
#endif
    void *_0currentMode = mem2server((void *)currentMode, 0);
    void *_0pendingMode = mem2server((void *)pendingMode, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMigMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0currentMode, sizeof(_0currentMode));
    rpc_write(client, &_0pendingMode, sizeof(_0pendingMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)currentMode, 0);
    mem2client((void *)pendingMode, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceProfileInfo called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstanceProfileInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profile, sizeof(profile));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceProfileInfoV called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstanceProfileInfoV);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profile, sizeof(profile));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t *placements, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstancePossiblePlacements_v2 called" << std::endl;
#endif
    void *_0placements = mem2server((void *)placements, 0);
    void *_0count = mem2server((void *)count, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstancePossiblePlacements_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0placements, sizeof(_0placements));
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)placements, 0);
    mem2client((void *)count, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceRemainingCapacity called" << std::endl;
#endif
    void *_0count = mem2server((void *)count, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstanceRemainingCapacity);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
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

extern "C" nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceCreateGpuInstance called" << std::endl;
#endif
    void *_0gpuInstance = mem2server((void *)gpuInstance, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceCreateGpuInstance);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0gpuInstance, sizeof(_0gpuInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)gpuInstance, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t *placement, nvmlGpuInstance_t *gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceCreateGpuInstanceWithPlacement called" << std::endl;
#endif
    void *_0placement = mem2server((void *)placement, 0);
    void *_0gpuInstance = mem2server((void *)gpuInstance, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceCreateGpuInstanceWithPlacement);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0placement, sizeof(_0placement));
    rpc_write(client, &_0gpuInstance, sizeof(_0gpuInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)placement, 0);
    mem2client((void *)gpuInstance, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceDestroy called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceDestroy);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstances, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstances called" << std::endl;
#endif
    void *_0gpuInstances = mem2server((void *)gpuInstances, 0);
    void *_0count = mem2server((void *)count, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstances);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0gpuInstances, sizeof(_0gpuInstances));
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)gpuInstances, 0);
    mem2client((void *)count, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t *gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceById called" << std::endl;
#endif
    void *_0gpuInstance = mem2server((void *)gpuInstance, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstanceById);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &id, sizeof(id));
    rpc_write(client, &_0gpuInstance, sizeof(_0gpuInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)gpuInstance, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetInfo called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetInfo);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceProfileInfo called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetComputeInstanceProfileInfo);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profile, sizeof(profile));
    rpc_write(client, &engProfile, sizeof(engProfile));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceProfileInfoV called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetComputeInstanceProfileInfoV);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profile, sizeof(profile));
    rpc_write(client, &engProfile, sizeof(engProfile));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceRemainingCapacity called" << std::endl;
#endif
    void *_0count = mem2server((void *)count, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetComputeInstanceRemainingCapacity);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profileId, sizeof(profileId));
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

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstancePlacement_t *placements, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstancePossiblePlacements called" << std::endl;
#endif
    void *_0placements = mem2server((void *)placements, 0);
    void *_0count = mem2server((void *)count, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetComputeInstancePossiblePlacements);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0placements, sizeof(_0placements));
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)placements, 0);
    mem2client((void *)count, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceCreateComputeInstance called" << std::endl;
#endif
    void *_0computeInstance = mem2server((void *)computeInstance, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceCreateComputeInstance);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0computeInstance, sizeof(_0computeInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)computeInstance, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceCreateComputeInstanceWithPlacement(nvmlGpuInstance_t gpuInstance, unsigned int profileId, const nvmlComputeInstancePlacement_t *placement, nvmlComputeInstance_t *computeInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceCreateComputeInstanceWithPlacement called" << std::endl;
#endif
    void *_0placement = mem2server((void *)placement, 0);
    void *_0computeInstance = mem2server((void *)computeInstance, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceCreateComputeInstanceWithPlacement);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0placement, sizeof(_0placement));
    rpc_write(client, &_0computeInstance, sizeof(_0computeInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)placement, 0);
    mem2client((void *)computeInstance, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlComputeInstanceDestroy called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlComputeInstanceDestroy);
    rpc_write(client, &computeInstance, sizeof(computeInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstances, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstances called" << std::endl;
#endif
    void *_0computeInstances = mem2server((void *)computeInstances, 0);
    void *_0count = mem2server((void *)count, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetComputeInstances);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, &_0computeInstances, sizeof(_0computeInstances));
    rpc_write(client, &_0count, sizeof(_0count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)computeInstances, 0);
    mem2client((void *)count, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t *computeInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceById called" << std::endl;
#endif
    void *_0computeInstance = mem2server((void *)computeInstance, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetComputeInstanceById);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &id, sizeof(id));
    rpc_write(client, &_0computeInstance, sizeof(_0computeInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)computeInstance, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlComputeInstanceGetInfo_v2 called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlComputeInstanceGetInfo_v2);
    rpc_write(client, &computeInstance, sizeof(computeInstance));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int *isMigDevice) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceIsMigDeviceHandle called" << std::endl;
#endif
    void *_0isMigDevice = mem2server((void *)isMigDevice, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceIsMigDeviceHandle);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0isMigDevice, sizeof(_0isMigDevice));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)isMigDevice, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int *id) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceId called" << std::endl;
#endif
    void *_0id = mem2server((void *)id, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstanceId);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0id, sizeof(_0id));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)id, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int *id) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetComputeInstanceId called" << std::endl;
#endif
    void *_0id = mem2server((void *)id, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetComputeInstanceId);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0id, sizeof(_0id));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)id, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxMigDeviceCount called" << std::endl;
#endif
    void *_0count = mem2server((void *)count, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxMigDeviceCount);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t *migDevice) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMigDeviceHandleByIndex called" << std::endl;
#endif
    void *_0migDevice = mem2server((void *)migDevice, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMigDeviceHandleByIndex);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &index, sizeof(index));
    rpc_write(client, &_0migDevice, sizeof(_0migDevice));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)migDevice, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDeviceHandleFromMigDeviceHandle called" << std::endl;
#endif
    void *_0device = mem2server((void *)device, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDeviceHandleFromMigDeviceHandle);
    rpc_write(client, &migDevice, sizeof(migDevice));
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

extern "C" nvmlReturn_t nvmlGpmMetricsGet(nvmlGpmMetricsGet_t *metricsGet) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmMetricsGet called" << std::endl;
#endif
    void *_0metricsGet = mem2server((void *)metricsGet, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpmMetricsGet);
    rpc_write(client, &_0metricsGet, sizeof(_0metricsGet));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)metricsGet, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpmSampleFree(nvmlGpmSample_t gpmSample) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmSampleFree called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpmSampleFree);
    rpc_write(client, &gpmSample, sizeof(gpmSample));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpmSampleAlloc(nvmlGpmSample_t *gpmSample) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmSampleAlloc called" << std::endl;
#endif
    void *_0gpmSample = mem2server((void *)gpmSample, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpmSampleAlloc);
    rpc_write(client, &_0gpmSample, sizeof(_0gpmSample));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)gpmSample, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpmSampleGet(nvmlDevice_t device, nvmlGpmSample_t gpmSample) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmSampleGet called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpmSampleGet);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &gpmSample, sizeof(gpmSample));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpmMigSampleGet(nvmlDevice_t device, unsigned int gpuInstanceId, nvmlGpmSample_t gpmSample) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmMigSampleGet called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpmMigSampleGet);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &gpuInstanceId, sizeof(gpuInstanceId));
    rpc_write(client, &gpmSample, sizeof(gpmSample));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpmQueryDeviceSupport(nvmlDevice_t device, nvmlGpmSupport_t *gpmSupport) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmQueryDeviceSupport called" << std::endl;
#endif
    void *_0gpmSupport = mem2server((void *)gpmSupport, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpmQueryDeviceSupport);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0gpmSupport, sizeof(_0gpmSupport));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)gpmSupport, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpmQueryIfStreamingEnabled(nvmlDevice_t device, unsigned int *state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmQueryIfStreamingEnabled called" << std::endl;
#endif
    void *_0state = mem2server((void *)state, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpmQueryIfStreamingEnabled);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlGpmSetStreamingEnabled(nvmlDevice_t device, unsigned int state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmSetStreamingEnabled called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpmSetStreamingEnabled);
    rpc_write(client, &device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetCapabilities(nvmlDevice_t device, nvmlDeviceCapabilities_t *caps) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCapabilities called" << std::endl;
#endif
    void *_0caps = mem2server((void *)caps, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCapabilities);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0caps, sizeof(_0caps));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)caps, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceWorkloadPowerProfileGetProfilesInfo(nvmlDevice_t device, nvmlWorkloadPowerProfileProfilesInfo_t *profilesInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceWorkloadPowerProfileGetProfilesInfo called" << std::endl;
#endif
    void *_0profilesInfo = mem2server((void *)profilesInfo, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceWorkloadPowerProfileGetProfilesInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0profilesInfo, sizeof(_0profilesInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)profilesInfo, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceWorkloadPowerProfileGetCurrentProfiles(nvmlDevice_t device, nvmlWorkloadPowerProfileCurrentProfiles_t *currentProfiles) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceWorkloadPowerProfileGetCurrentProfiles called" << std::endl;
#endif
    void *_0currentProfiles = mem2server((void *)currentProfiles, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceWorkloadPowerProfileGetCurrentProfiles);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0currentProfiles, sizeof(_0currentProfiles));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)currentProfiles, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceWorkloadPowerProfileSetRequestedProfiles(nvmlDevice_t device, nvmlWorkloadPowerProfileRequestedProfiles_t *requestedProfiles) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceWorkloadPowerProfileSetRequestedProfiles called" << std::endl;
#endif
    void *_0requestedProfiles = mem2server((void *)requestedProfiles, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceWorkloadPowerProfileSetRequestedProfiles);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0requestedProfiles, sizeof(_0requestedProfiles));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)requestedProfiles, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceWorkloadPowerProfileClearRequestedProfiles(nvmlDevice_t device, nvmlWorkloadPowerProfileRequestedProfiles_t *requestedProfiles) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceWorkloadPowerProfileClearRequestedProfiles called" << std::endl;
#endif
    void *_0requestedProfiles = mem2server((void *)requestedProfiles, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceWorkloadPowerProfileClearRequestedProfiles);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0requestedProfiles, sizeof(_0requestedProfiles));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)requestedProfiles, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDevicePowerSmoothingActivatePresetProfile(nvmlDevice_t device, nvmlPowerSmoothingProfile_t *profile) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDevicePowerSmoothingActivatePresetProfile called" << std::endl;
#endif
    void *_0profile = mem2server((void *)profile, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDevicePowerSmoothingActivatePresetProfile);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0profile, sizeof(_0profile));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)profile, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDevicePowerSmoothingUpdatePresetProfileParam(nvmlDevice_t device, nvmlPowerSmoothingProfile_t *profile) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDevicePowerSmoothingUpdatePresetProfileParam called" << std::endl;
#endif
    void *_0profile = mem2server((void *)profile, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDevicePowerSmoothingUpdatePresetProfileParam);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_0profile, sizeof(_0profile));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)profile, 0);
    return _result;
}

extern "C" nvmlReturn_t nvmlDevicePowerSmoothingSetState(nvmlDevice_t device, nvmlPowerSmoothingState_t *state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDevicePowerSmoothingSetState called" << std::endl;
#endif
    void *_0state = mem2server((void *)state, 0);
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDevicePowerSmoothingSetState);
    rpc_write(client, &device, sizeof(device));
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

