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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetCudaDriverVersion);
    rpc_read(client, cudaDriverVersion, sizeof(*cudaDriverVersion));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetCudaDriverVersion_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetCudaDriverVersion_v2);
    rpc_read(client, cudaDriverVersion, sizeof(*cudaDriverVersion));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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

extern "C" nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetCount called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetCount);
    rpc_read(client, unitCount, sizeof(*unitCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t *unit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetHandleByIndex called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetHandleByIndex);
    rpc_write(client, &index, sizeof(index));
    rpc_read(client, unit, sizeof(*unit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetUnitInfo called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetUnitInfo);
    rpc_write(client, &unit, sizeof(unit));
    rpc_read(client, info, sizeof(*info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t *state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetLedState called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetLedState);
    rpc_write(client, &unit, sizeof(unit));
    rpc_read(client, state, sizeof(*state));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t *psu) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetPsuInfo called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetPsuInfo);
    rpc_write(client, &unit, sizeof(unit));
    rpc_read(client, psu, sizeof(*psu));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetTemperature called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetTemperature);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &type, sizeof(type));
    rpc_read(client, temp, sizeof(*temp));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t *fanSpeeds) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetFanSpeedInfo called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetFanSpeedInfo);
    rpc_write(client, &unit, sizeof(unit));
    rpc_read(client, fanSpeeds, sizeof(*fanSpeeds));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int *deviceCount, nvmlDevice_t *devices) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetDevices called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlUnitGetDevices);
    rpc_write(client, &unit, sizeof(unit));
    rpc_read(client, deviceCount, sizeof(*deviceCount));
    rpc_read(client, devices, sizeof(*devices));
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetHicVersion);
    rpc_read(client, hwbcCount, sizeof(*hwbcCount));
    rpc_read(client, hwbcEntries, sizeof(*hwbcEntries));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCount_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCount_v2);
    rpc_read(client, deviceCount, sizeof(*deviceCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t *attributes) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAttributes_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAttributes_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, attributes, sizeof(*attributes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleByIndex_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetHandleByIndex_v2);
    rpc_write(client, &index, sizeof(index));
    rpc_read(client, device, sizeof(*device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleBySerial(const char *serial, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleBySerial called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetHandleBySerial);
    rpc_write(client, serial, strlen(serial) + 1, true);
    rpc_read(client, device, sizeof(*device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleByUUID called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetHandleByUUID);
    rpc_write(client, uuid, strlen(uuid) + 1, true);
    rpc_read(client, device, sizeof(*device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleByPciBusId_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetHandleByPciBusId_v2);
    rpc_write(client, pciBusId, strlen(pciBusId) + 1, true);
    rpc_read(client, device, sizeof(*device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetBrand);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, type, sizeof(*type));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetIndex called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetIndex);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, index, sizeof(*index));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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

extern "C" nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long *nodeSet, nvmlAffinityScope_t scope) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryAffinity called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMemoryAffinity);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &nodeSetSize, sizeof(nodeSetSize));
    rpc_read(client, nodeSet, sizeof(*nodeSet));
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

extern "C" nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet, nvmlAffinityScope_t scope) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCpuAffinityWithinScope called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCpuAffinityWithinScope);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &cpuSetSize, sizeof(cpuSetSize));
    rpc_read(client, cpuSet, sizeof(*cpuSet));
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

extern "C" nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCpuAffinity called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCpuAffinity);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &cpuSetSize, sizeof(cpuSetSize));
    rpc_read(client, cpuSet, sizeof(*cpuSet));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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

extern "C" nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t *pathInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTopologyCommonAncestor called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetTopologyCommonAncestor);
    rpc_write(client, &device1, sizeof(device1));
    rpc_write(client, &device2, sizeof(device2));
    rpc_read(client, pathInfo, sizeof(*pathInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int *count, nvmlDevice_t *deviceArray) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTopologyNearestGpus called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetTopologyNearestGpus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &level, sizeof(level));
    rpc_read(client, count, sizeof(*count));
    rpc_read(client, deviceArray, sizeof(*deviceArray));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int *count, nvmlDevice_t *deviceArray) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetTopologyGpuSet called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSystemGetTopologyGpuSet);
    rpc_write(client, &cpuNumber, sizeof(cpuNumber));
    rpc_read(client, count, sizeof(*count));
    rpc_read(client, deviceArray, sizeof(*deviceArray));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t *p2pStatus) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetP2PStatus called" << std::endl;
#endif
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
    rpc_read(client, p2pStatus, sizeof(*p2pStatus));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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

extern "C" nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minorNumber) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMinorNumber called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMinorNumber);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, minorNumber, sizeof(*minorNumber));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetInforomConfigurationChecksum);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, checksum, sizeof(*checksum));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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

extern "C" nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t *display) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDisplayMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDisplayMode);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, display, sizeof(*display));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t *isActive) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDisplayActive called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDisplayActive);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, isActive, sizeof(*isActive));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPersistenceMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPersistenceMode);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, mode, sizeof(*mode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPciInfo_v3 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPciInfo_v3);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, pci, sizeof(*pci));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGen) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxPcieLinkGeneration called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxPcieLinkGeneration);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, maxLinkGen, sizeof(*maxLinkGen));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int *maxLinkWidth) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxPcieLinkWidth called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxPcieLinkWidth);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, maxLinkWidth, sizeof(*maxLinkWidth));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int *currLinkGen) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrPcieLinkGeneration called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCurrPcieLinkGeneration);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, currLinkGen, sizeof(*currLinkGen));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int *currLinkWidth) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrPcieLinkWidth called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCurrPcieLinkWidth);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, currLinkWidth, sizeof(*currLinkWidth));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int *value) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPcieThroughput called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPcieThroughput);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &counter, sizeof(counter));
    rpc_read(client, value, sizeof(*value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int *value) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPcieReplayCounter called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPcieReplayCounter);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, value, sizeof(*value));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClockInfo called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetClockInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &type, sizeof(type));
    rpc_read(client, clock, sizeof(*clock));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxClockInfo called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxClockInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &type, sizeof(type));
    rpc_read(client, clock, sizeof(*clock));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetApplicationsClock called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetApplicationsClock);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &clockType, sizeof(clockType));
    rpc_read(client, clockMHz, sizeof(*clockMHz));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDefaultApplicationsClock called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDefaultApplicationsClock);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &clockType, sizeof(clockType));
    rpc_read(client, clockMHz, sizeof(*clockMHz));
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

extern "C" nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClock called" << std::endl;
#endif
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
    rpc_read(client, clockMHz, sizeof(*clockMHz));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxCustomerBoostClock called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxCustomerBoostClock);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &clockType, sizeof(clockType));
    rpc_read(client, clockMHz, sizeof(*clockMHz));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int *count, unsigned int *clocksMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedMemoryClocks called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedMemoryClocks);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, count, sizeof(*count));
    rpc_read(client, clocksMHz, sizeof(*clocksMHz));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int *count, unsigned int *clocksMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedGraphicsClocks called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedGraphicsClocks);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &memoryClockMHz, sizeof(memoryClockMHz));
    rpc_read(client, count, sizeof(*count));
    rpc_read(client, clocksMHz, sizeof(*clocksMHz));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t *isEnabled, nvmlEnableState_t *defaultIsEnabled) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAutoBoostedClocksEnabled called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAutoBoostedClocksEnabled);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, isEnabled, sizeof(*isEnabled));
    rpc_read(client, defaultIsEnabled, sizeof(*defaultIsEnabled));
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

extern "C" nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFanSpeed called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetFanSpeed);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, speed, sizeof(*speed));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int *speed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFanSpeed_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetFanSpeed_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &fan, sizeof(fan));
    rpc_read(client, speed, sizeof(*speed));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTemperature called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetTemperature);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &sensorType, sizeof(sensorType));
    rpc_read(client, temp, sizeof(*temp));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTemperatureThreshold called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetTemperatureThreshold);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &thresholdType, sizeof(thresholdType));
    rpc_read(client, temp, sizeof(*temp));
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetTemperatureThreshold);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &thresholdType, sizeof(thresholdType));
    rpc_read(client, temp, sizeof(*temp));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t *pState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPerformanceState called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPerformanceState);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, pState, sizeof(*pState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long *clocksThrottleReasons) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrentClocksThrottleReasons called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCurrentClocksThrottleReasons);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, clocksThrottleReasons, sizeof(*clocksThrottleReasons));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long *supportedClocksThrottleReasons) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedClocksThrottleReasons called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedClocksThrottleReasons);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, supportedClocksThrottleReasons, sizeof(*supportedClocksThrottleReasons));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t *pState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerState called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerState);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, pState, sizeof(*pState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerManagementMode);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, mode, sizeof(*mode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int *limit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementLimit called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerManagementLimit);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, limit, sizeof(*limit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementLimitConstraints called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerManagementLimitConstraints);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, minLimit, sizeof(*minLimit));
    rpc_read(client, maxLimit, sizeof(*maxLimit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int *defaultLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementDefaultLimit called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerManagementDefaultLimit);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, defaultLimit, sizeof(*defaultLimit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerUsage called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPowerUsage);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, power, sizeof(*power));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long *energy) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTotalEnergyConsumption called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetTotalEnergyConsumption);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, energy, sizeof(*energy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEnforcedPowerLimit called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetEnforcedPowerLimit);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, limit, sizeof(*limit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t *current, nvmlGpuOperationMode_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuOperationMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuOperationMode);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, current, sizeof(*current));
    rpc_read(client, pending, sizeof(*pending));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryInfo called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMemoryInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, memory, sizeof(*memory));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetComputeMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetComputeMode);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, mode, sizeof(*mode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCudaComputeCapability called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCudaComputeCapability);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, major, sizeof(*major));
    rpc_read(client, minor, sizeof(*minor));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t *current, nvmlEnableState_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEccMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetEccMode);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, current, sizeof(*current));
    rpc_read(client, pending, sizeof(*pending));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int *boardId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBoardId called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetBoardId);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, boardId, sizeof(*boardId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int *multiGpuBool) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMultiGpuBoard called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMultiGpuBoard);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, multiGpuBool, sizeof(*multiGpuBool));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long *eccCounts) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTotalEccErrors called" << std::endl;
#endif
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
    rpc_read(client, eccCounts, sizeof(*eccCounts));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t *eccCounts) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDetailedEccErrors called" << std::endl;
#endif
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
    rpc_read(client, eccCounts, sizeof(*eccCounts));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryErrorCounter called" << std::endl;
#endif
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
    rpc_read(client, count, sizeof(*count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetUtilizationRates called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetUtilizationRates);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, utilization, sizeof(*utilization));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderUtilization called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetEncoderUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, utilization, sizeof(*utilization));
    rpc_read(client, samplingPeriodUs, sizeof(*samplingPeriodUs));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int *encoderCapacity) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderCapacity called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetEncoderCapacity);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &encoderQueryType, sizeof(encoderQueryType));
    rpc_read(client, encoderCapacity, sizeof(*encoderCapacity));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderStats called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetEncoderStats);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, sessionCount, sizeof(*sessionCount));
    rpc_read(client, averageFps, sizeof(*averageFps));
    rpc_read(client, averageLatency, sizeof(*averageLatency));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderSessions called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetEncoderSessions);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, sessionCount, sizeof(*sessionCount));
    rpc_read(client, sessionInfos, sizeof(*sessionInfos));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDecoderUtilization called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDecoderUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, utilization, sizeof(*utilization));
    rpc_read(client, samplingPeriodUs, sizeof(*samplingPeriodUs));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t *fbcStats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFBCStats called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetFBCStats);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, fbcStats, sizeof(*fbcStats));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFBCSessions called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetFBCSessions);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, sessionCount, sizeof(*sessionCount));
    rpc_read(client, sessionInfo, sizeof(*sessionInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t device, nvmlDriverModel_t *current, nvmlDriverModel_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDriverModel called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDriverModel);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, current, sizeof(*current));
    rpc_read(client, pending, sizeof(*pending));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetBridgeChipInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, bridgeHierarchy, sizeof(*bridgeHierarchy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetComputeRunningProcesses_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetComputeRunningProcesses_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, infoCount, sizeof(*infoCount));
    rpc_read(client, infos, sizeof(*infos));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGraphicsRunningProcesses_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGraphicsRunningProcesses_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, infoCount, sizeof(*infoCount));
    rpc_read(client, infos, sizeof(*infos));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMPSComputeRunningProcesses_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMPSComputeRunningProcesses_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, infoCount, sizeof(*infoCount));
    rpc_read(client, infos, sizeof(*infos));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int *onSameBoard) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceOnSameBoard called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceOnSameBoard);
    rpc_write(client, &device1, sizeof(device1));
    rpc_write(client, &device2, sizeof(device2));
    rpc_read(client, onSameBoard, sizeof(*onSameBoard));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t *isRestricted) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAPIRestriction called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAPIRestriction);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &apiType, sizeof(apiType));
    rpc_read(client, isRestricted, sizeof(*isRestricted));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *sampleCount, nvmlSample_t *samples) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSamples called" << std::endl;
#endif
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
    rpc_read(client, sampleValType, sizeof(*sampleValType));
    rpc_read(client, sampleCount, sizeof(*sampleCount));
    rpc_read(client, samples, sizeof(*samples));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t *bar1Memory) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBAR1MemoryInfo called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetBAR1MemoryInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, bar1Memory, sizeof(*bar1Memory));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t *violTime) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetViolationStatus called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetViolationStatus);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &perfPolicyType, sizeof(perfPolicyType));
    rpc_read(client, violTime, sizeof(*violTime));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAccountingMode);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, mode, sizeof(*mode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t *stats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingStats called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAccountingStats);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &pid, sizeof(pid));
    rpc_read(client, stats, sizeof(*stats));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int *count, unsigned int *pids) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingPids called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAccountingPids);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, count, sizeof(*count));
    rpc_read(client, pids, sizeof(*pids));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingBufferSize called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetAccountingBufferSize);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, bufferSize, sizeof(*bufferSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRetiredPages called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetRetiredPages);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &cause, sizeof(cause));
    rpc_read(client, pageCount, sizeof(*pageCount));
    rpc_read(client, addresses, sizeof(*addresses));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses, unsigned long long *timestamps) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRetiredPages_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetRetiredPages_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &cause, sizeof(cause));
    rpc_read(client, pageCount, sizeof(*pageCount));
    rpc_read(client, addresses, sizeof(*addresses));
    rpc_read(client, timestamps, sizeof(*timestamps));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t *isPending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRetiredPagesPendingStatus called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetRetiredPagesPendingStatus);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, isPending, sizeof(*isPending));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int *corrRows, unsigned int *uncRows, unsigned int *isPending, unsigned int *failureOccurred) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRemappedRows called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetRemappedRows);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, corrRows, sizeof(*corrRows));
    rpc_read(client, uncRows, sizeof(*uncRows));
    rpc_read(client, isPending, sizeof(*isPending));
    rpc_read(client, failureOccurred, sizeof(*failureOccurred));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t *values) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRowRemapperHistogram called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetRowRemapperHistogram);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, values, sizeof(*values));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t *arch) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetArchitecture called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetArchitecture);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, arch, sizeof(*arch));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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

extern "C" nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t *status) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClkMonStatus called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetClkMonStatus);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, status, sizeof(*status));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkState called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkState);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_read(client, isActive, sizeof(*isActive));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int *version) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkVersion called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkVersion);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_read(client, version, sizeof(*version));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int *capResult) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkCapability called" << std::endl;
#endif
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
    rpc_read(client, capResult, sizeof(*capResult));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkRemotePciInfo_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_read(client, pci, sizeof(*pci));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long *counterValue) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkErrorCounter called" << std::endl;
#endif
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
    rpc_read(client, counterValue, sizeof(*counterValue));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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
    rpc_read(client, control, sizeof(*control));
    rpc_write(client, &reset, sizeof(reset));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkUtilizationControl called" << std::endl;
#endif
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
    rpc_read(client, control, sizeof(*control));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, unsigned long long *rxcounter, unsigned long long *txcounter) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkUtilizationCounter called" << std::endl;
#endif
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
    rpc_read(client, rxcounter, sizeof(*rxcounter));
    rpc_read(client, txcounter, sizeof(*txcounter));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetNvLinkRemoteDeviceType);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &link, sizeof(link));
    rpc_read(client, pNvLinkDeviceType, sizeof(*pNvLinkDeviceType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t *set) {
#ifdef DEBUG
    std::cout << "Hook: nvmlEventSetCreate called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlEventSetCreate);
    rpc_read(client, set, sizeof(*set));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedEventTypes);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, eventTypes, sizeof(*eventTypes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t *data, unsigned int timeoutms) {
#ifdef DEBUG
    std::cout << "Hook: nvmlEventSetWait_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlEventSetWait_v2);
    rpc_write(client, &set, sizeof(set));
    rpc_read(client, data, sizeof(*data));
    rpc_write(client, &timeoutms, sizeof(timeoutms));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceModifyDrainState);
    rpc_read(client, pciInfo, sizeof(*pciInfo));
    rpc_write(client, &newState, sizeof(newState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t *pciInfo, nvmlEnableState_t *currentState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceQueryDrainState called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceQueryDrainState);
    rpc_read(client, pciInfo, sizeof(*pciInfo));
    rpc_read(client, currentState, sizeof(*currentState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t *pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceRemoveGpu_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceRemoveGpu_v2);
    rpc_read(client, pciInfo, sizeof(*pciInfo));
    rpc_write(client, &gpuState, sizeof(gpuState));
    rpc_write(client, &linkState, sizeof(linkState));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t *pciInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceDiscoverGpus called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceDiscoverGpus);
    rpc_read(client, pciInfo, sizeof(*pciInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFieldValues called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetFieldValues);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &valuesCount, sizeof(valuesCount));
    rpc_read(client, values, sizeof(*values));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t *pVirtualMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVirtualizationMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVirtualizationMode);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, pVirtualMode, sizeof(*pVirtualMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t *pHostVgpuMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHostVgpuMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetHostVgpuMode);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, pHostVgpuMode, sizeof(*pHostVgpuMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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

extern "C" nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v3(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGridLicensableFeatures_v3 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGridLicensableFeatures_v3);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, pGridLicensableFeatures, sizeof(*pGridLicensableFeatures));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization, unsigned int *processSamplesCount, unsigned long long lastSeenTimeStamp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetProcessUtilization called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetProcessUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, utilization, sizeof(*utilization));
    rpc_read(client, processSamplesCount, sizeof(*processSamplesCount));
    rpc_write(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedVgpus called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetSupportedVgpus);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, vgpuCount, sizeof(*vgpuCount));
    rpc_read(client, vgpuTypeIds, sizeof(*vgpuTypeIds));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCreatableVgpus called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetCreatableVgpus);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, vgpuCount, sizeof(*vgpuCount));
    rpc_read(client, vgpuTypeIds, sizeof(*vgpuTypeIds));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeClass, unsigned int *size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetClass called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetClass);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_read(client, vgpuTypeClass, *size, true);
    rpc_read(client, size, sizeof(*size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeName, unsigned int *size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetName called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetName);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_read(client, vgpuTypeName, *size, true);
    rpc_read(client, size, sizeof(*size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *gpuInstanceProfileId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetGpuInstanceProfileId called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetGpuInstanceProfileId);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_read(client, gpuInstanceProfileId, sizeof(*gpuInstanceProfileId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *deviceID, unsigned long long *subsystemID) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetDeviceID called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetDeviceID);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_read(client, deviceID, sizeof(*deviceID));
    rpc_read(client, subsystemID, sizeof(*subsystemID));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *fbSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetFramebufferSize called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetFramebufferSize);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_read(client, fbSize, sizeof(*fbSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *numDisplayHeads) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetNumDisplayHeads called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetNumDisplayHeads);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_read(client, numDisplayHeads, sizeof(*numDisplayHeads));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int *xdim, unsigned int *ydim) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetResolution called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetResolution);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &displayIndex, sizeof(displayIndex));
    rpc_read(client, xdim, sizeof(*xdim));
    rpc_read(client, ydim, sizeof(*ydim));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetFrameRateLimit);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_read(client, frameRateLimit, sizeof(*frameRateLimit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetMaxInstances called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetMaxInstances);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_read(client, vgpuInstanceCount, sizeof(*vgpuInstanceCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCountPerVm) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetMaxInstancesPerVm called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuTypeGetMaxInstancesPerVm);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_read(client, vgpuInstanceCountPerVm, sizeof(*vgpuInstanceCountPerVm));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuInstance_t *vgpuInstances) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetActiveVgpus called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetActiveVgpus);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, vgpuCount, sizeof(*vgpuCount));
    rpc_read(client, vgpuInstances, sizeof(*vgpuInstances));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char *vmId, unsigned int size, nvmlVgpuVmIdType_t *vmIdType) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetVmID called" << std::endl;
#endif
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
    rpc_read(client, vmIdType, sizeof(*vmIdType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetFbUsage);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, fbUsage, sizeof(*fbUsage));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int *licensed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetLicenseStatus called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetLicenseStatus);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, licensed, sizeof(*licensed));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t *vgpuTypeId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetType called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetType);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, vgpuTypeId, sizeof(*vgpuTypeId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int *frameRateLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFrameRateLimit called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetFrameRateLimit);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, frameRateLimit, sizeof(*frameRateLimit));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *eccMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEccMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetEccMode);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, eccMode, sizeof(*eccMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int *encoderCapacity) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEncoderCapacity called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetEncoderCapacity);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, encoderCapacity, sizeof(*encoderCapacity));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetEncoderStats);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, sessionCount, sizeof(*sessionCount));
    rpc_read(client, averageFps, sizeof(*averageFps));
    rpc_read(client, averageLatency, sizeof(*averageLatency));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEncoderSessions called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetEncoderSessions);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, sessionCount, sizeof(*sessionCount));
    rpc_read(client, sessionInfo, sizeof(*sessionInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t *fbcStats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFBCStats called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetFBCStats);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, fbcStats, sizeof(*fbcStats));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFBCSessions called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetFBCSessions);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, sessionCount, sizeof(*sessionCount));
    rpc_read(client, sessionInfo, sizeof(*sessionInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int *gpuInstanceId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetGpuInstanceId called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetGpuInstanceId);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, gpuInstanceId, sizeof(*gpuInstanceId));
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetMetadata);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, vgpuMetadata, sizeof(*vgpuMetadata));
    rpc_read(client, bufferSize, sizeof(*bufferSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t *pgpuMetadata, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuMetadata called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuMetadata);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, pgpuMetadata, sizeof(*pgpuMetadata));
    rpc_read(client, bufferSize, sizeof(*bufferSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t *vgpuMetadata, nvmlVgpuPgpuMetadata_t *pgpuMetadata, nvmlVgpuPgpuCompatibility_t *compatibilityInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetVgpuCompatibility called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGetVgpuCompatibility);
    rpc_read(client, vgpuMetadata, sizeof(*vgpuMetadata));
    rpc_read(client, pgpuMetadata, sizeof(*pgpuMetadata));
    rpc_read(client, compatibilityInfo, sizeof(*compatibilityInfo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char *pgpuMetadata, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPgpuMetadataString called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetPgpuMetadataString);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, pgpuMetadata, *bufferSize, true);
    rpc_read(client, bufferSize, sizeof(*bufferSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t *supported, nvmlVgpuVersion_t *current) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetVgpuVersion called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGetVgpuVersion);
    rpc_read(client, supported, sizeof(*supported));
    rpc_read(client, current, sizeof(*current));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t *vgpuVersion) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSetVgpuVersion called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlSetVgpuVersion);
    rpc_read(client, vgpuVersion, sizeof(*vgpuVersion));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t *utilizationSamples) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuUtilization called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    rpc_read(client, sampleValType, sizeof(*sampleValType));
    rpc_read(client, vgpuInstanceSamplesCount, sizeof(*vgpuInstanceSamplesCount));
    rpc_read(client, utilizationSamples, sizeof(*utilizationSamples));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int *vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t *utilizationSamples) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuProcessUtilization called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetVgpuProcessUtilization);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    rpc_read(client, vgpuProcessSamplesCount, sizeof(*vgpuProcessSamplesCount));
    rpc_read(client, utilizationSamples, sizeof(*utilizationSamples));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetAccountingMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetAccountingMode);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, mode, sizeof(*mode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int *count, unsigned int *pids) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetAccountingPids called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetAccountingPids);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_read(client, count, sizeof(*count));
    rpc_read(client, pids, sizeof(*pids));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t *stats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetAccountingStats called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlVgpuInstanceGetAccountingStats);
    rpc_write(client, &vgpuInstance, sizeof(vgpuInstance));
    rpc_write(client, &pid, sizeof(pid));
    rpc_read(client, stats, sizeof(*stats));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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

extern "C" nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int *deviceCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetExcludedDeviceCount called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGetExcludedDeviceCount);
    rpc_read(client, deviceCount, sizeof(*deviceCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetExcludedDeviceInfoByIndex called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGetExcludedDeviceInfoByIndex);
    rpc_write(client, &index, sizeof(index));
    rpc_read(client, info, sizeof(*info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t *activationStatus) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetMigMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceSetMigMode);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &mode, sizeof(mode));
    rpc_read(client, activationStatus, sizeof(*activationStatus));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *currentMode, unsigned int *pendingMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMigMode called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMigMode);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, currentMode, sizeof(*currentMode));
    rpc_read(client, pendingMode, sizeof(*pendingMode));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceProfileInfo called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstanceProfileInfo);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profile, sizeof(profile));
    rpc_read(client, info, sizeof(*info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t *placements, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstancePossiblePlacements_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstancePossiblePlacements_v2);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_read(client, placements, sizeof(*placements));
    rpc_read(client, count, sizeof(*count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceRemainingCapacity called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstanceRemainingCapacity);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_read(client, count, sizeof(*count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceCreateGpuInstance called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceCreateGpuInstance);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_read(client, gpuInstance, sizeof(*gpuInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t *placement, nvmlGpuInstance_t *gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceCreateGpuInstanceWithPlacement called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceCreateGpuInstanceWithPlacement);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_write(client, placement, sizeof(*placement));
    rpc_read(client, gpuInstance, sizeof(*gpuInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstances);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_read(client, gpuInstances, sizeof(*gpuInstances));
    rpc_read(client, count, sizeof(*count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t *gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceById called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstanceById);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &id, sizeof(id));
    rpc_read(client, gpuInstance, sizeof(*gpuInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetInfo called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetInfo);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_read(client, info, sizeof(*info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceProfileInfo called" << std::endl;
#endif
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
    rpc_read(client, info, sizeof(*info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceRemainingCapacity called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetComputeInstanceRemainingCapacity);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_read(client, count, sizeof(*count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceCreateComputeInstance called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceCreateComputeInstance);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_read(client, computeInstance, sizeof(*computeInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
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
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetComputeInstances);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &profileId, sizeof(profileId));
    rpc_read(client, computeInstances, sizeof(*computeInstances));
    rpc_read(client, count, sizeof(*count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t *computeInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceById called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlGpuInstanceGetComputeInstanceById);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &id, sizeof(id));
    rpc_read(client, computeInstance, sizeof(*computeInstance));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlComputeInstanceGetInfo_v2 called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlComputeInstanceGetInfo_v2);
    rpc_write(client, &computeInstance, sizeof(computeInstance));
    rpc_read(client, info, sizeof(*info));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int *isMigDevice) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceIsMigDeviceHandle called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceIsMigDeviceHandle);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, isMigDevice, sizeof(*isMigDevice));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int *id) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceId called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetGpuInstanceId);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, id, sizeof(*id));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int *id) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetComputeInstanceId called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetComputeInstanceId);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, id, sizeof(*id));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxMigDeviceCount called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMaxMigDeviceCount);
    rpc_write(client, &device, sizeof(device));
    rpc_read(client, count, sizeof(*count));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t *migDevice) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMigDeviceHandleByIndex called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetMigDeviceHandleByIndex);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &index, sizeof(index));
    rpc_read(client, migDevice, sizeof(*migDevice));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDeviceHandleFromMigDeviceHandle called" << std::endl;
#endif
    nvmlReturn_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlDeviceGetDeviceHandleFromMigDeviceHandle);
    rpc_write(client, &migDevice, sizeof(migDevice));
    rpc_read(client, device, sizeof(*device));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

