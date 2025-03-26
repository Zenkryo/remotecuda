#include <iostream>
#include <unordered_map>
#include "hook_api.h"
#include "handle_server.h"
#include "../rpc.h"
#include "nvml.h"

int handle_nvmlInit_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlInit_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlInit_v2();
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlInitWithFlags(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlInitWithFlags called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlInitWithFlags(flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlShutdown(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlShutdown called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlShutdown();
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetDriverVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetDriverVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    char version[1024];
    unsigned int length;
    rpc_read(client, &length, sizeof(length));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetDriverVersion(version, length);
    rpc_write(client, version, strlen(version) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetNVMLVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetNVMLVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    char version[1024];
    unsigned int length;
    rpc_read(client, &length, sizeof(length));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetNVMLVersion(version, length);
    rpc_write(client, version, strlen(version) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetCudaDriverVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetCudaDriverVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*cudaDriverVersion;
    rpc_read(client, &cudaDriverVersion, sizeof(cudaDriverVersion));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetCudaDriverVersion(cudaDriverVersion);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetCudaDriverVersion_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetCudaDriverVersion_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int*cudaDriverVersion;
    rpc_read(client, &cudaDriverVersion, sizeof(cudaDriverVersion));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetCudaDriverVersion_v2(cudaDriverVersion);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetProcessName(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetProcessName called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int pid;
    rpc_read(client, &pid, sizeof(pid));
    char name[1024];
    unsigned int length;
    rpc_read(client, &length, sizeof(length));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetProcessName(pid, name, length);
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

int handle_nvmlUnitGetCount(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlUnitGetCount called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int*unitCount;
    rpc_read(client, &unitCount, sizeof(unitCount));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetCount(unitCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlUnitGetHandleByIndex(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlUnitGetHandleByIndex called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int index;
    rpc_read(client, &index, sizeof(index));
    nvmlUnit_t*unit;
    rpc_read(client, &unit, sizeof(unit));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetHandleByIndex(index, unit);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlUnitGetUnitInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlUnitGetUnitInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlUnit_t unit;
    rpc_read(client, &unit, sizeof(unit));
    nvmlUnitInfo_t*info;
    rpc_read(client, &info, sizeof(info));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetUnitInfo(unit, info);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlUnitGetLedState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlUnitGetLedState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlUnit_t unit;
    rpc_read(client, &unit, sizeof(unit));
    nvmlLedState_t*state;
    rpc_read(client, &state, sizeof(state));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetLedState(unit, state);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlUnitGetPsuInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlUnitGetPsuInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlUnit_t unit;
    rpc_read(client, &unit, sizeof(unit));
    nvmlPSUInfo_t*psu;
    rpc_read(client, &psu, sizeof(psu));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetPsuInfo(unit, psu);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlUnitGetTemperature(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlUnitGetTemperature called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlUnit_t unit;
    rpc_read(client, &unit, sizeof(unit));
    unsigned int type;
    rpc_read(client, &type, sizeof(type));
    unsigned int*temp;
    rpc_read(client, &temp, sizeof(temp));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetTemperature(unit, type, temp);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlUnitGetFanSpeedInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlUnitGetFanSpeedInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlUnit_t unit;
    rpc_read(client, &unit, sizeof(unit));
    nvmlUnitFanSpeeds_t*fanSpeeds;
    rpc_read(client, &fanSpeeds, sizeof(fanSpeeds));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetFanSpeedInfo(unit, fanSpeeds);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlUnitGetDevices(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlUnitGetDevices called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlUnit_t unit;
    rpc_read(client, &unit, sizeof(unit));
    unsigned int*deviceCount;
    rpc_read(client, &deviceCount, sizeof(deviceCount));
    nvmlDevice_t*devices;
    rpc_read(client, &devices, sizeof(devices));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetDevices(unit, deviceCount, devices);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetHicVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetHicVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int*hwbcCount;
    rpc_read(client, &hwbcCount, sizeof(hwbcCount));
    nvmlHwbcEntry_t*hwbcEntries;
    rpc_read(client, &hwbcEntries, sizeof(hwbcEntries));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetHicVersion(hwbcCount, hwbcEntries);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetCount_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetCount_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int*deviceCount;
    rpc_read(client, &deviceCount, sizeof(deviceCount));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCount_v2(deviceCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetAttributes_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetAttributes_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlDeviceAttributes_t*attributes;
    rpc_read(client, &attributes, sizeof(attributes));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAttributes_v2(device, attributes);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetHandleByIndex_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetHandleByIndex_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int index;
    rpc_read(client, &index, sizeof(index));
    nvmlDevice_t*device;
    rpc_read(client, &device, sizeof(device));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetHandleByIndex_v2(index, device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetHandleBySerial(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetHandleBySerial called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    char *serial = nullptr;
    rpc_read(client, &serial, 0, true);
    nvmlDevice_t*device;
    rpc_read(client, &device, sizeof(device));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(serial);
    _result = nvmlDeviceGetHandleBySerial(serial, device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetHandleByUUID(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetHandleByUUID called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    char *uuid = nullptr;
    rpc_read(client, &uuid, 0, true);
    nvmlDevice_t*device;
    rpc_read(client, &device, sizeof(device));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(uuid);
    _result = nvmlDeviceGetHandleByUUID(uuid, device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetHandleByPciBusId_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetHandleByPciBusId_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    char *pciBusId = nullptr;
    rpc_read(client, &pciBusId, 0, true);
    nvmlDevice_t*device;
    rpc_read(client, &device, sizeof(device));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(pciBusId);
    _result = nvmlDeviceGetHandleByPciBusId_v2(pciBusId, device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetName(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetName called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    char name[1024];
    unsigned int length;
    rpc_read(client, &length, sizeof(length));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetName(device, name, length);
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

int handle_nvmlDeviceGetBrand(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetBrand called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlBrandType_t*type;
    rpc_read(client, &type, sizeof(type));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBrand(device, type);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetIndex(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetIndex called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*index;
    rpc_read(client, &index, sizeof(index));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetIndex(device, index);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetSerial(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetSerial called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    char serial[1024];
    unsigned int length;
    rpc_read(client, &length, sizeof(length));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSerial(device, serial, length);
    rpc_write(client, serial, strlen(serial) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMemoryAffinity(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMemoryAffinity called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int nodeSetSize;
    rpc_read(client, &nodeSetSize, sizeof(nodeSetSize));
    unsigned long*nodeSet;
    rpc_read(client, &nodeSet, sizeof(nodeSet));
    nvmlAffinityScope_t scope;
    rpc_read(client, &scope, sizeof(scope));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMemoryAffinity(device, nodeSetSize, nodeSet, scope);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetCpuAffinityWithinScope(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetCpuAffinityWithinScope called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int cpuSetSize;
    rpc_read(client, &cpuSetSize, sizeof(cpuSetSize));
    unsigned long*cpuSet;
    rpc_read(client, &cpuSet, sizeof(cpuSet));
    nvmlAffinityScope_t scope;
    rpc_read(client, &scope, sizeof(scope));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCpuAffinityWithinScope(device, cpuSetSize, cpuSet, scope);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetCpuAffinity(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetCpuAffinity called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int cpuSetSize;
    rpc_read(client, &cpuSetSize, sizeof(cpuSetSize));
    unsigned long*cpuSet;
    rpc_read(client, &cpuSet, sizeof(cpuSet));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCpuAffinity(device, cpuSetSize, cpuSet);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetCpuAffinity(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetCpuAffinity called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetCpuAffinity(device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceClearCpuAffinity(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceClearCpuAffinity called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceClearCpuAffinity(device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetTopologyCommonAncestor(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetTopologyCommonAncestor called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device1;
    rpc_read(client, &device1, sizeof(device1));
    nvmlDevice_t device2;
    rpc_read(client, &device2, sizeof(device2));
    nvmlGpuTopologyLevel_t*pathInfo;
    rpc_read(client, &pathInfo, sizeof(pathInfo));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTopologyCommonAncestor(device1, device2, pathInfo);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetTopologyNearestGpus(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetTopologyNearestGpus called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlGpuTopologyLevel_t level;
    rpc_read(client, &level, sizeof(level));
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    nvmlDevice_t*deviceArray;
    rpc_read(client, &deviceArray, sizeof(deviceArray));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTopologyNearestGpus(device, level, count, deviceArray);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetTopologyGpuSet(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetTopologyGpuSet called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int cpuNumber;
    rpc_read(client, &cpuNumber, sizeof(cpuNumber));
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    nvmlDevice_t*deviceArray;
    rpc_read(client, &deviceArray, sizeof(deviceArray));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetTopologyGpuSet(cpuNumber, count, deviceArray);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetP2PStatus(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetP2PStatus called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device1;
    rpc_read(client, &device1, sizeof(device1));
    nvmlDevice_t device2;
    rpc_read(client, &device2, sizeof(device2));
    nvmlGpuP2PCapsIndex_t p2pIndex;
    rpc_read(client, &p2pIndex, sizeof(p2pIndex));
    nvmlGpuP2PStatus_t*p2pStatus;
    rpc_read(client, &p2pStatus, sizeof(p2pStatus));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetP2PStatus(device1, device2, p2pIndex, p2pStatus);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetUUID(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetUUID called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    char uuid[1024];
    unsigned int length;
    rpc_read(client, &length, sizeof(length));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetUUID(device, uuid, length);
    rpc_write(client, uuid, strlen(uuid) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetMdevUUID(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetMdevUUID called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    char mdevUuid[1024];
    unsigned int size;
    rpc_read(client, &size, sizeof(size));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetMdevUUID(vgpuInstance, mdevUuid, size);
    rpc_write(client, mdevUuid, strlen(mdevUuid) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMinorNumber(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMinorNumber called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*minorNumber;
    rpc_read(client, &minorNumber, sizeof(minorNumber));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMinorNumber(device, minorNumber);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetBoardPartNumber(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetBoardPartNumber called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    char partNumber[1024];
    unsigned int length;
    rpc_read(client, &length, sizeof(length));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBoardPartNumber(device, partNumber, length);
    rpc_write(client, partNumber, strlen(partNumber) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetInforomVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetInforomVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlInforomObject_t object;
    rpc_read(client, &object, sizeof(object));
    char version[1024];
    unsigned int length;
    rpc_read(client, &length, sizeof(length));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetInforomVersion(device, object, version, length);
    rpc_write(client, version, strlen(version) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetInforomImageVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetInforomImageVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    char version[1024];
    unsigned int length;
    rpc_read(client, &length, sizeof(length));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetInforomImageVersion(device, version, length);
    rpc_write(client, version, strlen(version) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetInforomConfigurationChecksum(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetInforomConfigurationChecksum called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*checksum;
    rpc_read(client, &checksum, sizeof(checksum));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetInforomConfigurationChecksum(device, checksum);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceValidateInforom(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceValidateInforom called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceValidateInforom(device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetDisplayMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetDisplayMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t*display;
    rpc_read(client, &display, sizeof(display));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDisplayMode(device, display);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetDisplayActive(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetDisplayActive called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t*isActive;
    rpc_read(client, &isActive, sizeof(isActive));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDisplayActive(device, isActive);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPersistenceMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPersistenceMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t*mode;
    rpc_read(client, &mode, sizeof(mode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPersistenceMode(device, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPciInfo_v3(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPciInfo_v3 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPciInfo_t*pci;
    rpc_read(client, &pci, sizeof(pci));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPciInfo_v3(device, pci);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMaxPcieLinkGeneration(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMaxPcieLinkGeneration called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*maxLinkGen;
    rpc_read(client, &maxLinkGen, sizeof(maxLinkGen));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxPcieLinkGeneration(device, maxLinkGen);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMaxPcieLinkWidth(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMaxPcieLinkWidth called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*maxLinkWidth;
    rpc_read(client, &maxLinkWidth, sizeof(maxLinkWidth));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxPcieLinkWidth(device, maxLinkWidth);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetCurrPcieLinkGeneration(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetCurrPcieLinkGeneration called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*currLinkGen;
    rpc_read(client, &currLinkGen, sizeof(currLinkGen));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCurrPcieLinkGeneration(device, currLinkGen);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetCurrPcieLinkWidth(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetCurrPcieLinkWidth called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*currLinkWidth;
    rpc_read(client, &currLinkWidth, sizeof(currLinkWidth));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCurrPcieLinkWidth(device, currLinkWidth);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPcieThroughput(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPcieThroughput called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPcieUtilCounter_t counter;
    rpc_read(client, &counter, sizeof(counter));
    unsigned int*value;
    rpc_read(client, &value, sizeof(value));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPcieThroughput(device, counter, value);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPcieReplayCounter(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPcieReplayCounter called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*value;
    rpc_read(client, &value, sizeof(value));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPcieReplayCounter(device, value);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetClockInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetClockInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlClockType_t type;
    rpc_read(client, &type, sizeof(type));
    unsigned int*clock;
    rpc_read(client, &clock, sizeof(clock));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetClockInfo(device, type, clock);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMaxClockInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMaxClockInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlClockType_t type;
    rpc_read(client, &type, sizeof(type));
    unsigned int*clock;
    rpc_read(client, &clock, sizeof(clock));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxClockInfo(device, type, clock);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetApplicationsClock(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetApplicationsClock called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlClockType_t clockType;
    rpc_read(client, &clockType, sizeof(clockType));
    unsigned int*clockMHz;
    rpc_read(client, &clockMHz, sizeof(clockMHz));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetApplicationsClock(device, clockType, clockMHz);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetDefaultApplicationsClock(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetDefaultApplicationsClock called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlClockType_t clockType;
    rpc_read(client, &clockType, sizeof(clockType));
    unsigned int*clockMHz;
    rpc_read(client, &clockMHz, sizeof(clockMHz));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDefaultApplicationsClock(device, clockType, clockMHz);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceResetApplicationsClocks(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceResetApplicationsClocks called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceResetApplicationsClocks(device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetClock(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetClock called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlClockType_t clockType;
    rpc_read(client, &clockType, sizeof(clockType));
    nvmlClockId_t clockId;
    rpc_read(client, &clockId, sizeof(clockId));
    unsigned int*clockMHz;
    rpc_read(client, &clockMHz, sizeof(clockMHz));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetClock(device, clockType, clockId, clockMHz);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMaxCustomerBoostClock(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMaxCustomerBoostClock called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlClockType_t clockType;
    rpc_read(client, &clockType, sizeof(clockType));
    unsigned int*clockMHz;
    rpc_read(client, &clockMHz, sizeof(clockMHz));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxCustomerBoostClock(device, clockType, clockMHz);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetSupportedMemoryClocks(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetSupportedMemoryClocks called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    unsigned int*clocksMHz;
    rpc_read(client, &clocksMHz, sizeof(clocksMHz));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedMemoryClocks(device, count, clocksMHz);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetSupportedGraphicsClocks(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetSupportedGraphicsClocks called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int memoryClockMHz;
    rpc_read(client, &memoryClockMHz, sizeof(memoryClockMHz));
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    unsigned int*clocksMHz;
    rpc_read(client, &clocksMHz, sizeof(clocksMHz));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, count, clocksMHz);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetAutoBoostedClocksEnabled(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetAutoBoostedClocksEnabled called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t*isEnabled;
    rpc_read(client, &isEnabled, sizeof(isEnabled));
    nvmlEnableState_t*defaultIsEnabled;
    rpc_read(client, &defaultIsEnabled, sizeof(defaultIsEnabled));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAutoBoostedClocksEnabled(device, isEnabled, defaultIsEnabled);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetAutoBoostedClocksEnabled(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetAutoBoostedClocksEnabled called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t enabled;
    rpc_read(client, &enabled, sizeof(enabled));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetAutoBoostedClocksEnabled(device, enabled);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetDefaultAutoBoostedClocksEnabled(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetDefaultAutoBoostedClocksEnabled called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t enabled;
    rpc_read(client, &enabled, sizeof(enabled));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetDefaultAutoBoostedClocksEnabled(device, enabled, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetFanSpeed(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetFanSpeed called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*speed;
    rpc_read(client, &speed, sizeof(speed));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFanSpeed(device, speed);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetFanSpeed_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetFanSpeed_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int fan;
    rpc_read(client, &fan, sizeof(fan));
    unsigned int*speed;
    rpc_read(client, &speed, sizeof(speed));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFanSpeed_v2(device, fan, speed);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetTemperature(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetTemperature called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlTemperatureSensors_t sensorType;
    rpc_read(client, &sensorType, sizeof(sensorType));
    unsigned int*temp;
    rpc_read(client, &temp, sizeof(temp));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTemperature(device, sensorType, temp);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetTemperatureThreshold(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetTemperatureThreshold called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlTemperatureThresholds_t thresholdType;
    rpc_read(client, &thresholdType, sizeof(thresholdType));
    unsigned int*temp;
    rpc_read(client, &temp, sizeof(temp));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTemperatureThreshold(device, thresholdType, temp);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetTemperatureThreshold(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetTemperatureThreshold called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlTemperatureThresholds_t thresholdType;
    rpc_read(client, &thresholdType, sizeof(thresholdType));
    int*temp;
    rpc_read(client, &temp, sizeof(temp));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetTemperatureThreshold(device, thresholdType, temp);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPerformanceState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPerformanceState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPstates_t*pState;
    rpc_read(client, &pState, sizeof(pState));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPerformanceState(device, pState);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetCurrentClocksThrottleReasons(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetCurrentClocksThrottleReasons called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned long long*clocksThrottleReasons;
    rpc_read(client, &clocksThrottleReasons, sizeof(clocksThrottleReasons));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCurrentClocksThrottleReasons(device, clocksThrottleReasons);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetSupportedClocksThrottleReasons(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetSupportedClocksThrottleReasons called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned long long*supportedClocksThrottleReasons;
    rpc_read(client, &supportedClocksThrottleReasons, sizeof(supportedClocksThrottleReasons));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedClocksThrottleReasons(device, supportedClocksThrottleReasons);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPowerState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPowerState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPstates_t*pState;
    rpc_read(client, &pState, sizeof(pState));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerState(device, pState);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPowerManagementMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPowerManagementMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t*mode;
    rpc_read(client, &mode, sizeof(mode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerManagementMode(device, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPowerManagementLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPowerManagementLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*limit;
    rpc_read(client, &limit, sizeof(limit));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerManagementLimit(device, limit);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPowerManagementLimitConstraints(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPowerManagementLimitConstraints called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*minLimit;
    rpc_read(client, &minLimit, sizeof(minLimit));
    unsigned int*maxLimit;
    rpc_read(client, &maxLimit, sizeof(maxLimit));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerManagementLimitConstraints(device, minLimit, maxLimit);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPowerManagementDefaultLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPowerManagementDefaultLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*defaultLimit;
    rpc_read(client, &defaultLimit, sizeof(defaultLimit));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerManagementDefaultLimit(device, defaultLimit);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPowerUsage(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPowerUsage called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*power;
    rpc_read(client, &power, sizeof(power));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerUsage(device, power);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetTotalEnergyConsumption(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetTotalEnergyConsumption called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned long long*energy;
    rpc_read(client, &energy, sizeof(energy));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTotalEnergyConsumption(device, energy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetEnforcedPowerLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetEnforcedPowerLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*limit;
    rpc_read(client, &limit, sizeof(limit));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEnforcedPowerLimit(device, limit);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGpuOperationMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGpuOperationMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlGpuOperationMode_t*current;
    rpc_read(client, &current, sizeof(current));
    nvmlGpuOperationMode_t*pending;
    rpc_read(client, &pending, sizeof(pending));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuOperationMode(device, current, pending);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMemoryInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMemoryInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlMemory_t*memory;
    rpc_read(client, &memory, sizeof(memory));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMemoryInfo(device, memory);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetComputeMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetComputeMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlComputeMode_t*mode;
    rpc_read(client, &mode, sizeof(mode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetComputeMode(device, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetCudaComputeCapability(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetCudaComputeCapability called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    int*major;
    rpc_read(client, &major, sizeof(major));
    int*minor;
    rpc_read(client, &minor, sizeof(minor));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCudaComputeCapability(device, major, minor);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetEccMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetEccMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t*current;
    rpc_read(client, &current, sizeof(current));
    nvmlEnableState_t*pending;
    rpc_read(client, &pending, sizeof(pending));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEccMode(device, current, pending);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetBoardId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetBoardId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*boardId;
    rpc_read(client, &boardId, sizeof(boardId));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBoardId(device, boardId);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMultiGpuBoard(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMultiGpuBoard called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*multiGpuBool;
    rpc_read(client, &multiGpuBool, sizeof(multiGpuBool));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMultiGpuBoard(device, multiGpuBool);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetTotalEccErrors(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetTotalEccErrors called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlMemoryErrorType_t errorType;
    rpc_read(client, &errorType, sizeof(errorType));
    nvmlEccCounterType_t counterType;
    rpc_read(client, &counterType, sizeof(counterType));
    unsigned long long*eccCounts;
    rpc_read(client, &eccCounts, sizeof(eccCounts));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTotalEccErrors(device, errorType, counterType, eccCounts);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetDetailedEccErrors(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetDetailedEccErrors called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlMemoryErrorType_t errorType;
    rpc_read(client, &errorType, sizeof(errorType));
    nvmlEccCounterType_t counterType;
    rpc_read(client, &counterType, sizeof(counterType));
    nvmlEccErrorCounts_t*eccCounts;
    rpc_read(client, &eccCounts, sizeof(eccCounts));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDetailedEccErrors(device, errorType, counterType, eccCounts);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMemoryErrorCounter(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMemoryErrorCounter called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlMemoryErrorType_t errorType;
    rpc_read(client, &errorType, sizeof(errorType));
    nvmlEccCounterType_t counterType;
    rpc_read(client, &counterType, sizeof(counterType));
    nvmlMemoryLocation_t locationType;
    rpc_read(client, &locationType, sizeof(locationType));
    unsigned long long*count;
    rpc_read(client, &count, sizeof(count));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMemoryErrorCounter(device, errorType, counterType, locationType, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetUtilizationRates(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetUtilizationRates called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlUtilization_t*utilization;
    rpc_read(client, &utilization, sizeof(utilization));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetUtilizationRates(device, utilization);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetEncoderUtilization(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetEncoderUtilization called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*utilization;
    rpc_read(client, &utilization, sizeof(utilization));
    unsigned int*samplingPeriodUs;
    rpc_read(client, &samplingPeriodUs, sizeof(samplingPeriodUs));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEncoderUtilization(device, utilization, samplingPeriodUs);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetEncoderCapacity(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetEncoderCapacity called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEncoderType_t encoderQueryType;
    rpc_read(client, &encoderQueryType, sizeof(encoderQueryType));
    unsigned int*encoderCapacity;
    rpc_read(client, &encoderCapacity, sizeof(encoderCapacity));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEncoderCapacity(device, encoderQueryType, encoderCapacity);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetEncoderStats(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetEncoderStats called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*sessionCount;
    rpc_read(client, &sessionCount, sizeof(sessionCount));
    unsigned int*averageFps;
    rpc_read(client, &averageFps, sizeof(averageFps));
    unsigned int*averageLatency;
    rpc_read(client, &averageLatency, sizeof(averageLatency));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEncoderStats(device, sessionCount, averageFps, averageLatency);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetEncoderSessions(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetEncoderSessions called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*sessionCount;
    rpc_read(client, &sessionCount, sizeof(sessionCount));
    nvmlEncoderSessionInfo_t*sessionInfos;
    rpc_read(client, &sessionInfos, sizeof(sessionInfos));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEncoderSessions(device, sessionCount, sessionInfos);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetDecoderUtilization(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetDecoderUtilization called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*utilization;
    rpc_read(client, &utilization, sizeof(utilization));
    unsigned int*samplingPeriodUs;
    rpc_read(client, &samplingPeriodUs, sizeof(samplingPeriodUs));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDecoderUtilization(device, utilization, samplingPeriodUs);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetFBCStats(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetFBCStats called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlFBCStats_t*fbcStats;
    rpc_read(client, &fbcStats, sizeof(fbcStats));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFBCStats(device, fbcStats);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetFBCSessions(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetFBCSessions called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*sessionCount;
    rpc_read(client, &sessionCount, sizeof(sessionCount));
    nvmlFBCSessionInfo_t*sessionInfo;
    rpc_read(client, &sessionInfo, sizeof(sessionInfo));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFBCSessions(device, sessionCount, sessionInfo);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetDriverModel(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetDriverModel called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlDriverModel_t*current;
    rpc_read(client, &current, sizeof(current));
    nvmlDriverModel_t*pending;
    rpc_read(client, &pending, sizeof(pending));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDriverModel(device, current, pending);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetVbiosVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVbiosVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    char version[1024];
    unsigned int length;
    rpc_read(client, &length, sizeof(length));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVbiosVersion(device, version, length);
    rpc_write(client, version, strlen(version) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetBridgeChipInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetBridgeChipInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlBridgeChipHierarchy_t*bridgeHierarchy;
    rpc_read(client, &bridgeHierarchy, sizeof(bridgeHierarchy));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBridgeChipInfo(device, bridgeHierarchy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetComputeRunningProcesses_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetComputeRunningProcesses_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*infoCount;
    rpc_read(client, &infoCount, sizeof(infoCount));
    nvmlProcessInfo_t*infos;
    rpc_read(client, &infos, sizeof(infos));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetComputeRunningProcesses_v2(device, infoCount, infos);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGraphicsRunningProcesses_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGraphicsRunningProcesses_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*infoCount;
    rpc_read(client, &infoCount, sizeof(infoCount));
    nvmlProcessInfo_t*infos;
    rpc_read(client, &infos, sizeof(infos));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGraphicsRunningProcesses_v2(device, infoCount, infos);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMPSComputeRunningProcesses_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMPSComputeRunningProcesses_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*infoCount;
    rpc_read(client, &infoCount, sizeof(infoCount));
    nvmlProcessInfo_t*infos;
    rpc_read(client, &infos, sizeof(infos));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMPSComputeRunningProcesses_v2(device, infoCount, infos);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceOnSameBoard(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceOnSameBoard called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device1;
    rpc_read(client, &device1, sizeof(device1));
    nvmlDevice_t device2;
    rpc_read(client, &device2, sizeof(device2));
    int*onSameBoard;
    rpc_read(client, &onSameBoard, sizeof(onSameBoard));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceOnSameBoard(device1, device2, onSameBoard);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetAPIRestriction(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetAPIRestriction called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlRestrictedAPI_t apiType;
    rpc_read(client, &apiType, sizeof(apiType));
    nvmlEnableState_t*isRestricted;
    rpc_read(client, &isRestricted, sizeof(isRestricted));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAPIRestriction(device, apiType, isRestricted);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetSamples(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetSamples called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlSamplingType_t type;
    rpc_read(client, &type, sizeof(type));
    unsigned long long lastSeenTimeStamp;
    rpc_read(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    nvmlValueType_t*sampleValType;
    rpc_read(client, &sampleValType, sizeof(sampleValType));
    unsigned int*sampleCount;
    rpc_read(client, &sampleCount, sizeof(sampleCount));
    nvmlSample_t*samples;
    rpc_read(client, &samples, sizeof(samples));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, sampleValType, sampleCount, samples);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetBAR1MemoryInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetBAR1MemoryInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlBAR1Memory_t*bar1Memory;
    rpc_read(client, &bar1Memory, sizeof(bar1Memory));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBAR1MemoryInfo(device, bar1Memory);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetViolationStatus(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetViolationStatus called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPerfPolicyType_t perfPolicyType;
    rpc_read(client, &perfPolicyType, sizeof(perfPolicyType));
    nvmlViolationTime_t*violTime;
    rpc_read(client, &violTime, sizeof(violTime));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetViolationStatus(device, perfPolicyType, violTime);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetAccountingMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetAccountingMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t*mode;
    rpc_read(client, &mode, sizeof(mode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAccountingMode(device, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetAccountingStats(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetAccountingStats called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int pid;
    rpc_read(client, &pid, sizeof(pid));
    nvmlAccountingStats_t*stats;
    rpc_read(client, &stats, sizeof(stats));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAccountingStats(device, pid, stats);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetAccountingPids(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetAccountingPids called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    unsigned int*pids;
    rpc_read(client, &pids, sizeof(pids));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAccountingPids(device, count, pids);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetAccountingBufferSize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetAccountingBufferSize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*bufferSize;
    rpc_read(client, &bufferSize, sizeof(bufferSize));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAccountingBufferSize(device, bufferSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetRetiredPages(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetRetiredPages called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPageRetirementCause_t cause;
    rpc_read(client, &cause, sizeof(cause));
    unsigned int*pageCount;
    rpc_read(client, &pageCount, sizeof(pageCount));
    unsigned long long*addresses;
    rpc_read(client, &addresses, sizeof(addresses));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRetiredPages(device, cause, pageCount, addresses);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetRetiredPages_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetRetiredPages_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPageRetirementCause_t cause;
    rpc_read(client, &cause, sizeof(cause));
    unsigned int*pageCount;
    rpc_read(client, &pageCount, sizeof(pageCount));
    unsigned long long*addresses;
    rpc_read(client, &addresses, sizeof(addresses));
    unsigned long long*timestamps;
    rpc_read(client, &timestamps, sizeof(timestamps));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRetiredPages_v2(device, cause, pageCount, addresses, timestamps);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetRetiredPagesPendingStatus(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetRetiredPagesPendingStatus called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t*isPending;
    rpc_read(client, &isPending, sizeof(isPending));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRetiredPagesPendingStatus(device, isPending);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetRemappedRows(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetRemappedRows called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*corrRows;
    rpc_read(client, &corrRows, sizeof(corrRows));
    unsigned int*uncRows;
    rpc_read(client, &uncRows, sizeof(uncRows));
    unsigned int*isPending;
    rpc_read(client, &isPending, sizeof(isPending));
    unsigned int*failureOccurred;
    rpc_read(client, &failureOccurred, sizeof(failureOccurred));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRemappedRows(device, corrRows, uncRows, isPending, failureOccurred);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetRowRemapperHistogram(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetRowRemapperHistogram called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlRowRemapperHistogramValues_t*values;
    rpc_read(client, &values, sizeof(values));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRowRemapperHistogram(device, values);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetArchitecture(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetArchitecture called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlDeviceArchitecture_t*arch;
    rpc_read(client, &arch, sizeof(arch));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetArchitecture(device, arch);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlUnitSetLedState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlUnitSetLedState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlUnit_t unit;
    rpc_read(client, &unit, sizeof(unit));
    nvmlLedColor_t color;
    rpc_read(client, &color, sizeof(color));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitSetLedState(unit, color);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetPersistenceMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetPersistenceMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t mode;
    rpc_read(client, &mode, sizeof(mode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetPersistenceMode(device, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetComputeMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetComputeMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlComputeMode_t mode;
    rpc_read(client, &mode, sizeof(mode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetComputeMode(device, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetEccMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetEccMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t ecc;
    rpc_read(client, &ecc, sizeof(ecc));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetEccMode(device, ecc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceClearEccErrorCounts(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceClearEccErrorCounts called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEccCounterType_t counterType;
    rpc_read(client, &counterType, sizeof(counterType));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceClearEccErrorCounts(device, counterType);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetDriverModel(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetDriverModel called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlDriverModel_t driverModel;
    rpc_read(client, &driverModel, sizeof(driverModel));
    unsigned int flags;
    rpc_read(client, &flags, sizeof(flags));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetDriverModel(device, driverModel, flags);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetGpuLockedClocks(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetGpuLockedClocks called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int minGpuClockMHz;
    rpc_read(client, &minGpuClockMHz, sizeof(minGpuClockMHz));
    unsigned int maxGpuClockMHz;
    rpc_read(client, &maxGpuClockMHz, sizeof(maxGpuClockMHz));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetGpuLockedClocks(device, minGpuClockMHz, maxGpuClockMHz);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceResetGpuLockedClocks(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceResetGpuLockedClocks called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceResetGpuLockedClocks(device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetMemoryLockedClocks(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetMemoryLockedClocks called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int minMemClockMHz;
    rpc_read(client, &minMemClockMHz, sizeof(minMemClockMHz));
    unsigned int maxMemClockMHz;
    rpc_read(client, &maxMemClockMHz, sizeof(maxMemClockMHz));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetMemoryLockedClocks(device, minMemClockMHz, maxMemClockMHz);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceResetMemoryLockedClocks(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceResetMemoryLockedClocks called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceResetMemoryLockedClocks(device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetApplicationsClocks(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetApplicationsClocks called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int memClockMHz;
    rpc_read(client, &memClockMHz, sizeof(memClockMHz));
    unsigned int graphicsClockMHz;
    rpc_read(client, &graphicsClockMHz, sizeof(graphicsClockMHz));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetApplicationsClocks(device, memClockMHz, graphicsClockMHz);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetClkMonStatus(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetClkMonStatus called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlClkMonStatus_t*status;
    rpc_read(client, &status, sizeof(status));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetClkMonStatus(device, status);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetPowerManagementLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetPowerManagementLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int limit;
    rpc_read(client, &limit, sizeof(limit));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetPowerManagementLimit(device, limit);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetGpuOperationMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetGpuOperationMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlGpuOperationMode_t mode;
    rpc_read(client, &mode, sizeof(mode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetGpuOperationMode(device, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetAPIRestriction(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetAPIRestriction called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlRestrictedAPI_t apiType;
    rpc_read(client, &apiType, sizeof(apiType));
    nvmlEnableState_t isRestricted;
    rpc_read(client, &isRestricted, sizeof(isRestricted));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetAPIRestriction(device, apiType, isRestricted);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetAccountingMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetAccountingMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t mode;
    rpc_read(client, &mode, sizeof(mode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetAccountingMode(device, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceClearAccountingPids(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceClearAccountingPids called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceClearAccountingPids(device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetNvLinkState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetNvLinkState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int link;
    rpc_read(client, &link, sizeof(link));
    nvmlEnableState_t*isActive;
    rpc_read(client, &isActive, sizeof(isActive));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkState(device, link, isActive);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetNvLinkVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetNvLinkVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int link;
    rpc_read(client, &link, sizeof(link));
    unsigned int*version;
    rpc_read(client, &version, sizeof(version));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkVersion(device, link, version);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetNvLinkCapability(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetNvLinkCapability called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int link;
    rpc_read(client, &link, sizeof(link));
    nvmlNvLinkCapability_t capability;
    rpc_read(client, &capability, sizeof(capability));
    unsigned int*capResult;
    rpc_read(client, &capResult, sizeof(capResult));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkCapability(device, link, capability, capResult);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetNvLinkRemotePciInfo_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetNvLinkRemotePciInfo_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int link;
    rpc_read(client, &link, sizeof(link));
    nvmlPciInfo_t*pci;
    rpc_read(client, &pci, sizeof(pci));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, pci);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetNvLinkErrorCounter(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetNvLinkErrorCounter called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int link;
    rpc_read(client, &link, sizeof(link));
    nvmlNvLinkErrorCounter_t counter;
    rpc_read(client, &counter, sizeof(counter));
    unsigned long long*counterValue;
    rpc_read(client, &counterValue, sizeof(counterValue));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkErrorCounter(device, link, counter, counterValue);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceResetNvLinkErrorCounters(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceResetNvLinkErrorCounters called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int link;
    rpc_read(client, &link, sizeof(link));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceResetNvLinkErrorCounters(device, link);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetNvLinkUtilizationControl(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetNvLinkUtilizationControl called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int link;
    rpc_read(client, &link, sizeof(link));
    unsigned int counter;
    rpc_read(client, &counter, sizeof(counter));
    nvmlNvLinkUtilizationControl_t*control;
    rpc_read(client, &control, sizeof(control));
    unsigned int reset;
    rpc_read(client, &reset, sizeof(reset));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetNvLinkUtilizationControl(device, link, counter, control, reset);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetNvLinkUtilizationControl(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetNvLinkUtilizationControl called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int link;
    rpc_read(client, &link, sizeof(link));
    unsigned int counter;
    rpc_read(client, &counter, sizeof(counter));
    nvmlNvLinkUtilizationControl_t*control;
    rpc_read(client, &control, sizeof(control));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkUtilizationControl(device, link, counter, control);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetNvLinkUtilizationCounter(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetNvLinkUtilizationCounter called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int link;
    rpc_read(client, &link, sizeof(link));
    unsigned int counter;
    rpc_read(client, &counter, sizeof(counter));
    unsigned long long*rxcounter;
    rpc_read(client, &rxcounter, sizeof(rxcounter));
    unsigned long long*txcounter;
    rpc_read(client, &txcounter, sizeof(txcounter));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter, rxcounter, txcounter);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceFreezeNvLinkUtilizationCounter(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceFreezeNvLinkUtilizationCounter called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int link;
    rpc_read(client, &link, sizeof(link));
    unsigned int counter;
    rpc_read(client, &counter, sizeof(counter));
    nvmlEnableState_t freeze;
    rpc_read(client, &freeze, sizeof(freeze));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceFreezeNvLinkUtilizationCounter(device, link, counter, freeze);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceResetNvLinkUtilizationCounter(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceResetNvLinkUtilizationCounter called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int link;
    rpc_read(client, &link, sizeof(link));
    unsigned int counter;
    rpc_read(client, &counter, sizeof(counter));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceResetNvLinkUtilizationCounter(device, link, counter);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetNvLinkRemoteDeviceType(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetNvLinkRemoteDeviceType called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int link;
    rpc_read(client, &link, sizeof(link));
    nvmlIntNvLinkDeviceType_t*pNvLinkDeviceType;
    rpc_read(client, &pNvLinkDeviceType, sizeof(pNvLinkDeviceType));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkRemoteDeviceType(device, link, pNvLinkDeviceType);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlEventSetCreate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlEventSetCreate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlEventSet_t*set;
    rpc_read(client, &set, sizeof(set));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlEventSetCreate(set);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceRegisterEvents(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceRegisterEvents called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned long long eventTypes;
    rpc_read(client, &eventTypes, sizeof(eventTypes));
    nvmlEventSet_t set;
    rpc_read(client, &set, sizeof(set));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceRegisterEvents(device, eventTypes, set);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetSupportedEventTypes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetSupportedEventTypes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned long long*eventTypes;
    rpc_read(client, &eventTypes, sizeof(eventTypes));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedEventTypes(device, eventTypes);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlEventSetWait_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlEventSetWait_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlEventSet_t set;
    rpc_read(client, &set, sizeof(set));
    nvmlEventData_t*data;
    rpc_read(client, &data, sizeof(data));
    unsigned int timeoutms;
    rpc_read(client, &timeoutms, sizeof(timeoutms));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlEventSetWait_v2(set, data, timeoutms);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlEventSetFree(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlEventSetFree called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlEventSet_t set;
    rpc_read(client, &set, sizeof(set));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlEventSetFree(set);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceModifyDrainState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceModifyDrainState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlPciInfo_t*pciInfo;
    rpc_read(client, &pciInfo, sizeof(pciInfo));
    nvmlEnableState_t newState;
    rpc_read(client, &newState, sizeof(newState));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceModifyDrainState(pciInfo, newState);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceQueryDrainState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceQueryDrainState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlPciInfo_t*pciInfo;
    rpc_read(client, &pciInfo, sizeof(pciInfo));
    nvmlEnableState_t*currentState;
    rpc_read(client, &currentState, sizeof(currentState));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceQueryDrainState(pciInfo, currentState);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceRemoveGpu_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceRemoveGpu_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlPciInfo_t*pciInfo;
    rpc_read(client, &pciInfo, sizeof(pciInfo));
    nvmlDetachGpuState_t gpuState;
    rpc_read(client, &gpuState, sizeof(gpuState));
    nvmlPcieLinkState_t linkState;
    rpc_read(client, &linkState, sizeof(linkState));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceRemoveGpu_v2(pciInfo, gpuState, linkState);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceDiscoverGpus(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceDiscoverGpus called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlPciInfo_t*pciInfo;
    rpc_read(client, &pciInfo, sizeof(pciInfo));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceDiscoverGpus(pciInfo);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetFieldValues(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetFieldValues called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    int valuesCount;
    rpc_read(client, &valuesCount, sizeof(valuesCount));
    nvmlFieldValue_t*values;
    rpc_read(client, &values, sizeof(values));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFieldValues(device, valuesCount, values);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetVirtualizationMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVirtualizationMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlGpuVirtualizationMode_t*pVirtualMode;
    rpc_read(client, &pVirtualMode, sizeof(pVirtualMode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVirtualizationMode(device, pVirtualMode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetHostVgpuMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetHostVgpuMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlHostVgpuMode_t*pHostVgpuMode;
    rpc_read(client, &pHostVgpuMode, sizeof(pHostVgpuMode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetHostVgpuMode(device, pHostVgpuMode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetVirtualizationMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetVirtualizationMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlGpuVirtualizationMode_t virtualMode;
    rpc_read(client, &virtualMode, sizeof(virtualMode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetVirtualizationMode(device, virtualMode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGridLicensableFeatures_v3(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGridLicensableFeatures_v3 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlGridLicensableFeatures_t*pGridLicensableFeatures;
    rpc_read(client, &pGridLicensableFeatures, sizeof(pGridLicensableFeatures));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGridLicensableFeatures_v3(device, pGridLicensableFeatures);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetProcessUtilization(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetProcessUtilization called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlProcessUtilizationSample_t*utilization;
    rpc_read(client, &utilization, sizeof(utilization));
    unsigned int*processSamplesCount;
    rpc_read(client, &processSamplesCount, sizeof(processSamplesCount));
    unsigned long long lastSeenTimeStamp;
    rpc_read(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetProcessUtilization(device, utilization, processSamplesCount, lastSeenTimeStamp);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetSupportedVgpus(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetSupportedVgpus called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*vgpuCount;
    rpc_read(client, &vgpuCount, sizeof(vgpuCount));
    nvmlVgpuTypeId_t*vgpuTypeIds;
    rpc_read(client, &vgpuTypeIds, sizeof(vgpuTypeIds));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedVgpus(device, vgpuCount, vgpuTypeIds);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetCreatableVgpus(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetCreatableVgpus called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*vgpuCount;
    rpc_read(client, &vgpuCount, sizeof(vgpuCount));
    nvmlVgpuTypeId_t*vgpuTypeIds;
    rpc_read(client, &vgpuTypeIds, sizeof(vgpuTypeIds));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCreatableVgpus(device, vgpuCount, vgpuTypeIds);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetClass(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetClass called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    char vgpuTypeClass[1024];
    unsigned int*size;
    rpc_read(client, &size, sizeof(size));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetClass(vgpuTypeId, vgpuTypeClass, size);
    rpc_write(client, vgpuTypeClass, strlen(vgpuTypeClass) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetName(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetName called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    char vgpuTypeName[1024];
    unsigned int*size;
    rpc_read(client, &size, sizeof(size));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetName(vgpuTypeId, vgpuTypeName, size);
    rpc_write(client, vgpuTypeName, strlen(vgpuTypeName) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetGpuInstanceProfileId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetGpuInstanceProfileId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    unsigned int*gpuInstanceProfileId;
    rpc_read(client, &gpuInstanceProfileId, sizeof(gpuInstanceProfileId));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId, gpuInstanceProfileId);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetDeviceID(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetDeviceID called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    unsigned long long*deviceID;
    rpc_read(client, &deviceID, sizeof(deviceID));
    unsigned long long*subsystemID;
    rpc_read(client, &subsystemID, sizeof(subsystemID));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetDeviceID(vgpuTypeId, deviceID, subsystemID);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetFramebufferSize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetFramebufferSize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    unsigned long long*fbSize;
    rpc_read(client, &fbSize, sizeof(fbSize));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetFramebufferSize(vgpuTypeId, fbSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetNumDisplayHeads(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetNumDisplayHeads called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    unsigned int*numDisplayHeads;
    rpc_read(client, &numDisplayHeads, sizeof(numDisplayHeads));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId, numDisplayHeads);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetResolution(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetResolution called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    unsigned int displayIndex;
    rpc_read(client, &displayIndex, sizeof(displayIndex));
    unsigned int*xdim;
    rpc_read(client, &xdim, sizeof(xdim));
    unsigned int*ydim;
    rpc_read(client, &ydim, sizeof(ydim));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetResolution(vgpuTypeId, displayIndex, xdim, ydim);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetLicense(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetLicense called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    char vgpuTypeLicenseString[1024];
    unsigned int size;
    rpc_read(client, &size, sizeof(size));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetLicense(vgpuTypeId, vgpuTypeLicenseString, size);
    rpc_write(client, vgpuTypeLicenseString, strlen(vgpuTypeLicenseString) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetFrameRateLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetFrameRateLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    unsigned int*frameRateLimit;
    rpc_read(client, &frameRateLimit, sizeof(frameRateLimit));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId, frameRateLimit);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetMaxInstances(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetMaxInstances called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    unsigned int*vgpuInstanceCount;
    rpc_read(client, &vgpuInstanceCount, sizeof(vgpuInstanceCount));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetMaxInstances(device, vgpuTypeId, vgpuInstanceCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetMaxInstancesPerVm(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetMaxInstancesPerVm called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    unsigned int*vgpuInstanceCountPerVm;
    rpc_read(client, &vgpuInstanceCountPerVm, sizeof(vgpuInstanceCountPerVm));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId, vgpuInstanceCountPerVm);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetActiveVgpus(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetActiveVgpus called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*vgpuCount;
    rpc_read(client, &vgpuCount, sizeof(vgpuCount));
    nvmlVgpuInstance_t*vgpuInstances;
    rpc_read(client, &vgpuInstances, sizeof(vgpuInstances));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetActiveVgpus(device, vgpuCount, vgpuInstances);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetVmID(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetVmID called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    char vmId[1024];
    unsigned int size;
    rpc_read(client, &size, sizeof(size));
    nvmlVgpuVmIdType_t*vmIdType;
    rpc_read(client, &vmIdType, sizeof(vmIdType));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetVmID(vgpuInstance, vmId, size, vmIdType);
    rpc_write(client, vmId, strlen(vmId) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetUUID(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetUUID called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    char uuid[1024];
    unsigned int size;
    rpc_read(client, &size, sizeof(size));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetUUID(vgpuInstance, uuid, size);
    rpc_write(client, uuid, strlen(uuid) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetVmDriverVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetVmDriverVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    char version[1024];
    unsigned int length;
    rpc_read(client, &length, sizeof(length));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance, version, length);
    rpc_write(client, version, strlen(version) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetFbUsage(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetFbUsage called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    unsigned long long*fbUsage;
    rpc_read(client, &fbUsage, sizeof(fbUsage));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetFbUsage(vgpuInstance, fbUsage);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetLicenseStatus(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetLicenseStatus called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    unsigned int*licensed;
    rpc_read(client, &licensed, sizeof(licensed));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetLicenseStatus(vgpuInstance, licensed);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetType(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetType called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    nvmlVgpuTypeId_t*vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetType(vgpuInstance, vgpuTypeId);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetFrameRateLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetFrameRateLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    unsigned int*frameRateLimit;
    rpc_read(client, &frameRateLimit, sizeof(frameRateLimit));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance, frameRateLimit);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetEccMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetEccMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    nvmlEnableState_t*eccMode;
    rpc_read(client, &eccMode, sizeof(eccMode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetEccMode(vgpuInstance, eccMode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetEncoderCapacity(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetEncoderCapacity called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    unsigned int*encoderCapacity;
    rpc_read(client, &encoderCapacity, sizeof(encoderCapacity));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance, encoderCapacity);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceSetEncoderCapacity(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceSetEncoderCapacity called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    unsigned int encoderCapacity;
    rpc_read(client, &encoderCapacity, sizeof(encoderCapacity));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, encoderCapacity);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetEncoderStats(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetEncoderStats called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    unsigned int*sessionCount;
    rpc_read(client, &sessionCount, sizeof(sessionCount));
    unsigned int*averageFps;
    rpc_read(client, &averageFps, sizeof(averageFps));
    unsigned int*averageLatency;
    rpc_read(client, &averageLatency, sizeof(averageLatency));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetEncoderStats(vgpuInstance, sessionCount, averageFps, averageLatency);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetEncoderSessions(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetEncoderSessions called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    unsigned int*sessionCount;
    rpc_read(client, &sessionCount, sizeof(sessionCount));
    nvmlEncoderSessionInfo_t*sessionInfo;
    rpc_read(client, &sessionInfo, sizeof(sessionInfo));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetEncoderSessions(vgpuInstance, sessionCount, sessionInfo);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetFBCStats(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetFBCStats called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    nvmlFBCStats_t*fbcStats;
    rpc_read(client, &fbcStats, sizeof(fbcStats));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetFBCStats(vgpuInstance, fbcStats);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetFBCSessions(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetFBCSessions called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    unsigned int*sessionCount;
    rpc_read(client, &sessionCount, sizeof(sessionCount));
    nvmlFBCSessionInfo_t*sessionInfo;
    rpc_read(client, &sessionInfo, sizeof(sessionInfo));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetFBCSessions(vgpuInstance, sessionCount, sessionInfo);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetGpuInstanceId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetGpuInstanceId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    unsigned int*gpuInstanceId;
    rpc_read(client, &gpuInstanceId, sizeof(gpuInstanceId));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance, gpuInstanceId);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetMetadata(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetMetadata called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    nvmlVgpuMetadata_t*vgpuMetadata;
    rpc_read(client, &vgpuMetadata, sizeof(vgpuMetadata));
    unsigned int*bufferSize;
    rpc_read(client, &bufferSize, sizeof(bufferSize));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetMetadata(vgpuInstance, vgpuMetadata, bufferSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetVgpuMetadata(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVgpuMetadata called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlVgpuPgpuMetadata_t*pgpuMetadata;
    rpc_read(client, &pgpuMetadata, sizeof(pgpuMetadata));
    unsigned int*bufferSize;
    rpc_read(client, &bufferSize, sizeof(bufferSize));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuMetadata(device, pgpuMetadata, bufferSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGetVgpuCompatibility(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGetVgpuCompatibility called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuMetadata_t*vgpuMetadata;
    rpc_read(client, &vgpuMetadata, sizeof(vgpuMetadata));
    nvmlVgpuPgpuMetadata_t*pgpuMetadata;
    rpc_read(client, &pgpuMetadata, sizeof(pgpuMetadata));
    nvmlVgpuPgpuCompatibility_t*compatibilityInfo;
    rpc_read(client, &compatibilityInfo, sizeof(compatibilityInfo));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGetVgpuCompatibility(vgpuMetadata, pgpuMetadata, compatibilityInfo);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPgpuMetadataString(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPgpuMetadataString called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    char pgpuMetadata[1024];
    unsigned int*bufferSize;
    rpc_read(client, &bufferSize, sizeof(bufferSize));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPgpuMetadataString(device, pgpuMetadata, bufferSize);
    rpc_write(client, pgpuMetadata, strlen(pgpuMetadata) + 1, true);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGetVgpuVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGetVgpuVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuVersion_t*supported;
    rpc_read(client, &supported, sizeof(supported));
    nvmlVgpuVersion_t*current;
    rpc_read(client, &current, sizeof(current));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGetVgpuVersion(supported, current);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSetVgpuVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSetVgpuVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuVersion_t*vgpuVersion;
    rpc_read(client, &vgpuVersion, sizeof(vgpuVersion));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSetVgpuVersion(vgpuVersion);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetVgpuUtilization(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVgpuUtilization called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned long long lastSeenTimeStamp;
    rpc_read(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    nvmlValueType_t*sampleValType;
    rpc_read(client, &sampleValType, sizeof(sampleValType));
    unsigned int*vgpuInstanceSamplesCount;
    rpc_read(client, &vgpuInstanceSamplesCount, sizeof(vgpuInstanceSamplesCount));
    nvmlVgpuInstanceUtilizationSample_t*utilizationSamples;
    rpc_read(client, &utilizationSamples, sizeof(utilizationSamples));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuUtilization(device, lastSeenTimeStamp, sampleValType, vgpuInstanceSamplesCount, utilizationSamples);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetVgpuProcessUtilization(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVgpuProcessUtilization called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned long long lastSeenTimeStamp;
    rpc_read(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    unsigned int*vgpuProcessSamplesCount;
    rpc_read(client, &vgpuProcessSamplesCount, sizeof(vgpuProcessSamplesCount));
    nvmlVgpuProcessUtilizationSample_t*utilizationSamples;
    rpc_read(client, &utilizationSamples, sizeof(utilizationSamples));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuProcessUtilization(device, lastSeenTimeStamp, vgpuProcessSamplesCount, utilizationSamples);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetAccountingMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetAccountingMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    nvmlEnableState_t*mode;
    rpc_read(client, &mode, sizeof(mode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetAccountingMode(vgpuInstance, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetAccountingPids(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetAccountingPids called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    unsigned int*pids;
    rpc_read(client, &pids, sizeof(pids));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetAccountingPids(vgpuInstance, count, pids);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetAccountingStats(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetAccountingStats called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    unsigned int pid;
    rpc_read(client, &pid, sizeof(pid));
    nvmlAccountingStats_t*stats;
    rpc_read(client, &stats, sizeof(stats));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid, stats);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceClearAccountingPids(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceClearAccountingPids called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceClearAccountingPids(vgpuInstance);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGetExcludedDeviceCount(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGetExcludedDeviceCount called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int*deviceCount;
    rpc_read(client, &deviceCount, sizeof(deviceCount));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGetExcludedDeviceCount(deviceCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGetExcludedDeviceInfoByIndex(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGetExcludedDeviceInfoByIndex called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int index;
    rpc_read(client, &index, sizeof(index));
    nvmlExcludedDeviceInfo_t*info;
    rpc_read(client, &info, sizeof(info));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGetExcludedDeviceInfoByIndex(index, info);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetMigMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetMigMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int mode;
    rpc_read(client, &mode, sizeof(mode));
    nvmlReturn_t*activationStatus;
    rpc_read(client, &activationStatus, sizeof(activationStatus));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetMigMode(device, mode, activationStatus);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMigMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMigMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*currentMode;
    rpc_read(client, &currentMode, sizeof(currentMode));
    unsigned int*pendingMode;
    rpc_read(client, &pendingMode, sizeof(pendingMode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMigMode(device, currentMode, pendingMode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGpuInstanceProfileInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGpuInstanceProfileInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int profile;
    rpc_read(client, &profile, sizeof(profile));
    nvmlGpuInstanceProfileInfo_t*info;
    rpc_read(client, &info, sizeof(info));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstanceProfileInfo(device, profile, info);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGpuInstancePossiblePlacements_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGpuInstancePossiblePlacements_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int profileId;
    rpc_read(client, &profileId, sizeof(profileId));
    nvmlGpuInstancePlacement_t*placements;
    rpc_read(client, &placements, sizeof(placements));
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstancePossiblePlacements_v2(device, profileId, placements, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGpuInstanceRemainingCapacity(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGpuInstanceRemainingCapacity called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int profileId;
    rpc_read(client, &profileId, sizeof(profileId));
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstanceRemainingCapacity(device, profileId, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceCreateGpuInstance(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceCreateGpuInstance called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int profileId;
    rpc_read(client, &profileId, sizeof(profileId));
    nvmlGpuInstance_t*gpuInstance;
    rpc_read(client, &gpuInstance, sizeof(gpuInstance));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceCreateGpuInstance(device, profileId, gpuInstance);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceCreateGpuInstanceWithPlacement(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceCreateGpuInstanceWithPlacement called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int profileId;
    rpc_read(client, &profileId, sizeof(profileId));
    nvmlGpuInstancePlacement_t *placement;
    rpc_read(client, &placement, sizeof(placement));
    nvmlGpuInstance_t*gpuInstance;
    rpc_read(client, &gpuInstance, sizeof(gpuInstance));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, placement, gpuInstance);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpuInstanceDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpuInstanceDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlGpuInstance_t gpuInstance;
    rpc_read(client, &gpuInstance, sizeof(gpuInstance));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceDestroy(gpuInstance);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGpuInstances(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGpuInstances called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int profileId;
    rpc_read(client, &profileId, sizeof(profileId));
    nvmlGpuInstance_t*gpuInstances;
    rpc_read(client, &gpuInstances, sizeof(gpuInstances));
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstances(device, profileId, gpuInstances, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGpuInstanceById(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGpuInstanceById called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int id;
    rpc_read(client, &id, sizeof(id));
    nvmlGpuInstance_t*gpuInstance;
    rpc_read(client, &gpuInstance, sizeof(gpuInstance));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstanceById(device, id, gpuInstance);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpuInstanceGetInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpuInstanceGetInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlGpuInstance_t gpuInstance;
    rpc_read(client, &gpuInstance, sizeof(gpuInstance));
    nvmlGpuInstanceInfo_t*info;
    rpc_read(client, &info, sizeof(info));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetInfo(gpuInstance, info);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpuInstanceGetComputeInstanceProfileInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpuInstanceGetComputeInstanceProfileInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlGpuInstance_t gpuInstance;
    rpc_read(client, &gpuInstance, sizeof(gpuInstance));
    unsigned int profile;
    rpc_read(client, &profile, sizeof(profile));
    unsigned int engProfile;
    rpc_read(client, &engProfile, sizeof(engProfile));
    nvmlComputeInstanceProfileInfo_t*info;
    rpc_read(client, &info, sizeof(info));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstanceProfileInfo(gpuInstance, profile, engProfile, info);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpuInstanceGetComputeInstanceRemainingCapacity(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpuInstanceGetComputeInstanceRemainingCapacity called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlGpuInstance_t gpuInstance;
    rpc_read(client, &gpuInstance, sizeof(gpuInstance));
    unsigned int profileId;
    rpc_read(client, &profileId, sizeof(profileId));
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, profileId, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpuInstanceCreateComputeInstance(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpuInstanceCreateComputeInstance called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlGpuInstance_t gpuInstance;
    rpc_read(client, &gpuInstance, sizeof(gpuInstance));
    unsigned int profileId;
    rpc_read(client, &profileId, sizeof(profileId));
    nvmlComputeInstance_t*computeInstance;
    rpc_read(client, &computeInstance, sizeof(computeInstance));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId, computeInstance);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlComputeInstanceDestroy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlComputeInstanceDestroy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlComputeInstance_t computeInstance;
    rpc_read(client, &computeInstance, sizeof(computeInstance));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlComputeInstanceDestroy(computeInstance);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpuInstanceGetComputeInstances(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpuInstanceGetComputeInstances called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlGpuInstance_t gpuInstance;
    rpc_read(client, &gpuInstance, sizeof(gpuInstance));
    unsigned int profileId;
    rpc_read(client, &profileId, sizeof(profileId));
    nvmlComputeInstance_t*computeInstances;
    rpc_read(client, &computeInstances, sizeof(computeInstances));
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId, computeInstances, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpuInstanceGetComputeInstanceById(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpuInstanceGetComputeInstanceById called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlGpuInstance_t gpuInstance;
    rpc_read(client, &gpuInstance, sizeof(gpuInstance));
    unsigned int id;
    rpc_read(client, &id, sizeof(id));
    nvmlComputeInstance_t*computeInstance;
    rpc_read(client, &computeInstance, sizeof(computeInstance));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstanceById(gpuInstance, id, computeInstance);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlComputeInstanceGetInfo_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlComputeInstanceGetInfo_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlComputeInstance_t computeInstance;
    rpc_read(client, &computeInstance, sizeof(computeInstance));
    nvmlComputeInstanceInfo_t*info;
    rpc_read(client, &info, sizeof(info));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlComputeInstanceGetInfo_v2(computeInstance, info);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceIsMigDeviceHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceIsMigDeviceHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*isMigDevice;
    rpc_read(client, &isMigDevice, sizeof(isMigDevice));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceIsMigDeviceHandle(device, isMigDevice);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGpuInstanceId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGpuInstanceId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*id;
    rpc_read(client, &id, sizeof(id));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstanceId(device, id);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetComputeInstanceId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetComputeInstanceId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*id;
    rpc_read(client, &id, sizeof(id));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetComputeInstanceId(device, id);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMaxMigDeviceCount(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMaxMigDeviceCount called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int*count;
    rpc_read(client, &count, sizeof(count));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxMigDeviceCount(device, count);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMigDeviceHandleByIndex(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMigDeviceHandleByIndex called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int index;
    rpc_read(client, &index, sizeof(index));
    nvmlDevice_t*migDevice;
    rpc_read(client, &migDevice, sizeof(migDevice));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMigDeviceHandleByIndex(device, index, migDevice);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetDeviceHandleFromMigDeviceHandle(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetDeviceHandleFromMigDeviceHandle called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t migDevice;
    rpc_read(client, &migDevice, sizeof(migDevice));
    nvmlDevice_t*device;
    rpc_read(client, &device, sizeof(device));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice, device);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

