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
    int cudaDriverVersion;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetCudaDriverVersion(&cudaDriverVersion);
    rpc_write(client, &cudaDriverVersion, sizeof(cudaDriverVersion));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    int cudaDriverVersion;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetCudaDriverVersion_v2(&cudaDriverVersion);
    rpc_write(client, &cudaDriverVersion, sizeof(cudaDriverVersion));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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

int handle_nvmlSystemGetHicVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetHicVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int hwbcCount;
    nvmlHwbcEntry_t hwbcEntries;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetHicVersion(&hwbcCount, &hwbcEntries);
    rpc_write(client, &hwbcCount, sizeof(hwbcCount));
    rpc_write(client, &hwbcEntries, sizeof(hwbcEntries));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int count;
    nvmlDevice_t deviceArray;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetTopologyGpuSet(cpuNumber, &count, &deviceArray);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &deviceArray, sizeof(deviceArray));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetDriverBranch(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetDriverBranch called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlSystemDriverBranchInfo_t branchInfo;
    unsigned int length;
    rpc_read(client, &length, sizeof(length));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetDriverBranch(&branchInfo, length);
    rpc_write(client, &branchInfo, sizeof(branchInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int unitCount;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetCount(&unitCount);
    rpc_write(client, &unitCount, sizeof(unitCount));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlUnit_t unit;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetHandleByIndex(index, &unit);
    rpc_write(client, &unit, sizeof(unit));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlUnitInfo_t info;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetUnitInfo(unit, &info);
    rpc_write(client, &info, sizeof(info));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlLedState_t state;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetLedState(unit, &state);
    rpc_write(client, &state, sizeof(state));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlPSUInfo_t psu;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetPsuInfo(unit, &psu);
    rpc_write(client, &psu, sizeof(psu));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int temp;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetTemperature(unit, type, &temp);
    rpc_write(client, &temp, sizeof(temp));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlUnitFanSpeeds_t fanSpeeds;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetFanSpeedInfo(unit, &fanSpeeds);
    rpc_write(client, &fanSpeeds, sizeof(fanSpeeds));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int deviceCount;
    nvmlDevice_t devices;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetDevices(unit, &deviceCount, &devices);
    rpc_write(client, &deviceCount, sizeof(deviceCount));
    rpc_write(client, &devices, sizeof(devices));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int deviceCount;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCount_v2(&deviceCount);
    rpc_write(client, &deviceCount, sizeof(deviceCount));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlDeviceAttributes_t attributes;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAttributes_v2(device, &attributes);
    rpc_write(client, &attributes, sizeof(attributes));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlDevice_t device;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetHandleByIndex_v2(index, &device);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlDevice_t device;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(serial);
    _result = nvmlDeviceGetHandleBySerial(serial, &device);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlDevice_t device;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(uuid);
    _result = nvmlDeviceGetHandleByUUID(uuid, &device);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlDevice_t device;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(pciBusId);
    _result = nvmlDeviceGetHandleByPciBusId_v2(pciBusId, &device);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlBrandType_t type;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBrand(device, &type);
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int index;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetIndex(device, &index);
    rpc_write(client, &index, sizeof(index));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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

int handle_nvmlDeviceGetModuleId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetModuleId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int moduleId;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetModuleId(device, &moduleId);
    rpc_write(client, &moduleId, sizeof(moduleId));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetC2cModeInfoV(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetC2cModeInfoV called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlC2cModeInfo_v1_t c2cModeInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetC2cModeInfoV(device, &c2cModeInfo);
    rpc_write(client, &c2cModeInfo, sizeof(c2cModeInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long nodeSet;
    nvmlAffinityScope_t scope;
    rpc_read(client, &scope, sizeof(scope));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMemoryAffinity(device, nodeSetSize, &nodeSet, scope);
    rpc_write(client, &nodeSet, sizeof(nodeSet));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long cpuSet;
    nvmlAffinityScope_t scope;
    rpc_read(client, &scope, sizeof(scope));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCpuAffinityWithinScope(device, cpuSetSize, &cpuSet, scope);
    rpc_write(client, &cpuSet, sizeof(cpuSet));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long cpuSet;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCpuAffinity(device, cpuSetSize, &cpuSet);
    rpc_write(client, &cpuSet, sizeof(cpuSet));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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

int handle_nvmlDeviceGetNumaNodeId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetNumaNodeId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int node;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNumaNodeId(device, &node);
    rpc_write(client, &node, sizeof(node));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlGpuTopologyLevel_t pathInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTopologyCommonAncestor(device1, device2, &pathInfo);
    rpc_write(client, &pathInfo, sizeof(pathInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int count;
    nvmlDevice_t deviceArray;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTopologyNearestGpus(device, level, &count, &deviceArray);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &deviceArray, sizeof(deviceArray));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlGpuP2PStatus_t p2pStatus;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetP2PStatus(device1, device2, p2pIndex, &p2pStatus);
    rpc_write(client, &p2pStatus, sizeof(p2pStatus));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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

int handle_nvmlDeviceGetMinorNumber(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMinorNumber called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int minorNumber;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMinorNumber(device, &minorNumber);
    rpc_write(client, &minorNumber, sizeof(minorNumber));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int checksum;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetInforomConfigurationChecksum(device, &checksum);
    rpc_write(client, &checksum, sizeof(checksum));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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

int handle_nvmlDeviceGetLastBBXFlushTime(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetLastBBXFlushTime called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned long long timestamp;
    unsigned long durationUs;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetLastBBXFlushTime(device, &timestamp, &durationUs);
    rpc_write(client, &timestamp, sizeof(timestamp));
    rpc_write(client, &durationUs, sizeof(durationUs));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEnableState_t display;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDisplayMode(device, &display);
    rpc_write(client, &display, sizeof(display));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEnableState_t isActive;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDisplayActive(device, &isActive);
    rpc_write(client, &isActive, sizeof(isActive));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEnableState_t mode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPersistenceMode(device, &mode);
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPciInfoExt(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPciInfoExt called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPciInfoExt_t pci;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPciInfoExt(device, &pci);
    rpc_write(client, &pci, sizeof(pci));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlPciInfo_t pci;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPciInfo_v3(device, &pci);
    rpc_write(client, &pci, sizeof(pci));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int maxLinkGen;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxPcieLinkGeneration(device, &maxLinkGen);
    rpc_write(client, &maxLinkGen, sizeof(maxLinkGen));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGpuMaxPcieLinkGeneration(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGpuMaxPcieLinkGeneration called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int maxLinkGenDevice;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuMaxPcieLinkGeneration(device, &maxLinkGenDevice);
    rpc_write(client, &maxLinkGenDevice, sizeof(maxLinkGenDevice));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int maxLinkWidth;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxPcieLinkWidth(device, &maxLinkWidth);
    rpc_write(client, &maxLinkWidth, sizeof(maxLinkWidth));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int currLinkGen;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCurrPcieLinkGeneration(device, &currLinkGen);
    rpc_write(client, &currLinkGen, sizeof(currLinkGen));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int currLinkWidth;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCurrPcieLinkWidth(device, &currLinkWidth);
    rpc_write(client, &currLinkWidth, sizeof(currLinkWidth));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int value;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPcieThroughput(device, counter, &value);
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int value;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPcieReplayCounter(device, &value);
    rpc_write(client, &value, sizeof(value));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int clock;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetClockInfo(device, type, &clock);
    rpc_write(client, &clock, sizeof(clock));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int clock;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxClockInfo(device, type, &clock);
    rpc_write(client, &clock, sizeof(clock));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGpcClkVfOffset(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGpcClkVfOffset called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    int offset;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpcClkVfOffset(device, &offset);
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int clockMHz;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetApplicationsClock(device, clockType, &clockMHz);
    rpc_write(client, &clockMHz, sizeof(clockMHz));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int clockMHz;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDefaultApplicationsClock(device, clockType, &clockMHz);
    rpc_write(client, &clockMHz, sizeof(clockMHz));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int clockMHz;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetClock(device, clockType, clockId, &clockMHz);
    rpc_write(client, &clockMHz, sizeof(clockMHz));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int clockMHz;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxCustomerBoostClock(device, clockType, &clockMHz);
    rpc_write(client, &clockMHz, sizeof(clockMHz));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int count;
    unsigned int clocksMHz;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedMemoryClocks(device, &count, &clocksMHz);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &clocksMHz, sizeof(clocksMHz));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int count;
    unsigned int clocksMHz;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, &count, &clocksMHz);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &clocksMHz, sizeof(clocksMHz));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEnableState_t isEnabled;
    nvmlEnableState_t defaultIsEnabled;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAutoBoostedClocksEnabled(device, &isEnabled, &defaultIsEnabled);
    rpc_write(client, &isEnabled, sizeof(isEnabled));
    rpc_write(client, &defaultIsEnabled, sizeof(defaultIsEnabled));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int speed;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFanSpeed(device, &speed);
    rpc_write(client, &speed, sizeof(speed));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int speed;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFanSpeed_v2(device, fan, &speed);
    rpc_write(client, &speed, sizeof(speed));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetFanSpeedRPM(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetFanSpeedRPM called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlFanSpeedInfo_t fanSpeed;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFanSpeedRPM(device, &fanSpeed);
    rpc_write(client, &fanSpeed, sizeof(fanSpeed));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetTargetFanSpeed(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetTargetFanSpeed called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int fan;
    rpc_read(client, &fan, sizeof(fan));
    unsigned int targetSpeed;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTargetFanSpeed(device, fan, &targetSpeed);
    rpc_write(client, &targetSpeed, sizeof(targetSpeed));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMinMaxFanSpeed(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMinMaxFanSpeed called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int minSpeed;
    unsigned int maxSpeed;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMinMaxFanSpeed(device, &minSpeed, &maxSpeed);
    rpc_write(client, &minSpeed, sizeof(minSpeed));
    rpc_write(client, &maxSpeed, sizeof(maxSpeed));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetFanControlPolicy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetFanControlPolicy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int fan;
    rpc_read(client, &fan, sizeof(fan));
    nvmlFanControlPolicy_t policy;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFanControlPolicy_v2(device, fan, &policy);
    rpc_write(client, &policy, sizeof(policy));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetNumFans(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetNumFans called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int numFans;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNumFans(device, &numFans);
    rpc_write(client, &numFans, sizeof(numFans));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int temp;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTemperature(device, sensorType, &temp);
    rpc_write(client, &temp, sizeof(temp));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetCoolerInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetCoolerInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlCoolerInfo_t coolerInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCoolerInfo(device, &coolerInfo);
    rpc_write(client, &coolerInfo, sizeof(coolerInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetTemperatureV(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetTemperatureV called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlTemperature_t temperature;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTemperatureV(device, &temperature);
    rpc_write(client, &temperature, sizeof(temperature));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int temp;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTemperatureThreshold(device, thresholdType, &temp);
    rpc_write(client, &temp, sizeof(temp));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMarginTemperature(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMarginTemperature called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlMarginTemperature_t marginTempInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMarginTemperature(device, &marginTempInfo);
    rpc_write(client, &marginTempInfo, sizeof(marginTempInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetThermalSettings(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetThermalSettings called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int sensorIndex;
    rpc_read(client, &sensorIndex, sizeof(sensorIndex));
    nvmlGpuThermalSettings_t pThermalSettings;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetThermalSettings(device, sensorIndex, &pThermalSettings);
    rpc_write(client, &pThermalSettings, sizeof(pThermalSettings));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlPstates_t pState;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPerformanceState(device, &pState);
    rpc_write(client, &pState, sizeof(pState));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetCurrentClocksEventReasons(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetCurrentClocksEventReasons called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned long long clocksEventReasons;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCurrentClocksEventReasons(device, &clocksEventReasons);
    rpc_write(client, &clocksEventReasons, sizeof(clocksEventReasons));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long long clocksThrottleReasons;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCurrentClocksThrottleReasons(device, &clocksThrottleReasons);
    rpc_write(client, &clocksThrottleReasons, sizeof(clocksThrottleReasons));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetSupportedClocksEventReasons(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetSupportedClocksEventReasons called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned long long supportedClocksEventReasons;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedClocksEventReasons(device, &supportedClocksEventReasons);
    rpc_write(client, &supportedClocksEventReasons, sizeof(supportedClocksEventReasons));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long long supportedClocksThrottleReasons;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedClocksThrottleReasons(device, &supportedClocksThrottleReasons);
    rpc_write(client, &supportedClocksThrottleReasons, sizeof(supportedClocksThrottleReasons));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlPstates_t pState;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerState(device, &pState);
    rpc_write(client, &pState, sizeof(pState));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetDynamicPstatesInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetDynamicPstatesInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlGpuDynamicPstatesInfo_t pDynamicPstatesInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDynamicPstatesInfo(device, &pDynamicPstatesInfo);
    rpc_write(client, &pDynamicPstatesInfo, sizeof(pDynamicPstatesInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMemClkVfOffset(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMemClkVfOffset called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    int offset;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMemClkVfOffset(device, &offset);
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMinMaxClockOfPState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMinMaxClockOfPState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlClockType_t type;
    rpc_read(client, &type, sizeof(type));
    nvmlPstates_t pstate;
    rpc_read(client, &pstate, sizeof(pstate));
    unsigned int minClockMHz;
    unsigned int maxClockMHz;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, &minClockMHz, &maxClockMHz);
    rpc_write(client, &minClockMHz, sizeof(minClockMHz));
    rpc_write(client, &maxClockMHz, sizeof(maxClockMHz));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetSupportedPerformanceStates(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetSupportedPerformanceStates called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPstates_t pstates;
    unsigned int size;
    rpc_read(client, &size, sizeof(size));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedPerformanceStates(device, &pstates, size);
    rpc_write(client, &pstates, sizeof(pstates));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGpcClkMinMaxVfOffset(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGpcClkMinMaxVfOffset called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    int minOffset;
    int maxOffset;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpcClkMinMaxVfOffset(device, &minOffset, &maxOffset);
    rpc_write(client, &minOffset, sizeof(minOffset));
    rpc_write(client, &maxOffset, sizeof(maxOffset));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMemClkMinMaxVfOffset(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMemClkMinMaxVfOffset called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    int minOffset;
    int maxOffset;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMemClkMinMaxVfOffset(device, &minOffset, &maxOffset);
    rpc_write(client, &minOffset, sizeof(minOffset));
    rpc_write(client, &maxOffset, sizeof(maxOffset));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetClockOffsets(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetClockOffsets called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlClockOffset_t info;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetClockOffsets(device, &info);
    rpc_write(client, &info, sizeof(info));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetClockOffsets(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetClockOffsets called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlClockOffset_t info;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetClockOffsets(device, &info);
    rpc_write(client, &info, sizeof(info));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPerformanceModes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPerformanceModes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlDevicePerfModes_t perfModes;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPerformanceModes(device, &perfModes);
    rpc_write(client, &perfModes, sizeof(perfModes));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetCurrentClockFreqs(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetCurrentClockFreqs called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlDeviceCurrentClockFreqs_t currentClockFreqs;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCurrentClockFreqs(device, &currentClockFreqs);
    rpc_write(client, &currentClockFreqs, sizeof(currentClockFreqs));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEnableState_t mode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerManagementMode(device, &mode);
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int limit;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerManagementLimit(device, &limit);
    rpc_write(client, &limit, sizeof(limit));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int minLimit;
    unsigned int maxLimit;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerManagementLimitConstraints(device, &minLimit, &maxLimit);
    rpc_write(client, &minLimit, sizeof(minLimit));
    rpc_write(client, &maxLimit, sizeof(maxLimit));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int defaultLimit;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerManagementDefaultLimit(device, &defaultLimit);
    rpc_write(client, &defaultLimit, sizeof(defaultLimit));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int power;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerUsage(device, &power);
    rpc_write(client, &power, sizeof(power));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long long energy;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTotalEnergyConsumption(device, &energy);
    rpc_write(client, &energy, sizeof(energy));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int limit;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEnforcedPowerLimit(device, &limit);
    rpc_write(client, &limit, sizeof(limit));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlGpuOperationMode_t current;
    nvmlGpuOperationMode_t pending;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuOperationMode(device, &current, &pending);
    rpc_write(client, &current, sizeof(current));
    rpc_write(client, &pending, sizeof(pending));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlMemory_t memory;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMemoryInfo(device, &memory);
    rpc_write(client, &memory, sizeof(memory));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMemoryInfo_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMemoryInfo_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlMemory_v2_t memory;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMemoryInfo_v2(device, &memory);
    rpc_write(client, &memory, sizeof(memory));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlComputeMode_t mode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetComputeMode(device, &mode);
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    int major;
    int minor;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);
    rpc_write(client, &major, sizeof(major));
    rpc_write(client, &minor, sizeof(minor));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetDramEncryptionMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetDramEncryptionMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlDramEncryptionInfo_t current;
    nvmlDramEncryptionInfo_t pending;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDramEncryptionMode(device, &current, &pending);
    rpc_write(client, &current, sizeof(current));
    rpc_write(client, &pending, sizeof(pending));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetDramEncryptionMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetDramEncryptionMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlDramEncryptionInfo_t dramEncryption;
    rpc_read(client, &dramEncryption, sizeof(dramEncryption));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetDramEncryptionMode(device, &dramEncryption);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEnableState_t current;
    nvmlEnableState_t pending;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEccMode(device, &current, &pending);
    rpc_write(client, &current, sizeof(current));
    rpc_write(client, &pending, sizeof(pending));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetDefaultEccMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetDefaultEccMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEnableState_t defaultMode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDefaultEccMode(device, &defaultMode);
    rpc_write(client, &defaultMode, sizeof(defaultMode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int boardId;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBoardId(device, &boardId);
    rpc_write(client, &boardId, sizeof(boardId));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int multiGpuBool;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMultiGpuBoard(device, &multiGpuBool);
    rpc_write(client, &multiGpuBool, sizeof(multiGpuBool));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long long eccCounts;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTotalEccErrors(device, errorType, counterType, &eccCounts);
    rpc_write(client, &eccCounts, sizeof(eccCounts));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEccErrorCounts_t eccCounts;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDetailedEccErrors(device, errorType, counterType, &eccCounts);
    rpc_write(client, &eccCounts, sizeof(eccCounts));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long long count;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMemoryErrorCounter(device, errorType, counterType, locationType, &count);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlUtilization_t utilization;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetUtilizationRates(device, &utilization);
    rpc_write(client, &utilization, sizeof(utilization));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int utilization;
    unsigned int samplingPeriodUs;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEncoderUtilization(device, &utilization, &samplingPeriodUs);
    rpc_write(client, &utilization, sizeof(utilization));
    rpc_write(client, &samplingPeriodUs, sizeof(samplingPeriodUs));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int encoderCapacity;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEncoderCapacity(device, encoderQueryType, &encoderCapacity);
    rpc_write(client, &encoderCapacity, sizeof(encoderCapacity));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int sessionCount;
    unsigned int averageFps;
    unsigned int averageLatency;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEncoderStats(device, &sessionCount, &averageFps, &averageLatency);
    rpc_write(client, &sessionCount, sizeof(sessionCount));
    rpc_write(client, &averageFps, sizeof(averageFps));
    rpc_write(client, &averageLatency, sizeof(averageLatency));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int sessionCount;
    nvmlEncoderSessionInfo_t sessionInfos;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEncoderSessions(device, &sessionCount, &sessionInfos);
    rpc_write(client, &sessionCount, sizeof(sessionCount));
    rpc_write(client, &sessionInfos, sizeof(sessionInfos));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int utilization;
    unsigned int samplingPeriodUs;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDecoderUtilization(device, &utilization, &samplingPeriodUs);
    rpc_write(client, &utilization, sizeof(utilization));
    rpc_write(client, &samplingPeriodUs, sizeof(samplingPeriodUs));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetJpgUtilization(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetJpgUtilization called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int utilization;
    unsigned int samplingPeriodUs;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetJpgUtilization(device, &utilization, &samplingPeriodUs);
    rpc_write(client, &utilization, sizeof(utilization));
    rpc_write(client, &samplingPeriodUs, sizeof(samplingPeriodUs));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetOfaUtilization(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetOfaUtilization called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int utilization;
    unsigned int samplingPeriodUs;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetOfaUtilization(device, &utilization, &samplingPeriodUs);
    rpc_write(client, &utilization, sizeof(utilization));
    rpc_write(client, &samplingPeriodUs, sizeof(samplingPeriodUs));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlFBCStats_t fbcStats;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFBCStats(device, &fbcStats);
    rpc_write(client, &fbcStats, sizeof(fbcStats));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int sessionCount;
    nvmlFBCSessionInfo_t sessionInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFBCSessions(device, &sessionCount, &sessionInfo);
    rpc_write(client, &sessionCount, sizeof(sessionCount));
    rpc_write(client, &sessionInfo, sizeof(sessionInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetDriverModel_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetDriverModel_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlDriverModel_t current;
    nvmlDriverModel_t pending;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDriverModel_v2(device, &current, &pending);
    rpc_write(client, &current, sizeof(current));
    rpc_write(client, &pending, sizeof(pending));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlBridgeChipHierarchy_t bridgeHierarchy;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBridgeChipInfo(device, &bridgeHierarchy);
    rpc_write(client, &bridgeHierarchy, sizeof(bridgeHierarchy));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetComputeRunningProcesses_v3(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetComputeRunningProcesses_v3 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int infoCount;
    nvmlProcessInfo_t infos;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetComputeRunningProcesses_v3(device, &infoCount, &infos);
    rpc_write(client, &infoCount, sizeof(infoCount));
    rpc_write(client, &infos, sizeof(infos));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGraphicsRunningProcesses_v3(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGraphicsRunningProcesses_v3 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int infoCount;
    nvmlProcessInfo_t infos;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGraphicsRunningProcesses_v3(device, &infoCount, &infos);
    rpc_write(client, &infoCount, sizeof(infoCount));
    rpc_write(client, &infos, sizeof(infos));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMPSComputeRunningProcesses_v3(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMPSComputeRunningProcesses_v3 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int infoCount;
    nvmlProcessInfo_t infos;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMPSComputeRunningProcesses_v3(device, &infoCount, &infos);
    rpc_write(client, &infoCount, sizeof(infoCount));
    rpc_write(client, &infos, sizeof(infos));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetRunningProcessDetailList(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetRunningProcessDetailList called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlProcessDetailList_t plist;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRunningProcessDetailList(device, &plist);
    rpc_write(client, &plist, sizeof(plist));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    int onSameBoard;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceOnSameBoard(device1, device2, &onSameBoard);
    rpc_write(client, &onSameBoard, sizeof(onSameBoard));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEnableState_t isRestricted;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAPIRestriction(device, apiType, &isRestricted);
    rpc_write(client, &isRestricted, sizeof(isRestricted));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlValueType_t sampleValType;
    unsigned int sampleCount;
    nvmlSample_t samples;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, &sampleValType, &sampleCount, &samples);
    rpc_write(client, &sampleValType, sizeof(sampleValType));
    rpc_write(client, &sampleCount, sizeof(sampleCount));
    rpc_write(client, &samples, sizeof(samples));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlBAR1Memory_t bar1Memory;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBAR1MemoryInfo(device, &bar1Memory);
    rpc_write(client, &bar1Memory, sizeof(bar1Memory));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlViolationTime_t violTime;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetViolationStatus(device, perfPolicyType, &violTime);
    rpc_write(client, &violTime, sizeof(violTime));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetIrqNum(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetIrqNum called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int irqNum;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetIrqNum(device, &irqNum);
    rpc_write(client, &irqNum, sizeof(irqNum));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetNumGpuCores(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetNumGpuCores called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int numCores;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNumGpuCores(device, &numCores);
    rpc_write(client, &numCores, sizeof(numCores));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPowerSource(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPowerSource called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPowerSource_t powerSource;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerSource(device, &powerSource);
    rpc_write(client, &powerSource, sizeof(powerSource));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetMemoryBusWidth(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMemoryBusWidth called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int busWidth;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMemoryBusWidth(device, &busWidth);
    rpc_write(client, &busWidth, sizeof(busWidth));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPcieLinkMaxSpeed(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPcieLinkMaxSpeed called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int maxSpeed;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPcieLinkMaxSpeed(device, &maxSpeed);
    rpc_write(client, &maxSpeed, sizeof(maxSpeed));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPcieSpeed(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPcieSpeed called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int pcieSpeed;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPcieSpeed(device, &pcieSpeed);
    rpc_write(client, &pcieSpeed, sizeof(pcieSpeed));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetAdaptiveClockInfoStatus(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetAdaptiveClockInfoStatus called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int adaptiveClockStatus;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAdaptiveClockInfoStatus(device, &adaptiveClockStatus);
    rpc_write(client, &adaptiveClockStatus, sizeof(adaptiveClockStatus));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetBusType(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetBusType called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlBusType_t type;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBusType(device, &type);
    rpc_write(client, &type, sizeof(type));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGpuFabricInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGpuFabricInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlGpuFabricInfo_t gpuFabricInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuFabricInfo(device, &gpuFabricInfo);
    rpc_write(client, &gpuFabricInfo, sizeof(gpuFabricInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGpuFabricInfoV(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGpuFabricInfoV called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlGpuFabricInfoV_t gpuFabricInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuFabricInfoV(device, &gpuFabricInfo);
    rpc_write(client, &gpuFabricInfo, sizeof(gpuFabricInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetConfComputeCapabilities(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetConfComputeCapabilities called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlConfComputeSystemCaps_t capabilities;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetConfComputeCapabilities(&capabilities);
    rpc_write(client, &capabilities, sizeof(capabilities));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetConfComputeState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetConfComputeState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlConfComputeSystemState_t state;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetConfComputeState(&state);
    rpc_write(client, &state, sizeof(state));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetConfComputeMemSizeInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetConfComputeMemSizeInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlConfComputeMemSizeInfo_t memInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetConfComputeMemSizeInfo(device, &memInfo);
    rpc_write(client, &memInfo, sizeof(memInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetConfComputeGpusReadyState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetConfComputeGpusReadyState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int isAcceptingWork;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetConfComputeGpusReadyState(&isAcceptingWork);
    rpc_write(client, &isAcceptingWork, sizeof(isAcceptingWork));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetConfComputeProtectedMemoryUsage(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetConfComputeProtectedMemoryUsage called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlMemory_t memory;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetConfComputeProtectedMemoryUsage(device, &memory);
    rpc_write(client, &memory, sizeof(memory));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetConfComputeGpuCertificate(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetConfComputeGpuCertificate called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlConfComputeGpuCertificate_t gpuCert;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetConfComputeGpuCertificate(device, &gpuCert);
    rpc_write(client, &gpuCert, sizeof(gpuCert));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetConfComputeGpuAttestationReport(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetConfComputeGpuAttestationReport called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlConfComputeGpuAttestationReport_t gpuAtstReport;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetConfComputeGpuAttestationReport(device, &gpuAtstReport);
    rpc_write(client, &gpuAtstReport, sizeof(gpuAtstReport));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetConfComputeKeyRotationThresholdInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetConfComputeKeyRotationThresholdInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlConfComputeGetKeyRotationThresholdInfo_t pKeyRotationThrInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetConfComputeKeyRotationThresholdInfo(&pKeyRotationThrInfo);
    rpc_write(client, &pKeyRotationThrInfo, sizeof(pKeyRotationThrInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetConfComputeUnprotectedMemSize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetConfComputeUnprotectedMemSize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned long long sizeKiB;
    rpc_read(client, &sizeKiB, sizeof(sizeKiB));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetConfComputeUnprotectedMemSize(device, sizeKiB);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemSetConfComputeGpusReadyState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemSetConfComputeGpusReadyState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int isAcceptingWork;
    rpc_read(client, &isAcceptingWork, sizeof(isAcceptingWork));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemSetConfComputeGpusReadyState(isAcceptingWork);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemSetConfComputeKeyRotationThresholdInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemSetConfComputeKeyRotationThresholdInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlConfComputeSetKeyRotationThresholdInfo_t pKeyRotationThrInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemSetConfComputeKeyRotationThresholdInfo(&pKeyRotationThrInfo);
    rpc_write(client, &pKeyRotationThrInfo, sizeof(pKeyRotationThrInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetConfComputeSettings(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetConfComputeSettings called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlSystemConfComputeSettings_t settings;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetConfComputeSettings(&settings);
    rpc_write(client, &settings, sizeof(settings));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGspFirmwareVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGspFirmwareVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    char version[1024];
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGspFirmwareVersion(device, version);
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

int handle_nvmlDeviceGetGspFirmwareMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGspFirmwareMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int isEnabled;
    unsigned int defaultMode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGspFirmwareMode(device, &isEnabled, &defaultMode);
    rpc_write(client, &isEnabled, sizeof(isEnabled));
    rpc_write(client, &defaultMode, sizeof(defaultMode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetSramEccErrorStatus(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetSramEccErrorStatus called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlEccSramErrorStatus_t status;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSramEccErrorStatus(device, &status);
    rpc_write(client, &status, sizeof(status));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEnableState_t mode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAccountingMode(device, &mode);
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlAccountingStats_t stats;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAccountingStats(device, pid, &stats);
    rpc_write(client, &stats, sizeof(stats));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int count;
    unsigned int pids;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAccountingPids(device, &count, &pids);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &pids, sizeof(pids));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int bufferSize;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAccountingBufferSize(device, &bufferSize);
    rpc_write(client, &bufferSize, sizeof(bufferSize));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int pageCount;
    unsigned long long addresses;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRetiredPages(device, cause, &pageCount, &addresses);
    rpc_write(client, &pageCount, sizeof(pageCount));
    rpc_write(client, &addresses, sizeof(addresses));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int pageCount;
    unsigned long long addresses;
    unsigned long long timestamps;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRetiredPages_v2(device, cause, &pageCount, &addresses, &timestamps);
    rpc_write(client, &pageCount, sizeof(pageCount));
    rpc_write(client, &addresses, sizeof(addresses));
    rpc_write(client, &timestamps, sizeof(timestamps));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEnableState_t isPending;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRetiredPagesPendingStatus(device, &isPending);
    rpc_write(client, &isPending, sizeof(isPending));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int corrRows;
    unsigned int uncRows;
    unsigned int isPending;
    unsigned int failureOccurred;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRemappedRows(device, &corrRows, &uncRows, &isPending, &failureOccurred);
    rpc_write(client, &corrRows, sizeof(corrRows));
    rpc_write(client, &uncRows, sizeof(uncRows));
    rpc_write(client, &isPending, sizeof(isPending));
    rpc_write(client, &failureOccurred, sizeof(failureOccurred));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlRowRemapperHistogramValues_t values;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRowRemapperHistogram(device, &values);
    rpc_write(client, &values, sizeof(values));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlDeviceArchitecture_t arch;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetArchitecture(device, &arch);
    rpc_write(client, &arch, sizeof(arch));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlClkMonStatus_t status;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetClkMonStatus(device, &status);
    rpc_write(client, &status, sizeof(status));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlProcessUtilizationSample_t utilization;
    unsigned int processSamplesCount;
    unsigned long long lastSeenTimeStamp;
    rpc_read(client, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetProcessUtilization(device, &utilization, &processSamplesCount, lastSeenTimeStamp);
    rpc_write(client, &utilization, sizeof(utilization));
    rpc_write(client, &processSamplesCount, sizeof(processSamplesCount));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetProcessesUtilizationInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetProcessesUtilizationInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlProcessesUtilizationInfo_t procesesUtilInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetProcessesUtilizationInfo(device, &procesesUtilInfo);
    rpc_write(client, &procesesUtilInfo, sizeof(procesesUtilInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetPlatformInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetPlatformInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPlatformInfo_t platformInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPlatformInfo(device, &platformInfo);
    rpc_write(client, &platformInfo, sizeof(platformInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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

int handle_nvmlDeviceSetDefaultFanSpeed_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetDefaultFanSpeed_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int fan;
    rpc_read(client, &fan, sizeof(fan));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetDefaultFanSpeed_v2(device, fan);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetFanControlPolicy(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetFanControlPolicy called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int fan;
    rpc_read(client, &fan, sizeof(fan));
    nvmlFanControlPolicy_t policy;
    rpc_read(client, &policy, sizeof(policy));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetFanControlPolicy(device, fan, policy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    int temp;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetTemperatureThreshold(device, thresholdType, &temp);
    rpc_write(client, &temp, sizeof(temp));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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

int handle_nvmlDeviceSetFanSpeed_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetFanSpeed_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int fan;
    rpc_read(client, &fan, sizeof(fan));
    unsigned int speed;
    rpc_read(client, &speed, sizeof(speed));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetFanSpeed_v2(device, fan, speed);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetGpcClkVfOffset(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetGpcClkVfOffset called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    int offset;
    rpc_read(client, &offset, sizeof(offset));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetGpcClkVfOffset(device, offset);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetMemClkVfOffset(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetMemClkVfOffset called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    int offset;
    rpc_read(client, &offset, sizeof(offset));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetMemClkVfOffset(device, offset);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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

int handle_nvmlDeviceSetPowerManagementLimit_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetPowerManagementLimit_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPowerValue_v2_t powerValue;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetPowerManagementLimit_v2(device, &powerValue);
    rpc_write(client, &powerValue, sizeof(powerValue));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEnableState_t isActive;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkState(device, link, &isActive);
    rpc_write(client, &isActive, sizeof(isActive));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int version;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkVersion(device, link, &version);
    rpc_write(client, &version, sizeof(version));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int capResult;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkCapability(device, link, capability, &capResult);
    rpc_write(client, &capResult, sizeof(capResult));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlPciInfo_t pci;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, &pci);
    rpc_write(client, &pci, sizeof(pci));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long long counterValue;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkErrorCounter(device, link, counter, &counterValue);
    rpc_write(client, &counterValue, sizeof(counterValue));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlNvLinkUtilizationControl_t control;
    unsigned int reset;
    rpc_read(client, &reset, sizeof(reset));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetNvLinkUtilizationControl(device, link, counter, &control, reset);
    rpc_write(client, &control, sizeof(control));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlNvLinkUtilizationControl_t control;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkUtilizationControl(device, link, counter, &control);
    rpc_write(client, &control, sizeof(control));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long long rxcounter;
    unsigned long long txcounter;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter, &rxcounter, &txcounter);
    rpc_write(client, &rxcounter, sizeof(rxcounter));
    rpc_write(client, &txcounter, sizeof(txcounter));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlIntNvLinkDeviceType_t pNvLinkDeviceType;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkRemoteDeviceType(device, link, &pNvLinkDeviceType);
    rpc_write(client, &pNvLinkDeviceType, sizeof(pNvLinkDeviceType));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetNvLinkDeviceLowPowerThreshold(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetNvLinkDeviceLowPowerThreshold called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlNvLinkPowerThres_t info;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetNvLinkDeviceLowPowerThreshold(device, &info);
    rpc_write(client, &info, sizeof(info));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemSetNvlinkBwMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemSetNvlinkBwMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int nvlinkBwMode;
    rpc_read(client, &nvlinkBwMode, sizeof(nvlinkBwMode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemSetNvlinkBwMode(nvlinkBwMode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlSystemGetNvlinkBwMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetNvlinkBwMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    unsigned int nvlinkBwMode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetNvlinkBwMode(&nvlinkBwMode);
    rpc_write(client, &nvlinkBwMode, sizeof(nvlinkBwMode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetNvlinkSupportedBwModes(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetNvlinkSupportedBwModes called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlNvlinkSupportedBwModes_t supportedBwMode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvlinkSupportedBwModes(device, &supportedBwMode);
    rpc_write(client, &supportedBwMode, sizeof(supportedBwMode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetNvlinkBwMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetNvlinkBwMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlNvlinkGetBwMode_t getBwMode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvlinkBwMode(device, &getBwMode);
    rpc_write(client, &getBwMode, sizeof(getBwMode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetNvlinkBwMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetNvlinkBwMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlNvlinkSetBwMode_t setBwMode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetNvlinkBwMode(device, &setBwMode);
    rpc_write(client, &setBwMode, sizeof(setBwMode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEventSet_t set;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlEventSetCreate(&set);
    rpc_write(client, &set, sizeof(set));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long long eventTypes;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedEventTypes(device, &eventTypes);
    rpc_write(client, &eventTypes, sizeof(eventTypes));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEventData_t data;
    unsigned int timeoutms;
    rpc_read(client, &timeoutms, sizeof(timeoutms));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlEventSetWait_v2(set, &data, timeoutms);
    rpc_write(client, &data, sizeof(data));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlPciInfo_t pciInfo;
    nvmlEnableState_t newState;
    rpc_read(client, &newState, sizeof(newState));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceModifyDrainState(&pciInfo, newState);
    rpc_write(client, &pciInfo, sizeof(pciInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlPciInfo_t pciInfo;
    nvmlEnableState_t currentState;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceQueryDrainState(&pciInfo, &currentState);
    rpc_write(client, &pciInfo, sizeof(pciInfo));
    rpc_write(client, &currentState, sizeof(currentState));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlPciInfo_t pciInfo;
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
    _result = nvmlDeviceRemoveGpu_v2(&pciInfo, gpuState, linkState);
    rpc_write(client, &pciInfo, sizeof(pciInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlPciInfo_t pciInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceDiscoverGpus(&pciInfo);
    rpc_write(client, &pciInfo, sizeof(pciInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlFieldValue_t values;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFieldValues(device, valuesCount, &values);
    rpc_write(client, &values, sizeof(values));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceClearFieldValues(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceClearFieldValues called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    int valuesCount;
    rpc_read(client, &valuesCount, sizeof(valuesCount));
    nvmlFieldValue_t values;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceClearFieldValues(device, valuesCount, &values);
    rpc_write(client, &values, sizeof(values));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlGpuVirtualizationMode_t pVirtualMode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVirtualizationMode(device, &pVirtualMode);
    rpc_write(client, &pVirtualMode, sizeof(pVirtualMode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlHostVgpuMode_t pHostVgpuMode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetHostVgpuMode(device, &pHostVgpuMode);
    rpc_write(client, &pHostVgpuMode, sizeof(pHostVgpuMode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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

int handle_nvmlDeviceGetVgpuHeterogeneousMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVgpuHeterogeneousMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlVgpuHeterogeneousMode_t pHeterogeneousMode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuHeterogeneousMode(device, &pHeterogeneousMode);
    rpc_write(client, &pHeterogeneousMode, sizeof(pHeterogeneousMode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetVgpuHeterogeneousMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetVgpuHeterogeneousMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlVgpuHeterogeneousMode_t pHeterogeneousMode;
    rpc_read(client, &pHeterogeneousMode, sizeof(pHeterogeneousMode));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetVgpuHeterogeneousMode(device, &pHeterogeneousMode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetPlacementId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetPlacementId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    nvmlVgpuPlacementId_t pPlacement;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetPlacementId(vgpuInstance, &pPlacement);
    rpc_write(client, &pPlacement, sizeof(pPlacement));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetVgpuTypeSupportedPlacements(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVgpuTypeSupportedPlacements called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    nvmlVgpuPlacementList_t pPlacementList;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuTypeSupportedPlacements(device, vgpuTypeId, &pPlacementList);
    rpc_write(client, &pPlacementList, sizeof(pPlacementList));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetVgpuTypeCreatablePlacements(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVgpuTypeCreatablePlacements called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    nvmlVgpuPlacementList_t pPlacementList;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuTypeCreatablePlacements(device, vgpuTypeId, &pPlacementList);
    rpc_write(client, &pPlacementList, sizeof(pPlacementList));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetGspHeapSize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetGspHeapSize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    unsigned long long gspHeapSize;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetGspHeapSize(vgpuTypeId, &gspHeapSize);
    rpc_write(client, &gspHeapSize, sizeof(gspHeapSize));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetFbReservation(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetFbReservation called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    unsigned long long fbReservation;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetFbReservation(vgpuTypeId, &fbReservation);
    rpc_write(client, &fbReservation, sizeof(fbReservation));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetRuntimeStateSize(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetRuntimeStateSize called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    nvmlVgpuRuntimeState_t pState;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetRuntimeStateSize(vgpuInstance, &pState);
    rpc_write(client, &pState, sizeof(pState));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetVgpuCapabilities(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetVgpuCapabilities called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlDeviceVgpuCapability_t capability;
    rpc_read(client, &capability, sizeof(capability));
    nvmlEnableState_t state;
    rpc_read(client, &state, sizeof(state));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetVgpuCapabilities(device, capability, state);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGridLicensableFeatures_v4(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGridLicensableFeatures_v4 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlGridLicensableFeatures_t pGridLicensableFeatures;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGridLicensableFeatures_v4(device, &pGridLicensableFeatures);
    rpc_write(client, &pGridLicensableFeatures, sizeof(pGridLicensableFeatures));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGetVgpuDriverCapabilities(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGetVgpuDriverCapabilities called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuDriverCapability_t capability;
    rpc_read(client, &capability, sizeof(capability));
    unsigned int capResult;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGetVgpuDriverCapabilities(capability, &capResult);
    rpc_write(client, &capResult, sizeof(capResult));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetVgpuCapabilities(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVgpuCapabilities called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlDeviceVgpuCapability_t capability;
    rpc_read(client, &capability, sizeof(capability));
    unsigned int capResult;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuCapabilities(device, capability, &capResult);
    rpc_write(client, &capResult, sizeof(capResult));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int vgpuCount;
    nvmlVgpuTypeId_t vgpuTypeIds;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedVgpus(device, &vgpuCount, &vgpuTypeIds);
    rpc_write(client, &vgpuCount, sizeof(vgpuCount));
    rpc_write(client, &vgpuTypeIds, sizeof(vgpuTypeIds));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int vgpuCount;
    nvmlVgpuTypeId_t vgpuTypeIds;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCreatableVgpus(device, &vgpuCount, &vgpuTypeIds);
    rpc_write(client, &vgpuCount, sizeof(vgpuCount));
    rpc_write(client, &vgpuTypeIds, sizeof(vgpuTypeIds));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int size;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetClass(vgpuTypeId, vgpuTypeClass, &size);
    rpc_write(client, vgpuTypeClass, strlen(vgpuTypeClass) + 1, true);
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int size;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetName(vgpuTypeId, vgpuTypeName, &size);
    rpc_write(client, vgpuTypeName, strlen(vgpuTypeName) + 1, true);
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int gpuInstanceProfileId;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId, &gpuInstanceProfileId);
    rpc_write(client, &gpuInstanceProfileId, sizeof(gpuInstanceProfileId));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long long deviceID;
    unsigned long long subsystemID;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetDeviceID(vgpuTypeId, &deviceID, &subsystemID);
    rpc_write(client, &deviceID, sizeof(deviceID));
    rpc_write(client, &subsystemID, sizeof(subsystemID));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long long fbSize;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetFramebufferSize(vgpuTypeId, &fbSize);
    rpc_write(client, &fbSize, sizeof(fbSize));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int numDisplayHeads;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId, &numDisplayHeads);
    rpc_write(client, &numDisplayHeads, sizeof(numDisplayHeads));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int xdim;
    unsigned int ydim;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetResolution(vgpuTypeId, displayIndex, &xdim, &ydim);
    rpc_write(client, &xdim, sizeof(xdim));
    rpc_write(client, &ydim, sizeof(ydim));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int frameRateLimit;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId, &frameRateLimit);
    rpc_write(client, &frameRateLimit, sizeof(frameRateLimit));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int vgpuInstanceCount;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetMaxInstances(device, vgpuTypeId, &vgpuInstanceCount);
    rpc_write(client, &vgpuInstanceCount, sizeof(vgpuInstanceCount));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int vgpuInstanceCountPerVm;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId, &vgpuInstanceCountPerVm);
    rpc_write(client, &vgpuInstanceCountPerVm, sizeof(vgpuInstanceCountPerVm));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetBAR1Info(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetBAR1Info called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    nvmlVgpuTypeBar1Info_t bar1Info;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetBAR1Info(vgpuTypeId, &bar1Info);
    rpc_write(client, &bar1Info, sizeof(bar1Info));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int vgpuCount;
    nvmlVgpuInstance_t vgpuInstances;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetActiveVgpus(device, &vgpuCount, &vgpuInstances);
    rpc_write(client, &vgpuCount, sizeof(vgpuCount));
    rpc_write(client, &vgpuInstances, sizeof(vgpuInstances));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlVgpuVmIdType_t vmIdType;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetVmID(vgpuInstance, vmId, size, &vmIdType);
    rpc_write(client, vmId, strlen(vmId) + 1, true);
    rpc_write(client, &vmIdType, sizeof(vmIdType));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned long long fbUsage;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetFbUsage(vgpuInstance, &fbUsage);
    rpc_write(client, &fbUsage, sizeof(fbUsage));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int licensed;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetLicenseStatus(vgpuInstance, &licensed);
    rpc_write(client, &licensed, sizeof(licensed));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlVgpuTypeId_t vgpuTypeId;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetType(vgpuInstance, &vgpuTypeId);
    rpc_write(client, &vgpuTypeId, sizeof(vgpuTypeId));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int frameRateLimit;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance, &frameRateLimit);
    rpc_write(client, &frameRateLimit, sizeof(frameRateLimit));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEnableState_t eccMode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetEccMode(vgpuInstance, &eccMode);
    rpc_write(client, &eccMode, sizeof(eccMode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int encoderCapacity;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance, &encoderCapacity);
    rpc_write(client, &encoderCapacity, sizeof(encoderCapacity));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int sessionCount;
    unsigned int averageFps;
    unsigned int averageLatency;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetEncoderStats(vgpuInstance, &sessionCount, &averageFps, &averageLatency);
    rpc_write(client, &sessionCount, sizeof(sessionCount));
    rpc_write(client, &averageFps, sizeof(averageFps));
    rpc_write(client, &averageLatency, sizeof(averageLatency));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int sessionCount;
    nvmlEncoderSessionInfo_t sessionInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetEncoderSessions(vgpuInstance, &sessionCount, &sessionInfo);
    rpc_write(client, &sessionCount, sizeof(sessionCount));
    rpc_write(client, &sessionInfo, sizeof(sessionInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlFBCStats_t fbcStats;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetFBCStats(vgpuInstance, &fbcStats);
    rpc_write(client, &fbcStats, sizeof(fbcStats));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int sessionCount;
    nvmlFBCSessionInfo_t sessionInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetFBCSessions(vgpuInstance, &sessionCount, &sessionInfo);
    rpc_write(client, &sessionCount, sizeof(sessionCount));
    rpc_write(client, &sessionInfo, sizeof(sessionInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int gpuInstanceId;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance, &gpuInstanceId);
    rpc_write(client, &gpuInstanceId, sizeof(gpuInstanceId));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuInstanceGetGpuPciId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetGpuPciId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    char vgpuPciId[1024];
    unsigned int length;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetGpuPciId(vgpuInstance, vgpuPciId, &length);
    rpc_write(client, vgpuPciId, strlen(vgpuPciId) + 1, true);
    rpc_write(client, &length, sizeof(length));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlVgpuTypeGetCapabilities(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetCapabilities called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    rpc_read(client, &vgpuTypeId, sizeof(vgpuTypeId));
    nvmlVgpuCapability_t capability;
    rpc_read(client, &capability, sizeof(capability));
    unsigned int capResult;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetCapabilities(vgpuTypeId, capability, &capResult);
    rpc_write(client, &capResult, sizeof(capResult));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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

int handle_nvmlVgpuInstanceGetMetadata(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetMetadata called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    nvmlVgpuMetadata_t vgpuMetadata;
    unsigned int bufferSize;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetMetadata(vgpuInstance, &vgpuMetadata, &bufferSize);
    rpc_write(client, &vgpuMetadata, sizeof(vgpuMetadata));
    rpc_write(client, &bufferSize, sizeof(bufferSize));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlVgpuPgpuMetadata_t pgpuMetadata;
    unsigned int bufferSize;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuMetadata(device, &pgpuMetadata, &bufferSize);
    rpc_write(client, &pgpuMetadata, sizeof(pgpuMetadata));
    rpc_write(client, &bufferSize, sizeof(bufferSize));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlVgpuMetadata_t vgpuMetadata;
    nvmlVgpuPgpuMetadata_t pgpuMetadata;
    nvmlVgpuPgpuCompatibility_t compatibilityInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGetVgpuCompatibility(&vgpuMetadata, &pgpuMetadata, &compatibilityInfo);
    rpc_write(client, &vgpuMetadata, sizeof(vgpuMetadata));
    rpc_write(client, &pgpuMetadata, sizeof(pgpuMetadata));
    rpc_write(client, &compatibilityInfo, sizeof(compatibilityInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int bufferSize;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPgpuMetadataString(device, pgpuMetadata, &bufferSize);
    rpc_write(client, pgpuMetadata, strlen(pgpuMetadata) + 1, true);
    rpc_write(client, &bufferSize, sizeof(bufferSize));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetVgpuSchedulerLog(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVgpuSchedulerLog called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlVgpuSchedulerLog_t pSchedulerLog;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuSchedulerLog(device, &pSchedulerLog);
    rpc_write(client, &pSchedulerLog, sizeof(pSchedulerLog));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetVgpuSchedulerState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVgpuSchedulerState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlVgpuSchedulerGetState_t pSchedulerState;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuSchedulerState(device, &pSchedulerState);
    rpc_write(client, &pSchedulerState, sizeof(pSchedulerState));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetVgpuSchedulerCapabilities(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVgpuSchedulerCapabilities called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlVgpuSchedulerCapabilities_t pCapabilities;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuSchedulerCapabilities(device, &pCapabilities);
    rpc_write(client, &pCapabilities, sizeof(pCapabilities));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceSetVgpuSchedulerState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceSetVgpuSchedulerState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlVgpuSchedulerSetState_t pSchedulerState;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetVgpuSchedulerState(device, &pSchedulerState);
    rpc_write(client, &pSchedulerState, sizeof(pSchedulerState));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlVgpuVersion_t supported;
    nvmlVgpuVersion_t current;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGetVgpuVersion(&supported, &current);
    rpc_write(client, &supported, sizeof(supported));
    rpc_write(client, &current, sizeof(current));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlVgpuVersion_t vgpuVersion;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSetVgpuVersion(&vgpuVersion);
    rpc_write(client, &vgpuVersion, sizeof(vgpuVersion));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlValueType_t sampleValType;
    unsigned int vgpuInstanceSamplesCount;
    nvmlVgpuInstanceUtilizationSample_t utilizationSamples;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuUtilization(device, lastSeenTimeStamp, &sampleValType, &vgpuInstanceSamplesCount, &utilizationSamples);
    rpc_write(client, &sampleValType, sizeof(sampleValType));
    rpc_write(client, &vgpuInstanceSamplesCount, sizeof(vgpuInstanceSamplesCount));
    rpc_write(client, &utilizationSamples, sizeof(utilizationSamples));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetVgpuInstancesUtilizationInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVgpuInstancesUtilizationInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlVgpuInstancesUtilizationInfo_t vgpuUtilInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuInstancesUtilizationInfo(device, &vgpuUtilInfo);
    rpc_write(client, &vgpuUtilInfo, sizeof(vgpuUtilInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int vgpuProcessSamplesCount;
    nvmlVgpuProcessUtilizationSample_t utilizationSamples;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuProcessUtilization(device, lastSeenTimeStamp, &vgpuProcessSamplesCount, &utilizationSamples);
    rpc_write(client, &vgpuProcessSamplesCount, sizeof(vgpuProcessSamplesCount));
    rpc_write(client, &utilizationSamples, sizeof(utilizationSamples));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetVgpuProcessesUtilizationInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetVgpuProcessesUtilizationInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlVgpuProcessesUtilizationInfo_t vgpuProcUtilInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuProcessesUtilizationInfo(device, &vgpuProcUtilInfo);
    rpc_write(client, &vgpuProcUtilInfo, sizeof(vgpuProcUtilInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlEnableState_t mode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetAccountingMode(vgpuInstance, &mode);
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int count;
    unsigned int pids;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetAccountingPids(vgpuInstance, &count, &pids);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &pids, sizeof(pids));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlAccountingStats_t stats;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid, &stats);
    rpc_write(client, &stats, sizeof(stats));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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

int handle_nvmlVgpuInstanceGetLicenseInfo_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetLicenseInfo_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    rpc_read(client, &vgpuInstance, sizeof(vgpuInstance));
    nvmlVgpuLicenseInfo_t licenseInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance, &licenseInfo);
    rpc_write(client, &licenseInfo, sizeof(licenseInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int deviceCount;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGetExcludedDeviceCount(&deviceCount);
    rpc_write(client, &deviceCount, sizeof(deviceCount));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlExcludedDeviceInfo_t info;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGetExcludedDeviceInfoByIndex(index, &info);
    rpc_write(client, &info, sizeof(info));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlReturn_t activationStatus;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetMigMode(device, mode, &activationStatus);
    rpc_write(client, &activationStatus, sizeof(activationStatus));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int currentMode;
    unsigned int pendingMode;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMigMode(device, &currentMode, &pendingMode);
    rpc_write(client, &currentMode, sizeof(currentMode));
    rpc_write(client, &pendingMode, sizeof(pendingMode));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlGpuInstanceProfileInfo_t info;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstanceProfileInfo(device, profile, &info);
    rpc_write(client, &info, sizeof(info));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetGpuInstanceProfileInfoV(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetGpuInstanceProfileInfoV called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int profile;
    rpc_read(client, &profile, sizeof(profile));
    nvmlGpuInstanceProfileInfo_v2_t info;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstanceProfileInfoV(device, profile, &info);
    rpc_write(client, &info, sizeof(info));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlGpuInstancePlacement_t placements;
    unsigned int count;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstancePossiblePlacements_v2(device, profileId, &placements, &count);
    rpc_write(client, &placements, sizeof(placements));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int count;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstanceRemainingCapacity(device, profileId, &count);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlGpuInstance_t gpuInstance;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceCreateGpuInstance(device, profileId, &gpuInstance);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlGpuInstancePlacement_t placement;
    rpc_read(client, &placement, sizeof(placement));
    nvmlGpuInstance_t gpuInstance;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, &placement, &gpuInstance);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlGpuInstance_t gpuInstances;
    unsigned int count;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstances(device, profileId, &gpuInstances, &count);
    rpc_write(client, &gpuInstances, sizeof(gpuInstances));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlGpuInstance_t gpuInstance;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstanceById(device, id, &gpuInstance);
    rpc_write(client, &gpuInstance, sizeof(gpuInstance));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlGpuInstanceInfo_t info;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetInfo(gpuInstance, &info);
    rpc_write(client, &info, sizeof(info));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlComputeInstanceProfileInfo_t info;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstanceProfileInfo(gpuInstance, profile, engProfile, &info);
    rpc_write(client, &info, sizeof(info));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpuInstanceGetComputeInstanceProfileInfoV(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpuInstanceGetComputeInstanceProfileInfoV called" << std::endl;
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
    nvmlComputeInstanceProfileInfo_v2_t info;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstanceProfileInfoV(gpuInstance, profile, engProfile, &info);
    rpc_write(client, &info, sizeof(info));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int count;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, profileId, &count);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpuInstanceGetComputeInstancePossiblePlacements(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpuInstanceGetComputeInstancePossiblePlacements called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlGpuInstance_t gpuInstance;
    rpc_read(client, &gpuInstance, sizeof(gpuInstance));
    unsigned int profileId;
    rpc_read(client, &profileId, sizeof(profileId));
    nvmlComputeInstancePlacement_t placements;
    unsigned int count;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstancePossiblePlacements(gpuInstance, profileId, &placements, &count);
    rpc_write(client, &placements, sizeof(placements));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlComputeInstance_t computeInstance;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId, &computeInstance);
    rpc_write(client, &computeInstance, sizeof(computeInstance));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpuInstanceCreateComputeInstanceWithPlacement(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpuInstanceCreateComputeInstanceWithPlacement called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlGpuInstance_t gpuInstance;
    rpc_read(client, &gpuInstance, sizeof(gpuInstance));
    unsigned int profileId;
    rpc_read(client, &profileId, sizeof(profileId));
    nvmlComputeInstancePlacement_t placement;
    rpc_read(client, &placement, sizeof(placement));
    nvmlComputeInstance_t computeInstance;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceCreateComputeInstanceWithPlacement(gpuInstance, profileId, &placement, &computeInstance);
    rpc_write(client, &computeInstance, sizeof(computeInstance));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlComputeInstance_t computeInstances;
    unsigned int count;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId, &computeInstances, &count);
    rpc_write(client, &computeInstances, sizeof(computeInstances));
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlComputeInstance_t computeInstance;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstanceById(gpuInstance, id, &computeInstance);
    rpc_write(client, &computeInstance, sizeof(computeInstance));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlComputeInstanceInfo_t info;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlComputeInstanceGetInfo_v2(computeInstance, &info);
    rpc_write(client, &info, sizeof(info));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int isMigDevice;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceIsMigDeviceHandle(device, &isMigDevice);
    rpc_write(client, &isMigDevice, sizeof(isMigDevice));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int id;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstanceId(device, &id);
    rpc_write(client, &id, sizeof(id));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int id;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetComputeInstanceId(device, &id);
    rpc_write(client, &id, sizeof(id));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    unsigned int count;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxMigDeviceCount(device, &count);
    rpc_write(client, &count, sizeof(count));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlDevice_t migDevice;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMigDeviceHandleByIndex(device, index, &migDevice);
    rpc_write(client, &migDevice, sizeof(migDevice));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
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
    nvmlDevice_t device;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice, &device);
    rpc_write(client, &device, sizeof(device));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpmMetricsGet(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpmMetricsGet called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlGpmMetricsGet_t metricsGet;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpmMetricsGet(&metricsGet);
    rpc_write(client, &metricsGet, sizeof(metricsGet));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpmSampleFree(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpmSampleFree called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlGpmSample_t gpmSample;
    rpc_read(client, &gpmSample, sizeof(gpmSample));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpmSampleFree(gpmSample);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpmSampleAlloc(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpmSampleAlloc called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlGpmSample_t gpmSample;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpmSampleAlloc(&gpmSample);
    rpc_write(client, &gpmSample, sizeof(gpmSample));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpmSampleGet(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpmSampleGet called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlGpmSample_t gpmSample;
    rpc_read(client, &gpmSample, sizeof(gpmSample));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpmSampleGet(device, gpmSample);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpmMigSampleGet(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpmMigSampleGet called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int gpuInstanceId;
    rpc_read(client, &gpuInstanceId, sizeof(gpuInstanceId));
    nvmlGpmSample_t gpmSample;
    rpc_read(client, &gpmSample, sizeof(gpmSample));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpmMigSampleGet(device, gpuInstanceId, gpmSample);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpmQueryDeviceSupport(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpmQueryDeviceSupport called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlGpmSupport_t gpmSupport;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpmQueryDeviceSupport(device, &gpmSupport);
    rpc_write(client, &gpmSupport, sizeof(gpmSupport));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpmQueryIfStreamingEnabled(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpmQueryIfStreamingEnabled called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int state;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpmQueryIfStreamingEnabled(device, &state);
    rpc_write(client, &state, sizeof(state));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlGpmSetStreamingEnabled(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGpmSetStreamingEnabled called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    unsigned int state;
    rpc_read(client, &state, sizeof(state));
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpmSetStreamingEnabled(device, state);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceGetCapabilities(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetCapabilities called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlDeviceCapabilities_t caps;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCapabilities(device, &caps);
    rpc_write(client, &caps, sizeof(caps));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceWorkloadPowerProfileGetProfilesInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceWorkloadPowerProfileGetProfilesInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlWorkloadPowerProfileProfilesInfo_t profilesInfo;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceWorkloadPowerProfileGetProfilesInfo(device, &profilesInfo);
    rpc_write(client, &profilesInfo, sizeof(profilesInfo));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceWorkloadPowerProfileGetCurrentProfiles(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceWorkloadPowerProfileGetCurrentProfiles called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlWorkloadPowerProfileCurrentProfiles_t currentProfiles;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceWorkloadPowerProfileGetCurrentProfiles(device, &currentProfiles);
    rpc_write(client, &currentProfiles, sizeof(currentProfiles));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceWorkloadPowerProfileSetRequestedProfiles(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceWorkloadPowerProfileSetRequestedProfiles called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlWorkloadPowerProfileRequestedProfiles_t requestedProfiles;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceWorkloadPowerProfileSetRequestedProfiles(device, &requestedProfiles);
    rpc_write(client, &requestedProfiles, sizeof(requestedProfiles));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDeviceWorkloadPowerProfileClearRequestedProfiles(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceWorkloadPowerProfileClearRequestedProfiles called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlWorkloadPowerProfileRequestedProfiles_t requestedProfiles;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceWorkloadPowerProfileClearRequestedProfiles(device, &requestedProfiles);
    rpc_write(client, &requestedProfiles, sizeof(requestedProfiles));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDevicePowerSmoothingActivatePresetProfile(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDevicePowerSmoothingActivatePresetProfile called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPowerSmoothingProfile_t profile;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDevicePowerSmoothingActivatePresetProfile(device, &profile);
    rpc_write(client, &profile, sizeof(profile));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDevicePowerSmoothingUpdatePresetProfileParam(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDevicePowerSmoothingUpdatePresetProfileParam called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPowerSmoothingProfile_t profile;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDevicePowerSmoothingUpdatePresetProfileParam(device, &profile);
    rpc_write(client, &profile, sizeof(profile));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_nvmlDevicePowerSmoothingSetState(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDevicePowerSmoothingSetState called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    nvmlDevice_t device;
    rpc_read(client, &device, sizeof(device));
    nvmlPowerSmoothingState_t state;
    nvmlReturn_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDevicePowerSmoothingSetState(device, &state);
    rpc_write(client, &state, sizeof(state));
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

