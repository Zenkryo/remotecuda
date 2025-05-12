#include <iostream>
#include <map>
#include <string.h>
#include "hook_api.h"
#include "handle_server.h"
#include "rpc/rpc_core.h"
#include "nvml.h"

using namespace rpc;
int handle_nvmlInit_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlInit_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlInit_v2();
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlInitWithFlags(flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlShutdown();
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    char version[1024];
    unsigned int length;
    conn->read(&length, sizeof(length));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetDriverVersion(version, length);
    if(length > 0) {
        conn->write(version, strlen(version) + 1, true);
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

int handle_nvmlSystemGetNVMLVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetNVMLVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    char version[1024];
    unsigned int length;
    conn->read(&length, sizeof(length));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetNVMLVersion(version, length);
    if(length > 0) {
        conn->write(version, strlen(version) + 1, true);
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

int handle_nvmlSystemGetCudaDriverVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlSystemGetCudaDriverVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int *cudaDriverVersion;
    conn->read(&cudaDriverVersion, sizeof(cudaDriverVersion));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetCudaDriverVersion(cudaDriverVersion);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    int *cudaDriverVersion;
    conn->read(&cudaDriverVersion, sizeof(cudaDriverVersion));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetCudaDriverVersion_v2(cudaDriverVersion);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int pid;
    conn->read(&pid, sizeof(pid));
    char name[1024];
    unsigned int length;
    conn->read(&length, sizeof(length));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetProcessName(pid, name, length);
    if(length > 0) {
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

int handle_nvmlUnitGetCount(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlUnitGetCount called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    unsigned int *unitCount;
    conn->read(&unitCount, sizeof(unitCount));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetCount(unitCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int index;
    conn->read(&index, sizeof(index));
    nvmlUnit_t *unit;
    conn->read(&unit, sizeof(unit));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetHandleByIndex(index, unit);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlUnit_t unit;
    conn->read(&unit, sizeof(unit));
    nvmlUnitInfo_t *info;
    conn->read(&info, sizeof(info));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetUnitInfo(unit, info);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlUnit_t unit;
    conn->read(&unit, sizeof(unit));
    nvmlLedState_t *state;
    conn->read(&state, sizeof(state));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetLedState(unit, state);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlUnit_t unit;
    conn->read(&unit, sizeof(unit));
    nvmlPSUInfo_t *psu;
    conn->read(&psu, sizeof(psu));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetPsuInfo(unit, psu);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlUnit_t unit;
    conn->read(&unit, sizeof(unit));
    unsigned int type;
    conn->read(&type, sizeof(type));
    unsigned int *temp;
    conn->read(&temp, sizeof(temp));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetTemperature(unit, type, temp);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlUnit_t unit;
    conn->read(&unit, sizeof(unit));
    nvmlUnitFanSpeeds_t *fanSpeeds;
    conn->read(&fanSpeeds, sizeof(fanSpeeds));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetFanSpeedInfo(unit, fanSpeeds);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlUnit_t unit;
    conn->read(&unit, sizeof(unit));
    unsigned int *deviceCount;
    conn->read(&deviceCount, sizeof(deviceCount));
    nvmlDevice_t *devices;
    conn->read(&devices, sizeof(devices));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitGetDevices(unit, deviceCount, devices);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int *hwbcCount;
    conn->read(&hwbcCount, sizeof(hwbcCount));
    nvmlHwbcEntry_t *hwbcEntries;
    conn->read(&hwbcEntries, sizeof(hwbcEntries));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetHicVersion(hwbcCount, hwbcEntries);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int *deviceCount;
    conn->read(&deviceCount, sizeof(deviceCount));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCount_v2(deviceCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlDeviceAttributes_t *attributes;
    conn->read(&attributes, sizeof(attributes));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAttributes_v2(device, attributes);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int index;
    conn->read(&index, sizeof(index));
    nvmlDevice_t *device;
    conn->read(&device, sizeof(device));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetHandleByIndex_v2(index, device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    char *serial = nullptr;
    conn->read(&serial, 0, true);
    nvmlDevice_t *device;
    conn->read(&device, sizeof(device));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(serial);
    _result = nvmlDeviceGetHandleBySerial(serial, device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    char *uuid = nullptr;
    conn->read(&uuid, 0, true);
    nvmlDevice_t *device;
    conn->read(&device, sizeof(device));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(uuid);
    _result = nvmlDeviceGetHandleByUUID(uuid, device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    char *pciBusId = nullptr;
    conn->read(&pciBusId, 0, true);
    nvmlDevice_t *device;
    conn->read(&device, sizeof(device));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(pciBusId);
    _result = nvmlDeviceGetHandleByPciBusId_v2(pciBusId, device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    char name[1024];
    unsigned int length;
    conn->read(&length, sizeof(length));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetName(device, name, length);
    if(length > 0) {
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

int handle_nvmlDeviceGetBrand(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetBrand called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlBrandType_t *type;
    conn->read(&type, sizeof(type));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBrand(device, type);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *index;
    conn->read(&index, sizeof(index));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetIndex(device, index);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    char serial[1024];
    unsigned int length;
    conn->read(&length, sizeof(length));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSerial(device, serial, length);
    if(length > 0) {
        conn->write(serial, strlen(serial) + 1, true);
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

int handle_nvmlDeviceGetMemoryAffinity(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMemoryAffinity called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int nodeSetSize;
    conn->read(&nodeSetSize, sizeof(nodeSetSize));
    unsigned long *nodeSet;
    conn->read(&nodeSet, sizeof(nodeSet));
    nvmlAffinityScope_t scope;
    conn->read(&scope, sizeof(scope));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMemoryAffinity(device, nodeSetSize, nodeSet, scope);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int cpuSetSize;
    conn->read(&cpuSetSize, sizeof(cpuSetSize));
    unsigned long *cpuSet;
    conn->read(&cpuSet, sizeof(cpuSet));
    nvmlAffinityScope_t scope;
    conn->read(&scope, sizeof(scope));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCpuAffinityWithinScope(device, cpuSetSize, cpuSet, scope);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int cpuSetSize;
    conn->read(&cpuSetSize, sizeof(cpuSetSize));
    unsigned long *cpuSet;
    conn->read(&cpuSet, sizeof(cpuSet));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCpuAffinity(device, cpuSetSize, cpuSet);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetCpuAffinity(device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceClearCpuAffinity(device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device1;
    conn->read(&device1, sizeof(device1));
    nvmlDevice_t device2;
    conn->read(&device2, sizeof(device2));
    nvmlGpuTopologyLevel_t *pathInfo;
    conn->read(&pathInfo, sizeof(pathInfo));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTopologyCommonAncestor(device1, device2, pathInfo);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlGpuTopologyLevel_t level;
    conn->read(&level, sizeof(level));
    unsigned int *count;
    conn->read(&count, sizeof(count));
    nvmlDevice_t *deviceArray;
    conn->read(&deviceArray, sizeof(deviceArray));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTopologyNearestGpus(device, level, count, deviceArray);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int cpuNumber;
    conn->read(&cpuNumber, sizeof(cpuNumber));
    unsigned int *count;
    conn->read(&count, sizeof(count));
    nvmlDevice_t *deviceArray;
    conn->read(&deviceArray, sizeof(deviceArray));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSystemGetTopologyGpuSet(cpuNumber, count, deviceArray);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device1;
    conn->read(&device1, sizeof(device1));
    nvmlDevice_t device2;
    conn->read(&device2, sizeof(device2));
    nvmlGpuP2PCapsIndex_t p2pIndex;
    conn->read(&p2pIndex, sizeof(p2pIndex));
    nvmlGpuP2PStatus_t *p2pStatus;
    conn->read(&p2pStatus, sizeof(p2pStatus));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetP2PStatus(device1, device2, p2pIndex, p2pStatus);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    char uuid[1024];
    unsigned int length;
    conn->read(&length, sizeof(length));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetUUID(device, uuid, length);
    if(length > 0) {
        conn->write(uuid, strlen(uuid) + 1, true);
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

int handle_nvmlVgpuInstanceGetMdevUUID(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetMdevUUID called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    char mdevUuid[1024];
    unsigned int size;
    conn->read(&size, sizeof(size));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetMdevUUID(vgpuInstance, mdevUuid, size);
    if(size > 0) {
        conn->write(mdevUuid, strlen(mdevUuid) + 1, true);
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

int handle_nvmlDeviceGetMinorNumber(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetMinorNumber called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *minorNumber;
    conn->read(&minorNumber, sizeof(minorNumber));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMinorNumber(device, minorNumber);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    char partNumber[1024];
    unsigned int length;
    conn->read(&length, sizeof(length));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBoardPartNumber(device, partNumber, length);
    if(length > 0) {
        conn->write(partNumber, strlen(partNumber) + 1, true);
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

int handle_nvmlDeviceGetInforomVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetInforomVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlInforomObject_t object;
    conn->read(&object, sizeof(object));
    char version[1024];
    unsigned int length;
    conn->read(&length, sizeof(length));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetInforomVersion(device, object, version, length);
    if(length > 0) {
        conn->write(version, strlen(version) + 1, true);
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

int handle_nvmlDeviceGetInforomImageVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetInforomImageVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    char version[1024];
    unsigned int length;
    conn->read(&length, sizeof(length));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetInforomImageVersion(device, version, length);
    if(length > 0) {
        conn->write(version, strlen(version) + 1, true);
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

int handle_nvmlDeviceGetInforomConfigurationChecksum(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetInforomConfigurationChecksum called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *checksum;
    conn->read(&checksum, sizeof(checksum));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetInforomConfigurationChecksum(device, checksum);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceValidateInforom(device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEnableState_t *display;
    conn->read(&display, sizeof(display));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDisplayMode(device, display);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEnableState_t *isActive;
    conn->read(&isActive, sizeof(isActive));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDisplayActive(device, isActive);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEnableState_t *mode;
    conn->read(&mode, sizeof(mode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPersistenceMode(device, mode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlPciInfo_t *pci;
    conn->read(&pci, sizeof(pci));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPciInfo_v3(device, pci);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *maxLinkGen;
    conn->read(&maxLinkGen, sizeof(maxLinkGen));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxPcieLinkGeneration(device, maxLinkGen);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *maxLinkWidth;
    conn->read(&maxLinkWidth, sizeof(maxLinkWidth));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxPcieLinkWidth(device, maxLinkWidth);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *currLinkGen;
    conn->read(&currLinkGen, sizeof(currLinkGen));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCurrPcieLinkGeneration(device, currLinkGen);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *currLinkWidth;
    conn->read(&currLinkWidth, sizeof(currLinkWidth));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCurrPcieLinkWidth(device, currLinkWidth);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlPcieUtilCounter_t counter;
    conn->read(&counter, sizeof(counter));
    unsigned int *value;
    conn->read(&value, sizeof(value));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPcieThroughput(device, counter, value);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *value;
    conn->read(&value, sizeof(value));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPcieReplayCounter(device, value);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlClockType_t type;
    conn->read(&type, sizeof(type));
    unsigned int *clock;
    conn->read(&clock, sizeof(clock));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetClockInfo(device, type, clock);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlClockType_t type;
    conn->read(&type, sizeof(type));
    unsigned int *clock;
    conn->read(&clock, sizeof(clock));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxClockInfo(device, type, clock);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlClockType_t clockType;
    conn->read(&clockType, sizeof(clockType));
    unsigned int *clockMHz;
    conn->read(&clockMHz, sizeof(clockMHz));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetApplicationsClock(device, clockType, clockMHz);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlClockType_t clockType;
    conn->read(&clockType, sizeof(clockType));
    unsigned int *clockMHz;
    conn->read(&clockMHz, sizeof(clockMHz));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDefaultApplicationsClock(device, clockType, clockMHz);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceResetApplicationsClocks(device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlClockType_t clockType;
    conn->read(&clockType, sizeof(clockType));
    nvmlClockId_t clockId;
    conn->read(&clockId, sizeof(clockId));
    unsigned int *clockMHz;
    conn->read(&clockMHz, sizeof(clockMHz));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetClock(device, clockType, clockId, clockMHz);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlClockType_t clockType;
    conn->read(&clockType, sizeof(clockType));
    unsigned int *clockMHz;
    conn->read(&clockMHz, sizeof(clockMHz));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxCustomerBoostClock(device, clockType, clockMHz);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *count;
    conn->read(&count, sizeof(count));
    unsigned int *clocksMHz;
    conn->read(&clocksMHz, sizeof(clocksMHz));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedMemoryClocks(device, count, clocksMHz);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int memoryClockMHz;
    conn->read(&memoryClockMHz, sizeof(memoryClockMHz));
    unsigned int *count;
    conn->read(&count, sizeof(count));
    unsigned int *clocksMHz;
    conn->read(&clocksMHz, sizeof(clocksMHz));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, count, clocksMHz);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEnableState_t *isEnabled;
    conn->read(&isEnabled, sizeof(isEnabled));
    nvmlEnableState_t *defaultIsEnabled;
    conn->read(&defaultIsEnabled, sizeof(defaultIsEnabled));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAutoBoostedClocksEnabled(device, isEnabled, defaultIsEnabled);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEnableState_t enabled;
    conn->read(&enabled, sizeof(enabled));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetAutoBoostedClocksEnabled(device, enabled);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEnableState_t enabled;
    conn->read(&enabled, sizeof(enabled));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetDefaultAutoBoostedClocksEnabled(device, enabled, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *speed;
    conn->read(&speed, sizeof(speed));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFanSpeed(device, speed);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int fan;
    conn->read(&fan, sizeof(fan));
    unsigned int *speed;
    conn->read(&speed, sizeof(speed));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFanSpeed_v2(device, fan, speed);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlTemperatureSensors_t sensorType;
    conn->read(&sensorType, sizeof(sensorType));
    unsigned int *temp;
    conn->read(&temp, sizeof(temp));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTemperature(device, sensorType, temp);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlTemperatureThresholds_t thresholdType;
    conn->read(&thresholdType, sizeof(thresholdType));
    unsigned int *temp;
    conn->read(&temp, sizeof(temp));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTemperatureThreshold(device, thresholdType, temp);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlTemperatureThresholds_t thresholdType;
    conn->read(&thresholdType, sizeof(thresholdType));
    int *temp;
    conn->read(&temp, sizeof(temp));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetTemperatureThreshold(device, thresholdType, temp);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlPstates_t *pState;
    conn->read(&pState, sizeof(pState));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPerformanceState(device, pState);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned long long *clocksThrottleReasons;
    conn->read(&clocksThrottleReasons, sizeof(clocksThrottleReasons));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCurrentClocksThrottleReasons(device, clocksThrottleReasons);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned long long *supportedClocksThrottleReasons;
    conn->read(&supportedClocksThrottleReasons, sizeof(supportedClocksThrottleReasons));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedClocksThrottleReasons(device, supportedClocksThrottleReasons);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlPstates_t *pState;
    conn->read(&pState, sizeof(pState));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerState(device, pState);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEnableState_t *mode;
    conn->read(&mode, sizeof(mode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerManagementMode(device, mode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *limit;
    conn->read(&limit, sizeof(limit));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerManagementLimit(device, limit);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *minLimit;
    conn->read(&minLimit, sizeof(minLimit));
    unsigned int *maxLimit;
    conn->read(&maxLimit, sizeof(maxLimit));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerManagementLimitConstraints(device, minLimit, maxLimit);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *defaultLimit;
    conn->read(&defaultLimit, sizeof(defaultLimit));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerManagementDefaultLimit(device, defaultLimit);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *power;
    conn->read(&power, sizeof(power));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPowerUsage(device, power);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned long long *energy;
    conn->read(&energy, sizeof(energy));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTotalEnergyConsumption(device, energy);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *limit;
    conn->read(&limit, sizeof(limit));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEnforcedPowerLimit(device, limit);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlGpuOperationMode_t *current;
    conn->read(&current, sizeof(current));
    nvmlGpuOperationMode_t *pending;
    conn->read(&pending, sizeof(pending));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuOperationMode(device, current, pending);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlMemory_t *memory;
    conn->read(&memory, sizeof(memory));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMemoryInfo(device, memory);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlComputeMode_t *mode;
    conn->read(&mode, sizeof(mode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetComputeMode(device, mode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    int *major;
    conn->read(&major, sizeof(major));
    int *minor;
    conn->read(&minor, sizeof(minor));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCudaComputeCapability(device, major, minor);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEnableState_t *current;
    conn->read(&current, sizeof(current));
    nvmlEnableState_t *pending;
    conn->read(&pending, sizeof(pending));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEccMode(device, current, pending);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *boardId;
    conn->read(&boardId, sizeof(boardId));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBoardId(device, boardId);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *multiGpuBool;
    conn->read(&multiGpuBool, sizeof(multiGpuBool));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMultiGpuBoard(device, multiGpuBool);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlMemoryErrorType_t errorType;
    conn->read(&errorType, sizeof(errorType));
    nvmlEccCounterType_t counterType;
    conn->read(&counterType, sizeof(counterType));
    unsigned long long *eccCounts;
    conn->read(&eccCounts, sizeof(eccCounts));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetTotalEccErrors(device, errorType, counterType, eccCounts);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlMemoryErrorType_t errorType;
    conn->read(&errorType, sizeof(errorType));
    nvmlEccCounterType_t counterType;
    conn->read(&counterType, sizeof(counterType));
    nvmlEccErrorCounts_t *eccCounts;
    conn->read(&eccCounts, sizeof(eccCounts));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDetailedEccErrors(device, errorType, counterType, eccCounts);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlMemoryErrorType_t errorType;
    conn->read(&errorType, sizeof(errorType));
    nvmlEccCounterType_t counterType;
    conn->read(&counterType, sizeof(counterType));
    nvmlMemoryLocation_t locationType;
    conn->read(&locationType, sizeof(locationType));
    unsigned long long *count;
    conn->read(&count, sizeof(count));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMemoryErrorCounter(device, errorType, counterType, locationType, count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlUtilization_t *utilization;
    conn->read(&utilization, sizeof(utilization));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetUtilizationRates(device, utilization);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *utilization;
    conn->read(&utilization, sizeof(utilization));
    unsigned int *samplingPeriodUs;
    conn->read(&samplingPeriodUs, sizeof(samplingPeriodUs));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEncoderUtilization(device, utilization, samplingPeriodUs);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEncoderType_t encoderQueryType;
    conn->read(&encoderQueryType, sizeof(encoderQueryType));
    unsigned int *encoderCapacity;
    conn->read(&encoderCapacity, sizeof(encoderCapacity));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEncoderCapacity(device, encoderQueryType, encoderCapacity);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *sessionCount;
    conn->read(&sessionCount, sizeof(sessionCount));
    unsigned int *averageFps;
    conn->read(&averageFps, sizeof(averageFps));
    unsigned int *averageLatency;
    conn->read(&averageLatency, sizeof(averageLatency));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEncoderStats(device, sessionCount, averageFps, averageLatency);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *sessionCount;
    conn->read(&sessionCount, sizeof(sessionCount));
    nvmlEncoderSessionInfo_t *sessionInfos;
    conn->read(&sessionInfos, sizeof(sessionInfos));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetEncoderSessions(device, sessionCount, sessionInfos);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *utilization;
    conn->read(&utilization, sizeof(utilization));
    unsigned int *samplingPeriodUs;
    conn->read(&samplingPeriodUs, sizeof(samplingPeriodUs));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDecoderUtilization(device, utilization, samplingPeriodUs);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlFBCStats_t *fbcStats;
    conn->read(&fbcStats, sizeof(fbcStats));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFBCStats(device, fbcStats);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *sessionCount;
    conn->read(&sessionCount, sizeof(sessionCount));
    nvmlFBCSessionInfo_t *sessionInfo;
    conn->read(&sessionInfo, sizeof(sessionInfo));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFBCSessions(device, sessionCount, sessionInfo);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlDriverModel_t *current;
    conn->read(&current, sizeof(current));
    nvmlDriverModel_t *pending;
    conn->read(&pending, sizeof(pending));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDriverModel(device, current, pending);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    char version[1024];
    unsigned int length;
    conn->read(&length, sizeof(length));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVbiosVersion(device, version, length);
    if(length > 0) {
        conn->write(version, strlen(version) + 1, true);
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

int handle_nvmlDeviceGetBridgeChipInfo(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlDeviceGetBridgeChipInfo called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlBridgeChipHierarchy_t *bridgeHierarchy;
    conn->read(&bridgeHierarchy, sizeof(bridgeHierarchy));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBridgeChipInfo(device, bridgeHierarchy);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *infoCount;
    conn->read(&infoCount, sizeof(infoCount));
    nvmlProcessInfo_t *infos;
    conn->read(&infos, sizeof(infos));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetComputeRunningProcesses_v2(device, infoCount, infos);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *infoCount;
    conn->read(&infoCount, sizeof(infoCount));
    nvmlProcessInfo_t *infos;
    conn->read(&infos, sizeof(infos));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGraphicsRunningProcesses_v2(device, infoCount, infos);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *infoCount;
    conn->read(&infoCount, sizeof(infoCount));
    nvmlProcessInfo_t *infos;
    conn->read(&infos, sizeof(infos));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMPSComputeRunningProcesses_v2(device, infoCount, infos);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device1;
    conn->read(&device1, sizeof(device1));
    nvmlDevice_t device2;
    conn->read(&device2, sizeof(device2));
    int *onSameBoard;
    conn->read(&onSameBoard, sizeof(onSameBoard));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceOnSameBoard(device1, device2, onSameBoard);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlRestrictedAPI_t apiType;
    conn->read(&apiType, sizeof(apiType));
    nvmlEnableState_t *isRestricted;
    conn->read(&isRestricted, sizeof(isRestricted));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAPIRestriction(device, apiType, isRestricted);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlSamplingType_t type;
    conn->read(&type, sizeof(type));
    unsigned long long lastSeenTimeStamp;
    conn->read(&lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    nvmlValueType_t *sampleValType;
    conn->read(&sampleValType, sizeof(sampleValType));
    unsigned int *sampleCount;
    conn->read(&sampleCount, sizeof(sampleCount));
    nvmlSample_t *samples;
    conn->read(&samples, sizeof(samples));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, sampleValType, sampleCount, samples);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlBAR1Memory_t *bar1Memory;
    conn->read(&bar1Memory, sizeof(bar1Memory));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetBAR1MemoryInfo(device, bar1Memory);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlPerfPolicyType_t perfPolicyType;
    conn->read(&perfPolicyType, sizeof(perfPolicyType));
    nvmlViolationTime_t *violTime;
    conn->read(&violTime, sizeof(violTime));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetViolationStatus(device, perfPolicyType, violTime);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEnableState_t *mode;
    conn->read(&mode, sizeof(mode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAccountingMode(device, mode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int pid;
    conn->read(&pid, sizeof(pid));
    nvmlAccountingStats_t *stats;
    conn->read(&stats, sizeof(stats));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAccountingStats(device, pid, stats);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *count;
    conn->read(&count, sizeof(count));
    unsigned int *pids;
    conn->read(&pids, sizeof(pids));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAccountingPids(device, count, pids);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *bufferSize;
    conn->read(&bufferSize, sizeof(bufferSize));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetAccountingBufferSize(device, bufferSize);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlPageRetirementCause_t cause;
    conn->read(&cause, sizeof(cause));
    unsigned int *pageCount;
    conn->read(&pageCount, sizeof(pageCount));
    unsigned long long *addresses;
    conn->read(&addresses, sizeof(addresses));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRetiredPages(device, cause, pageCount, addresses);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlPageRetirementCause_t cause;
    conn->read(&cause, sizeof(cause));
    unsigned int *pageCount;
    conn->read(&pageCount, sizeof(pageCount));
    unsigned long long *addresses;
    conn->read(&addresses, sizeof(addresses));
    unsigned long long *timestamps;
    conn->read(&timestamps, sizeof(timestamps));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRetiredPages_v2(device, cause, pageCount, addresses, timestamps);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEnableState_t *isPending;
    conn->read(&isPending, sizeof(isPending));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRetiredPagesPendingStatus(device, isPending);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *corrRows;
    conn->read(&corrRows, sizeof(corrRows));
    unsigned int *uncRows;
    conn->read(&uncRows, sizeof(uncRows));
    unsigned int *isPending;
    conn->read(&isPending, sizeof(isPending));
    unsigned int *failureOccurred;
    conn->read(&failureOccurred, sizeof(failureOccurred));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRemappedRows(device, corrRows, uncRows, isPending, failureOccurred);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlRowRemapperHistogramValues_t *values;
    conn->read(&values, sizeof(values));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetRowRemapperHistogram(device, values);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlDeviceArchitecture_t *arch;
    conn->read(&arch, sizeof(arch));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetArchitecture(device, arch);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlUnit_t unit;
    conn->read(&unit, sizeof(unit));
    nvmlLedColor_t color;
    conn->read(&color, sizeof(color));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlUnitSetLedState(unit, color);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEnableState_t mode;
    conn->read(&mode, sizeof(mode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetPersistenceMode(device, mode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlComputeMode_t mode;
    conn->read(&mode, sizeof(mode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetComputeMode(device, mode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEnableState_t ecc;
    conn->read(&ecc, sizeof(ecc));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetEccMode(device, ecc);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEccCounterType_t counterType;
    conn->read(&counterType, sizeof(counterType));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceClearEccErrorCounts(device, counterType);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlDriverModel_t driverModel;
    conn->read(&driverModel, sizeof(driverModel));
    unsigned int flags;
    conn->read(&flags, sizeof(flags));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetDriverModel(device, driverModel, flags);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int minGpuClockMHz;
    conn->read(&minGpuClockMHz, sizeof(minGpuClockMHz));
    unsigned int maxGpuClockMHz;
    conn->read(&maxGpuClockMHz, sizeof(maxGpuClockMHz));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetGpuLockedClocks(device, minGpuClockMHz, maxGpuClockMHz);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceResetGpuLockedClocks(device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int minMemClockMHz;
    conn->read(&minMemClockMHz, sizeof(minMemClockMHz));
    unsigned int maxMemClockMHz;
    conn->read(&maxMemClockMHz, sizeof(maxMemClockMHz));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetMemoryLockedClocks(device, minMemClockMHz, maxMemClockMHz);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceResetMemoryLockedClocks(device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int memClockMHz;
    conn->read(&memClockMHz, sizeof(memClockMHz));
    unsigned int graphicsClockMHz;
    conn->read(&graphicsClockMHz, sizeof(graphicsClockMHz));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetApplicationsClocks(device, memClockMHz, graphicsClockMHz);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlClkMonStatus_t *status;
    conn->read(&status, sizeof(status));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetClkMonStatus(device, status);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int limit;
    conn->read(&limit, sizeof(limit));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetPowerManagementLimit(device, limit);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlGpuOperationMode_t mode;
    conn->read(&mode, sizeof(mode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetGpuOperationMode(device, mode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlRestrictedAPI_t apiType;
    conn->read(&apiType, sizeof(apiType));
    nvmlEnableState_t isRestricted;
    conn->read(&isRestricted, sizeof(isRestricted));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetAPIRestriction(device, apiType, isRestricted);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlEnableState_t mode;
    conn->read(&mode, sizeof(mode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetAccountingMode(device, mode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceClearAccountingPids(device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int link;
    conn->read(&link, sizeof(link));
    nvmlEnableState_t *isActive;
    conn->read(&isActive, sizeof(isActive));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkState(device, link, isActive);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int link;
    conn->read(&link, sizeof(link));
    unsigned int *version;
    conn->read(&version, sizeof(version));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkVersion(device, link, version);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int link;
    conn->read(&link, sizeof(link));
    nvmlNvLinkCapability_t capability;
    conn->read(&capability, sizeof(capability));
    unsigned int *capResult;
    conn->read(&capResult, sizeof(capResult));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkCapability(device, link, capability, capResult);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int link;
    conn->read(&link, sizeof(link));
    nvmlPciInfo_t *pci;
    conn->read(&pci, sizeof(pci));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, pci);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int link;
    conn->read(&link, sizeof(link));
    nvmlNvLinkErrorCounter_t counter;
    conn->read(&counter, sizeof(counter));
    unsigned long long *counterValue;
    conn->read(&counterValue, sizeof(counterValue));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkErrorCounter(device, link, counter, counterValue);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int link;
    conn->read(&link, sizeof(link));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceResetNvLinkErrorCounters(device, link);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int link;
    conn->read(&link, sizeof(link));
    unsigned int counter;
    conn->read(&counter, sizeof(counter));
    nvmlNvLinkUtilizationControl_t *control;
    conn->read(&control, sizeof(control));
    unsigned int reset;
    conn->read(&reset, sizeof(reset));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetNvLinkUtilizationControl(device, link, counter, control, reset);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int link;
    conn->read(&link, sizeof(link));
    unsigned int counter;
    conn->read(&counter, sizeof(counter));
    nvmlNvLinkUtilizationControl_t *control;
    conn->read(&control, sizeof(control));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkUtilizationControl(device, link, counter, control);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int link;
    conn->read(&link, sizeof(link));
    unsigned int counter;
    conn->read(&counter, sizeof(counter));
    unsigned long long *rxcounter;
    conn->read(&rxcounter, sizeof(rxcounter));
    unsigned long long *txcounter;
    conn->read(&txcounter, sizeof(txcounter));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter, rxcounter, txcounter);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int link;
    conn->read(&link, sizeof(link));
    unsigned int counter;
    conn->read(&counter, sizeof(counter));
    nvmlEnableState_t freeze;
    conn->read(&freeze, sizeof(freeze));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceFreezeNvLinkUtilizationCounter(device, link, counter, freeze);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int link;
    conn->read(&link, sizeof(link));
    unsigned int counter;
    conn->read(&counter, sizeof(counter));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceResetNvLinkUtilizationCounter(device, link, counter);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int link;
    conn->read(&link, sizeof(link));
    nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType;
    conn->read(&pNvLinkDeviceType, sizeof(pNvLinkDeviceType));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetNvLinkRemoteDeviceType(device, link, pNvLinkDeviceType);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlEventSet_t *set;
    conn->read(&set, sizeof(set));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlEventSetCreate(set);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned long long eventTypes;
    conn->read(&eventTypes, sizeof(eventTypes));
    nvmlEventSet_t set;
    conn->read(&set, sizeof(set));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceRegisterEvents(device, eventTypes, set);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned long long *eventTypes;
    conn->read(&eventTypes, sizeof(eventTypes));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedEventTypes(device, eventTypes);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlEventSet_t set;
    conn->read(&set, sizeof(set));
    nvmlEventData_t *data;
    conn->read(&data, sizeof(data));
    unsigned int timeoutms;
    conn->read(&timeoutms, sizeof(timeoutms));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlEventSetWait_v2(set, data, timeoutms);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlEventSet_t set;
    conn->read(&set, sizeof(set));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlEventSetFree(set);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlPciInfo_t *pciInfo;
    conn->read(&pciInfo, sizeof(pciInfo));
    nvmlEnableState_t newState;
    conn->read(&newState, sizeof(newState));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceModifyDrainState(pciInfo, newState);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlPciInfo_t *pciInfo;
    conn->read(&pciInfo, sizeof(pciInfo));
    nvmlEnableState_t *currentState;
    conn->read(&currentState, sizeof(currentState));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceQueryDrainState(pciInfo, currentState);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlPciInfo_t *pciInfo;
    conn->read(&pciInfo, sizeof(pciInfo));
    nvmlDetachGpuState_t gpuState;
    conn->read(&gpuState, sizeof(gpuState));
    nvmlPcieLinkState_t linkState;
    conn->read(&linkState, sizeof(linkState));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceRemoveGpu_v2(pciInfo, gpuState, linkState);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlPciInfo_t *pciInfo;
    conn->read(&pciInfo, sizeof(pciInfo));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceDiscoverGpus(pciInfo);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    int valuesCount;
    conn->read(&valuesCount, sizeof(valuesCount));
    nvmlFieldValue_t *values;
    conn->read(&values, sizeof(values));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetFieldValues(device, valuesCount, values);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlGpuVirtualizationMode_t *pVirtualMode;
    conn->read(&pVirtualMode, sizeof(pVirtualMode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVirtualizationMode(device, pVirtualMode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlHostVgpuMode_t *pHostVgpuMode;
    conn->read(&pHostVgpuMode, sizeof(pHostVgpuMode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetHostVgpuMode(device, pHostVgpuMode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlGpuVirtualizationMode_t virtualMode;
    conn->read(&virtualMode, sizeof(virtualMode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetVirtualizationMode(device, virtualMode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlGridLicensableFeatures_t *pGridLicensableFeatures;
    conn->read(&pGridLicensableFeatures, sizeof(pGridLicensableFeatures));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGridLicensableFeatures_v3(device, pGridLicensableFeatures);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlProcessUtilizationSample_t *utilization;
    conn->read(&utilization, sizeof(utilization));
    unsigned int *processSamplesCount;
    conn->read(&processSamplesCount, sizeof(processSamplesCount));
    unsigned long long lastSeenTimeStamp;
    conn->read(&lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetProcessUtilization(device, utilization, processSamplesCount, lastSeenTimeStamp);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *vgpuCount;
    conn->read(&vgpuCount, sizeof(vgpuCount));
    nvmlVgpuTypeId_t *vgpuTypeIds;
    conn->read(&vgpuTypeIds, sizeof(vgpuTypeIds));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetSupportedVgpus(device, vgpuCount, vgpuTypeIds);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *vgpuCount;
    conn->read(&vgpuCount, sizeof(vgpuCount));
    nvmlVgpuTypeId_t *vgpuTypeIds;
    conn->read(&vgpuTypeIds, sizeof(vgpuTypeIds));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetCreatableVgpus(device, vgpuCount, vgpuTypeIds);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    conn->read(&vgpuTypeId, sizeof(vgpuTypeId));
    char vgpuTypeClass[1024];
    unsigned int *size;
    conn->read(&size, sizeof(size));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetClass(vgpuTypeId, vgpuTypeClass, size);
    if(*size > 0) {
        conn->write(vgpuTypeClass, strlen(vgpuTypeClass) + 1, true);
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

int handle_nvmlVgpuTypeGetName(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetName called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    conn->read(&vgpuTypeId, sizeof(vgpuTypeId));
    char vgpuTypeName[1024];
    unsigned int *size;
    conn->read(&size, sizeof(size));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetName(vgpuTypeId, vgpuTypeName, size);
    if(*size > 0) {
        conn->write(vgpuTypeName, strlen(vgpuTypeName) + 1, true);
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

int handle_nvmlVgpuTypeGetGpuInstanceProfileId(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetGpuInstanceProfileId called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    conn->read(&vgpuTypeId, sizeof(vgpuTypeId));
    unsigned int *gpuInstanceProfileId;
    conn->read(&gpuInstanceProfileId, sizeof(gpuInstanceProfileId));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId, gpuInstanceProfileId);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    conn->read(&vgpuTypeId, sizeof(vgpuTypeId));
    unsigned long long *deviceID;
    conn->read(&deviceID, sizeof(deviceID));
    unsigned long long *subsystemID;
    conn->read(&subsystemID, sizeof(subsystemID));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetDeviceID(vgpuTypeId, deviceID, subsystemID);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    conn->read(&vgpuTypeId, sizeof(vgpuTypeId));
    unsigned long long *fbSize;
    conn->read(&fbSize, sizeof(fbSize));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetFramebufferSize(vgpuTypeId, fbSize);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    conn->read(&vgpuTypeId, sizeof(vgpuTypeId));
    unsigned int *numDisplayHeads;
    conn->read(&numDisplayHeads, sizeof(numDisplayHeads));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId, numDisplayHeads);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    conn->read(&vgpuTypeId, sizeof(vgpuTypeId));
    unsigned int displayIndex;
    conn->read(&displayIndex, sizeof(displayIndex));
    unsigned int *xdim;
    conn->read(&xdim, sizeof(xdim));
    unsigned int *ydim;
    conn->read(&ydim, sizeof(ydim));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetResolution(vgpuTypeId, displayIndex, xdim, ydim);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    conn->read(&vgpuTypeId, sizeof(vgpuTypeId));
    char vgpuTypeLicenseString[1024];
    unsigned int size;
    conn->read(&size, sizeof(size));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetLicense(vgpuTypeId, vgpuTypeLicenseString, size);
    if(size > 0) {
        conn->write(vgpuTypeLicenseString, strlen(vgpuTypeLicenseString) + 1, true);
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

int handle_nvmlVgpuTypeGetFrameRateLimit(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuTypeGetFrameRateLimit called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    conn->read(&vgpuTypeId, sizeof(vgpuTypeId));
    unsigned int *frameRateLimit;
    conn->read(&frameRateLimit, sizeof(frameRateLimit));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId, frameRateLimit);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlVgpuTypeId_t vgpuTypeId;
    conn->read(&vgpuTypeId, sizeof(vgpuTypeId));
    unsigned int *vgpuInstanceCount;
    conn->read(&vgpuInstanceCount, sizeof(vgpuInstanceCount));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetMaxInstances(device, vgpuTypeId, vgpuInstanceCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuTypeId_t vgpuTypeId;
    conn->read(&vgpuTypeId, sizeof(vgpuTypeId));
    unsigned int *vgpuInstanceCountPerVm;
    conn->read(&vgpuInstanceCountPerVm, sizeof(vgpuInstanceCountPerVm));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId, vgpuInstanceCountPerVm);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *vgpuCount;
    conn->read(&vgpuCount, sizeof(vgpuCount));
    nvmlVgpuInstance_t *vgpuInstances;
    conn->read(&vgpuInstances, sizeof(vgpuInstances));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetActiveVgpus(device, vgpuCount, vgpuInstances);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    char vmId[1024];
    unsigned int size;
    conn->read(&size, sizeof(size));
    nvmlVgpuVmIdType_t *vmIdType;
    conn->read(&vmIdType, sizeof(vmIdType));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetVmID(vgpuInstance, vmId, size, vmIdType);
    if(size > 0) {
        conn->write(vmId, strlen(vmId) + 1, true);
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

int handle_nvmlVgpuInstanceGetUUID(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetUUID called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    char uuid[1024];
    unsigned int size;
    conn->read(&size, sizeof(size));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetUUID(vgpuInstance, uuid, size);
    if(size > 0) {
        conn->write(uuid, strlen(uuid) + 1, true);
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

int handle_nvmlVgpuInstanceGetVmDriverVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetVmDriverVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    char version[1024];
    unsigned int length;
    conn->read(&length, sizeof(length));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance, version, length);
    if(length > 0) {
        conn->write(version, strlen(version) + 1, true);
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

int handle_nvmlVgpuInstanceGetFbUsage(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlVgpuInstanceGetFbUsage called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    unsigned long long *fbUsage;
    conn->read(&fbUsage, sizeof(fbUsage));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetFbUsage(vgpuInstance, fbUsage);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    unsigned int *licensed;
    conn->read(&licensed, sizeof(licensed));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetLicenseStatus(vgpuInstance, licensed);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    nvmlVgpuTypeId_t *vgpuTypeId;
    conn->read(&vgpuTypeId, sizeof(vgpuTypeId));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetType(vgpuInstance, vgpuTypeId);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    unsigned int *frameRateLimit;
    conn->read(&frameRateLimit, sizeof(frameRateLimit));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance, frameRateLimit);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    nvmlEnableState_t *eccMode;
    conn->read(&eccMode, sizeof(eccMode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetEccMode(vgpuInstance, eccMode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    unsigned int *encoderCapacity;
    conn->read(&encoderCapacity, sizeof(encoderCapacity));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance, encoderCapacity);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    unsigned int encoderCapacity;
    conn->read(&encoderCapacity, sizeof(encoderCapacity));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, encoderCapacity);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    unsigned int *sessionCount;
    conn->read(&sessionCount, sizeof(sessionCount));
    unsigned int *averageFps;
    conn->read(&averageFps, sizeof(averageFps));
    unsigned int *averageLatency;
    conn->read(&averageLatency, sizeof(averageLatency));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetEncoderStats(vgpuInstance, sessionCount, averageFps, averageLatency);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    unsigned int *sessionCount;
    conn->read(&sessionCount, sizeof(sessionCount));
    nvmlEncoderSessionInfo_t *sessionInfo;
    conn->read(&sessionInfo, sizeof(sessionInfo));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetEncoderSessions(vgpuInstance, sessionCount, sessionInfo);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    nvmlFBCStats_t *fbcStats;
    conn->read(&fbcStats, sizeof(fbcStats));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetFBCStats(vgpuInstance, fbcStats);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    unsigned int *sessionCount;
    conn->read(&sessionCount, sizeof(sessionCount));
    nvmlFBCSessionInfo_t *sessionInfo;
    conn->read(&sessionInfo, sizeof(sessionInfo));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetFBCSessions(vgpuInstance, sessionCount, sessionInfo);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    unsigned int *gpuInstanceId;
    conn->read(&gpuInstanceId, sizeof(gpuInstanceId));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance, gpuInstanceId);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    nvmlVgpuMetadata_t *vgpuMetadata;
    conn->read(&vgpuMetadata, sizeof(vgpuMetadata));
    unsigned int *bufferSize;
    conn->read(&bufferSize, sizeof(bufferSize));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetMetadata(vgpuInstance, vgpuMetadata, bufferSize);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    nvmlVgpuPgpuMetadata_t *pgpuMetadata;
    conn->read(&pgpuMetadata, sizeof(pgpuMetadata));
    unsigned int *bufferSize;
    conn->read(&bufferSize, sizeof(bufferSize));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuMetadata(device, pgpuMetadata, bufferSize);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuMetadata_t *vgpuMetadata;
    conn->read(&vgpuMetadata, sizeof(vgpuMetadata));
    nvmlVgpuPgpuMetadata_t *pgpuMetadata;
    conn->read(&pgpuMetadata, sizeof(pgpuMetadata));
    nvmlVgpuPgpuCompatibility_t *compatibilityInfo;
    conn->read(&compatibilityInfo, sizeof(compatibilityInfo));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGetVgpuCompatibility(vgpuMetadata, pgpuMetadata, compatibilityInfo);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    char pgpuMetadata[1024];
    unsigned int *bufferSize;
    conn->read(&bufferSize, sizeof(bufferSize));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetPgpuMetadataString(device, pgpuMetadata, bufferSize);
    if(*bufferSize > 0) {
        conn->write(pgpuMetadata, strlen(pgpuMetadata) + 1, true);
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

int handle_nvmlGetVgpuVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function nvmlGetVgpuVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuVersion_t *supported;
    conn->read(&supported, sizeof(supported));
    nvmlVgpuVersion_t *current;
    conn->read(&current, sizeof(current));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGetVgpuVersion(supported, current);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuVersion_t *vgpuVersion;
    conn->read(&vgpuVersion, sizeof(vgpuVersion));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlSetVgpuVersion(vgpuVersion);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned long long lastSeenTimeStamp;
    conn->read(&lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    nvmlValueType_t *sampleValType;
    conn->read(&sampleValType, sizeof(sampleValType));
    unsigned int *vgpuInstanceSamplesCount;
    conn->read(&vgpuInstanceSamplesCount, sizeof(vgpuInstanceSamplesCount));
    nvmlVgpuInstanceUtilizationSample_t *utilizationSamples;
    conn->read(&utilizationSamples, sizeof(utilizationSamples));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuUtilization(device, lastSeenTimeStamp, sampleValType, vgpuInstanceSamplesCount, utilizationSamples);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned long long lastSeenTimeStamp;
    conn->read(&lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    unsigned int *vgpuProcessSamplesCount;
    conn->read(&vgpuProcessSamplesCount, sizeof(vgpuProcessSamplesCount));
    nvmlVgpuProcessUtilizationSample_t *utilizationSamples;
    conn->read(&utilizationSamples, sizeof(utilizationSamples));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetVgpuProcessUtilization(device, lastSeenTimeStamp, vgpuProcessSamplesCount, utilizationSamples);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    nvmlEnableState_t *mode;
    conn->read(&mode, sizeof(mode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetAccountingMode(vgpuInstance, mode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    unsigned int *count;
    conn->read(&count, sizeof(count));
    unsigned int *pids;
    conn->read(&pids, sizeof(pids));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetAccountingPids(vgpuInstance, count, pids);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    unsigned int pid;
    conn->read(&pid, sizeof(pid));
    nvmlAccountingStats_t *stats;
    conn->read(&stats, sizeof(stats));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid, stats);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlVgpuInstance_t vgpuInstance;
    conn->read(&vgpuInstance, sizeof(vgpuInstance));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlVgpuInstanceClearAccountingPids(vgpuInstance);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int *deviceCount;
    conn->read(&deviceCount, sizeof(deviceCount));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGetExcludedDeviceCount(deviceCount);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    unsigned int index;
    conn->read(&index, sizeof(index));
    nvmlExcludedDeviceInfo_t *info;
    conn->read(&info, sizeof(info));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGetExcludedDeviceInfoByIndex(index, info);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int mode;
    conn->read(&mode, sizeof(mode));
    nvmlReturn_t *activationStatus;
    conn->read(&activationStatus, sizeof(activationStatus));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceSetMigMode(device, mode, activationStatus);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *currentMode;
    conn->read(&currentMode, sizeof(currentMode));
    unsigned int *pendingMode;
    conn->read(&pendingMode, sizeof(pendingMode));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMigMode(device, currentMode, pendingMode);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int profile;
    conn->read(&profile, sizeof(profile));
    nvmlGpuInstanceProfileInfo_t *info;
    conn->read(&info, sizeof(info));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstanceProfileInfo(device, profile, info);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int profileId;
    conn->read(&profileId, sizeof(profileId));
    nvmlGpuInstancePlacement_t *placements;
    conn->read(&placements, sizeof(placements));
    unsigned int *count;
    conn->read(&count, sizeof(count));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstancePossiblePlacements_v2(device, profileId, placements, count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int profileId;
    conn->read(&profileId, sizeof(profileId));
    unsigned int *count;
    conn->read(&count, sizeof(count));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstanceRemainingCapacity(device, profileId, count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int profileId;
    conn->read(&profileId, sizeof(profileId));
    nvmlGpuInstance_t *gpuInstance;
    conn->read(&gpuInstance, sizeof(gpuInstance));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceCreateGpuInstance(device, profileId, gpuInstance);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int profileId;
    conn->read(&profileId, sizeof(profileId));
    nvmlGpuInstancePlacement_t *placement = nullptr;
    conn->read(&placement, sizeof(placement));
    nvmlGpuInstance_t *gpuInstance;
    conn->read(&gpuInstance, sizeof(gpuInstance));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, placement, gpuInstance);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlGpuInstance_t gpuInstance;
    conn->read(&gpuInstance, sizeof(gpuInstance));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceDestroy(gpuInstance);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int profileId;
    conn->read(&profileId, sizeof(profileId));
    nvmlGpuInstance_t *gpuInstances;
    conn->read(&gpuInstances, sizeof(gpuInstances));
    unsigned int *count;
    conn->read(&count, sizeof(count));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstances(device, profileId, gpuInstances, count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int id;
    conn->read(&id, sizeof(id));
    nvmlGpuInstance_t *gpuInstance;
    conn->read(&gpuInstance, sizeof(gpuInstance));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstanceById(device, id, gpuInstance);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlGpuInstance_t gpuInstance;
    conn->read(&gpuInstance, sizeof(gpuInstance));
    nvmlGpuInstanceInfo_t *info;
    conn->read(&info, sizeof(info));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetInfo(gpuInstance, info);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlGpuInstance_t gpuInstance;
    conn->read(&gpuInstance, sizeof(gpuInstance));
    unsigned int profile;
    conn->read(&profile, sizeof(profile));
    unsigned int engProfile;
    conn->read(&engProfile, sizeof(engProfile));
    nvmlComputeInstanceProfileInfo_t *info;
    conn->read(&info, sizeof(info));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstanceProfileInfo(gpuInstance, profile, engProfile, info);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlGpuInstance_t gpuInstance;
    conn->read(&gpuInstance, sizeof(gpuInstance));
    unsigned int profileId;
    conn->read(&profileId, sizeof(profileId));
    unsigned int *count;
    conn->read(&count, sizeof(count));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, profileId, count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlGpuInstance_t gpuInstance;
    conn->read(&gpuInstance, sizeof(gpuInstance));
    unsigned int profileId;
    conn->read(&profileId, sizeof(profileId));
    nvmlComputeInstance_t *computeInstance;
    conn->read(&computeInstance, sizeof(computeInstance));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId, computeInstance);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlComputeInstance_t computeInstance;
    conn->read(&computeInstance, sizeof(computeInstance));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlComputeInstanceDestroy(computeInstance);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlGpuInstance_t gpuInstance;
    conn->read(&gpuInstance, sizeof(gpuInstance));
    unsigned int profileId;
    conn->read(&profileId, sizeof(profileId));
    nvmlComputeInstance_t *computeInstances;
    conn->read(&computeInstances, sizeof(computeInstances));
    unsigned int *count;
    conn->read(&count, sizeof(count));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId, computeInstances, count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlGpuInstance_t gpuInstance;
    conn->read(&gpuInstance, sizeof(gpuInstance));
    unsigned int id;
    conn->read(&id, sizeof(id));
    nvmlComputeInstance_t *computeInstance;
    conn->read(&computeInstance, sizeof(computeInstance));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlGpuInstanceGetComputeInstanceById(gpuInstance, id, computeInstance);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlComputeInstance_t computeInstance;
    conn->read(&computeInstance, sizeof(computeInstance));
    nvmlComputeInstanceInfo_t *info;
    conn->read(&info, sizeof(info));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlComputeInstanceGetInfo_v2(computeInstance, info);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *isMigDevice;
    conn->read(&isMigDevice, sizeof(isMigDevice));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceIsMigDeviceHandle(device, isMigDevice);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *id;
    conn->read(&id, sizeof(id));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetGpuInstanceId(device, id);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *id;
    conn->read(&id, sizeof(id));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetComputeInstanceId(device, id);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int *count;
    conn->read(&count, sizeof(count));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMaxMigDeviceCount(device, count);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t device;
    conn->read(&device, sizeof(device));
    unsigned int index;
    conn->read(&index, sizeof(index));
    nvmlDevice_t *migDevice;
    conn->read(&migDevice, sizeof(migDevice));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetMigDeviceHandleByIndex(device, index, migDevice);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
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
    RpcConn *conn = (RpcConn *)args0;
    nvmlDevice_t migDevice;
    conn->read(&migDevice, sizeof(migDevice));
    nvmlDevice_t *device;
    conn->read(&device, sizeof(device));
    nvmlReturn_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice, device);
    conn->write(&_result, sizeof(_result));
    if(conn->submit_response() != RpcError::OK) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}
