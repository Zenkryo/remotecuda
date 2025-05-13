#include <iostream>
#include <map>
#include "nvml.h"

#include "hook_api.h"
#include "client.h"
extern "C" nvmlReturn_t nvmlInit_v2() {
#ifdef DEBUG
    std::cout << "Hook: nvmlInit_v2 called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlInit_v2);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: nvmlInitWithFlags called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlInitWithFlags);
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlShutdown() {
#ifdef DEBUG
    std::cout << "Hook: nvmlShutdown called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlShutdown);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetDriverVersion called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlSystemGetDriverVersion);
    if(length > 0) {
        conn->read(version, length, true);
    }
    conn->write(&length, sizeof(length));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetNVMLVersion called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlSystemGetNVMLVersion);
    if(length > 0) {
        conn->read(version, length, true);
    }
    conn->write(&length, sizeof(length));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetCudaDriverVersion called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0cudaDriverVersion;
    mem2server(conn, &_0cudaDriverVersion, (void *)cudaDriverVersion, sizeof(*cudaDriverVersion));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlSystemGetCudaDriverVersion);
    conn->write(&_0cudaDriverVersion, sizeof(_0cudaDriverVersion));
    updateTmpPtr((void *)cudaDriverVersion, _0cudaDriverVersion);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)cudaDriverVersion, sizeof(*cudaDriverVersion), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetCudaDriverVersion_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0cudaDriverVersion;
    mem2server(conn, &_0cudaDriverVersion, (void *)cudaDriverVersion, sizeof(*cudaDriverVersion));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlSystemGetCudaDriverVersion_v2);
    conn->write(&_0cudaDriverVersion, sizeof(_0cudaDriverVersion));
    updateTmpPtr((void *)cudaDriverVersion, _0cudaDriverVersion);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)cudaDriverVersion, sizeof(*cudaDriverVersion), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char *name, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetProcessName called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlSystemGetProcessName);
    conn->write(&pid, sizeof(pid));
    if(length > 0) {
        conn->read(name, length, true);
    }
    conn->write(&length, sizeof(length));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetCount called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0unitCount;
    mem2server(conn, &_0unitCount, (void *)unitCount, sizeof(*unitCount));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlUnitGetCount);
    conn->write(&_0unitCount, sizeof(_0unitCount));
    updateTmpPtr((void *)unitCount, _0unitCount);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)unitCount, sizeof(*unitCount), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t *unit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetHandleByIndex called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0unit;
    mem2server(conn, &_0unit, (void *)unit, sizeof(*unit));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlUnitGetHandleByIndex);
    conn->write(&index, sizeof(index));
    conn->write(&_0unit, sizeof(_0unit));
    updateTmpPtr((void *)unit, _0unit);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)unit, sizeof(*unit), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetUnitInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0info;
    mem2server(conn, &_0info, (void *)info, sizeof(*info));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlUnitGetUnitInfo);
    conn->write(&unit, sizeof(unit));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)info, sizeof(*info), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t *state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetLedState called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0state;
    mem2server(conn, &_0state, (void *)state, sizeof(*state));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlUnitGetLedState);
    conn->write(&unit, sizeof(unit));
    conn->write(&_0state, sizeof(_0state));
    updateTmpPtr((void *)state, _0state);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)state, sizeof(*state), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t *psu) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetPsuInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0psu;
    mem2server(conn, &_0psu, (void *)psu, sizeof(*psu));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlUnitGetPsuInfo);
    conn->write(&unit, sizeof(unit));
    conn->write(&_0psu, sizeof(_0psu));
    updateTmpPtr((void *)psu, _0psu);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)psu, sizeof(*psu), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetTemperature called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0temp;
    mem2server(conn, &_0temp, (void *)temp, sizeof(*temp));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlUnitGetTemperature);
    conn->write(&unit, sizeof(unit));
    conn->write(&type, sizeof(type));
    conn->write(&_0temp, sizeof(_0temp));
    updateTmpPtr((void *)temp, _0temp);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)temp, sizeof(*temp), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t *fanSpeeds) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetFanSpeedInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0fanSpeeds;
    mem2server(conn, &_0fanSpeeds, (void *)fanSpeeds, sizeof(*fanSpeeds));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlUnitGetFanSpeedInfo);
    conn->write(&unit, sizeof(unit));
    conn->write(&_0fanSpeeds, sizeof(_0fanSpeeds));
    updateTmpPtr((void *)fanSpeeds, _0fanSpeeds);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)fanSpeeds, sizeof(*fanSpeeds), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int *deviceCount, nvmlDevice_t *devices) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitGetDevices called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0deviceCount;
    mem2server(conn, &_0deviceCount, (void *)deviceCount, sizeof(*deviceCount));
    void *_0devices;
    mem2server(conn, &_0devices, (void *)devices, sizeof(*devices));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlUnitGetDevices);
    conn->write(&unit, sizeof(unit));
    conn->write(&_0deviceCount, sizeof(_0deviceCount));
    updateTmpPtr((void *)deviceCount, _0deviceCount);
    conn->write(&_0devices, sizeof(_0devices));
    updateTmpPtr((void *)devices, _0devices);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)deviceCount, sizeof(*deviceCount), true);
    mem2client(conn, (void *)devices, sizeof(*devices), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetHicVersion(unsigned int *hwbcCount, nvmlHwbcEntry_t *hwbcEntries) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetHicVersion called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0hwbcCount;
    mem2server(conn, &_0hwbcCount, (void *)hwbcCount, sizeof(*hwbcCount));
    void *_0hwbcEntries;
    mem2server(conn, &_0hwbcEntries, (void *)hwbcEntries, sizeof(*hwbcEntries));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlSystemGetHicVersion);
    conn->write(&_0hwbcCount, sizeof(_0hwbcCount));
    updateTmpPtr((void *)hwbcCount, _0hwbcCount);
    conn->write(&_0hwbcEntries, sizeof(_0hwbcEntries));
    updateTmpPtr((void *)hwbcEntries, _0hwbcEntries);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)hwbcCount, sizeof(*hwbcCount), true);
    mem2client(conn, (void *)hwbcEntries, sizeof(*hwbcEntries), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCount_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0deviceCount;
    mem2server(conn, &_0deviceCount, (void *)deviceCount, sizeof(*deviceCount));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetCount_v2);
    conn->write(&_0deviceCount, sizeof(_0deviceCount));
    updateTmpPtr((void *)deviceCount, _0deviceCount);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)deviceCount, sizeof(*deviceCount), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t *attributes) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAttributes_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0attributes;
    mem2server(conn, &_0attributes, (void *)attributes, sizeof(*attributes));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetAttributes_v2);
    conn->write(&device, sizeof(device));
    conn->write(&_0attributes, sizeof(_0attributes));
    updateTmpPtr((void *)attributes, _0attributes);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)attributes, sizeof(*attributes), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleByIndex_v2 called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetHandleByIndex_v2);
    conn->write(&index, sizeof(index));
    conn->write(&_0device, sizeof(_0device));
    updateTmpPtr((void *)device, _0device);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)device, sizeof(*device), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleBySerial(const char *serial, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleBySerial called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetHandleBySerial);
    conn->write(serial, strlen(serial) + 1, true);
    conn->write(&_0device, sizeof(_0device));
    updateTmpPtr((void *)device, _0device);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)device, sizeof(*device), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleByUUID called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetHandleByUUID);
    conn->write(uuid, strlen(uuid) + 1, true);
    conn->write(&_0device, sizeof(_0device));
    updateTmpPtr((void *)device, _0device);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)device, sizeof(*device), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleByPciBusId_v2 called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetHandleByPciBusId_v2);
    conn->write(pciBusId, strlen(pciBusId) + 1, true);
    conn->write(&_0device, sizeof(_0device));
    updateTmpPtr((void *)device, _0device);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)device, sizeof(*device), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetName called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetName);
    conn->write(&device, sizeof(device));
    if(length > 0) {
        conn->read(name, length, true);
    }
    conn->write(&length, sizeof(length));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t *type) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBrand called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetBrand);
    conn->write(&device, sizeof(device));
    conn->write(&_0type, sizeof(_0type));
    updateTmpPtr((void *)type, _0type);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)type, sizeof(*type), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetIndex called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0index;
    mem2server(conn, &_0index, (void *)index, sizeof(*index));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetIndex);
    conn->write(&device, sizeof(device));
    conn->write(&_0index, sizeof(_0index));
    updateTmpPtr((void *)index, _0index);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)index, sizeof(*index), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char *serial, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSerial called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetSerial);
    conn->write(&device, sizeof(device));
    if(length > 0) {
        conn->read(serial, length, true);
    }
    conn->write(&length, sizeof(length));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long *nodeSet, nvmlAffinityScope_t scope) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryAffinity called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0nodeSet;
    mem2server(conn, &_0nodeSet, (void *)nodeSet, sizeof(*nodeSet));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetMemoryAffinity);
    conn->write(&device, sizeof(device));
    conn->write(&nodeSetSize, sizeof(nodeSetSize));
    conn->write(&_0nodeSet, sizeof(_0nodeSet));
    updateTmpPtr((void *)nodeSet, _0nodeSet);
    conn->write(&scope, sizeof(scope));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)nodeSet, sizeof(*nodeSet), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet, nvmlAffinityScope_t scope) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCpuAffinityWithinScope called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0cpuSet;
    mem2server(conn, &_0cpuSet, (void *)cpuSet, sizeof(*cpuSet));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetCpuAffinityWithinScope);
    conn->write(&device, sizeof(device));
    conn->write(&cpuSetSize, sizeof(cpuSetSize));
    conn->write(&_0cpuSet, sizeof(_0cpuSet));
    updateTmpPtr((void *)cpuSet, _0cpuSet);
    conn->write(&scope, sizeof(scope));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)cpuSet, sizeof(*cpuSet), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCpuAffinity called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0cpuSet;
    mem2server(conn, &_0cpuSet, (void *)cpuSet, sizeof(*cpuSet));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetCpuAffinity);
    conn->write(&device, sizeof(device));
    conn->write(&cpuSetSize, sizeof(cpuSetSize));
    conn->write(&_0cpuSet, sizeof(_0cpuSet));
    updateTmpPtr((void *)cpuSet, _0cpuSet);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)cpuSet, sizeof(*cpuSet), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetCpuAffinity called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetCpuAffinity);
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceClearCpuAffinity called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceClearCpuAffinity);
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t *pathInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTopologyCommonAncestor called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pathInfo;
    mem2server(conn, &_0pathInfo, (void *)pathInfo, sizeof(*pathInfo));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetTopologyCommonAncestor);
    conn->write(&device1, sizeof(device1));
    conn->write(&device2, sizeof(device2));
    conn->write(&_0pathInfo, sizeof(_0pathInfo));
    updateTmpPtr((void *)pathInfo, _0pathInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pathInfo, sizeof(*pathInfo), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int *count, nvmlDevice_t *deviceArray) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTopologyNearestGpus called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0count;
    mem2server(conn, &_0count, (void *)count, sizeof(*count));
    void *_0deviceArray;
    mem2server(conn, &_0deviceArray, (void *)deviceArray, sizeof(*deviceArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetTopologyNearestGpus);
    conn->write(&device, sizeof(device));
    conn->write(&level, sizeof(level));
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->write(&_0deviceArray, sizeof(_0deviceArray));
    updateTmpPtr((void *)deviceArray, _0deviceArray);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)count, sizeof(*count), true);
    mem2client(conn, (void *)deviceArray, sizeof(*deviceArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int *count, nvmlDevice_t *deviceArray) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetTopologyGpuSet called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0count;
    mem2server(conn, &_0count, (void *)count, sizeof(*count));
    void *_0deviceArray;
    mem2server(conn, &_0deviceArray, (void *)deviceArray, sizeof(*deviceArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlSystemGetTopologyGpuSet);
    conn->write(&cpuNumber, sizeof(cpuNumber));
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->write(&_0deviceArray, sizeof(_0deviceArray));
    updateTmpPtr((void *)deviceArray, _0deviceArray);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)count, sizeof(*count), true);
    mem2client(conn, (void *)deviceArray, sizeof(*deviceArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t *p2pStatus) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetP2PStatus called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0p2pStatus;
    mem2server(conn, &_0p2pStatus, (void *)p2pStatus, sizeof(*p2pStatus));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetP2PStatus);
    conn->write(&device1, sizeof(device1));
    conn->write(&device2, sizeof(device2));
    conn->write(&p2pIndex, sizeof(p2pIndex));
    conn->write(&_0p2pStatus, sizeof(_0p2pStatus));
    updateTmpPtr((void *)p2pStatus, _0p2pStatus);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)p2pStatus, sizeof(*p2pStatus), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetUUID called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetUUID);
    conn->write(&device, sizeof(device));
    if(length > 0) {
        conn->read(uuid, length, true);
    }
    conn->write(&length, sizeof(length));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char *mdevUuid, unsigned int size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetMdevUUID called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetMdevUUID);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    if(size > 0) {
        conn->read(mdevUuid, size, true);
    }
    conn->write(&size, sizeof(size));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minorNumber) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMinorNumber called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0minorNumber;
    mem2server(conn, &_0minorNumber, (void *)minorNumber, sizeof(*minorNumber));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetMinorNumber);
    conn->write(&device, sizeof(device));
    conn->write(&_0minorNumber, sizeof(_0minorNumber));
    updateTmpPtr((void *)minorNumber, _0minorNumber);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)minorNumber, sizeof(*minorNumber), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char *partNumber, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBoardPartNumber called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetBoardPartNumber);
    conn->write(&device, sizeof(device));
    if(length > 0) {
        conn->read(partNumber, length, true);
    }
    conn->write(&length, sizeof(length));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetInforomVersion called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetInforomVersion);
    conn->write(&device, sizeof(device));
    conn->write(&object, sizeof(object));
    if(length > 0) {
        conn->read(version, length, true);
    }
    conn->write(&length, sizeof(length));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetInforomImageVersion called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetInforomImageVersion);
    conn->write(&device, sizeof(device));
    if(length > 0) {
        conn->read(version, length, true);
    }
    conn->write(&length, sizeof(length));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int *checksum) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetInforomConfigurationChecksum called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0checksum;
    mem2server(conn, &_0checksum, (void *)checksum, sizeof(*checksum));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetInforomConfigurationChecksum);
    conn->write(&device, sizeof(device));
    conn->write(&_0checksum, sizeof(_0checksum));
    updateTmpPtr((void *)checksum, _0checksum);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)checksum, sizeof(*checksum), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceValidateInforom called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceValidateInforom);
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t *display) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDisplayMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0display;
    mem2server(conn, &_0display, (void *)display, sizeof(*display));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetDisplayMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0display, sizeof(_0display));
    updateTmpPtr((void *)display, _0display);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)display, sizeof(*display), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t *isActive) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDisplayActive called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0isActive;
    mem2server(conn, &_0isActive, (void *)isActive, sizeof(*isActive));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetDisplayActive);
    conn->write(&device, sizeof(device));
    conn->write(&_0isActive, sizeof(_0isActive));
    updateTmpPtr((void *)isActive, _0isActive);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)isActive, sizeof(*isActive), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPersistenceMode called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetPersistenceMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0mode, sizeof(_0mode));
    updateTmpPtr((void *)mode, _0mode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)mode, sizeof(*mode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPciInfo_v3 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pci;
    mem2server(conn, &_0pci, (void *)pci, sizeof(*pci));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetPciInfo_v3);
    conn->write(&device, sizeof(device));
    conn->write(&_0pci, sizeof(_0pci));
    updateTmpPtr((void *)pci, _0pci);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pci, sizeof(*pci), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGen) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxPcieLinkGeneration called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0maxLinkGen;
    mem2server(conn, &_0maxLinkGen, (void *)maxLinkGen, sizeof(*maxLinkGen));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetMaxPcieLinkGeneration);
    conn->write(&device, sizeof(device));
    conn->write(&_0maxLinkGen, sizeof(_0maxLinkGen));
    updateTmpPtr((void *)maxLinkGen, _0maxLinkGen);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)maxLinkGen, sizeof(*maxLinkGen), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int *maxLinkWidth) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxPcieLinkWidth called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0maxLinkWidth;
    mem2server(conn, &_0maxLinkWidth, (void *)maxLinkWidth, sizeof(*maxLinkWidth));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetMaxPcieLinkWidth);
    conn->write(&device, sizeof(device));
    conn->write(&_0maxLinkWidth, sizeof(_0maxLinkWidth));
    updateTmpPtr((void *)maxLinkWidth, _0maxLinkWidth);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)maxLinkWidth, sizeof(*maxLinkWidth), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int *currLinkGen) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrPcieLinkGeneration called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0currLinkGen;
    mem2server(conn, &_0currLinkGen, (void *)currLinkGen, sizeof(*currLinkGen));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetCurrPcieLinkGeneration);
    conn->write(&device, sizeof(device));
    conn->write(&_0currLinkGen, sizeof(_0currLinkGen));
    updateTmpPtr((void *)currLinkGen, _0currLinkGen);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)currLinkGen, sizeof(*currLinkGen), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int *currLinkWidth) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrPcieLinkWidth called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0currLinkWidth;
    mem2server(conn, &_0currLinkWidth, (void *)currLinkWidth, sizeof(*currLinkWidth));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetCurrPcieLinkWidth);
    conn->write(&device, sizeof(device));
    conn->write(&_0currLinkWidth, sizeof(_0currLinkWidth));
    updateTmpPtr((void *)currLinkWidth, _0currLinkWidth);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)currLinkWidth, sizeof(*currLinkWidth), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int *value) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPcieThroughput called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetPcieThroughput);
    conn->write(&device, sizeof(device));
    conn->write(&counter, sizeof(counter));
    conn->write(&_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)value, sizeof(*value), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int *value) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPcieReplayCounter called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetPcieReplayCounter);
    conn->write(&device, sizeof(device));
    conn->write(&_0value, sizeof(_0value));
    updateTmpPtr((void *)value, _0value);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)value, sizeof(*value), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClockInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0clock;
    mem2server(conn, &_0clock, (void *)clock, sizeof(*clock));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetClockInfo);
    conn->write(&device, sizeof(device));
    conn->write(&type, sizeof(type));
    conn->write(&_0clock, sizeof(_0clock));
    updateTmpPtr((void *)clock, _0clock);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)clock, sizeof(*clock), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxClockInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0clock;
    mem2server(conn, &_0clock, (void *)clock, sizeof(*clock));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetMaxClockInfo);
    conn->write(&device, sizeof(device));
    conn->write(&type, sizeof(type));
    conn->write(&_0clock, sizeof(_0clock));
    updateTmpPtr((void *)clock, _0clock);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)clock, sizeof(*clock), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetApplicationsClock called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0clockMHz;
    mem2server(conn, &_0clockMHz, (void *)clockMHz, sizeof(*clockMHz));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetApplicationsClock);
    conn->write(&device, sizeof(device));
    conn->write(&clockType, sizeof(clockType));
    conn->write(&_0clockMHz, sizeof(_0clockMHz));
    updateTmpPtr((void *)clockMHz, _0clockMHz);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)clockMHz, sizeof(*clockMHz), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDefaultApplicationsClock called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0clockMHz;
    mem2server(conn, &_0clockMHz, (void *)clockMHz, sizeof(*clockMHz));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetDefaultApplicationsClock);
    conn->write(&device, sizeof(device));
    conn->write(&clockType, sizeof(clockType));
    conn->write(&_0clockMHz, sizeof(_0clockMHz));
    updateTmpPtr((void *)clockMHz, _0clockMHz);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)clockMHz, sizeof(*clockMHz), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceResetApplicationsClocks(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetApplicationsClocks called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceResetApplicationsClocks);
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClock called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0clockMHz;
    mem2server(conn, &_0clockMHz, (void *)clockMHz, sizeof(*clockMHz));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetClock);
    conn->write(&device, sizeof(device));
    conn->write(&clockType, sizeof(clockType));
    conn->write(&clockId, sizeof(clockId));
    conn->write(&_0clockMHz, sizeof(_0clockMHz));
    updateTmpPtr((void *)clockMHz, _0clockMHz);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)clockMHz, sizeof(*clockMHz), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxCustomerBoostClock called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0clockMHz;
    mem2server(conn, &_0clockMHz, (void *)clockMHz, sizeof(*clockMHz));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetMaxCustomerBoostClock);
    conn->write(&device, sizeof(device));
    conn->write(&clockType, sizeof(clockType));
    conn->write(&_0clockMHz, sizeof(_0clockMHz));
    updateTmpPtr((void *)clockMHz, _0clockMHz);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)clockMHz, sizeof(*clockMHz), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int *count, unsigned int *clocksMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedMemoryClocks called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0count;
    mem2server(conn, &_0count, (void *)count, sizeof(*count));
    void *_0clocksMHz;
    mem2server(conn, &_0clocksMHz, (void *)clocksMHz, sizeof(*clocksMHz));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetSupportedMemoryClocks);
    conn->write(&device, sizeof(device));
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->write(&_0clocksMHz, sizeof(_0clocksMHz));
    updateTmpPtr((void *)clocksMHz, _0clocksMHz);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)count, sizeof(*count), true);
    mem2client(conn, (void *)clocksMHz, sizeof(*clocksMHz), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int *count, unsigned int *clocksMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedGraphicsClocks called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0count;
    mem2server(conn, &_0count, (void *)count, sizeof(*count));
    void *_0clocksMHz;
    mem2server(conn, &_0clocksMHz, (void *)clocksMHz, sizeof(*clocksMHz));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetSupportedGraphicsClocks);
    conn->write(&device, sizeof(device));
    conn->write(&memoryClockMHz, sizeof(memoryClockMHz));
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->write(&_0clocksMHz, sizeof(_0clocksMHz));
    updateTmpPtr((void *)clocksMHz, _0clocksMHz);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)count, sizeof(*count), true);
    mem2client(conn, (void *)clocksMHz, sizeof(*clocksMHz), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t *isEnabled, nvmlEnableState_t *defaultIsEnabled) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAutoBoostedClocksEnabled called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0isEnabled;
    mem2server(conn, &_0isEnabled, (void *)isEnabled, sizeof(*isEnabled));
    void *_0defaultIsEnabled;
    mem2server(conn, &_0defaultIsEnabled, (void *)defaultIsEnabled, sizeof(*defaultIsEnabled));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetAutoBoostedClocksEnabled);
    conn->write(&device, sizeof(device));
    conn->write(&_0isEnabled, sizeof(_0isEnabled));
    updateTmpPtr((void *)isEnabled, _0isEnabled);
    conn->write(&_0defaultIsEnabled, sizeof(_0defaultIsEnabled));
    updateTmpPtr((void *)defaultIsEnabled, _0defaultIsEnabled);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)isEnabled, sizeof(*isEnabled), true);
    mem2client(conn, (void *)defaultIsEnabled, sizeof(*defaultIsEnabled), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetAutoBoostedClocksEnabled called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetAutoBoostedClocksEnabled);
    conn->write(&device, sizeof(device));
    conn->write(&enabled, sizeof(enabled));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetDefaultAutoBoostedClocksEnabled called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetDefaultAutoBoostedClocksEnabled);
    conn->write(&device, sizeof(device));
    conn->write(&enabled, sizeof(enabled));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFanSpeed called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0speed;
    mem2server(conn, &_0speed, (void *)speed, sizeof(*speed));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetFanSpeed);
    conn->write(&device, sizeof(device));
    conn->write(&_0speed, sizeof(_0speed));
    updateTmpPtr((void *)speed, _0speed);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)speed, sizeof(*speed), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int *speed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFanSpeed_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0speed;
    mem2server(conn, &_0speed, (void *)speed, sizeof(*speed));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetFanSpeed_v2);
    conn->write(&device, sizeof(device));
    conn->write(&fan, sizeof(fan));
    conn->write(&_0speed, sizeof(_0speed));
    updateTmpPtr((void *)speed, _0speed);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)speed, sizeof(*speed), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTemperature called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0temp;
    mem2server(conn, &_0temp, (void *)temp, sizeof(*temp));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetTemperature);
    conn->write(&device, sizeof(device));
    conn->write(&sensorType, sizeof(sensorType));
    conn->write(&_0temp, sizeof(_0temp));
    updateTmpPtr((void *)temp, _0temp);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)temp, sizeof(*temp), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTemperatureThreshold called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0temp;
    mem2server(conn, &_0temp, (void *)temp, sizeof(*temp));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetTemperatureThreshold);
    conn->write(&device, sizeof(device));
    conn->write(&thresholdType, sizeof(thresholdType));
    conn->write(&_0temp, sizeof(_0temp));
    updateTmpPtr((void *)temp, _0temp);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)temp, sizeof(*temp), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int *temp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetTemperatureThreshold called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0temp;
    mem2server(conn, &_0temp, (void *)temp, sizeof(*temp));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetTemperatureThreshold);
    conn->write(&device, sizeof(device));
    conn->write(&thresholdType, sizeof(thresholdType));
    conn->write(&_0temp, sizeof(_0temp));
    updateTmpPtr((void *)temp, _0temp);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)temp, sizeof(*temp), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t *pState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPerformanceState called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pState;
    mem2server(conn, &_0pState, (void *)pState, sizeof(*pState));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetPerformanceState);
    conn->write(&device, sizeof(device));
    conn->write(&_0pState, sizeof(_0pState));
    updateTmpPtr((void *)pState, _0pState);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pState, sizeof(*pState), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long *clocksThrottleReasons) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrentClocksThrottleReasons called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0clocksThrottleReasons;
    mem2server(conn, &_0clocksThrottleReasons, (void *)clocksThrottleReasons, sizeof(*clocksThrottleReasons));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetCurrentClocksThrottleReasons);
    conn->write(&device, sizeof(device));
    conn->write(&_0clocksThrottleReasons, sizeof(_0clocksThrottleReasons));
    updateTmpPtr((void *)clocksThrottleReasons, _0clocksThrottleReasons);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)clocksThrottleReasons, sizeof(*clocksThrottleReasons), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long *supportedClocksThrottleReasons) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedClocksThrottleReasons called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0supportedClocksThrottleReasons;
    mem2server(conn, &_0supportedClocksThrottleReasons, (void *)supportedClocksThrottleReasons, sizeof(*supportedClocksThrottleReasons));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetSupportedClocksThrottleReasons);
    conn->write(&device, sizeof(device));
    conn->write(&_0supportedClocksThrottleReasons, sizeof(_0supportedClocksThrottleReasons));
    updateTmpPtr((void *)supportedClocksThrottleReasons, _0supportedClocksThrottleReasons);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)supportedClocksThrottleReasons, sizeof(*supportedClocksThrottleReasons), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t *pState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerState called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pState;
    mem2server(conn, &_0pState, (void *)pState, sizeof(*pState));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetPowerState);
    conn->write(&device, sizeof(device));
    conn->write(&_0pState, sizeof(_0pState));
    updateTmpPtr((void *)pState, _0pState);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pState, sizeof(*pState), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementMode called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetPowerManagementMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0mode, sizeof(_0mode));
    updateTmpPtr((void *)mode, _0mode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)mode, sizeof(*mode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int *limit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementLimit called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0limit;
    mem2server(conn, &_0limit, (void *)limit, sizeof(*limit));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetPowerManagementLimit);
    conn->write(&device, sizeof(device));
    conn->write(&_0limit, sizeof(_0limit));
    updateTmpPtr((void *)limit, _0limit);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)limit, sizeof(*limit), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementLimitConstraints called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0minLimit;
    mem2server(conn, &_0minLimit, (void *)minLimit, sizeof(*minLimit));
    void *_0maxLimit;
    mem2server(conn, &_0maxLimit, (void *)maxLimit, sizeof(*maxLimit));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetPowerManagementLimitConstraints);
    conn->write(&device, sizeof(device));
    conn->write(&_0minLimit, sizeof(_0minLimit));
    updateTmpPtr((void *)minLimit, _0minLimit);
    conn->write(&_0maxLimit, sizeof(_0maxLimit));
    updateTmpPtr((void *)maxLimit, _0maxLimit);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)minLimit, sizeof(*minLimit), true);
    mem2client(conn, (void *)maxLimit, sizeof(*maxLimit), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int *defaultLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerManagementDefaultLimit called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0defaultLimit;
    mem2server(conn, &_0defaultLimit, (void *)defaultLimit, sizeof(*defaultLimit));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetPowerManagementDefaultLimit);
    conn->write(&device, sizeof(device));
    conn->write(&_0defaultLimit, sizeof(_0defaultLimit));
    updateTmpPtr((void *)defaultLimit, _0defaultLimit);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)defaultLimit, sizeof(*defaultLimit), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerUsage called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0power;
    mem2server(conn, &_0power, (void *)power, sizeof(*power));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetPowerUsage);
    conn->write(&device, sizeof(device));
    conn->write(&_0power, sizeof(_0power));
    updateTmpPtr((void *)power, _0power);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)power, sizeof(*power), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long *energy) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTotalEnergyConsumption called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0energy;
    mem2server(conn, &_0energy, (void *)energy, sizeof(*energy));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetTotalEnergyConsumption);
    conn->write(&device, sizeof(device));
    conn->write(&_0energy, sizeof(_0energy));
    updateTmpPtr((void *)energy, _0energy);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)energy, sizeof(*energy), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEnforcedPowerLimit called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0limit;
    mem2server(conn, &_0limit, (void *)limit, sizeof(*limit));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetEnforcedPowerLimit);
    conn->write(&device, sizeof(device));
    conn->write(&_0limit, sizeof(_0limit));
    updateTmpPtr((void *)limit, _0limit);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)limit, sizeof(*limit), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t *current, nvmlGpuOperationMode_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuOperationMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0current;
    mem2server(conn, &_0current, (void *)current, sizeof(*current));
    void *_0pending;
    mem2server(conn, &_0pending, (void *)pending, sizeof(*pending));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetGpuOperationMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0current, sizeof(_0current));
    updateTmpPtr((void *)current, _0current);
    conn->write(&_0pending, sizeof(_0pending));
    updateTmpPtr((void *)pending, _0pending);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)current, sizeof(*current), true);
    mem2client(conn, (void *)pending, sizeof(*pending), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0memory;
    mem2server(conn, &_0memory, (void *)memory, sizeof(*memory));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetMemoryInfo);
    conn->write(&device, sizeof(device));
    conn->write(&_0memory, sizeof(_0memory));
    updateTmpPtr((void *)memory, _0memory);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)memory, sizeof(*memory), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetComputeMode called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetComputeMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0mode, sizeof(_0mode));
    updateTmpPtr((void *)mode, _0mode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)mode, sizeof(*mode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCudaComputeCapability called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetCudaComputeCapability);
    conn->write(&device, sizeof(device));
    conn->write(&_0major, sizeof(_0major));
    updateTmpPtr((void *)major, _0major);
    conn->write(&_0minor, sizeof(_0minor));
    updateTmpPtr((void *)minor, _0minor);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)major, sizeof(*major), true);
    mem2client(conn, (void *)minor, sizeof(*minor), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t *current, nvmlEnableState_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEccMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0current;
    mem2server(conn, &_0current, (void *)current, sizeof(*current));
    void *_0pending;
    mem2server(conn, &_0pending, (void *)pending, sizeof(*pending));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetEccMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0current, sizeof(_0current));
    updateTmpPtr((void *)current, _0current);
    conn->write(&_0pending, sizeof(_0pending));
    updateTmpPtr((void *)pending, _0pending);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)current, sizeof(*current), true);
    mem2client(conn, (void *)pending, sizeof(*pending), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int *boardId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBoardId called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0boardId;
    mem2server(conn, &_0boardId, (void *)boardId, sizeof(*boardId));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetBoardId);
    conn->write(&device, sizeof(device));
    conn->write(&_0boardId, sizeof(_0boardId));
    updateTmpPtr((void *)boardId, _0boardId);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)boardId, sizeof(*boardId), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int *multiGpuBool) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMultiGpuBoard called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0multiGpuBool;
    mem2server(conn, &_0multiGpuBool, (void *)multiGpuBool, sizeof(*multiGpuBool));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetMultiGpuBoard);
    conn->write(&device, sizeof(device));
    conn->write(&_0multiGpuBool, sizeof(_0multiGpuBool));
    updateTmpPtr((void *)multiGpuBool, _0multiGpuBool);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)multiGpuBool, sizeof(*multiGpuBool), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long *eccCounts) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTotalEccErrors called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0eccCounts;
    mem2server(conn, &_0eccCounts, (void *)eccCounts, sizeof(*eccCounts));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetTotalEccErrors);
    conn->write(&device, sizeof(device));
    conn->write(&errorType, sizeof(errorType));
    conn->write(&counterType, sizeof(counterType));
    conn->write(&_0eccCounts, sizeof(_0eccCounts));
    updateTmpPtr((void *)eccCounts, _0eccCounts);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)eccCounts, sizeof(*eccCounts), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t *eccCounts) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDetailedEccErrors called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0eccCounts;
    mem2server(conn, &_0eccCounts, (void *)eccCounts, sizeof(*eccCounts));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetDetailedEccErrors);
    conn->write(&device, sizeof(device));
    conn->write(&errorType, sizeof(errorType));
    conn->write(&counterType, sizeof(counterType));
    conn->write(&_0eccCounts, sizeof(_0eccCounts));
    updateTmpPtr((void *)eccCounts, _0eccCounts);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)eccCounts, sizeof(*eccCounts), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryErrorCounter called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetMemoryErrorCounter);
    conn->write(&device, sizeof(device));
    conn->write(&errorType, sizeof(errorType));
    conn->write(&counterType, sizeof(counterType));
    conn->write(&locationType, sizeof(locationType));
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)count, sizeof(*count), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetUtilizationRates called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0utilization;
    mem2server(conn, &_0utilization, (void *)utilization, sizeof(*utilization));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetUtilizationRates);
    conn->write(&device, sizeof(device));
    conn->write(&_0utilization, sizeof(_0utilization));
    updateTmpPtr((void *)utilization, _0utilization);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)utilization, sizeof(*utilization), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderUtilization called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0utilization;
    mem2server(conn, &_0utilization, (void *)utilization, sizeof(*utilization));
    void *_0samplingPeriodUs;
    mem2server(conn, &_0samplingPeriodUs, (void *)samplingPeriodUs, sizeof(*samplingPeriodUs));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetEncoderUtilization);
    conn->write(&device, sizeof(device));
    conn->write(&_0utilization, sizeof(_0utilization));
    updateTmpPtr((void *)utilization, _0utilization);
    conn->write(&_0samplingPeriodUs, sizeof(_0samplingPeriodUs));
    updateTmpPtr((void *)samplingPeriodUs, _0samplingPeriodUs);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)utilization, sizeof(*utilization), true);
    mem2client(conn, (void *)samplingPeriodUs, sizeof(*samplingPeriodUs), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int *encoderCapacity) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderCapacity called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0encoderCapacity;
    mem2server(conn, &_0encoderCapacity, (void *)encoderCapacity, sizeof(*encoderCapacity));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetEncoderCapacity);
    conn->write(&device, sizeof(device));
    conn->write(&encoderQueryType, sizeof(encoderQueryType));
    conn->write(&_0encoderCapacity, sizeof(_0encoderCapacity));
    updateTmpPtr((void *)encoderCapacity, _0encoderCapacity);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)encoderCapacity, sizeof(*encoderCapacity), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderStats called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0sessionCount;
    mem2server(conn, &_0sessionCount, (void *)sessionCount, sizeof(*sessionCount));
    void *_0averageFps;
    mem2server(conn, &_0averageFps, (void *)averageFps, sizeof(*averageFps));
    void *_0averageLatency;
    mem2server(conn, &_0averageLatency, (void *)averageLatency, sizeof(*averageLatency));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetEncoderStats);
    conn->write(&device, sizeof(device));
    conn->write(&_0sessionCount, sizeof(_0sessionCount));
    updateTmpPtr((void *)sessionCount, _0sessionCount);
    conn->write(&_0averageFps, sizeof(_0averageFps));
    updateTmpPtr((void *)averageFps, _0averageFps);
    conn->write(&_0averageLatency, sizeof(_0averageLatency));
    updateTmpPtr((void *)averageLatency, _0averageLatency);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)sessionCount, sizeof(*sessionCount), true);
    mem2client(conn, (void *)averageFps, sizeof(*averageFps), true);
    mem2client(conn, (void *)averageLatency, sizeof(*averageLatency), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetEncoderSessions called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0sessionCount;
    mem2server(conn, &_0sessionCount, (void *)sessionCount, sizeof(*sessionCount));
    void *_0sessionInfos;
    mem2server(conn, &_0sessionInfos, (void *)sessionInfos, sizeof(*sessionInfos));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetEncoderSessions);
    conn->write(&device, sizeof(device));
    conn->write(&_0sessionCount, sizeof(_0sessionCount));
    updateTmpPtr((void *)sessionCount, _0sessionCount);
    conn->write(&_0sessionInfos, sizeof(_0sessionInfos));
    updateTmpPtr((void *)sessionInfos, _0sessionInfos);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)sessionCount, sizeof(*sessionCount), true);
    mem2client(conn, (void *)sessionInfos, sizeof(*sessionInfos), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDecoderUtilization called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0utilization;
    mem2server(conn, &_0utilization, (void *)utilization, sizeof(*utilization));
    void *_0samplingPeriodUs;
    mem2server(conn, &_0samplingPeriodUs, (void *)samplingPeriodUs, sizeof(*samplingPeriodUs));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetDecoderUtilization);
    conn->write(&device, sizeof(device));
    conn->write(&_0utilization, sizeof(_0utilization));
    updateTmpPtr((void *)utilization, _0utilization);
    conn->write(&_0samplingPeriodUs, sizeof(_0samplingPeriodUs));
    updateTmpPtr((void *)samplingPeriodUs, _0samplingPeriodUs);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)utilization, sizeof(*utilization), true);
    mem2client(conn, (void *)samplingPeriodUs, sizeof(*samplingPeriodUs), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t *fbcStats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFBCStats called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0fbcStats;
    mem2server(conn, &_0fbcStats, (void *)fbcStats, sizeof(*fbcStats));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetFBCStats);
    conn->write(&device, sizeof(device));
    conn->write(&_0fbcStats, sizeof(_0fbcStats));
    updateTmpPtr((void *)fbcStats, _0fbcStats);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)fbcStats, sizeof(*fbcStats), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFBCSessions called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0sessionCount;
    mem2server(conn, &_0sessionCount, (void *)sessionCount, sizeof(*sessionCount));
    void *_0sessionInfo;
    mem2server(conn, &_0sessionInfo, (void *)sessionInfo, sizeof(*sessionInfo));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetFBCSessions);
    conn->write(&device, sizeof(device));
    conn->write(&_0sessionCount, sizeof(_0sessionCount));
    updateTmpPtr((void *)sessionCount, _0sessionCount);
    conn->write(&_0sessionInfo, sizeof(_0sessionInfo));
    updateTmpPtr((void *)sessionInfo, _0sessionInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)sessionCount, sizeof(*sessionCount), true);
    mem2client(conn, (void *)sessionInfo, sizeof(*sessionInfo), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t device, nvmlDriverModel_t *current, nvmlDriverModel_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDriverModel called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0current;
    mem2server(conn, &_0current, (void *)current, sizeof(*current));
    void *_0pending;
    mem2server(conn, &_0pending, (void *)pending, sizeof(*pending));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetDriverModel);
    conn->write(&device, sizeof(device));
    conn->write(&_0current, sizeof(_0current));
    updateTmpPtr((void *)current, _0current);
    conn->write(&_0pending, sizeof(_0pending));
    updateTmpPtr((void *)pending, _0pending);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)current, sizeof(*current), true);
    mem2client(conn, (void *)pending, sizeof(*pending), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVbiosVersion called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetVbiosVersion);
    conn->write(&device, sizeof(device));
    if(length > 0) {
        conn->read(version, length, true);
    }
    conn->write(&length, sizeof(length));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t *bridgeHierarchy) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBridgeChipInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0bridgeHierarchy;
    mem2server(conn, &_0bridgeHierarchy, (void *)bridgeHierarchy, sizeof(*bridgeHierarchy));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetBridgeChipInfo);
    conn->write(&device, sizeof(device));
    conn->write(&_0bridgeHierarchy, sizeof(_0bridgeHierarchy));
    updateTmpPtr((void *)bridgeHierarchy, _0bridgeHierarchy);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)bridgeHierarchy, sizeof(*bridgeHierarchy), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetComputeRunningProcesses_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0infoCount;
    mem2server(conn, &_0infoCount, (void *)infoCount, sizeof(*infoCount));
    void *_0infos;
    mem2server(conn, &_0infos, (void *)infos, sizeof(*infos));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetComputeRunningProcesses_v2);
    conn->write(&device, sizeof(device));
    conn->write(&_0infoCount, sizeof(_0infoCount));
    updateTmpPtr((void *)infoCount, _0infoCount);
    conn->write(&_0infos, sizeof(_0infos));
    updateTmpPtr((void *)infos, _0infos);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)infoCount, sizeof(*infoCount), true);
    mem2client(conn, (void *)infos, sizeof(*infos), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGraphicsRunningProcesses_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0infoCount;
    mem2server(conn, &_0infoCount, (void *)infoCount, sizeof(*infoCount));
    void *_0infos;
    mem2server(conn, &_0infos, (void *)infos, sizeof(*infos));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetGraphicsRunningProcesses_v2);
    conn->write(&device, sizeof(device));
    conn->write(&_0infoCount, sizeof(_0infoCount));
    updateTmpPtr((void *)infoCount, _0infoCount);
    conn->write(&_0infos, sizeof(_0infos));
    updateTmpPtr((void *)infos, _0infos);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)infoCount, sizeof(*infoCount), true);
    mem2client(conn, (void *)infos, sizeof(*infos), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMPSComputeRunningProcesses_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0infoCount;
    mem2server(conn, &_0infoCount, (void *)infoCount, sizeof(*infoCount));
    void *_0infos;
    mem2server(conn, &_0infos, (void *)infos, sizeof(*infos));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetMPSComputeRunningProcesses_v2);
    conn->write(&device, sizeof(device));
    conn->write(&_0infoCount, sizeof(_0infoCount));
    updateTmpPtr((void *)infoCount, _0infoCount);
    conn->write(&_0infos, sizeof(_0infos));
    updateTmpPtr((void *)infos, _0infos);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)infoCount, sizeof(*infoCount), true);
    mem2client(conn, (void *)infos, sizeof(*infos), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int *onSameBoard) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceOnSameBoard called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0onSameBoard;
    mem2server(conn, &_0onSameBoard, (void *)onSameBoard, sizeof(*onSameBoard));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceOnSameBoard);
    conn->write(&device1, sizeof(device1));
    conn->write(&device2, sizeof(device2));
    conn->write(&_0onSameBoard, sizeof(_0onSameBoard));
    updateTmpPtr((void *)onSameBoard, _0onSameBoard);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)onSameBoard, sizeof(*onSameBoard), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t *isRestricted) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAPIRestriction called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0isRestricted;
    mem2server(conn, &_0isRestricted, (void *)isRestricted, sizeof(*isRestricted));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetAPIRestriction);
    conn->write(&device, sizeof(device));
    conn->write(&apiType, sizeof(apiType));
    conn->write(&_0isRestricted, sizeof(_0isRestricted));
    updateTmpPtr((void *)isRestricted, _0isRestricted);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)isRestricted, sizeof(*isRestricted), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *sampleCount, nvmlSample_t *samples) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSamples called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0sampleValType;
    mem2server(conn, &_0sampleValType, (void *)sampleValType, sizeof(*sampleValType));
    void *_0sampleCount;
    mem2server(conn, &_0sampleCount, (void *)sampleCount, sizeof(*sampleCount));
    void *_0samples;
    mem2server(conn, &_0samples, (void *)samples, sizeof(*samples));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetSamples);
    conn->write(&device, sizeof(device));
    conn->write(&type, sizeof(type));
    conn->write(&lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    conn->write(&_0sampleValType, sizeof(_0sampleValType));
    updateTmpPtr((void *)sampleValType, _0sampleValType);
    conn->write(&_0sampleCount, sizeof(_0sampleCount));
    updateTmpPtr((void *)sampleCount, _0sampleCount);
    conn->write(&_0samples, sizeof(_0samples));
    updateTmpPtr((void *)samples, _0samples);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)sampleValType, sizeof(*sampleValType), true);
    mem2client(conn, (void *)sampleCount, sizeof(*sampleCount), true);
    mem2client(conn, (void *)samples, sizeof(*samples), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t *bar1Memory) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBAR1MemoryInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0bar1Memory;
    mem2server(conn, &_0bar1Memory, (void *)bar1Memory, sizeof(*bar1Memory));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetBAR1MemoryInfo);
    conn->write(&device, sizeof(device));
    conn->write(&_0bar1Memory, sizeof(_0bar1Memory));
    updateTmpPtr((void *)bar1Memory, _0bar1Memory);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)bar1Memory, sizeof(*bar1Memory), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t *violTime) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetViolationStatus called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0violTime;
    mem2server(conn, &_0violTime, (void *)violTime, sizeof(*violTime));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetViolationStatus);
    conn->write(&device, sizeof(device));
    conn->write(&perfPolicyType, sizeof(perfPolicyType));
    conn->write(&_0violTime, sizeof(_0violTime));
    updateTmpPtr((void *)violTime, _0violTime);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)violTime, sizeof(*violTime), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingMode called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetAccountingMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0mode, sizeof(_0mode));
    updateTmpPtr((void *)mode, _0mode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)mode, sizeof(*mode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t *stats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingStats called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0stats;
    mem2server(conn, &_0stats, (void *)stats, sizeof(*stats));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetAccountingStats);
    conn->write(&device, sizeof(device));
    conn->write(&pid, sizeof(pid));
    conn->write(&_0stats, sizeof(_0stats));
    updateTmpPtr((void *)stats, _0stats);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)stats, sizeof(*stats), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int *count, unsigned int *pids) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingPids called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0count;
    mem2server(conn, &_0count, (void *)count, sizeof(*count));
    void *_0pids;
    mem2server(conn, &_0pids, (void *)pids, sizeof(*pids));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetAccountingPids);
    conn->write(&device, sizeof(device));
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->write(&_0pids, sizeof(_0pids));
    updateTmpPtr((void *)pids, _0pids);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)count, sizeof(*count), true);
    mem2client(conn, (void *)pids, sizeof(*pids), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAccountingBufferSize called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0bufferSize;
    mem2server(conn, &_0bufferSize, (void *)bufferSize, sizeof(*bufferSize));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetAccountingBufferSize);
    conn->write(&device, sizeof(device));
    conn->write(&_0bufferSize, sizeof(_0bufferSize));
    updateTmpPtr((void *)bufferSize, _0bufferSize);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)bufferSize, sizeof(*bufferSize), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRetiredPages called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pageCount;
    mem2server(conn, &_0pageCount, (void *)pageCount, sizeof(*pageCount));
    void *_0addresses;
    mem2server(conn, &_0addresses, (void *)addresses, sizeof(*addresses));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetRetiredPages);
    conn->write(&device, sizeof(device));
    conn->write(&cause, sizeof(cause));
    conn->write(&_0pageCount, sizeof(_0pageCount));
    updateTmpPtr((void *)pageCount, _0pageCount);
    conn->write(&_0addresses, sizeof(_0addresses));
    updateTmpPtr((void *)addresses, _0addresses);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pageCount, sizeof(*pageCount), true);
    mem2client(conn, (void *)addresses, sizeof(*addresses), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses, unsigned long long *timestamps) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRetiredPages_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pageCount;
    mem2server(conn, &_0pageCount, (void *)pageCount, sizeof(*pageCount));
    void *_0addresses;
    mem2server(conn, &_0addresses, (void *)addresses, sizeof(*addresses));
    void *_0timestamps;
    mem2server(conn, &_0timestamps, (void *)timestamps, sizeof(*timestamps));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetRetiredPages_v2);
    conn->write(&device, sizeof(device));
    conn->write(&cause, sizeof(cause));
    conn->write(&_0pageCount, sizeof(_0pageCount));
    updateTmpPtr((void *)pageCount, _0pageCount);
    conn->write(&_0addresses, sizeof(_0addresses));
    updateTmpPtr((void *)addresses, _0addresses);
    conn->write(&_0timestamps, sizeof(_0timestamps));
    updateTmpPtr((void *)timestamps, _0timestamps);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pageCount, sizeof(*pageCount), true);
    mem2client(conn, (void *)addresses, sizeof(*addresses), true);
    mem2client(conn, (void *)timestamps, sizeof(*timestamps), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t *isPending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRetiredPagesPendingStatus called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0isPending;
    mem2server(conn, &_0isPending, (void *)isPending, sizeof(*isPending));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetRetiredPagesPendingStatus);
    conn->write(&device, sizeof(device));
    conn->write(&_0isPending, sizeof(_0isPending));
    updateTmpPtr((void *)isPending, _0isPending);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)isPending, sizeof(*isPending), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int *corrRows, unsigned int *uncRows, unsigned int *isPending, unsigned int *failureOccurred) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRemappedRows called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0corrRows;
    mem2server(conn, &_0corrRows, (void *)corrRows, sizeof(*corrRows));
    void *_0uncRows;
    mem2server(conn, &_0uncRows, (void *)uncRows, sizeof(*uncRows));
    void *_0isPending;
    mem2server(conn, &_0isPending, (void *)isPending, sizeof(*isPending));
    void *_0failureOccurred;
    mem2server(conn, &_0failureOccurred, (void *)failureOccurred, sizeof(*failureOccurred));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetRemappedRows);
    conn->write(&device, sizeof(device));
    conn->write(&_0corrRows, sizeof(_0corrRows));
    updateTmpPtr((void *)corrRows, _0corrRows);
    conn->write(&_0uncRows, sizeof(_0uncRows));
    updateTmpPtr((void *)uncRows, _0uncRows);
    conn->write(&_0isPending, sizeof(_0isPending));
    updateTmpPtr((void *)isPending, _0isPending);
    conn->write(&_0failureOccurred, sizeof(_0failureOccurred));
    updateTmpPtr((void *)failureOccurred, _0failureOccurred);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)corrRows, sizeof(*corrRows), true);
    mem2client(conn, (void *)uncRows, sizeof(*uncRows), true);
    mem2client(conn, (void *)isPending, sizeof(*isPending), true);
    mem2client(conn, (void *)failureOccurred, sizeof(*failureOccurred), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t *values) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRowRemapperHistogram called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0values;
    mem2server(conn, &_0values, (void *)values, sizeof(*values));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetRowRemapperHistogram);
    conn->write(&device, sizeof(device));
    conn->write(&_0values, sizeof(_0values));
    updateTmpPtr((void *)values, _0values);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)values, sizeof(*values), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t *arch) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetArchitecture called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0arch;
    mem2server(conn, &_0arch, (void *)arch, sizeof(*arch));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetArchitecture);
    conn->write(&device, sizeof(device));
    conn->write(&_0arch, sizeof(_0arch));
    updateTmpPtr((void *)arch, _0arch);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)arch, sizeof(*arch), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color) {
#ifdef DEBUG
    std::cout << "Hook: nvmlUnitSetLedState called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlUnitSetLedState);
    conn->write(&unit, sizeof(unit));
    conn->write(&color, sizeof(color));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetPersistenceMode called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetPersistenceMode);
    conn->write(&device, sizeof(device));
    conn->write(&mode, sizeof(mode));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetComputeMode called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetComputeMode);
    conn->write(&device, sizeof(device));
    conn->write(&mode, sizeof(mode));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetEccMode called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetEccMode);
    conn->write(&device, sizeof(device));
    conn->write(&ecc, sizeof(ecc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceClearEccErrorCounts called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceClearEccErrorCounts);
    conn->write(&device, sizeof(device));
    conn->write(&counterType, sizeof(counterType));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetDriverModel called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetDriverModel);
    conn->write(&device, sizeof(device));
    conn->write(&driverModel, sizeof(driverModel));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetGpuLockedClocks called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetGpuLockedClocks);
    conn->write(&device, sizeof(device));
    conn->write(&minGpuClockMHz, sizeof(minGpuClockMHz));
    conn->write(&maxGpuClockMHz, sizeof(maxGpuClockMHz));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetGpuLockedClocks called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceResetGpuLockedClocks);
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetMemoryLockedClocks called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetMemoryLockedClocks);
    conn->write(&device, sizeof(device));
    conn->write(&minMemClockMHz, sizeof(minMemClockMHz));
    conn->write(&maxMemClockMHz, sizeof(maxMemClockMHz));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetMemoryLockedClocks called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceResetMemoryLockedClocks);
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetApplicationsClocks called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetApplicationsClocks);
    conn->write(&device, sizeof(device));
    conn->write(&memClockMHz, sizeof(memClockMHz));
    conn->write(&graphicsClockMHz, sizeof(graphicsClockMHz));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t *status) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClkMonStatus called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0status;
    mem2server(conn, &_0status, (void *)status, sizeof(*status));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetClkMonStatus);
    conn->write(&device, sizeof(device));
    conn->write(&_0status, sizeof(_0status));
    updateTmpPtr((void *)status, _0status);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)status, sizeof(*status), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int limit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetPowerManagementLimit called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetPowerManagementLimit);
    conn->write(&device, sizeof(device));
    conn->write(&limit, sizeof(limit));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetGpuOperationMode called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetGpuOperationMode);
    conn->write(&device, sizeof(device));
    conn->write(&mode, sizeof(mode));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetAPIRestriction called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetAPIRestriction);
    conn->write(&device, sizeof(device));
    conn->write(&apiType, sizeof(apiType));
    conn->write(&isRestricted, sizeof(isRestricted));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetAccountingMode called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetAccountingMode);
    conn->write(&device, sizeof(device));
    conn->write(&mode, sizeof(mode));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceClearAccountingPids called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceClearAccountingPids);
    conn->write(&device, sizeof(device));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkState called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0isActive;
    mem2server(conn, &_0isActive, (void *)isActive, sizeof(*isActive));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetNvLinkState);
    conn->write(&device, sizeof(device));
    conn->write(&link, sizeof(link));
    conn->write(&_0isActive, sizeof(_0isActive));
    updateTmpPtr((void *)isActive, _0isActive);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)isActive, sizeof(*isActive), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int *version) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkVersion called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetNvLinkVersion);
    conn->write(&device, sizeof(device));
    conn->write(&link, sizeof(link));
    conn->write(&_0version, sizeof(_0version));
    updateTmpPtr((void *)version, _0version);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)version, sizeof(*version), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int *capResult) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkCapability called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0capResult;
    mem2server(conn, &_0capResult, (void *)capResult, sizeof(*capResult));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetNvLinkCapability);
    conn->write(&device, sizeof(device));
    conn->write(&link, sizeof(link));
    conn->write(&capability, sizeof(capability));
    conn->write(&_0capResult, sizeof(_0capResult));
    updateTmpPtr((void *)capResult, _0capResult);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)capResult, sizeof(*capResult), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkRemotePciInfo_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pci;
    mem2server(conn, &_0pci, (void *)pci, sizeof(*pci));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2);
    conn->write(&device, sizeof(device));
    conn->write(&link, sizeof(link));
    conn->write(&_0pci, sizeof(_0pci));
    updateTmpPtr((void *)pci, _0pci);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pci, sizeof(*pci), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long *counterValue) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkErrorCounter called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0counterValue;
    mem2server(conn, &_0counterValue, (void *)counterValue, sizeof(*counterValue));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetNvLinkErrorCounter);
    conn->write(&device, sizeof(device));
    conn->write(&link, sizeof(link));
    conn->write(&counter, sizeof(counter));
    conn->write(&_0counterValue, sizeof(_0counterValue));
    updateTmpPtr((void *)counterValue, _0counterValue);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)counterValue, sizeof(*counterValue), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetNvLinkErrorCounters called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceResetNvLinkErrorCounters);
    conn->write(&device, sizeof(device));
    conn->write(&link, sizeof(link));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control, unsigned int reset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetNvLinkUtilizationControl called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0control;
    mem2server(conn, &_0control, (void *)control, sizeof(*control));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetNvLinkUtilizationControl);
    conn->write(&device, sizeof(device));
    conn->write(&link, sizeof(link));
    conn->write(&counter, sizeof(counter));
    conn->write(&_0control, sizeof(_0control));
    updateTmpPtr((void *)control, _0control);
    conn->write(&reset, sizeof(reset));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)control, sizeof(*control), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkUtilizationControl called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0control;
    mem2server(conn, &_0control, (void *)control, sizeof(*control));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetNvLinkUtilizationControl);
    conn->write(&device, sizeof(device));
    conn->write(&link, sizeof(link));
    conn->write(&counter, sizeof(counter));
    conn->write(&_0control, sizeof(_0control));
    updateTmpPtr((void *)control, _0control);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)control, sizeof(*control), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, unsigned long long *rxcounter, unsigned long long *txcounter) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkUtilizationCounter called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0rxcounter;
    mem2server(conn, &_0rxcounter, (void *)rxcounter, sizeof(*rxcounter));
    void *_0txcounter;
    mem2server(conn, &_0txcounter, (void *)txcounter, sizeof(*txcounter));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetNvLinkUtilizationCounter);
    conn->write(&device, sizeof(device));
    conn->write(&link, sizeof(link));
    conn->write(&counter, sizeof(counter));
    conn->write(&_0rxcounter, sizeof(_0rxcounter));
    updateTmpPtr((void *)rxcounter, _0rxcounter);
    conn->write(&_0txcounter, sizeof(_0txcounter));
    updateTmpPtr((void *)txcounter, _0txcounter);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)rxcounter, sizeof(*rxcounter), true);
    mem2client(conn, (void *)txcounter, sizeof(*txcounter), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceFreezeNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlEnableState_t freeze) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceFreezeNvLinkUtilizationCounter called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceFreezeNvLinkUtilizationCounter);
    conn->write(&device, sizeof(device));
    conn->write(&link, sizeof(link));
    conn->write(&counter, sizeof(counter));
    conn->write(&freeze, sizeof(freeze));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceResetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceResetNvLinkUtilizationCounter called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceResetNvLinkUtilizationCounter);
    conn->write(&device, sizeof(device));
    conn->write(&link, sizeof(link));
    conn->write(&counter, sizeof(counter));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvLinkRemoteDeviceType called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pNvLinkDeviceType;
    mem2server(conn, &_0pNvLinkDeviceType, (void *)pNvLinkDeviceType, sizeof(*pNvLinkDeviceType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetNvLinkRemoteDeviceType);
    conn->write(&device, sizeof(device));
    conn->write(&link, sizeof(link));
    conn->write(&_0pNvLinkDeviceType, sizeof(_0pNvLinkDeviceType));
    updateTmpPtr((void *)pNvLinkDeviceType, _0pNvLinkDeviceType);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pNvLinkDeviceType, sizeof(*pNvLinkDeviceType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t *set) {
#ifdef DEBUG
    std::cout << "Hook: nvmlEventSetCreate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0set;
    mem2server(conn, &_0set, (void *)set, sizeof(*set));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlEventSetCreate);
    conn->write(&_0set, sizeof(_0set));
    updateTmpPtr((void *)set, _0set);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)set, sizeof(*set), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceRegisterEvents called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceRegisterEvents);
    conn->write(&device, sizeof(device));
    conn->write(&eventTypes, sizeof(eventTypes));
    conn->write(&set, sizeof(set));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long *eventTypes) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedEventTypes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0eventTypes;
    mem2server(conn, &_0eventTypes, (void *)eventTypes, sizeof(*eventTypes));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetSupportedEventTypes);
    conn->write(&device, sizeof(device));
    conn->write(&_0eventTypes, sizeof(_0eventTypes));
    updateTmpPtr((void *)eventTypes, _0eventTypes);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)eventTypes, sizeof(*eventTypes), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t *data, unsigned int timeoutms) {
#ifdef DEBUG
    std::cout << "Hook: nvmlEventSetWait_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0data;
    mem2server(conn, &_0data, (void *)data, 0);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlEventSetWait_v2);
    conn->write(&set, sizeof(set));
    conn->write(&_0data, sizeof(_0data));
    updateTmpPtr((void *)data, _0data);
    conn->write(&timeoutms, sizeof(timeoutms));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)data, 0, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set) {
#ifdef DEBUG
    std::cout << "Hook: nvmlEventSetFree called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlEventSetFree);
    conn->write(&set, sizeof(set));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t *pciInfo, nvmlEnableState_t newState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceModifyDrainState called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pciInfo;
    mem2server(conn, &_0pciInfo, (void *)pciInfo, sizeof(*pciInfo));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceModifyDrainState);
    conn->write(&_0pciInfo, sizeof(_0pciInfo));
    updateTmpPtr((void *)pciInfo, _0pciInfo);
    conn->write(&newState, sizeof(newState));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pciInfo, sizeof(*pciInfo), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t *pciInfo, nvmlEnableState_t *currentState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceQueryDrainState called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pciInfo;
    mem2server(conn, &_0pciInfo, (void *)pciInfo, sizeof(*pciInfo));
    void *_0currentState;
    mem2server(conn, &_0currentState, (void *)currentState, sizeof(*currentState));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceQueryDrainState);
    conn->write(&_0pciInfo, sizeof(_0pciInfo));
    updateTmpPtr((void *)pciInfo, _0pciInfo);
    conn->write(&_0currentState, sizeof(_0currentState));
    updateTmpPtr((void *)currentState, _0currentState);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pciInfo, sizeof(*pciInfo), true);
    mem2client(conn, (void *)currentState, sizeof(*currentState), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t *pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceRemoveGpu_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pciInfo;
    mem2server(conn, &_0pciInfo, (void *)pciInfo, sizeof(*pciInfo));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceRemoveGpu_v2);
    conn->write(&_0pciInfo, sizeof(_0pciInfo));
    updateTmpPtr((void *)pciInfo, _0pciInfo);
    conn->write(&gpuState, sizeof(gpuState));
    conn->write(&linkState, sizeof(linkState));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pciInfo, sizeof(*pciInfo), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t *pciInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceDiscoverGpus called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pciInfo;
    mem2server(conn, &_0pciInfo, (void *)pciInfo, sizeof(*pciInfo));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceDiscoverGpus);
    conn->write(&_0pciInfo, sizeof(_0pciInfo));
    updateTmpPtr((void *)pciInfo, _0pciInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pciInfo, sizeof(*pciInfo), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFieldValues called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0values;
    mem2server(conn, &_0values, (void *)values, sizeof(*values));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetFieldValues);
    conn->write(&device, sizeof(device));
    conn->write(&valuesCount, sizeof(valuesCount));
    conn->write(&_0values, sizeof(_0values));
    updateTmpPtr((void *)values, _0values);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)values, sizeof(*values), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t *pVirtualMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVirtualizationMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pVirtualMode;
    mem2server(conn, &_0pVirtualMode, (void *)pVirtualMode, sizeof(*pVirtualMode));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetVirtualizationMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0pVirtualMode, sizeof(_0pVirtualMode));
    updateTmpPtr((void *)pVirtualMode, _0pVirtualMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pVirtualMode, sizeof(*pVirtualMode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t *pHostVgpuMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHostVgpuMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pHostVgpuMode;
    mem2server(conn, &_0pHostVgpuMode, (void *)pHostVgpuMode, sizeof(*pHostVgpuMode));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetHostVgpuMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0pHostVgpuMode, sizeof(_0pHostVgpuMode));
    updateTmpPtr((void *)pHostVgpuMode, _0pHostVgpuMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pHostVgpuMode, sizeof(*pHostVgpuMode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetVirtualizationMode called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetVirtualizationMode);
    conn->write(&device, sizeof(device));
    conn->write(&virtualMode, sizeof(virtualMode));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v3(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGridLicensableFeatures_v3 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pGridLicensableFeatures;
    mem2server(conn, &_0pGridLicensableFeatures, (void *)pGridLicensableFeatures, sizeof(*pGridLicensableFeatures));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetGridLicensableFeatures_v3);
    conn->write(&device, sizeof(device));
    conn->write(&_0pGridLicensableFeatures, sizeof(_0pGridLicensableFeatures));
    updateTmpPtr((void *)pGridLicensableFeatures, _0pGridLicensableFeatures);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pGridLicensableFeatures, sizeof(*pGridLicensableFeatures), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization, unsigned int *processSamplesCount, unsigned long long lastSeenTimeStamp) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetProcessUtilization called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0utilization;
    mem2server(conn, &_0utilization, (void *)utilization, sizeof(*utilization));
    void *_0processSamplesCount;
    mem2server(conn, &_0processSamplesCount, (void *)processSamplesCount, sizeof(*processSamplesCount));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetProcessUtilization);
    conn->write(&device, sizeof(device));
    conn->write(&_0utilization, sizeof(_0utilization));
    updateTmpPtr((void *)utilization, _0utilization);
    conn->write(&_0processSamplesCount, sizeof(_0processSamplesCount));
    updateTmpPtr((void *)processSamplesCount, _0processSamplesCount);
    conn->write(&lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)utilization, sizeof(*utilization), true);
    mem2client(conn, (void *)processSamplesCount, sizeof(*processSamplesCount), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedVgpus called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0vgpuCount;
    mem2server(conn, &_0vgpuCount, (void *)vgpuCount, sizeof(*vgpuCount));
    void *_0vgpuTypeIds;
    mem2server(conn, &_0vgpuTypeIds, (void *)vgpuTypeIds, sizeof(*vgpuTypeIds));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetSupportedVgpus);
    conn->write(&device, sizeof(device));
    conn->write(&_0vgpuCount, sizeof(_0vgpuCount));
    updateTmpPtr((void *)vgpuCount, _0vgpuCount);
    conn->write(&_0vgpuTypeIds, sizeof(_0vgpuTypeIds));
    updateTmpPtr((void *)vgpuTypeIds, _0vgpuTypeIds);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)vgpuCount, sizeof(*vgpuCount), true);
    mem2client(conn, (void *)vgpuTypeIds, sizeof(*vgpuTypeIds), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCreatableVgpus called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0vgpuCount;
    mem2server(conn, &_0vgpuCount, (void *)vgpuCount, sizeof(*vgpuCount));
    void *_0vgpuTypeIds;
    mem2server(conn, &_0vgpuTypeIds, (void *)vgpuTypeIds, sizeof(*vgpuTypeIds));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetCreatableVgpus);
    conn->write(&device, sizeof(device));
    conn->write(&_0vgpuCount, sizeof(_0vgpuCount));
    updateTmpPtr((void *)vgpuCount, _0vgpuCount);
    conn->write(&_0vgpuTypeIds, sizeof(_0vgpuTypeIds));
    updateTmpPtr((void *)vgpuTypeIds, _0vgpuTypeIds);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)vgpuCount, sizeof(*vgpuCount), true);
    mem2client(conn, (void *)vgpuTypeIds, sizeof(*vgpuTypeIds), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeClass, unsigned int *size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetClass called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0size;
    mem2server(conn, &_0size, (void *)size, sizeof(*size));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuTypeGetClass);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    if(*size > 0) {
        conn->read(vgpuTypeClass, *size, true);
    }
    conn->write(&_0size, sizeof(_0size));
    updateTmpPtr((void *)size, _0size);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)size, sizeof(*size), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeName, unsigned int *size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetName called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0size;
    mem2server(conn, &_0size, (void *)size, sizeof(*size));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuTypeGetName);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    if(*size > 0) {
        conn->read(vgpuTypeName, *size, true);
    }
    conn->write(&_0size, sizeof(_0size));
    updateTmpPtr((void *)size, _0size);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)size, sizeof(*size), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *gpuInstanceProfileId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetGpuInstanceProfileId called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0gpuInstanceProfileId;
    mem2server(conn, &_0gpuInstanceProfileId, (void *)gpuInstanceProfileId, sizeof(*gpuInstanceProfileId));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuTypeGetGpuInstanceProfileId);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    conn->write(&_0gpuInstanceProfileId, sizeof(_0gpuInstanceProfileId));
    updateTmpPtr((void *)gpuInstanceProfileId, _0gpuInstanceProfileId);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)gpuInstanceProfileId, sizeof(*gpuInstanceProfileId), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *deviceID, unsigned long long *subsystemID) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetDeviceID called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0deviceID;
    mem2server(conn, &_0deviceID, (void *)deviceID, sizeof(*deviceID));
    void *_0subsystemID;
    mem2server(conn, &_0subsystemID, (void *)subsystemID, sizeof(*subsystemID));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuTypeGetDeviceID);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    conn->write(&_0deviceID, sizeof(_0deviceID));
    updateTmpPtr((void *)deviceID, _0deviceID);
    conn->write(&_0subsystemID, sizeof(_0subsystemID));
    updateTmpPtr((void *)subsystemID, _0subsystemID);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)deviceID, sizeof(*deviceID), true);
    mem2client(conn, (void *)subsystemID, sizeof(*subsystemID), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *fbSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetFramebufferSize called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0fbSize;
    mem2server(conn, &_0fbSize, (void *)fbSize, sizeof(*fbSize));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuTypeGetFramebufferSize);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    conn->write(&_0fbSize, sizeof(_0fbSize));
    updateTmpPtr((void *)fbSize, _0fbSize);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)fbSize, sizeof(*fbSize), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *numDisplayHeads) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetNumDisplayHeads called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0numDisplayHeads;
    mem2server(conn, &_0numDisplayHeads, (void *)numDisplayHeads, sizeof(*numDisplayHeads));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuTypeGetNumDisplayHeads);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    conn->write(&_0numDisplayHeads, sizeof(_0numDisplayHeads));
    updateTmpPtr((void *)numDisplayHeads, _0numDisplayHeads);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)numDisplayHeads, sizeof(*numDisplayHeads), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int *xdim, unsigned int *ydim) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetResolution called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0xdim;
    mem2server(conn, &_0xdim, (void *)xdim, sizeof(*xdim));
    void *_0ydim;
    mem2server(conn, &_0ydim, (void *)ydim, sizeof(*ydim));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuTypeGetResolution);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    conn->write(&displayIndex, sizeof(displayIndex));
    conn->write(&_0xdim, sizeof(_0xdim));
    updateTmpPtr((void *)xdim, _0xdim);
    conn->write(&_0ydim, sizeof(_0ydim));
    updateTmpPtr((void *)ydim, _0ydim);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)xdim, sizeof(*xdim), true);
    mem2client(conn, (void *)ydim, sizeof(*ydim), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeLicenseString, unsigned int size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetLicense called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuTypeGetLicense);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    if(size > 0) {
        conn->read(vgpuTypeLicenseString, size, true);
    }
    conn->write(&size, sizeof(size));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *frameRateLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetFrameRateLimit called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0frameRateLimit;
    mem2server(conn, &_0frameRateLimit, (void *)frameRateLimit, sizeof(*frameRateLimit));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuTypeGetFrameRateLimit);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    conn->write(&_0frameRateLimit, sizeof(_0frameRateLimit));
    updateTmpPtr((void *)frameRateLimit, _0frameRateLimit);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)frameRateLimit, sizeof(*frameRateLimit), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetMaxInstances called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0vgpuInstanceCount;
    mem2server(conn, &_0vgpuInstanceCount, (void *)vgpuInstanceCount, sizeof(*vgpuInstanceCount));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuTypeGetMaxInstances);
    conn->write(&device, sizeof(device));
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    conn->write(&_0vgpuInstanceCount, sizeof(_0vgpuInstanceCount));
    updateTmpPtr((void *)vgpuInstanceCount, _0vgpuInstanceCount);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)vgpuInstanceCount, sizeof(*vgpuInstanceCount), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCountPerVm) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetMaxInstancesPerVm called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0vgpuInstanceCountPerVm;
    mem2server(conn, &_0vgpuInstanceCountPerVm, (void *)vgpuInstanceCountPerVm, sizeof(*vgpuInstanceCountPerVm));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuTypeGetMaxInstancesPerVm);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    conn->write(&_0vgpuInstanceCountPerVm, sizeof(_0vgpuInstanceCountPerVm));
    updateTmpPtr((void *)vgpuInstanceCountPerVm, _0vgpuInstanceCountPerVm);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)vgpuInstanceCountPerVm, sizeof(*vgpuInstanceCountPerVm), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuInstance_t *vgpuInstances) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetActiveVgpus called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0vgpuCount;
    mem2server(conn, &_0vgpuCount, (void *)vgpuCount, sizeof(*vgpuCount));
    void *_0vgpuInstances;
    mem2server(conn, &_0vgpuInstances, (void *)vgpuInstances, sizeof(*vgpuInstances));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetActiveVgpus);
    conn->write(&device, sizeof(device));
    conn->write(&_0vgpuCount, sizeof(_0vgpuCount));
    updateTmpPtr((void *)vgpuCount, _0vgpuCount);
    conn->write(&_0vgpuInstances, sizeof(_0vgpuInstances));
    updateTmpPtr((void *)vgpuInstances, _0vgpuInstances);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)vgpuCount, sizeof(*vgpuCount), true);
    mem2client(conn, (void *)vgpuInstances, sizeof(*vgpuInstances), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char *vmId, unsigned int size, nvmlVgpuVmIdType_t *vmIdType) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetVmID called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0vmIdType;
    mem2server(conn, &_0vmIdType, (void *)vmIdType, sizeof(*vmIdType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetVmID);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    if(size > 0) {
        conn->read(vmId, size, true);
    }
    conn->write(&size, sizeof(size));
    conn->write(&_0vmIdType, sizeof(_0vmIdType));
    updateTmpPtr((void *)vmIdType, _0vmIdType);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)vmIdType, sizeof(*vmIdType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char *uuid, unsigned int size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetUUID called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetUUID);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    if(size > 0) {
        conn->read(uuid, size, true);
    }
    conn->write(&size, sizeof(size));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char *version, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetVmDriverVersion called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetVmDriverVersion);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    if(length > 0) {
        conn->read(version, length, true);
    }
    conn->write(&length, sizeof(length));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long *fbUsage) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFbUsage called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0fbUsage;
    mem2server(conn, &_0fbUsage, (void *)fbUsage, sizeof(*fbUsage));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetFbUsage);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0fbUsage, sizeof(_0fbUsage));
    updateTmpPtr((void *)fbUsage, _0fbUsage);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)fbUsage, sizeof(*fbUsage), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int *licensed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetLicenseStatus called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0licensed;
    mem2server(conn, &_0licensed, (void *)licensed, sizeof(*licensed));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetLicenseStatus);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0licensed, sizeof(_0licensed));
    updateTmpPtr((void *)licensed, _0licensed);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)licensed, sizeof(*licensed), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t *vgpuTypeId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetType called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0vgpuTypeId;
    mem2server(conn, &_0vgpuTypeId, (void *)vgpuTypeId, sizeof(*vgpuTypeId));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetType);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0vgpuTypeId, sizeof(_0vgpuTypeId));
    updateTmpPtr((void *)vgpuTypeId, _0vgpuTypeId);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)vgpuTypeId, sizeof(*vgpuTypeId), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int *frameRateLimit) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFrameRateLimit called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0frameRateLimit;
    mem2server(conn, &_0frameRateLimit, (void *)frameRateLimit, sizeof(*frameRateLimit));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetFrameRateLimit);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0frameRateLimit, sizeof(_0frameRateLimit));
    updateTmpPtr((void *)frameRateLimit, _0frameRateLimit);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)frameRateLimit, sizeof(*frameRateLimit), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *eccMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEccMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0eccMode;
    mem2server(conn, &_0eccMode, (void *)eccMode, sizeof(*eccMode));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetEccMode);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0eccMode, sizeof(_0eccMode));
    updateTmpPtr((void *)eccMode, _0eccMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)eccMode, sizeof(*eccMode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int *encoderCapacity) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEncoderCapacity called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0encoderCapacity;
    mem2server(conn, &_0encoderCapacity, (void *)encoderCapacity, sizeof(*encoderCapacity));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetEncoderCapacity);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0encoderCapacity, sizeof(_0encoderCapacity));
    updateTmpPtr((void *)encoderCapacity, _0encoderCapacity);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)encoderCapacity, sizeof(*encoderCapacity), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceSetEncoderCapacity called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceSetEncoderCapacity);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&encoderCapacity, sizeof(encoderCapacity));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEncoderStats called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0sessionCount;
    mem2server(conn, &_0sessionCount, (void *)sessionCount, sizeof(*sessionCount));
    void *_0averageFps;
    mem2server(conn, &_0averageFps, (void *)averageFps, sizeof(*averageFps));
    void *_0averageLatency;
    mem2server(conn, &_0averageLatency, (void *)averageLatency, sizeof(*averageLatency));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetEncoderStats);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0sessionCount, sizeof(_0sessionCount));
    updateTmpPtr((void *)sessionCount, _0sessionCount);
    conn->write(&_0averageFps, sizeof(_0averageFps));
    updateTmpPtr((void *)averageFps, _0averageFps);
    conn->write(&_0averageLatency, sizeof(_0averageLatency));
    updateTmpPtr((void *)averageLatency, _0averageLatency);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)sessionCount, sizeof(*sessionCount), true);
    mem2client(conn, (void *)averageFps, sizeof(*averageFps), true);
    mem2client(conn, (void *)averageLatency, sizeof(*averageLatency), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetEncoderSessions called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0sessionCount;
    mem2server(conn, &_0sessionCount, (void *)sessionCount, sizeof(*sessionCount));
    void *_0sessionInfo;
    mem2server(conn, &_0sessionInfo, (void *)sessionInfo, sizeof(*sessionInfo));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetEncoderSessions);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0sessionCount, sizeof(_0sessionCount));
    updateTmpPtr((void *)sessionCount, _0sessionCount);
    conn->write(&_0sessionInfo, sizeof(_0sessionInfo));
    updateTmpPtr((void *)sessionInfo, _0sessionInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)sessionCount, sizeof(*sessionCount), true);
    mem2client(conn, (void *)sessionInfo, sizeof(*sessionInfo), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t *fbcStats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFBCStats called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0fbcStats;
    mem2server(conn, &_0fbcStats, (void *)fbcStats, sizeof(*fbcStats));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetFBCStats);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0fbcStats, sizeof(_0fbcStats));
    updateTmpPtr((void *)fbcStats, _0fbcStats);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)fbcStats, sizeof(*fbcStats), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetFBCSessions called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0sessionCount;
    mem2server(conn, &_0sessionCount, (void *)sessionCount, sizeof(*sessionCount));
    void *_0sessionInfo;
    mem2server(conn, &_0sessionInfo, (void *)sessionInfo, sizeof(*sessionInfo));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetFBCSessions);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0sessionCount, sizeof(_0sessionCount));
    updateTmpPtr((void *)sessionCount, _0sessionCount);
    conn->write(&_0sessionInfo, sizeof(_0sessionInfo));
    updateTmpPtr((void *)sessionInfo, _0sessionInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)sessionCount, sizeof(*sessionCount), true);
    mem2client(conn, (void *)sessionInfo, sizeof(*sessionInfo), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int *gpuInstanceId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetGpuInstanceId called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0gpuInstanceId;
    mem2server(conn, &_0gpuInstanceId, (void *)gpuInstanceId, sizeof(*gpuInstanceId));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetGpuInstanceId);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0gpuInstanceId, sizeof(_0gpuInstanceId));
    updateTmpPtr((void *)gpuInstanceId, _0gpuInstanceId);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)gpuInstanceId, sizeof(*gpuInstanceId), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t *vgpuMetadata, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetMetadata called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0vgpuMetadata;
    mem2server(conn, &_0vgpuMetadata, (void *)vgpuMetadata, sizeof(*vgpuMetadata));
    void *_0bufferSize;
    mem2server(conn, &_0bufferSize, (void *)bufferSize, sizeof(*bufferSize));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetMetadata);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0vgpuMetadata, sizeof(_0vgpuMetadata));
    updateTmpPtr((void *)vgpuMetadata, _0vgpuMetadata);
    conn->write(&_0bufferSize, sizeof(_0bufferSize));
    updateTmpPtr((void *)bufferSize, _0bufferSize);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)vgpuMetadata, sizeof(*vgpuMetadata), true);
    mem2client(conn, (void *)bufferSize, sizeof(*bufferSize), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t *pgpuMetadata, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuMetadata called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pgpuMetadata;
    mem2server(conn, &_0pgpuMetadata, (void *)pgpuMetadata, sizeof(*pgpuMetadata));
    void *_0bufferSize;
    mem2server(conn, &_0bufferSize, (void *)bufferSize, sizeof(*bufferSize));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetVgpuMetadata);
    conn->write(&device, sizeof(device));
    conn->write(&_0pgpuMetadata, sizeof(_0pgpuMetadata));
    updateTmpPtr((void *)pgpuMetadata, _0pgpuMetadata);
    conn->write(&_0bufferSize, sizeof(_0bufferSize));
    updateTmpPtr((void *)bufferSize, _0bufferSize);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pgpuMetadata, sizeof(*pgpuMetadata), true);
    mem2client(conn, (void *)bufferSize, sizeof(*bufferSize), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t *vgpuMetadata, nvmlVgpuPgpuMetadata_t *pgpuMetadata, nvmlVgpuPgpuCompatibility_t *compatibilityInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetVgpuCompatibility called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0vgpuMetadata;
    mem2server(conn, &_0vgpuMetadata, (void *)vgpuMetadata, sizeof(*vgpuMetadata));
    void *_0pgpuMetadata;
    mem2server(conn, &_0pgpuMetadata, (void *)pgpuMetadata, sizeof(*pgpuMetadata));
    void *_0compatibilityInfo;
    mem2server(conn, &_0compatibilityInfo, (void *)compatibilityInfo, sizeof(*compatibilityInfo));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlGetVgpuCompatibility);
    conn->write(&_0vgpuMetadata, sizeof(_0vgpuMetadata));
    updateTmpPtr((void *)vgpuMetadata, _0vgpuMetadata);
    conn->write(&_0pgpuMetadata, sizeof(_0pgpuMetadata));
    updateTmpPtr((void *)pgpuMetadata, _0pgpuMetadata);
    conn->write(&_0compatibilityInfo, sizeof(_0compatibilityInfo));
    updateTmpPtr((void *)compatibilityInfo, _0compatibilityInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)vgpuMetadata, sizeof(*vgpuMetadata), true);
    mem2client(conn, (void *)pgpuMetadata, sizeof(*pgpuMetadata), true);
    mem2client(conn, (void *)compatibilityInfo, sizeof(*compatibilityInfo), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char *pgpuMetadata, unsigned int *bufferSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPgpuMetadataString called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0bufferSize;
    mem2server(conn, &_0bufferSize, (void *)bufferSize, sizeof(*bufferSize));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetPgpuMetadataString);
    conn->write(&device, sizeof(device));
    if(*bufferSize > 0) {
        conn->read(pgpuMetadata, *bufferSize, true);
    }
    conn->write(&_0bufferSize, sizeof(_0bufferSize));
    updateTmpPtr((void *)bufferSize, _0bufferSize);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)bufferSize, sizeof(*bufferSize), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t *supported, nvmlVgpuVersion_t *current) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetVgpuVersion called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0supported;
    mem2server(conn, &_0supported, (void *)supported, sizeof(*supported));
    void *_0current;
    mem2server(conn, &_0current, (void *)current, sizeof(*current));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlGetVgpuVersion);
    conn->write(&_0supported, sizeof(_0supported));
    updateTmpPtr((void *)supported, _0supported);
    conn->write(&_0current, sizeof(_0current));
    updateTmpPtr((void *)current, _0current);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)supported, sizeof(*supported), true);
    mem2client(conn, (void *)current, sizeof(*current), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t *vgpuVersion) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSetVgpuVersion called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0vgpuVersion;
    mem2server(conn, &_0vgpuVersion, (void *)vgpuVersion, sizeof(*vgpuVersion));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlSetVgpuVersion);
    conn->write(&_0vgpuVersion, sizeof(_0vgpuVersion));
    updateTmpPtr((void *)vgpuVersion, _0vgpuVersion);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)vgpuVersion, sizeof(*vgpuVersion), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t *utilizationSamples) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuUtilization called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0sampleValType;
    mem2server(conn, &_0sampleValType, (void *)sampleValType, sizeof(*sampleValType));
    void *_0vgpuInstanceSamplesCount;
    mem2server(conn, &_0vgpuInstanceSamplesCount, (void *)vgpuInstanceSamplesCount, sizeof(*vgpuInstanceSamplesCount));
    void *_0utilizationSamples;
    mem2server(conn, &_0utilizationSamples, (void *)utilizationSamples, sizeof(*utilizationSamples));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetVgpuUtilization);
    conn->write(&device, sizeof(device));
    conn->write(&lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    conn->write(&_0sampleValType, sizeof(_0sampleValType));
    updateTmpPtr((void *)sampleValType, _0sampleValType);
    conn->write(&_0vgpuInstanceSamplesCount, sizeof(_0vgpuInstanceSamplesCount));
    updateTmpPtr((void *)vgpuInstanceSamplesCount, _0vgpuInstanceSamplesCount);
    conn->write(&_0utilizationSamples, sizeof(_0utilizationSamples));
    updateTmpPtr((void *)utilizationSamples, _0utilizationSamples);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)sampleValType, sizeof(*sampleValType), true);
    mem2client(conn, (void *)vgpuInstanceSamplesCount, sizeof(*vgpuInstanceSamplesCount), true);
    mem2client(conn, (void *)utilizationSamples, sizeof(*utilizationSamples), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int *vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t *utilizationSamples) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuProcessUtilization called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0vgpuProcessSamplesCount;
    mem2server(conn, &_0vgpuProcessSamplesCount, (void *)vgpuProcessSamplesCount, sizeof(*vgpuProcessSamplesCount));
    void *_0utilizationSamples;
    mem2server(conn, &_0utilizationSamples, (void *)utilizationSamples, sizeof(*utilizationSamples));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetVgpuProcessUtilization);
    conn->write(&device, sizeof(device));
    conn->write(&lastSeenTimeStamp, sizeof(lastSeenTimeStamp));
    conn->write(&_0vgpuProcessSamplesCount, sizeof(_0vgpuProcessSamplesCount));
    updateTmpPtr((void *)vgpuProcessSamplesCount, _0vgpuProcessSamplesCount);
    conn->write(&_0utilizationSamples, sizeof(_0utilizationSamples));
    updateTmpPtr((void *)utilizationSamples, _0utilizationSamples);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)vgpuProcessSamplesCount, sizeof(*vgpuProcessSamplesCount), true);
    mem2client(conn, (void *)utilizationSamples, sizeof(*utilizationSamples), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetAccountingMode called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetAccountingMode);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0mode, sizeof(_0mode));
    updateTmpPtr((void *)mode, _0mode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)mode, sizeof(*mode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int *count, unsigned int *pids) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetAccountingPids called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0count;
    mem2server(conn, &_0count, (void *)count, sizeof(*count));
    void *_0pids;
    mem2server(conn, &_0pids, (void *)pids, sizeof(*pids));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetAccountingPids);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->write(&_0pids, sizeof(_0pids));
    updateTmpPtr((void *)pids, _0pids);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)count, sizeof(*count), true);
    mem2client(conn, (void *)pids, sizeof(*pids), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t *stats) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetAccountingStats called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0stats;
    mem2server(conn, &_0stats, (void *)stats, sizeof(*stats));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceGetAccountingStats);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&pid, sizeof(pid));
    conn->write(&_0stats, sizeof(_0stats));
    updateTmpPtr((void *)stats, _0stats);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)stats, sizeof(*stats), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceClearAccountingPids called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlVgpuInstanceClearAccountingPids);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int *deviceCount) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetExcludedDeviceCount called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0deviceCount;
    mem2server(conn, &_0deviceCount, (void *)deviceCount, sizeof(*deviceCount));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlGetExcludedDeviceCount);
    conn->write(&_0deviceCount, sizeof(_0deviceCount));
    updateTmpPtr((void *)deviceCount, _0deviceCount);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)deviceCount, sizeof(*deviceCount), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetExcludedDeviceInfoByIndex called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0info;
    mem2server(conn, &_0info, (void *)info, sizeof(*info));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlGetExcludedDeviceInfoByIndex);
    conn->write(&index, sizeof(index));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)info, sizeof(*info), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t *activationStatus) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetMigMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0activationStatus;
    mem2server(conn, &_0activationStatus, (void *)activationStatus, sizeof(*activationStatus));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceSetMigMode);
    conn->write(&device, sizeof(device));
    conn->write(&mode, sizeof(mode));
    conn->write(&_0activationStatus, sizeof(_0activationStatus));
    updateTmpPtr((void *)activationStatus, _0activationStatus);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)activationStatus, sizeof(*activationStatus), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *currentMode, unsigned int *pendingMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMigMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0currentMode;
    mem2server(conn, &_0currentMode, (void *)currentMode, sizeof(*currentMode));
    void *_0pendingMode;
    mem2server(conn, &_0pendingMode, (void *)pendingMode, sizeof(*pendingMode));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetMigMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0currentMode, sizeof(_0currentMode));
    updateTmpPtr((void *)currentMode, _0currentMode);
    conn->write(&_0pendingMode, sizeof(_0pendingMode));
    updateTmpPtr((void *)pendingMode, _0pendingMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)currentMode, sizeof(*currentMode), true);
    mem2client(conn, (void *)pendingMode, sizeof(*pendingMode), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceProfileInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0info;
    mem2server(conn, &_0info, (void *)info, sizeof(*info));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetGpuInstanceProfileInfo);
    conn->write(&device, sizeof(device));
    conn->write(&profile, sizeof(profile));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)info, sizeof(*info), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t *placements, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstancePossiblePlacements_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0placements;
    mem2server(conn, &_0placements, (void *)placements, sizeof(*placements));
    void *_0count;
    mem2server(conn, &_0count, (void *)count, sizeof(*count));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetGpuInstancePossiblePlacements_v2);
    conn->write(&device, sizeof(device));
    conn->write(&profileId, sizeof(profileId));
    conn->write(&_0placements, sizeof(_0placements));
    updateTmpPtr((void *)placements, _0placements);
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)placements, sizeof(*placements), true);
    mem2client(conn, (void *)count, sizeof(*count), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceRemainingCapacity called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetGpuInstanceRemainingCapacity);
    conn->write(&device, sizeof(device));
    conn->write(&profileId, sizeof(profileId));
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)count, sizeof(*count), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceCreateGpuInstance called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0gpuInstance;
    mem2server(conn, &_0gpuInstance, (void *)gpuInstance, sizeof(*gpuInstance));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceCreateGpuInstance);
    conn->write(&device, sizeof(device));
    conn->write(&profileId, sizeof(profileId));
    conn->write(&_0gpuInstance, sizeof(_0gpuInstance));
    updateTmpPtr((void *)gpuInstance, _0gpuInstance);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)gpuInstance, sizeof(*gpuInstance), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t *placement, nvmlGpuInstance_t *gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceCreateGpuInstanceWithPlacement called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0placement;
    mem2server(conn, &_0placement, (void *)placement, sizeof(*placement));
    void *_0gpuInstance;
    mem2server(conn, &_0gpuInstance, (void *)gpuInstance, sizeof(*gpuInstance));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceCreateGpuInstanceWithPlacement);
    conn->write(&device, sizeof(device));
    conn->write(&profileId, sizeof(profileId));
    conn->write(&_0placement, sizeof(_0placement));
    updateTmpPtr((void *)placement, _0placement);
    conn->write(&_0gpuInstance, sizeof(_0gpuInstance));
    updateTmpPtr((void *)gpuInstance, _0gpuInstance);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)placement, sizeof(*placement), true);
    mem2client(conn, (void *)gpuInstance, sizeof(*gpuInstance), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceDestroy called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlGpuInstanceDestroy);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstances, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstances called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0gpuInstances;
    mem2server(conn, &_0gpuInstances, (void *)gpuInstances, sizeof(*gpuInstances));
    void *_0count;
    mem2server(conn, &_0count, (void *)count, sizeof(*count));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetGpuInstances);
    conn->write(&device, sizeof(device));
    conn->write(&profileId, sizeof(profileId));
    conn->write(&_0gpuInstances, sizeof(_0gpuInstances));
    updateTmpPtr((void *)gpuInstances, _0gpuInstances);
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)gpuInstances, sizeof(*gpuInstances), true);
    mem2client(conn, (void *)count, sizeof(*count), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t *gpuInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceById called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0gpuInstance;
    mem2server(conn, &_0gpuInstance, (void *)gpuInstance, sizeof(*gpuInstance));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetGpuInstanceById);
    conn->write(&device, sizeof(device));
    conn->write(&id, sizeof(id));
    conn->write(&_0gpuInstance, sizeof(_0gpuInstance));
    updateTmpPtr((void *)gpuInstance, _0gpuInstance);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)gpuInstance, sizeof(*gpuInstance), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0info;
    mem2server(conn, &_0info, (void *)info, sizeof(*info));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlGpuInstanceGetInfo);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)info, sizeof(*info), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceProfileInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0info;
    mem2server(conn, &_0info, (void *)info, sizeof(*info));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlGpuInstanceGetComputeInstanceProfileInfo);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&profile, sizeof(profile));
    conn->write(&engProfile, sizeof(engProfile));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)info, sizeof(*info), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceRemainingCapacity called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlGpuInstanceGetComputeInstanceRemainingCapacity);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&profileId, sizeof(profileId));
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)count, sizeof(*count), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceCreateComputeInstance called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0computeInstance;
    mem2server(conn, &_0computeInstance, (void *)computeInstance, sizeof(*computeInstance));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlGpuInstanceCreateComputeInstance);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&profileId, sizeof(profileId));
    conn->write(&_0computeInstance, sizeof(_0computeInstance));
    updateTmpPtr((void *)computeInstance, _0computeInstance);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)computeInstance, sizeof(*computeInstance), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlComputeInstanceDestroy called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlComputeInstanceDestroy);
    conn->write(&computeInstance, sizeof(computeInstance));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstances, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstances called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0computeInstances;
    mem2server(conn, &_0computeInstances, (void *)computeInstances, sizeof(*computeInstances));
    void *_0count;
    mem2server(conn, &_0count, (void *)count, sizeof(*count));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlGpuInstanceGetComputeInstances);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&profileId, sizeof(profileId));
    conn->write(&_0computeInstances, sizeof(_0computeInstances));
    updateTmpPtr((void *)computeInstances, _0computeInstances);
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)computeInstances, sizeof(*computeInstances), true);
    mem2client(conn, (void *)count, sizeof(*count), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t *computeInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceById called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0computeInstance;
    mem2server(conn, &_0computeInstance, (void *)computeInstance, sizeof(*computeInstance));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlGpuInstanceGetComputeInstanceById);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&id, sizeof(id));
    conn->write(&_0computeInstance, sizeof(_0computeInstance));
    updateTmpPtr((void *)computeInstance, _0computeInstance);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)computeInstance, sizeof(*computeInstance), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlComputeInstanceGetInfo_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0info;
    mem2server(conn, &_0info, (void *)info, sizeof(*info));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlComputeInstanceGetInfo_v2);
    conn->write(&computeInstance, sizeof(computeInstance));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)info, sizeof(*info), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int *isMigDevice) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceIsMigDeviceHandle called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0isMigDevice;
    mem2server(conn, &_0isMigDevice, (void *)isMigDevice, sizeof(*isMigDevice));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceIsMigDeviceHandle);
    conn->write(&device, sizeof(device));
    conn->write(&_0isMigDevice, sizeof(_0isMigDevice));
    updateTmpPtr((void *)isMigDevice, _0isMigDevice);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)isMigDevice, sizeof(*isMigDevice), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int *id) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceId called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0id;
    mem2server(conn, &_0id, (void *)id, sizeof(*id));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetGpuInstanceId);
    conn->write(&device, sizeof(device));
    conn->write(&_0id, sizeof(_0id));
    updateTmpPtr((void *)id, _0id);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)id, sizeof(*id), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int *id) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetComputeInstanceId called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0id;
    mem2server(conn, &_0id, (void *)id, sizeof(*id));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetComputeInstanceId);
    conn->write(&device, sizeof(device));
    conn->write(&_0id, sizeof(_0id));
    updateTmpPtr((void *)id, _0id);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)id, sizeof(*id), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMaxMigDeviceCount called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetMaxMigDeviceCount);
    conn->write(&device, sizeof(device));
    conn->write(&_0count, sizeof(_0count));
    updateTmpPtr((void *)count, _0count);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)count, sizeof(*count), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t *migDevice) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMigDeviceHandleByIndex called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0migDevice;
    mem2server(conn, &_0migDevice, (void *)migDevice, sizeof(*migDevice));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetMigDeviceHandleByIndex);
    conn->write(&device, sizeof(device));
    conn->write(&index, sizeof(index));
    conn->write(&_0migDevice, sizeof(_0migDevice));
    updateTmpPtr((void *)migDevice, _0migDevice);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)migDevice, sizeof(*migDevice), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDeviceHandleFromMigDeviceHandle called" << std::endl;
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
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    nvmlReturn_t _result;
    conn->prepare_request(RPC_nvmlDeviceGetDeviceHandleFromMigDeviceHandle);
    conn->write(&migDevice, sizeof(migDevice));
    conn->write(&_0device, sizeof(_0device));
    updateTmpPtr((void *)device, _0device);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)device, sizeof(*device), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}
