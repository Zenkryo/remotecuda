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

extern "C" nvmlReturn_t nvmlSystemGetDriverBranch(nvmlSystemDriverBranchInfo_t *branchInfo, unsigned int length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetDriverBranch called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0branchInfo;
    mem2server(conn, &_0branchInfo, (void *)branchInfo, sizeof(*branchInfo));
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
    conn->prepare_request(RPC_nvmlSystemGetDriverBranch);
    conn->write(&_0branchInfo, sizeof(_0branchInfo));
    updateTmpPtr((void *)branchInfo, _0branchInfo);
    conn->write(&length, sizeof(length));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)branchInfo, sizeof(*branchInfo), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetHandleByUUIDV(const nvmlUUID_t *uuid, nvmlDevice_t *device) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetHandleByUUIDV called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0uuid;
    mem2server(conn, &_0uuid, (void *)uuid, sizeof(*uuid));
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
    conn->prepare_request(RPC_nvmlDeviceGetHandleByUUIDV);
    conn->write(&_0uuid, sizeof(_0uuid));
    updateTmpPtr((void *)uuid, _0uuid);
    conn->write(&_0device, sizeof(_0device));
    updateTmpPtr((void *)device, _0device);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)uuid, sizeof(*uuid), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetModuleId(nvmlDevice_t device, unsigned int *moduleId) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetModuleId called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0moduleId;
    mem2server(conn, &_0moduleId, (void *)moduleId, sizeof(*moduleId));
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
    conn->prepare_request(RPC_nvmlDeviceGetModuleId);
    conn->write(&device, sizeof(device));
    conn->write(&_0moduleId, sizeof(_0moduleId));
    updateTmpPtr((void *)moduleId, _0moduleId);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)moduleId, sizeof(*moduleId), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetC2cModeInfoV(nvmlDevice_t device, nvmlC2cModeInfo_v1_t *c2cModeInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetC2cModeInfoV called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0c2cModeInfo;
    mem2server(conn, &_0c2cModeInfo, (void *)c2cModeInfo, sizeof(*c2cModeInfo));
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
    conn->prepare_request(RPC_nvmlDeviceGetC2cModeInfoV);
    conn->write(&device, sizeof(device));
    conn->write(&_0c2cModeInfo, sizeof(_0c2cModeInfo));
    updateTmpPtr((void *)c2cModeInfo, _0c2cModeInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)c2cModeInfo, sizeof(*c2cModeInfo), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetNumaNodeId(nvmlDevice_t device, unsigned int *node) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNumaNodeId called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0node;
    mem2server(conn, &_0node, (void *)node, sizeof(*node));
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
    conn->prepare_request(RPC_nvmlDeviceGetNumaNodeId);
    conn->write(&device, sizeof(device));
    conn->write(&_0node, sizeof(_0node));
    updateTmpPtr((void *)node, _0node);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)node, sizeof(*node), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetLastBBXFlushTime(nvmlDevice_t device, unsigned long long *timestamp, unsigned long *durationUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetLastBBXFlushTime called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0timestamp;
    mem2server(conn, &_0timestamp, (void *)timestamp, sizeof(*timestamp));
    void *_0durationUs;
    mem2server(conn, &_0durationUs, (void *)durationUs, sizeof(*durationUs));
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
    conn->prepare_request(RPC_nvmlDeviceGetLastBBXFlushTime);
    conn->write(&device, sizeof(device));
    conn->write(&_0timestamp, sizeof(_0timestamp));
    updateTmpPtr((void *)timestamp, _0timestamp);
    conn->write(&_0durationUs, sizeof(_0durationUs));
    updateTmpPtr((void *)durationUs, _0durationUs);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)timestamp, sizeof(*timestamp), true);
    mem2client(conn, (void *)durationUs, sizeof(*durationUs), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetPciInfoExt(nvmlDevice_t device, nvmlPciInfoExt_t *pci) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPciInfoExt called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetPciInfoExt);
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

extern "C" nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGenDevice) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuMaxPcieLinkGeneration called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0maxLinkGenDevice;
    mem2server(conn, &_0maxLinkGenDevice, (void *)maxLinkGenDevice, sizeof(*maxLinkGenDevice));
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
    conn->prepare_request(RPC_nvmlDeviceGetGpuMaxPcieLinkGeneration);
    conn->write(&device, sizeof(device));
    conn->write(&_0maxLinkGenDevice, sizeof(_0maxLinkGenDevice));
    updateTmpPtr((void *)maxLinkGenDevice, _0maxLinkGenDevice);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)maxLinkGenDevice, sizeof(*maxLinkGenDevice), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int *offset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpcClkVfOffset called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0offset;
    mem2server(conn, &_0offset, (void *)offset, sizeof(*offset));
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
    conn->prepare_request(RPC_nvmlDeviceGetGpcClkVfOffset);
    conn->write(&device, sizeof(device));
    conn->write(&_0offset, sizeof(_0offset));
    updateTmpPtr((void *)offset, _0offset);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)offset, sizeof(*offset), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetFanSpeedRPM(nvmlDevice_t device, nvmlFanSpeedInfo_t *fanSpeed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFanSpeedRPM called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0fanSpeed;
    mem2server(conn, &_0fanSpeed, (void *)fanSpeed, sizeof(*fanSpeed));
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
    conn->prepare_request(RPC_nvmlDeviceGetFanSpeedRPM);
    conn->write(&device, sizeof(device));
    conn->write(&_0fanSpeed, sizeof(_0fanSpeed));
    updateTmpPtr((void *)fanSpeed, _0fanSpeed);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)fanSpeed, sizeof(*fanSpeed), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int *targetSpeed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTargetFanSpeed called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0targetSpeed;
    mem2server(conn, &_0targetSpeed, (void *)targetSpeed, sizeof(*targetSpeed));
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
    conn->prepare_request(RPC_nvmlDeviceGetTargetFanSpeed);
    conn->write(&device, sizeof(device));
    conn->write(&fan, sizeof(fan));
    conn->write(&_0targetSpeed, sizeof(_0targetSpeed));
    updateTmpPtr((void *)targetSpeed, _0targetSpeed);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)targetSpeed, sizeof(*targetSpeed), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int *minSpeed, unsigned int *maxSpeed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMinMaxFanSpeed called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0minSpeed;
    mem2server(conn, &_0minSpeed, (void *)minSpeed, sizeof(*minSpeed));
    void *_0maxSpeed;
    mem2server(conn, &_0maxSpeed, (void *)maxSpeed, sizeof(*maxSpeed));
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
    conn->prepare_request(RPC_nvmlDeviceGetMinMaxFanSpeed);
    conn->write(&device, sizeof(device));
    conn->write(&_0minSpeed, sizeof(_0minSpeed));
    updateTmpPtr((void *)minSpeed, _0minSpeed);
    conn->write(&_0maxSpeed, sizeof(_0maxSpeed));
    updateTmpPtr((void *)maxSpeed, _0maxSpeed);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)minSpeed, sizeof(*minSpeed), true);
    mem2client(conn, (void *)maxSpeed, sizeof(*maxSpeed), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t *policy) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetFanControlPolicy_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0policy;
    mem2server(conn, &_0policy, (void *)policy, sizeof(*policy));
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
    conn->prepare_request(RPC_nvmlDeviceGetFanControlPolicy_v2);
    conn->write(&device, sizeof(device));
    conn->write(&fan, sizeof(fan));
    conn->write(&_0policy, sizeof(_0policy));
    updateTmpPtr((void *)policy, _0policy);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)policy, sizeof(*policy), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int *numFans) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNumFans called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0numFans;
    mem2server(conn, &_0numFans, (void *)numFans, sizeof(*numFans));
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
    conn->prepare_request(RPC_nvmlDeviceGetNumFans);
    conn->write(&device, sizeof(device));
    conn->write(&_0numFans, sizeof(_0numFans));
    updateTmpPtr((void *)numFans, _0numFans);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)numFans, sizeof(*numFans), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetCoolerInfo(nvmlDevice_t device, nvmlCoolerInfo_t *coolerInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCoolerInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0coolerInfo;
    mem2server(conn, &_0coolerInfo, (void *)coolerInfo, sizeof(*coolerInfo));
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
    conn->prepare_request(RPC_nvmlDeviceGetCoolerInfo);
    conn->write(&device, sizeof(device));
    conn->write(&_0coolerInfo, sizeof(_0coolerInfo));
    updateTmpPtr((void *)coolerInfo, _0coolerInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)coolerInfo, sizeof(*coolerInfo), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetTemperatureV(nvmlDevice_t device, nvmlTemperature_t *temperature) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetTemperatureV called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0temperature;
    mem2server(conn, &_0temperature, (void *)temperature, sizeof(*temperature));
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
    conn->prepare_request(RPC_nvmlDeviceGetTemperatureV);
    conn->write(&device, sizeof(device));
    conn->write(&_0temperature, sizeof(_0temperature));
    updateTmpPtr((void *)temperature, _0temperature);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)temperature, sizeof(*temperature), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetMarginTemperature(nvmlDevice_t device, nvmlMarginTemperature_t *marginTempInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMarginTemperature called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0marginTempInfo;
    mem2server(conn, &_0marginTempInfo, (void *)marginTempInfo, sizeof(*marginTempInfo));
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
    conn->prepare_request(RPC_nvmlDeviceGetMarginTemperature);
    conn->write(&device, sizeof(device));
    conn->write(&_0marginTempInfo, sizeof(_0marginTempInfo));
    updateTmpPtr((void *)marginTempInfo, _0marginTempInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)marginTempInfo, sizeof(*marginTempInfo), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t *pThermalSettings) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetThermalSettings called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pThermalSettings;
    mem2server(conn, &_0pThermalSettings, (void *)pThermalSettings, sizeof(*pThermalSettings));
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
    conn->prepare_request(RPC_nvmlDeviceGetThermalSettings);
    conn->write(&device, sizeof(device));
    conn->write(&sensorIndex, sizeof(sensorIndex));
    conn->write(&_0pThermalSettings, sizeof(_0pThermalSettings));
    updateTmpPtr((void *)pThermalSettings, _0pThermalSettings);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pThermalSettings, sizeof(*pThermalSettings), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t device, unsigned long long *clocksEventReasons) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrentClocksEventReasons called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0clocksEventReasons;
    mem2server(conn, &_0clocksEventReasons, (void *)clocksEventReasons, sizeof(*clocksEventReasons));
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
    conn->prepare_request(RPC_nvmlDeviceGetCurrentClocksEventReasons);
    conn->write(&device, sizeof(device));
    conn->write(&_0clocksEventReasons, sizeof(_0clocksEventReasons));
    updateTmpPtr((void *)clocksEventReasons, _0clocksEventReasons);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)clocksEventReasons, sizeof(*clocksEventReasons), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetSupportedClocksEventReasons(nvmlDevice_t device, unsigned long long *supportedClocksEventReasons) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedClocksEventReasons called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0supportedClocksEventReasons;
    mem2server(conn, &_0supportedClocksEventReasons, (void *)supportedClocksEventReasons, sizeof(*supportedClocksEventReasons));
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
    conn->prepare_request(RPC_nvmlDeviceGetSupportedClocksEventReasons);
    conn->write(&device, sizeof(device));
    conn->write(&_0supportedClocksEventReasons, sizeof(_0supportedClocksEventReasons));
    updateTmpPtr((void *)supportedClocksEventReasons, _0supportedClocksEventReasons);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)supportedClocksEventReasons, sizeof(*supportedClocksEventReasons), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t *pDynamicPstatesInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDynamicPstatesInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pDynamicPstatesInfo;
    mem2server(conn, &_0pDynamicPstatesInfo, (void *)pDynamicPstatesInfo, sizeof(*pDynamicPstatesInfo));
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
    conn->prepare_request(RPC_nvmlDeviceGetDynamicPstatesInfo);
    conn->write(&device, sizeof(device));
    conn->write(&_0pDynamicPstatesInfo, sizeof(_0pDynamicPstatesInfo));
    updateTmpPtr((void *)pDynamicPstatesInfo, _0pDynamicPstatesInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pDynamicPstatesInfo, sizeof(*pDynamicPstatesInfo), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int *offset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemClkVfOffset called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0offset;
    mem2server(conn, &_0offset, (void *)offset, sizeof(*offset));
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
    conn->prepare_request(RPC_nvmlDeviceGetMemClkVfOffset);
    conn->write(&device, sizeof(device));
    conn->write(&_0offset, sizeof(_0offset));
    updateTmpPtr((void *)offset, _0offset);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)offset, sizeof(*offset), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int *minClockMHz, unsigned int *maxClockMHz) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMinMaxClockOfPState called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0minClockMHz;
    mem2server(conn, &_0minClockMHz, (void *)minClockMHz, sizeof(*minClockMHz));
    void *_0maxClockMHz;
    mem2server(conn, &_0maxClockMHz, (void *)maxClockMHz, sizeof(*maxClockMHz));
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
    conn->prepare_request(RPC_nvmlDeviceGetMinMaxClockOfPState);
    conn->write(&device, sizeof(device));
    conn->write(&type, sizeof(type));
    conn->write(&pstate, sizeof(pstate));
    conn->write(&_0minClockMHz, sizeof(_0minClockMHz));
    updateTmpPtr((void *)minClockMHz, _0minClockMHz);
    conn->write(&_0maxClockMHz, sizeof(_0maxClockMHz));
    updateTmpPtr((void *)maxClockMHz, _0maxClockMHz);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)minClockMHz, sizeof(*minClockMHz), true);
    mem2client(conn, (void *)maxClockMHz, sizeof(*maxClockMHz), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t *pstates, unsigned int size) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSupportedPerformanceStates called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pstates;
    mem2server(conn, &_0pstates, (void *)pstates, sizeof(*pstates));
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
    conn->prepare_request(RPC_nvmlDeviceGetSupportedPerformanceStates);
    conn->write(&device, sizeof(device));
    conn->write(&_0pstates, sizeof(_0pstates));
    updateTmpPtr((void *)pstates, _0pstates);
    conn->write(&size, sizeof(size));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pstates, sizeof(*pstates), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device, int *minOffset, int *maxOffset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpcClkMinMaxVfOffset called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0minOffset;
    mem2server(conn, &_0minOffset, (void *)minOffset, sizeof(*minOffset));
    void *_0maxOffset;
    mem2server(conn, &_0maxOffset, (void *)maxOffset, sizeof(*maxOffset));
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
    conn->prepare_request(RPC_nvmlDeviceGetGpcClkMinMaxVfOffset);
    conn->write(&device, sizeof(device));
    conn->write(&_0minOffset, sizeof(_0minOffset));
    updateTmpPtr((void *)minOffset, _0minOffset);
    conn->write(&_0maxOffset, sizeof(_0maxOffset));
    updateTmpPtr((void *)maxOffset, _0maxOffset);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)minOffset, sizeof(*minOffset), true);
    mem2client(conn, (void *)maxOffset, sizeof(*maxOffset), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device, int *minOffset, int *maxOffset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemClkMinMaxVfOffset called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0minOffset;
    mem2server(conn, &_0minOffset, (void *)minOffset, sizeof(*minOffset));
    void *_0maxOffset;
    mem2server(conn, &_0maxOffset, (void *)maxOffset, sizeof(*maxOffset));
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
    conn->prepare_request(RPC_nvmlDeviceGetMemClkMinMaxVfOffset);
    conn->write(&device, sizeof(device));
    conn->write(&_0minOffset, sizeof(_0minOffset));
    updateTmpPtr((void *)minOffset, _0minOffset);
    conn->write(&_0maxOffset, sizeof(_0maxOffset));
    updateTmpPtr((void *)maxOffset, _0maxOffset);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)minOffset, sizeof(*minOffset), true);
    mem2client(conn, (void *)maxOffset, sizeof(*maxOffset), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetClockOffsets called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetClockOffsets);
    conn->write(&device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceSetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetClockOffsets called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceSetClockOffsets);
    conn->write(&device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlDeviceGetPerformanceModes(nvmlDevice_t device, nvmlDevicePerfModes_t *perfModes) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPerformanceModes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0perfModes;
    mem2server(conn, &_0perfModes, (void *)perfModes, sizeof(*perfModes));
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
    conn->prepare_request(RPC_nvmlDeviceGetPerformanceModes);
    conn->write(&device, sizeof(device));
    conn->write(&_0perfModes, sizeof(_0perfModes));
    updateTmpPtr((void *)perfModes, _0perfModes);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)perfModes, sizeof(*perfModes), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetCurrentClockFreqs(nvmlDevice_t device, nvmlDeviceCurrentClockFreqs_t *currentClockFreqs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCurrentClockFreqs called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0currentClockFreqs;
    mem2server(conn, &_0currentClockFreqs, (void *)currentClockFreqs, sizeof(*currentClockFreqs));
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
    conn->prepare_request(RPC_nvmlDeviceGetCurrentClockFreqs);
    conn->write(&device, sizeof(device));
    conn->write(&_0currentClockFreqs, sizeof(_0currentClockFreqs));
    updateTmpPtr((void *)currentClockFreqs, _0currentClockFreqs);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)currentClockFreqs, sizeof(*currentClockFreqs), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryInfo_v2 called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetMemoryInfo_v2);
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

extern "C" nvmlReturn_t nvmlDeviceGetDramEncryptionMode(nvmlDevice_t device, nvmlDramEncryptionInfo_t *current, nvmlDramEncryptionInfo_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDramEncryptionMode called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetDramEncryptionMode);
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

extern "C" nvmlReturn_t nvmlDeviceSetDramEncryptionMode(nvmlDevice_t device, const nvmlDramEncryptionInfo_t *dramEncryption) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetDramEncryptionMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0dramEncryption;
    mem2server(conn, &_0dramEncryption, (void *)dramEncryption, sizeof(*dramEncryption));
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
    conn->prepare_request(RPC_nvmlDeviceSetDramEncryptionMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0dramEncryption, sizeof(_0dramEncryption));
    updateTmpPtr((void *)dramEncryption, _0dramEncryption);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)dramEncryption, sizeof(*dramEncryption), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t *defaultMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDefaultEccMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0defaultMode;
    mem2server(conn, &_0defaultMode, (void *)defaultMode, sizeof(*defaultMode));
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
    conn->prepare_request(RPC_nvmlDeviceGetDefaultEccMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0defaultMode, sizeof(_0defaultMode));
    updateTmpPtr((void *)defaultMode, _0defaultMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)defaultMode, sizeof(*defaultMode), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetJpgUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetJpgUtilization called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetJpgUtilization);
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

extern "C" nvmlReturn_t nvmlDeviceGetOfaUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetOfaUtilization called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetOfaUtilization);
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

extern "C" nvmlReturn_t nvmlDeviceGetDriverModel_v2(nvmlDevice_t device, nvmlDriverModel_t *current, nvmlDriverModel_t *pending) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetDriverModel_v2 called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetDriverModel_v2);
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

extern "C" nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetComputeRunningProcesses_v3 called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetComputeRunningProcesses_v3);
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

extern "C" nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGraphicsRunningProcesses_v3 called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetGraphicsRunningProcesses_v3);
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

extern "C" nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMPSComputeRunningProcesses_v3 called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetMPSComputeRunningProcesses_v3);
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

extern "C" nvmlReturn_t nvmlDeviceGetRunningProcessDetailList(nvmlDevice_t device, nvmlProcessDetailList_t *plist) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetRunningProcessDetailList called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0plist;
    mem2server(conn, &_0plist, (void *)plist, sizeof(*plist));
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
    conn->prepare_request(RPC_nvmlDeviceGetRunningProcessDetailList);
    conn->write(&device, sizeof(device));
    conn->write(&_0plist, sizeof(_0plist));
    updateTmpPtr((void *)plist, _0plist);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)plist, sizeof(*plist), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int *irqNum) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetIrqNum called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0irqNum;
    mem2server(conn, &_0irqNum, (void *)irqNum, sizeof(*irqNum));
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
    conn->prepare_request(RPC_nvmlDeviceGetIrqNum);
    conn->write(&device, sizeof(device));
    conn->write(&_0irqNum, sizeof(_0irqNum));
    updateTmpPtr((void *)irqNum, _0irqNum);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)irqNum, sizeof(*irqNum), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int *numCores) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNumGpuCores called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0numCores;
    mem2server(conn, &_0numCores, (void *)numCores, sizeof(*numCores));
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
    conn->prepare_request(RPC_nvmlDeviceGetNumGpuCores);
    conn->write(&device, sizeof(device));
    conn->write(&_0numCores, sizeof(_0numCores));
    updateTmpPtr((void *)numCores, _0numCores);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)numCores, sizeof(*numCores), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t *powerSource) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPowerSource called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0powerSource;
    mem2server(conn, &_0powerSource, (void *)powerSource, sizeof(*powerSource));
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
    conn->prepare_request(RPC_nvmlDeviceGetPowerSource);
    conn->write(&device, sizeof(device));
    conn->write(&_0powerSource, sizeof(_0powerSource));
    updateTmpPtr((void *)powerSource, _0powerSource);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)powerSource, sizeof(*powerSource), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int *busWidth) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetMemoryBusWidth called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0busWidth;
    mem2server(conn, &_0busWidth, (void *)busWidth, sizeof(*busWidth));
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
    conn->prepare_request(RPC_nvmlDeviceGetMemoryBusWidth);
    conn->write(&device, sizeof(device));
    conn->write(&_0busWidth, sizeof(_0busWidth));
    updateTmpPtr((void *)busWidth, _0busWidth);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)busWidth, sizeof(*busWidth), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int *maxSpeed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPcieLinkMaxSpeed called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0maxSpeed;
    mem2server(conn, &_0maxSpeed, (void *)maxSpeed, sizeof(*maxSpeed));
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
    conn->prepare_request(RPC_nvmlDeviceGetPcieLinkMaxSpeed);
    conn->write(&device, sizeof(device));
    conn->write(&_0maxSpeed, sizeof(_0maxSpeed));
    updateTmpPtr((void *)maxSpeed, _0maxSpeed);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)maxSpeed, sizeof(*maxSpeed), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int *pcieSpeed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPcieSpeed called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pcieSpeed;
    mem2server(conn, &_0pcieSpeed, (void *)pcieSpeed, sizeof(*pcieSpeed));
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
    conn->prepare_request(RPC_nvmlDeviceGetPcieSpeed);
    conn->write(&device, sizeof(device));
    conn->write(&_0pcieSpeed, sizeof(_0pcieSpeed));
    updateTmpPtr((void *)pcieSpeed, _0pcieSpeed);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pcieSpeed, sizeof(*pcieSpeed), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int *adaptiveClockStatus) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetAdaptiveClockInfoStatus called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0adaptiveClockStatus;
    mem2server(conn, &_0adaptiveClockStatus, (void *)adaptiveClockStatus, sizeof(*adaptiveClockStatus));
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
    conn->prepare_request(RPC_nvmlDeviceGetAdaptiveClockInfoStatus);
    conn->write(&device, sizeof(device));
    conn->write(&_0adaptiveClockStatus, sizeof(_0adaptiveClockStatus));
    updateTmpPtr((void *)adaptiveClockStatus, _0adaptiveClockStatus);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)adaptiveClockStatus, sizeof(*adaptiveClockStatus), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t *type) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetBusType called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetBusType);
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

extern "C" nvmlReturn_t nvmlDeviceGetGpuFabricInfo(nvmlDevice_t device, nvmlGpuFabricInfo_t *gpuFabricInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuFabricInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0gpuFabricInfo;
    mem2server(conn, &_0gpuFabricInfo, (void *)gpuFabricInfo, sizeof(*gpuFabricInfo));
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
    conn->prepare_request(RPC_nvmlDeviceGetGpuFabricInfo);
    conn->write(&device, sizeof(device));
    conn->write(&_0gpuFabricInfo, sizeof(_0gpuFabricInfo));
    updateTmpPtr((void *)gpuFabricInfo, _0gpuFabricInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)gpuFabricInfo, sizeof(*gpuFabricInfo), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t *gpuFabricInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuFabricInfoV called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0gpuFabricInfo;
    mem2server(conn, &_0gpuFabricInfo, (void *)gpuFabricInfo, sizeof(*gpuFabricInfo));
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
    conn->prepare_request(RPC_nvmlDeviceGetGpuFabricInfoV);
    conn->write(&device, sizeof(device));
    conn->write(&_0gpuFabricInfo, sizeof(_0gpuFabricInfo));
    updateTmpPtr((void *)gpuFabricInfo, _0gpuFabricInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)gpuFabricInfo, sizeof(*gpuFabricInfo), true);
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

extern "C" nvmlReturn_t nvmlSystemGetConfComputeCapabilities(nvmlConfComputeSystemCaps_t *capabilities) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetConfComputeCapabilities called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0capabilities;
    mem2server(conn, &_0capabilities, (void *)capabilities, sizeof(*capabilities));
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
    conn->prepare_request(RPC_nvmlSystemGetConfComputeCapabilities);
    conn->write(&_0capabilities, sizeof(_0capabilities));
    updateTmpPtr((void *)capabilities, _0capabilities);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)capabilities, sizeof(*capabilities), true);
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

extern "C" nvmlReturn_t nvmlSystemGetConfComputeState(nvmlConfComputeSystemState_t *state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetConfComputeState called" << std::endl;
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
    conn->prepare_request(RPC_nvmlSystemGetConfComputeState);
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

extern "C" nvmlReturn_t nvmlDeviceGetConfComputeMemSizeInfo(nvmlDevice_t device, nvmlConfComputeMemSizeInfo_t *memInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetConfComputeMemSizeInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0memInfo;
    mem2server(conn, &_0memInfo, (void *)memInfo, sizeof(*memInfo));
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
    conn->prepare_request(RPC_nvmlDeviceGetConfComputeMemSizeInfo);
    conn->write(&device, sizeof(device));
    conn->write(&_0memInfo, sizeof(_0memInfo));
    updateTmpPtr((void *)memInfo, _0memInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)memInfo, sizeof(*memInfo), true);
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

extern "C" nvmlReturn_t nvmlSystemGetConfComputeGpusReadyState(unsigned int *isAcceptingWork) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetConfComputeGpusReadyState called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0isAcceptingWork;
    mem2server(conn, &_0isAcceptingWork, (void *)isAcceptingWork, sizeof(*isAcceptingWork));
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
    conn->prepare_request(RPC_nvmlSystemGetConfComputeGpusReadyState);
    conn->write(&_0isAcceptingWork, sizeof(_0isAcceptingWork));
    updateTmpPtr((void *)isAcceptingWork, _0isAcceptingWork);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)isAcceptingWork, sizeof(*isAcceptingWork), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetConfComputeProtectedMemoryUsage(nvmlDevice_t device, nvmlMemory_t *memory) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetConfComputeProtectedMemoryUsage called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetConfComputeProtectedMemoryUsage);
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

extern "C" nvmlReturn_t nvmlDeviceGetConfComputeGpuCertificate(nvmlDevice_t device, nvmlConfComputeGpuCertificate_t *gpuCert) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetConfComputeGpuCertificate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0gpuCert;
    mem2server(conn, &_0gpuCert, (void *)gpuCert, sizeof(*gpuCert));
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
    conn->prepare_request(RPC_nvmlDeviceGetConfComputeGpuCertificate);
    conn->write(&device, sizeof(device));
    conn->write(&_0gpuCert, sizeof(_0gpuCert));
    updateTmpPtr((void *)gpuCert, _0gpuCert);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)gpuCert, sizeof(*gpuCert), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetConfComputeGpuAttestationReport(nvmlDevice_t device, nvmlConfComputeGpuAttestationReport_t *gpuAtstReport) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetConfComputeGpuAttestationReport called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0gpuAtstReport;
    mem2server(conn, &_0gpuAtstReport, (void *)gpuAtstReport, sizeof(*gpuAtstReport));
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
    conn->prepare_request(RPC_nvmlDeviceGetConfComputeGpuAttestationReport);
    conn->write(&device, sizeof(device));
    conn->write(&_0gpuAtstReport, sizeof(_0gpuAtstReport));
    updateTmpPtr((void *)gpuAtstReport, _0gpuAtstReport);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)gpuAtstReport, sizeof(*gpuAtstReport), true);
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

extern "C" nvmlReturn_t nvmlSystemGetConfComputeKeyRotationThresholdInfo(nvmlConfComputeGetKeyRotationThresholdInfo_t *pKeyRotationThrInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetConfComputeKeyRotationThresholdInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pKeyRotationThrInfo;
    mem2server(conn, &_0pKeyRotationThrInfo, (void *)pKeyRotationThrInfo, sizeof(*pKeyRotationThrInfo));
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
    conn->prepare_request(RPC_nvmlSystemGetConfComputeKeyRotationThresholdInfo);
    conn->write(&_0pKeyRotationThrInfo, sizeof(_0pKeyRotationThrInfo));
    updateTmpPtr((void *)pKeyRotationThrInfo, _0pKeyRotationThrInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pKeyRotationThrInfo, sizeof(*pKeyRotationThrInfo), true);
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

extern "C" nvmlReturn_t nvmlDeviceSetConfComputeUnprotectedMemSize(nvmlDevice_t device, unsigned long long sizeKiB) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetConfComputeUnprotectedMemSize called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceSetConfComputeUnprotectedMemSize);
    conn->write(&device, sizeof(device));
    conn->write(&sizeKiB, sizeof(sizeKiB));
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

extern "C" nvmlReturn_t nvmlSystemSetConfComputeGpusReadyState(unsigned int isAcceptingWork) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemSetConfComputeGpusReadyState called" << std::endl;
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
    conn->prepare_request(RPC_nvmlSystemSetConfComputeGpusReadyState);
    conn->write(&isAcceptingWork, sizeof(isAcceptingWork));
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

extern "C" nvmlReturn_t nvmlSystemSetConfComputeKeyRotationThresholdInfo(nvmlConfComputeSetKeyRotationThresholdInfo_t *pKeyRotationThrInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemSetConfComputeKeyRotationThresholdInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pKeyRotationThrInfo;
    mem2server(conn, &_0pKeyRotationThrInfo, (void *)pKeyRotationThrInfo, sizeof(*pKeyRotationThrInfo));
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
    conn->prepare_request(RPC_nvmlSystemSetConfComputeKeyRotationThresholdInfo);
    conn->write(&_0pKeyRotationThrInfo, sizeof(_0pKeyRotationThrInfo));
    updateTmpPtr((void *)pKeyRotationThrInfo, _0pKeyRotationThrInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pKeyRotationThrInfo, sizeof(*pKeyRotationThrInfo), true);
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

extern "C" nvmlReturn_t nvmlSystemGetConfComputeSettings(nvmlSystemConfComputeSettings_t *settings) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetConfComputeSettings called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0settings;
    mem2server(conn, &_0settings, (void *)settings, sizeof(*settings));
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
    conn->prepare_request(RPC_nvmlSystemGetConfComputeSettings);
    conn->write(&_0settings, sizeof(_0settings));
    updateTmpPtr((void *)settings, _0settings);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)settings, sizeof(*settings), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char *version) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGspFirmwareVersion called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetGspFirmwareVersion);
    conn->write(&device, sizeof(device));
    if(32 > 0) {
        conn->read(version, 32, true);
    }
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

extern "C" nvmlReturn_t nvmlDeviceGetGspFirmwareMode(nvmlDevice_t device, unsigned int *isEnabled, unsigned int *defaultMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGspFirmwareMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0isEnabled;
    mem2server(conn, &_0isEnabled, (void *)isEnabled, sizeof(*isEnabled));
    void *_0defaultMode;
    mem2server(conn, &_0defaultMode, (void *)defaultMode, sizeof(*defaultMode));
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
    conn->prepare_request(RPC_nvmlDeviceGetGspFirmwareMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0isEnabled, sizeof(_0isEnabled));
    updateTmpPtr((void *)isEnabled, _0isEnabled);
    conn->write(&_0defaultMode, sizeof(_0defaultMode));
    updateTmpPtr((void *)defaultMode, _0defaultMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)isEnabled, sizeof(*isEnabled), true);
    mem2client(conn, (void *)defaultMode, sizeof(*defaultMode), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetSramEccErrorStatus(nvmlDevice_t device, nvmlEccSramErrorStatus_t *status) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetSramEccErrorStatus called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetSramEccErrorStatus);
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

extern "C" nvmlReturn_t nvmlDeviceGetProcessesUtilizationInfo(nvmlDevice_t device, nvmlProcessesUtilizationInfo_t *procesesUtilInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetProcessesUtilizationInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0procesesUtilInfo;
    mem2server(conn, &_0procesesUtilInfo, (void *)procesesUtilInfo, sizeof(*procesesUtilInfo));
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
    conn->prepare_request(RPC_nvmlDeviceGetProcessesUtilizationInfo);
    conn->write(&device, sizeof(device));
    conn->write(&_0procesesUtilInfo, sizeof(_0procesesUtilInfo));
    updateTmpPtr((void *)procesesUtilInfo, _0procesesUtilInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)procesesUtilInfo, sizeof(*procesesUtilInfo), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetPlatformInfo(nvmlDevice_t device, nvmlPlatformInfo_t *platformInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetPlatformInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0platformInfo;
    mem2server(conn, &_0platformInfo, (void *)platformInfo, sizeof(*platformInfo));
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
    conn->prepare_request(RPC_nvmlDeviceGetPlatformInfo);
    conn->write(&device, sizeof(device));
    conn->write(&_0platformInfo, sizeof(_0platformInfo));
    updateTmpPtr((void *)platformInfo, _0platformInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)platformInfo, sizeof(*platformInfo), true);
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

extern "C" nvmlReturn_t nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device, unsigned int fan) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetDefaultFanSpeed_v2 called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceSetDefaultFanSpeed_v2);
    conn->write(&device, sizeof(device));
    conn->write(&fan, sizeof(fan));
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

extern "C" nvmlReturn_t nvmlDeviceSetFanControlPolicy(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t policy) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetFanControlPolicy called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceSetFanControlPolicy);
    conn->write(&device, sizeof(device));
    conn->write(&fan, sizeof(fan));
    conn->write(&policy, sizeof(policy));
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

extern "C" nvmlReturn_t nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int speed) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetFanSpeed_v2 called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceSetFanSpeed_v2);
    conn->write(&device, sizeof(device));
    conn->write(&fan, sizeof(fan));
    conn->write(&speed, sizeof(speed));
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

extern "C" nvmlReturn_t nvmlDeviceSetGpcClkVfOffset(nvmlDevice_t device, int offset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetGpcClkVfOffset called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceSetGpcClkVfOffset);
    conn->write(&device, sizeof(device));
    conn->write(&offset, sizeof(offset));
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

extern "C" nvmlReturn_t nvmlDeviceSetMemClkVfOffset(nvmlDevice_t device, int offset) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetMemClkVfOffset called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceSetMemClkVfOffset);
    conn->write(&device, sizeof(device));
    conn->write(&offset, sizeof(offset));
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

extern "C" nvmlReturn_t nvmlDeviceSetPowerManagementLimit_v2(nvmlDevice_t device, nvmlPowerValue_v2_t *powerValue) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetPowerManagementLimit_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0powerValue;
    mem2server(conn, &_0powerValue, (void *)powerValue, sizeof(*powerValue));
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
    conn->prepare_request(RPC_nvmlDeviceSetPowerManagementLimit_v2);
    conn->write(&device, sizeof(device));
    conn->write(&_0powerValue, sizeof(_0powerValue));
    updateTmpPtr((void *)powerValue, _0powerValue);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)powerValue, sizeof(*powerValue), true);
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

extern "C" nvmlReturn_t nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetNvLinkDeviceLowPowerThreshold called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceSetNvLinkDeviceLowPowerThreshold);
    conn->write(&device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlSystemSetNvlinkBwMode(unsigned int nvlinkBwMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemSetNvlinkBwMode called" << std::endl;
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
    conn->prepare_request(RPC_nvmlSystemSetNvlinkBwMode);
    conn->write(&nvlinkBwMode, sizeof(nvlinkBwMode));
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

extern "C" nvmlReturn_t nvmlSystemGetNvlinkBwMode(unsigned int *nvlinkBwMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemGetNvlinkBwMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0nvlinkBwMode;
    mem2server(conn, &_0nvlinkBwMode, (void *)nvlinkBwMode, sizeof(*nvlinkBwMode));
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
    conn->prepare_request(RPC_nvmlSystemGetNvlinkBwMode);
    conn->write(&_0nvlinkBwMode, sizeof(_0nvlinkBwMode));
    updateTmpPtr((void *)nvlinkBwMode, _0nvlinkBwMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)nvlinkBwMode, sizeof(*nvlinkBwMode), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetNvlinkSupportedBwModes(nvmlDevice_t device, nvmlNvlinkSupportedBwModes_t *supportedBwMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvlinkSupportedBwModes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0supportedBwMode;
    mem2server(conn, &_0supportedBwMode, (void *)supportedBwMode, sizeof(*supportedBwMode));
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
    conn->prepare_request(RPC_nvmlDeviceGetNvlinkSupportedBwModes);
    conn->write(&device, sizeof(device));
    conn->write(&_0supportedBwMode, sizeof(_0supportedBwMode));
    updateTmpPtr((void *)supportedBwMode, _0supportedBwMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)supportedBwMode, sizeof(*supportedBwMode), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkGetBwMode_t *getBwMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetNvlinkBwMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0getBwMode;
    mem2server(conn, &_0getBwMode, (void *)getBwMode, sizeof(*getBwMode));
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
    conn->prepare_request(RPC_nvmlDeviceGetNvlinkBwMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0getBwMode, sizeof(_0getBwMode));
    updateTmpPtr((void *)getBwMode, _0getBwMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)getBwMode, sizeof(*getBwMode), true);
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

extern "C" nvmlReturn_t nvmlDeviceSetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkSetBwMode_t *setBwMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetNvlinkBwMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0setBwMode;
    mem2server(conn, &_0setBwMode, (void *)setBwMode, sizeof(*setBwMode));
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
    conn->prepare_request(RPC_nvmlDeviceSetNvlinkBwMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0setBwMode, sizeof(_0setBwMode));
    updateTmpPtr((void *)setBwMode, _0setBwMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)setBwMode, sizeof(*setBwMode), true);
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

extern "C" nvmlReturn_t nvmlSystemEventSetCreate(nvmlSystemEventSetCreateRequest_t *request) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemEventSetCreate called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0request;
    mem2server(conn, &_0request, (void *)request, sizeof(*request));
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
    conn->prepare_request(RPC_nvmlSystemEventSetCreate);
    conn->write(&_0request, sizeof(_0request));
    updateTmpPtr((void *)request, _0request);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)request, sizeof(*request), true);
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

extern "C" nvmlReturn_t nvmlSystemEventSetFree(nvmlSystemEventSetFreeRequest_t *request) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemEventSetFree called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0request;
    mem2server(conn, &_0request, (void *)request, sizeof(*request));
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
    conn->prepare_request(RPC_nvmlSystemEventSetFree);
    conn->write(&_0request, sizeof(_0request));
    updateTmpPtr((void *)request, _0request);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)request, sizeof(*request), true);
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

extern "C" nvmlReturn_t nvmlSystemRegisterEvents(nvmlSystemRegisterEventRequest_t *request) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemRegisterEvents called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0request;
    mem2server(conn, &_0request, (void *)request, sizeof(*request));
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
    conn->prepare_request(RPC_nvmlSystemRegisterEvents);
    conn->write(&_0request, sizeof(_0request));
    updateTmpPtr((void *)request, _0request);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)request, sizeof(*request), true);
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

extern "C" nvmlReturn_t nvmlSystemEventSetWait(nvmlSystemEventSetWaitRequest_t *request) {
#ifdef DEBUG
    std::cout << "Hook: nvmlSystemEventSetWait called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0request;
    mem2server(conn, &_0request, (void *)request, sizeof(*request));
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
    conn->prepare_request(RPC_nvmlSystemEventSetWait);
    conn->write(&_0request, sizeof(_0request));
    updateTmpPtr((void *)request, _0request);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)request, sizeof(*request), true);
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

extern "C" nvmlReturn_t nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceClearFieldValues called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceClearFieldValues);
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

extern "C" nvmlReturn_t nvmlDeviceGetVgpuHeterogeneousMode(nvmlDevice_t device, nvmlVgpuHeterogeneousMode_t *pHeterogeneousMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuHeterogeneousMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pHeterogeneousMode;
    mem2server(conn, &_0pHeterogeneousMode, (void *)pHeterogeneousMode, sizeof(*pHeterogeneousMode));
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
    conn->prepare_request(RPC_nvmlDeviceGetVgpuHeterogeneousMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0pHeterogeneousMode, sizeof(_0pHeterogeneousMode));
    updateTmpPtr((void *)pHeterogeneousMode, _0pHeterogeneousMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pHeterogeneousMode, sizeof(*pHeterogeneousMode), true);
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

extern "C" nvmlReturn_t nvmlDeviceSetVgpuHeterogeneousMode(nvmlDevice_t device, const nvmlVgpuHeterogeneousMode_t *pHeterogeneousMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetVgpuHeterogeneousMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pHeterogeneousMode;
    mem2server(conn, &_0pHeterogeneousMode, (void *)pHeterogeneousMode, sizeof(*pHeterogeneousMode));
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
    conn->prepare_request(RPC_nvmlDeviceSetVgpuHeterogeneousMode);
    conn->write(&device, sizeof(device));
    conn->write(&_0pHeterogeneousMode, sizeof(_0pHeterogeneousMode));
    updateTmpPtr((void *)pHeterogeneousMode, _0pHeterogeneousMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pHeterogeneousMode, sizeof(*pHeterogeneousMode), true);
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

extern "C" nvmlReturn_t nvmlVgpuInstanceGetPlacementId(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuPlacementId_t *pPlacement) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetPlacementId called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pPlacement;
    mem2server(conn, &_0pPlacement, (void *)pPlacement, sizeof(*pPlacement));
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
    conn->prepare_request(RPC_nvmlVgpuInstanceGetPlacementId);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0pPlacement, sizeof(_0pPlacement));
    updateTmpPtr((void *)pPlacement, _0pPlacement);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pPlacement, sizeof(*pPlacement), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetVgpuTypeSupportedPlacements(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuPlacementList_t *pPlacementList) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuTypeSupportedPlacements called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pPlacementList;
    mem2server(conn, &_0pPlacementList, (void *)pPlacementList, sizeof(*pPlacementList));
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
    conn->prepare_request(RPC_nvmlDeviceGetVgpuTypeSupportedPlacements);
    conn->write(&device, sizeof(device));
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    conn->write(&_0pPlacementList, sizeof(_0pPlacementList));
    updateTmpPtr((void *)pPlacementList, _0pPlacementList);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pPlacementList, sizeof(*pPlacementList), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetVgpuTypeCreatablePlacements(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuPlacementList_t *pPlacementList) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuTypeCreatablePlacements called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pPlacementList;
    mem2server(conn, &_0pPlacementList, (void *)pPlacementList, sizeof(*pPlacementList));
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
    conn->prepare_request(RPC_nvmlDeviceGetVgpuTypeCreatablePlacements);
    conn->write(&device, sizeof(device));
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    conn->write(&_0pPlacementList, sizeof(_0pPlacementList));
    updateTmpPtr((void *)pPlacementList, _0pPlacementList);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pPlacementList, sizeof(*pPlacementList), true);
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

extern "C" nvmlReturn_t nvmlVgpuTypeGetGspHeapSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *gspHeapSize) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetGspHeapSize called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0gspHeapSize;
    mem2server(conn, &_0gspHeapSize, (void *)gspHeapSize, sizeof(*gspHeapSize));
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
    conn->prepare_request(RPC_nvmlVgpuTypeGetGspHeapSize);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    conn->write(&_0gspHeapSize, sizeof(_0gspHeapSize));
    updateTmpPtr((void *)gspHeapSize, _0gspHeapSize);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)gspHeapSize, sizeof(*gspHeapSize), true);
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

extern "C" nvmlReturn_t nvmlVgpuTypeGetFbReservation(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *fbReservation) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetFbReservation called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0fbReservation;
    mem2server(conn, &_0fbReservation, (void *)fbReservation, sizeof(*fbReservation));
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
    conn->prepare_request(RPC_nvmlVgpuTypeGetFbReservation);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    conn->write(&_0fbReservation, sizeof(_0fbReservation));
    updateTmpPtr((void *)fbReservation, _0fbReservation);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)fbReservation, sizeof(*fbReservation), true);
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

extern "C" nvmlReturn_t nvmlVgpuInstanceGetRuntimeStateSize(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuRuntimeState_t *pState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetRuntimeStateSize called" << std::endl;
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
    conn->prepare_request(RPC_nvmlVgpuInstanceGetRuntimeStateSize);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
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

extern "C" nvmlReturn_t nvmlDeviceSetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, nvmlEnableState_t state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetVgpuCapabilities called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceSetVgpuCapabilities);
    conn->write(&device, sizeof(device));
    conn->write(&capability, sizeof(capability));
    conn->write(&state, sizeof(state));
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

extern "C" nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGridLicensableFeatures_v4 called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetGridLicensableFeatures_v4);
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

extern "C" nvmlReturn_t nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t capability, unsigned int *capResult) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGetVgpuDriverCapabilities called" << std::endl;
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
    conn->prepare_request(RPC_nvmlGetVgpuDriverCapabilities);
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

extern "C" nvmlReturn_t nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, unsigned int *capResult) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuCapabilities called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetVgpuCapabilities);
    conn->write(&device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlVgpuTypeGetBAR1Info(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuTypeBar1Info_t *bar1Info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetBAR1Info called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0bar1Info;
    mem2server(conn, &_0bar1Info, (void *)bar1Info, sizeof(*bar1Info));
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
    conn->prepare_request(RPC_nvmlVgpuTypeGetBAR1Info);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
    conn->write(&_0bar1Info, sizeof(_0bar1Info));
    updateTmpPtr((void *)bar1Info, _0bar1Info);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)bar1Info, sizeof(*bar1Info), true);
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

extern "C" nvmlReturn_t nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char *vgpuPciId, unsigned int *length) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetGpuPciId called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0length;
    mem2server(conn, &_0length, (void *)length, sizeof(*length));
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
    conn->prepare_request(RPC_nvmlVgpuInstanceGetGpuPciId);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    if(*length > 0) {
        conn->read(vgpuPciId, *length, true);
    }
    conn->write(&_0length, sizeof(_0length));
    updateTmpPtr((void *)length, _0length);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)length, sizeof(*length), true);
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

extern "C" nvmlReturn_t nvmlVgpuTypeGetCapabilities(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int *capResult) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetCapabilities called" << std::endl;
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
    conn->prepare_request(RPC_nvmlVgpuTypeGetCapabilities);
    conn->write(&vgpuTypeId, sizeof(vgpuTypeId));
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

extern "C" nvmlReturn_t nvmlGpuInstanceGetCreatableVgpus(nvmlGpuInstance_t gpuInstance, nvmlVgpuTypeIdInfo_t *pVgpus) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetCreatableVgpus called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pVgpus;
    mem2server(conn, &_0pVgpus, (void *)pVgpus, sizeof(*pVgpus));
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
    conn->prepare_request(RPC_nvmlGpuInstanceGetCreatableVgpus);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&_0pVgpus, sizeof(_0pVgpus));
    updateTmpPtr((void *)pVgpus, _0pVgpus);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pVgpus, sizeof(*pVgpus), true);
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

extern "C" nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerGpuInstance(nvmlVgpuTypeMaxInstance_t *pMaxInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuTypeGetMaxInstancesPerGpuInstance called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pMaxInstance;
    mem2server(conn, &_0pMaxInstance, (void *)pMaxInstance, sizeof(*pMaxInstance));
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
    conn->prepare_request(RPC_nvmlVgpuTypeGetMaxInstancesPerGpuInstance);
    conn->write(&_0pMaxInstance, sizeof(_0pMaxInstance));
    updateTmpPtr((void *)pMaxInstance, _0pMaxInstance);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pMaxInstance, sizeof(*pMaxInstance), true);
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

extern "C" nvmlReturn_t nvmlGpuInstanceGetActiveVgpus(nvmlGpuInstance_t gpuInstance, nvmlActiveVgpuInstanceInfo_t *pVgpuInstanceInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetActiveVgpus called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pVgpuInstanceInfo;
    mem2server(conn, &_0pVgpuInstanceInfo, (void *)pVgpuInstanceInfo, sizeof(*pVgpuInstanceInfo));
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
    conn->prepare_request(RPC_nvmlGpuInstanceGetActiveVgpus);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&_0pVgpuInstanceInfo, sizeof(_0pVgpuInstanceInfo));
    updateTmpPtr((void *)pVgpuInstanceInfo, _0pVgpuInstanceInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pVgpuInstanceInfo, sizeof(*pVgpuInstanceInfo), true);
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

extern "C" nvmlReturn_t nvmlGpuInstanceSetVgpuSchedulerState(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerState_t *pScheduler) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceSetVgpuSchedulerState called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pScheduler;
    mem2server(conn, &_0pScheduler, (void *)pScheduler, sizeof(*pScheduler));
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
    conn->prepare_request(RPC_nvmlGpuInstanceSetVgpuSchedulerState);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&_0pScheduler, sizeof(_0pScheduler));
    updateTmpPtr((void *)pScheduler, _0pScheduler);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pScheduler, sizeof(*pScheduler), true);
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

extern "C" nvmlReturn_t nvmlGpuInstanceGetVgpuSchedulerState(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerStateInfo_t *pSchedulerStateInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetVgpuSchedulerState called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pSchedulerStateInfo;
    mem2server(conn, &_0pSchedulerStateInfo, (void *)pSchedulerStateInfo, sizeof(*pSchedulerStateInfo));
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
    conn->prepare_request(RPC_nvmlGpuInstanceGetVgpuSchedulerState);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&_0pSchedulerStateInfo, sizeof(_0pSchedulerStateInfo));
    updateTmpPtr((void *)pSchedulerStateInfo, _0pSchedulerStateInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pSchedulerStateInfo, sizeof(*pSchedulerStateInfo), true);
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

extern "C" nvmlReturn_t nvmlGpuInstanceGetVgpuSchedulerLog(nvmlGpuInstance_t gpuInstance, nvmlVgpuSchedulerLogInfo_t *pSchedulerLogInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetVgpuSchedulerLog called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pSchedulerLogInfo;
    mem2server(conn, &_0pSchedulerLogInfo, (void *)pSchedulerLogInfo, sizeof(*pSchedulerLogInfo));
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
    conn->prepare_request(RPC_nvmlGpuInstanceGetVgpuSchedulerLog);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&_0pSchedulerLogInfo, sizeof(_0pSchedulerLogInfo));
    updateTmpPtr((void *)pSchedulerLogInfo, _0pSchedulerLogInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pSchedulerLogInfo, sizeof(*pSchedulerLogInfo), true);
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

extern "C" nvmlReturn_t nvmlGpuInstanceGetVgpuTypeCreatablePlacements(nvmlGpuInstance_t gpuInstance, nvmlVgpuCreatablePlacementInfo_t *pCreatablePlacementInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetVgpuTypeCreatablePlacements called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pCreatablePlacementInfo;
    mem2server(conn, &_0pCreatablePlacementInfo, (void *)pCreatablePlacementInfo, sizeof(*pCreatablePlacementInfo));
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
    conn->prepare_request(RPC_nvmlGpuInstanceGetVgpuTypeCreatablePlacements);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&_0pCreatablePlacementInfo, sizeof(_0pCreatablePlacementInfo));
    updateTmpPtr((void *)pCreatablePlacementInfo, _0pCreatablePlacementInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pCreatablePlacementInfo, sizeof(*pCreatablePlacementInfo), true);
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

extern "C" nvmlReturn_t nvmlGpuInstanceGetVgpuHeterogeneousMode(nvmlGpuInstance_t gpuInstance, nvmlVgpuHeterogeneousMode_t *pHeterogeneousMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetVgpuHeterogeneousMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pHeterogeneousMode;
    mem2server(conn, &_0pHeterogeneousMode, (void *)pHeterogeneousMode, sizeof(*pHeterogeneousMode));
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
    conn->prepare_request(RPC_nvmlGpuInstanceGetVgpuHeterogeneousMode);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&_0pHeterogeneousMode, sizeof(_0pHeterogeneousMode));
    updateTmpPtr((void *)pHeterogeneousMode, _0pHeterogeneousMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pHeterogeneousMode, sizeof(*pHeterogeneousMode), true);
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

extern "C" nvmlReturn_t nvmlGpuInstanceSetVgpuHeterogeneousMode(nvmlGpuInstance_t gpuInstance, const nvmlVgpuHeterogeneousMode_t *pHeterogeneousMode) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceSetVgpuHeterogeneousMode called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pHeterogeneousMode;
    mem2server(conn, &_0pHeterogeneousMode, (void *)pHeterogeneousMode, sizeof(*pHeterogeneousMode));
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
    conn->prepare_request(RPC_nvmlGpuInstanceSetVgpuHeterogeneousMode);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&_0pHeterogeneousMode, sizeof(_0pHeterogeneousMode));
    updateTmpPtr((void *)pHeterogeneousMode, _0pHeterogeneousMode);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pHeterogeneousMode, sizeof(*pHeterogeneousMode), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetVgpuSchedulerLog(nvmlDevice_t device, nvmlVgpuSchedulerLog_t *pSchedulerLog) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuSchedulerLog called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pSchedulerLog;
    mem2server(conn, &_0pSchedulerLog, (void *)pSchedulerLog, sizeof(*pSchedulerLog));
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
    conn->prepare_request(RPC_nvmlDeviceGetVgpuSchedulerLog);
    conn->write(&device, sizeof(device));
    conn->write(&_0pSchedulerLog, sizeof(_0pSchedulerLog));
    updateTmpPtr((void *)pSchedulerLog, _0pSchedulerLog);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pSchedulerLog, sizeof(*pSchedulerLog), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerGetState_t *pSchedulerState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuSchedulerState called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pSchedulerState;
    mem2server(conn, &_0pSchedulerState, (void *)pSchedulerState, sizeof(*pSchedulerState));
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
    conn->prepare_request(RPC_nvmlDeviceGetVgpuSchedulerState);
    conn->write(&device, sizeof(device));
    conn->write(&_0pSchedulerState, sizeof(_0pSchedulerState));
    updateTmpPtr((void *)pSchedulerState, _0pSchedulerState);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pSchedulerState, sizeof(*pSchedulerState), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetVgpuSchedulerCapabilities(nvmlDevice_t device, nvmlVgpuSchedulerCapabilities_t *pCapabilities) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuSchedulerCapabilities called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pCapabilities;
    mem2server(conn, &_0pCapabilities, (void *)pCapabilities, sizeof(*pCapabilities));
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
    conn->prepare_request(RPC_nvmlDeviceGetVgpuSchedulerCapabilities);
    conn->write(&device, sizeof(device));
    conn->write(&_0pCapabilities, sizeof(_0pCapabilities));
    updateTmpPtr((void *)pCapabilities, _0pCapabilities);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pCapabilities, sizeof(*pCapabilities), true);
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

extern "C" nvmlReturn_t nvmlDeviceSetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerSetState_t *pSchedulerState) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceSetVgpuSchedulerState called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0pSchedulerState;
    mem2server(conn, &_0pSchedulerState, (void *)pSchedulerState, sizeof(*pSchedulerState));
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
    conn->prepare_request(RPC_nvmlDeviceSetVgpuSchedulerState);
    conn->write(&device, sizeof(device));
    conn->write(&_0pSchedulerState, sizeof(_0pSchedulerState));
    updateTmpPtr((void *)pSchedulerState, _0pSchedulerState);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)pSchedulerState, sizeof(*pSchedulerState), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetVgpuInstancesUtilizationInfo(nvmlDevice_t device, nvmlVgpuInstancesUtilizationInfo_t *vgpuUtilInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuInstancesUtilizationInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0vgpuUtilInfo;
    mem2server(conn, &_0vgpuUtilInfo, (void *)vgpuUtilInfo, sizeof(*vgpuUtilInfo));
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
    conn->prepare_request(RPC_nvmlDeviceGetVgpuInstancesUtilizationInfo);
    conn->write(&device, sizeof(device));
    conn->write(&_0vgpuUtilInfo, sizeof(_0vgpuUtilInfo));
    updateTmpPtr((void *)vgpuUtilInfo, _0vgpuUtilInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)vgpuUtilInfo, sizeof(*vgpuUtilInfo), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetVgpuProcessesUtilizationInfo(nvmlDevice_t device, nvmlVgpuProcessesUtilizationInfo_t *vgpuProcUtilInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetVgpuProcessesUtilizationInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0vgpuProcUtilInfo;
    mem2server(conn, &_0vgpuProcUtilInfo, (void *)vgpuProcUtilInfo, sizeof(*vgpuProcUtilInfo));
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
    conn->prepare_request(RPC_nvmlDeviceGetVgpuProcessesUtilizationInfo);
    conn->write(&device, sizeof(device));
    conn->write(&_0vgpuProcUtilInfo, sizeof(_0vgpuProcUtilInfo));
    updateTmpPtr((void *)vgpuProcUtilInfo, _0vgpuProcUtilInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)vgpuProcUtilInfo, sizeof(*vgpuProcUtilInfo), true);
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

extern "C" nvmlReturn_t nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t *licenseInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlVgpuInstanceGetLicenseInfo_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0licenseInfo;
    mem2server(conn, &_0licenseInfo, (void *)licenseInfo, sizeof(*licenseInfo));
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
    conn->prepare_request(RPC_nvmlVgpuInstanceGetLicenseInfo_v2);
    conn->write(&vgpuInstance, sizeof(vgpuInstance));
    conn->write(&_0licenseInfo, sizeof(_0licenseInfo));
    updateTmpPtr((void *)licenseInfo, _0licenseInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)licenseInfo, sizeof(*licenseInfo), true);
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

extern "C" nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetGpuInstanceProfileInfoV called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDeviceGetGpuInstanceProfileInfoV);
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

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t *info) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstanceProfileInfoV called" << std::endl;
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
    conn->prepare_request(RPC_nvmlGpuInstanceGetComputeInstanceProfileInfoV);
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

extern "C" nvmlReturn_t nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstancePlacement_t *placements, unsigned int *count) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceGetComputeInstancePossiblePlacements called" << std::endl;
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
    conn->prepare_request(RPC_nvmlGpuInstanceGetComputeInstancePossiblePlacements);
    conn->write(&gpuInstance, sizeof(gpuInstance));
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

extern "C" nvmlReturn_t nvmlGpuInstanceCreateComputeInstanceWithPlacement(nvmlGpuInstance_t gpuInstance, unsigned int profileId, const nvmlComputeInstancePlacement_t *placement, nvmlComputeInstance_t *computeInstance) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpuInstanceCreateComputeInstanceWithPlacement called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0placement;
    mem2server(conn, &_0placement, (void *)placement, sizeof(*placement));
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
    conn->prepare_request(RPC_nvmlGpuInstanceCreateComputeInstanceWithPlacement);
    conn->write(&gpuInstance, sizeof(gpuInstance));
    conn->write(&profileId, sizeof(profileId));
    conn->write(&_0placement, sizeof(_0placement));
    updateTmpPtr((void *)placement, _0placement);
    conn->write(&_0computeInstance, sizeof(_0computeInstance));
    updateTmpPtr((void *)computeInstance, _0computeInstance);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)placement, sizeof(*placement), true);
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

extern "C" nvmlReturn_t nvmlGpmMetricsGet(nvmlGpmMetricsGet_t *metricsGet) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmMetricsGet called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0metricsGet;
    mem2server(conn, &_0metricsGet, (void *)metricsGet, sizeof(*metricsGet));
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
    conn->prepare_request(RPC_nvmlGpmMetricsGet);
    conn->write(&_0metricsGet, sizeof(_0metricsGet));
    updateTmpPtr((void *)metricsGet, _0metricsGet);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)metricsGet, sizeof(*metricsGet), true);
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

extern "C" nvmlReturn_t nvmlGpmSampleFree(nvmlGpmSample_t gpmSample) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmSampleFree called" << std::endl;
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
    conn->prepare_request(RPC_nvmlGpmSampleFree);
    conn->write(&gpmSample, sizeof(gpmSample));
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

extern "C" nvmlReturn_t nvmlGpmSampleAlloc(nvmlGpmSample_t *gpmSample) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmSampleAlloc called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0gpmSample;
    mem2server(conn, &_0gpmSample, (void *)gpmSample, sizeof(*gpmSample));
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
    conn->prepare_request(RPC_nvmlGpmSampleAlloc);
    conn->write(&_0gpmSample, sizeof(_0gpmSample));
    updateTmpPtr((void *)gpmSample, _0gpmSample);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)gpmSample, sizeof(*gpmSample), true);
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

extern "C" nvmlReturn_t nvmlGpmSampleGet(nvmlDevice_t device, nvmlGpmSample_t gpmSample) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmSampleGet called" << std::endl;
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
    conn->prepare_request(RPC_nvmlGpmSampleGet);
    conn->write(&device, sizeof(device));
    conn->write(&gpmSample, sizeof(gpmSample));
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

extern "C" nvmlReturn_t nvmlGpmMigSampleGet(nvmlDevice_t device, unsigned int gpuInstanceId, nvmlGpmSample_t gpmSample) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmMigSampleGet called" << std::endl;
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
    conn->prepare_request(RPC_nvmlGpmMigSampleGet);
    conn->write(&device, sizeof(device));
    conn->write(&gpuInstanceId, sizeof(gpuInstanceId));
    conn->write(&gpmSample, sizeof(gpmSample));
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

extern "C" nvmlReturn_t nvmlGpmQueryDeviceSupport(nvmlDevice_t device, nvmlGpmSupport_t *gpmSupport) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmQueryDeviceSupport called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0gpmSupport;
    mem2server(conn, &_0gpmSupport, (void *)gpmSupport, sizeof(*gpmSupport));
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
    conn->prepare_request(RPC_nvmlGpmQueryDeviceSupport);
    conn->write(&device, sizeof(device));
    conn->write(&_0gpmSupport, sizeof(_0gpmSupport));
    updateTmpPtr((void *)gpmSupport, _0gpmSupport);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)gpmSupport, sizeof(*gpmSupport), true);
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

extern "C" nvmlReturn_t nvmlGpmQueryIfStreamingEnabled(nvmlDevice_t device, unsigned int *state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmQueryIfStreamingEnabled called" << std::endl;
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
    conn->prepare_request(RPC_nvmlGpmQueryIfStreamingEnabled);
    conn->write(&device, sizeof(device));
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

extern "C" nvmlReturn_t nvmlGpmSetStreamingEnabled(nvmlDevice_t device, unsigned int state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlGpmSetStreamingEnabled called" << std::endl;
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
    conn->prepare_request(RPC_nvmlGpmSetStreamingEnabled);
    conn->write(&device, sizeof(device));
    conn->write(&state, sizeof(state));
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

extern "C" nvmlReturn_t nvmlDeviceGetCapabilities(nvmlDevice_t device, nvmlDeviceCapabilities_t *caps) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceGetCapabilities called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0caps;
    mem2server(conn, &_0caps, (void *)caps, sizeof(*caps));
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
    conn->prepare_request(RPC_nvmlDeviceGetCapabilities);
    conn->write(&device, sizeof(device));
    conn->write(&_0caps, sizeof(_0caps));
    updateTmpPtr((void *)caps, _0caps);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)caps, sizeof(*caps), true);
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

extern "C" nvmlReturn_t nvmlDeviceWorkloadPowerProfileGetProfilesInfo(nvmlDevice_t device, nvmlWorkloadPowerProfileProfilesInfo_t *profilesInfo) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceWorkloadPowerProfileGetProfilesInfo called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0profilesInfo;
    mem2server(conn, &_0profilesInfo, (void *)profilesInfo, sizeof(*profilesInfo));
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
    conn->prepare_request(RPC_nvmlDeviceWorkloadPowerProfileGetProfilesInfo);
    conn->write(&device, sizeof(device));
    conn->write(&_0profilesInfo, sizeof(_0profilesInfo));
    updateTmpPtr((void *)profilesInfo, _0profilesInfo);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)profilesInfo, sizeof(*profilesInfo), true);
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

extern "C" nvmlReturn_t nvmlDeviceWorkloadPowerProfileGetCurrentProfiles(nvmlDevice_t device, nvmlWorkloadPowerProfileCurrentProfiles_t *currentProfiles) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceWorkloadPowerProfileGetCurrentProfiles called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0currentProfiles;
    mem2server(conn, &_0currentProfiles, (void *)currentProfiles, sizeof(*currentProfiles));
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
    conn->prepare_request(RPC_nvmlDeviceWorkloadPowerProfileGetCurrentProfiles);
    conn->write(&device, sizeof(device));
    conn->write(&_0currentProfiles, sizeof(_0currentProfiles));
    updateTmpPtr((void *)currentProfiles, _0currentProfiles);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)currentProfiles, sizeof(*currentProfiles), true);
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

extern "C" nvmlReturn_t nvmlDeviceWorkloadPowerProfileSetRequestedProfiles(nvmlDevice_t device, nvmlWorkloadPowerProfileRequestedProfiles_t *requestedProfiles) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceWorkloadPowerProfileSetRequestedProfiles called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0requestedProfiles;
    mem2server(conn, &_0requestedProfiles, (void *)requestedProfiles, sizeof(*requestedProfiles));
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
    conn->prepare_request(RPC_nvmlDeviceWorkloadPowerProfileSetRequestedProfiles);
    conn->write(&device, sizeof(device));
    conn->write(&_0requestedProfiles, sizeof(_0requestedProfiles));
    updateTmpPtr((void *)requestedProfiles, _0requestedProfiles);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)requestedProfiles, sizeof(*requestedProfiles), true);
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

extern "C" nvmlReturn_t nvmlDeviceWorkloadPowerProfileClearRequestedProfiles(nvmlDevice_t device, nvmlWorkloadPowerProfileRequestedProfiles_t *requestedProfiles) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDeviceWorkloadPowerProfileClearRequestedProfiles called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0requestedProfiles;
    mem2server(conn, &_0requestedProfiles, (void *)requestedProfiles, sizeof(*requestedProfiles));
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
    conn->prepare_request(RPC_nvmlDeviceWorkloadPowerProfileClearRequestedProfiles);
    conn->write(&device, sizeof(device));
    conn->write(&_0requestedProfiles, sizeof(_0requestedProfiles));
    updateTmpPtr((void *)requestedProfiles, _0requestedProfiles);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)requestedProfiles, sizeof(*requestedProfiles), true);
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

extern "C" nvmlReturn_t nvmlDevicePowerSmoothingActivatePresetProfile(nvmlDevice_t device, nvmlPowerSmoothingProfile_t *profile) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDevicePowerSmoothingActivatePresetProfile called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0profile;
    mem2server(conn, &_0profile, (void *)profile, sizeof(*profile));
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
    conn->prepare_request(RPC_nvmlDevicePowerSmoothingActivatePresetProfile);
    conn->write(&device, sizeof(device));
    conn->write(&_0profile, sizeof(_0profile));
    updateTmpPtr((void *)profile, _0profile);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)profile, sizeof(*profile), true);
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

extern "C" nvmlReturn_t nvmlDevicePowerSmoothingUpdatePresetProfileParam(nvmlDevice_t device, nvmlPowerSmoothingProfile_t *profile) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDevicePowerSmoothingUpdatePresetProfileParam called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0profile;
    mem2server(conn, &_0profile, (void *)profile, sizeof(*profile));
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
    conn->prepare_request(RPC_nvmlDevicePowerSmoothingUpdatePresetProfileParam);
    conn->write(&device, sizeof(device));
    conn->write(&_0profile, sizeof(_0profile));
    updateTmpPtr((void *)profile, _0profile);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)profile, sizeof(*profile), true);
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

extern "C" nvmlReturn_t nvmlDevicePowerSmoothingSetState(nvmlDevice_t device, nvmlPowerSmoothingState_t *state) {
#ifdef DEBUG
    std::cout << "Hook: nvmlDevicePowerSmoothingSetState called" << std::endl;
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
    conn->prepare_request(RPC_nvmlDevicePowerSmoothingSetState);
    conn->write(&device, sizeof(device));
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
