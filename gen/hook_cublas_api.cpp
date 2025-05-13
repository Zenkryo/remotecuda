#include <iostream>
#include <map>
#include "cublas_api.h"

#include "hook_api.h"
#include "client.h"
int sizeofType(cudaDataType type);
extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
#ifdef DEBUG
    std::cout << "Hook: cublasCreate_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0handle;
    mem2server(conn, &_0handle, (void *)handle, sizeof(*handle));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCreate_v2);
    conn->write(&_0handle, sizeof(_0handle));
    updateTmpPtr((void *)handle, _0handle);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)handle, sizeof(*handle), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
#ifdef DEBUG
    std::cout << "Hook: cublasDestroy_v2 called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDestroy_v2);
    conn->write(&handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetVersion_v2 called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasGetVersion_v2);
    conn->write(&handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetProperty called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasGetProperty);
    conn->write(&type, sizeof(type));
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

extern "C" size_t cublasGetCudartVersion() {
#ifdef DEBUG
    std::cout << "Hook: cublasGetCudartVersion called" << std::endl;
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
    size_t _result;
    conn->prepare_request(RPC_cublasGetCudartVersion);
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

extern "C" cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetWorkspace_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0workspace;
    mem2server(conn, &_0workspace, (void *)workspace, workspaceSizeInBytes);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSetWorkspace_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&_0workspace, sizeof(_0workspace));
    updateTmpPtr((void *)workspace, _0workspace);
    conn->write(&workspaceSizeInBytes, sizeof(workspaceSizeInBytes));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)workspace, workspaceSizeInBytes, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetStream_v2 called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSetStream_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&streamId, sizeof(streamId));
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

extern "C" cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *streamId) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetStream_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0streamId;
    mem2server(conn, &_0streamId, (void *)streamId, sizeof(*streamId));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasGetStream_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&_0streamId, sizeof(_0streamId));
    updateTmpPtr((void *)streamId, _0streamId);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)streamId, sizeof(*streamId), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetPointerMode_v2 called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasGetPointerMode_v2);
    conn->write(&handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetPointerMode_v2 called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSetPointerMode_v2);
    conn->write(&handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetAtomicsMode called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasGetAtomicsMode);
    conn->write(&handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetAtomicsMode called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSetAtomicsMode);
    conn->write(&handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetMathMode called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasGetMathMode);
    conn->write(&handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetMathMode called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSetMathMode);
    conn->write(&handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasGetSmCountTarget(cublasHandle_t handle, int *smCountTarget) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetSmCountTarget called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0smCountTarget;
    mem2server(conn, &_0smCountTarget, (void *)smCountTarget, sizeof(*smCountTarget));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasGetSmCountTarget);
    conn->write(&handle, sizeof(handle));
    conn->write(&_0smCountTarget, sizeof(_0smCountTarget));
    updateTmpPtr((void *)smCountTarget, _0smCountTarget);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)smCountTarget, sizeof(*smCountTarget), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetSmCountTarget called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSetSmCountTarget);
    conn->write(&handle, sizeof(handle));
    conn->write(&smCountTarget, sizeof(smCountTarget));
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

extern "C" cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char *logFileName) {
#ifdef DEBUG
    std::cout << "Hook: cublasLoggerConfigure called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasLoggerConfigure);
    conn->write(&logIsOn, sizeof(logIsOn));
    conn->write(&logToStdOut, sizeof(logToStdOut));
    conn->write(&logToStdErr, sizeof(logToStdErr));
    conn->write(logFileName, strlen(logFileName) + 1, true);
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

extern "C" cublasStatus_t cublasSetLoggerCallback(cublasLogCallback userCallback) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetLoggerCallback called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSetLoggerCallback);
    conn->write(&userCallback, sizeof(userCallback));
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

extern "C" cublasStatus_t cublasGetLoggerCallback(cublasLogCallback *userCallback) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetLoggerCallback called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0userCallback;
    mem2server(conn, &_0userCallback, (void *)userCallback, sizeof(*userCallback));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasGetLoggerCallback);
    conn->write(&_0userCallback, sizeof(_0userCallback));
    updateTmpPtr((void *)userCallback, _0userCallback);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)userCallback, sizeof(*userCallback), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx, void *devicePtr, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetVector called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * elemSize);
    void *_0devicePtr;
    mem2server(conn, &_0devicePtr, (void *)devicePtr, n * elemSize);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSetVector);
    conn->write(&n, sizeof(n));
    conn->write(&elemSize, sizeof(elemSize));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0devicePtr, sizeof(_0devicePtr));
    updateTmpPtr((void *)devicePtr, _0devicePtr);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * elemSize, true);
    mem2client(conn, (void *)devicePtr, n * elemSize, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetVector called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * elemSize);
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * elemSize);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasGetVector);
    conn->write(&n, sizeof(n));
    conn->write(&elemSize, sizeof(elemSize));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * elemSize, true);
    mem2client(conn, (void *)y, n * elemSize, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetMatrix called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, rows * cols * elemSize);
    void *_0B;
    mem2server(conn, &_0B, (void *)B, rows * cols * elemSize);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSetMatrix);
    conn->write(&rows, sizeof(rows));
    conn->write(&cols, sizeof(cols));
    conn->write(&elemSize, sizeof(elemSize));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, rows * cols * elemSize, true);
    mem2client(conn, (void *)B, rows * cols * elemSize, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetMatrix called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, rows * cols * elemSize);
    void *_0B;
    mem2server(conn, &_0B, (void *)B, rows * cols * elemSize);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasGetMatrix);
    conn->write(&rows, sizeof(rows));
    conn->write(&cols, sizeof(cols));
    conn->write(&elemSize, sizeof(elemSize));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, rows * cols * elemSize, true);
    mem2client(conn, (void *)B, rows * cols * elemSize, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetVectorAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0hostPtr;
    mem2server(conn, &_0hostPtr, (void *)hostPtr, n * elemSize);
    void *_0devicePtr;
    mem2server(conn, &_0devicePtr, (void *)devicePtr, n * elemSize);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSetVectorAsync);
    conn->write(&n, sizeof(n));
    conn->write(&elemSize, sizeof(elemSize));
    conn->write(&_0hostPtr, sizeof(_0hostPtr));
    updateTmpPtr((void *)hostPtr, _0hostPtr);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0devicePtr, sizeof(_0devicePtr));
    updateTmpPtr((void *)devicePtr, _0devicePtr);
    conn->write(&incy, sizeof(incy));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)hostPtr, n * elemSize, true);
    mem2client(conn, (void *)devicePtr, n * elemSize, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetVectorAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devicePtr;
    mem2server(conn, &_0devicePtr, (void *)devicePtr, n * elemSize);
    void *_0hostPtr;
    mem2server(conn, &_0hostPtr, (void *)hostPtr, n * elemSize);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasGetVectorAsync);
    conn->write(&n, sizeof(n));
    conn->write(&elemSize, sizeof(elemSize));
    conn->write(&_0devicePtr, sizeof(_0devicePtr));
    updateTmpPtr((void *)devicePtr, _0devicePtr);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0hostPtr, sizeof(_0hostPtr));
    updateTmpPtr((void *)hostPtr, _0hostPtr);
    conn->write(&incy, sizeof(incy));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devicePtr, n * elemSize, true);
    mem2client(conn, (void *)hostPtr, n * elemSize, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetMatrixAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, rows * cols * elemSize);
    void *_0B;
    mem2server(conn, &_0B, (void *)B, rows * cols * elemSize);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSetMatrixAsync);
    conn->write(&rows, sizeof(rows));
    conn->write(&cols, sizeof(cols));
    conn->write(&elemSize, sizeof(elemSize));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, rows * cols * elemSize, true);
    mem2client(conn, (void *)B, rows * cols * elemSize, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetMatrixAsync called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, rows * cols * elemSize);
    void *_0B;
    mem2server(conn, &_0B, (void *)B, rows * cols * elemSize);
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasGetMatrixAsync);
    conn->write(&rows, sizeof(rows));
    conn->write(&cols, sizeof(cols));
    conn->write(&elemSize, sizeof(elemSize));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, rows * cols * elemSize, true);
    mem2client(conn, (void *)B, rows * cols * elemSize, true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" void cublasXerbla(const char *srName, int info) {
#ifdef DEBUG
    std::cout << "Hook: cublasXerbla called" << std::endl;
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
    conn->prepare_request(RPC_cublasXerbla);
    conn->write(srName, strlen(srName) + 1, true);
    conn->write(&info, sizeof(info));
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
    return;
}

extern "C" cublasStatus_t cublasNrm2Ex(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, void *result, cudaDataType resultType, cudaDataType executionType) {
#ifdef DEBUG
    std::cout << "Hook: cublasNrm2Ex called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeofType(xType));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeofType(resultType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasNrm2Ex);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&xType, sizeof(xType));
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->write(&resultType, sizeof(resultType));
    conn->write(&executionType, sizeof(executionType));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeofType(xType), true);
    mem2client(conn, (void *)result, sizeofType(resultType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasSnrm2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSnrm2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDnrm2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDnrm2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasScnrm2_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasScnrm2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasScnrm2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDznrm2_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDznrm2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDznrm2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, const void *y, cudaDataType yType, int incy, void *result, cudaDataType resultType, cudaDataType executionType) {
#ifdef DEBUG
    std::cout << "Hook: cublasDotEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeofType(xType));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeofType(yType));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeofType(resultType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDotEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&xType, sizeof(xType));
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&yType, sizeof(yType));
    conn->write(&incy, sizeof(incy));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->write(&resultType, sizeof(resultType));
    conn->write(&executionType, sizeof(executionType));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeofType(xType), true);
    mem2client(conn, (void *)y, n * sizeofType(yType), true);
    mem2client(conn, (void *)result, sizeofType(resultType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDotcEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, const void *y, cudaDataType yType, int incy, void *result, cudaDataType resultType, cudaDataType executionType) {
#ifdef DEBUG
    std::cout << "Hook: cublasDotcEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeofType(xType));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeofType(yType));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeofType(resultType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDotcEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&xType, sizeof(xType));
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&yType, sizeof(yType));
    conn->write(&incy, sizeof(incy));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->write(&resultType, sizeof(resultType));
    conn->write(&executionType, sizeof(executionType));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeofType(xType), true);
    mem2client(conn, (void *)y, n * sizeofType(yType), true);
    mem2client(conn, (void *)result, sizeofType(resultType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasSdot_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSdot_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDdot_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDdot_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCdotu_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasCdotu_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCdotu_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasCdotc_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCdotc_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdotu_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZdotu_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdotc_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZdotc_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasScalEx(cublasHandle_t handle, int n, const void *alpha, cudaDataType alphaType, void *x, cudaDataType xType, int incx, cudaDataType executionType) {
#ifdef DEBUG
    std::cout << "Hook: cublasScalEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeofType(alphaType));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeofType(xType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasScalEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&alphaType, sizeof(alphaType));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&xType, sizeof(xType));
    conn->write(&incx, sizeof(incx));
    conn->write(&executionType, sizeof(executionType));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeofType(alphaType), true);
    mem2client(conn, (void *)x, n * sizeofType(xType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasSscal_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSscal_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDscal_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDscal_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCscal_v2(cublasHandle_t handle, int n, const cuComplex *alpha, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCscal_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCscal_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCsscal_v2(cublasHandle_t handle, int n, const float *alpha, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsscal_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCsscal_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZscal_v2(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZscal_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZscal_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZdscal_v2(cublasHandle_t handle, int n, const double *alpha, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdscal_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZdscal_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasAxpyEx(cublasHandle_t handle, int n, const void *alpha, cudaDataType alphaType, const void *x, cudaDataType xType, int incx, void *y, cudaDataType yType, int incy, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasAxpyEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeofType(alphaType));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeofType(xType));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeofType(yType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasAxpyEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&alphaType, sizeof(alphaType));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&xType, sizeof(xType));
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&yType, sizeof(yType));
    conn->write(&incy, sizeof(incy));
    conn->write(&executiontype, sizeof(executiontype));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeofType(alphaType), true);
    mem2client(conn, (void *)x, n * sizeofType(xType), true);
    mem2client(conn, (void *)y, n * sizeofType(yType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSaxpy_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSaxpy_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDaxpy_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDaxpy_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCaxpy_v2(cublasHandle_t handle, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCaxpy_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCaxpy_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZaxpy_v2(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZaxpy_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZaxpy_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCopyEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, void *y, cudaDataType yType, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCopyEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeofType(xType));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeofType(yType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCopyEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&xType, sizeof(xType));
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&yType, sizeof(yType));
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeofType(xType), true);
    mem2client(conn, (void *)y, n * sizeofType(yType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasScopy_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasScopy_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDcopy_v2(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDcopy_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDcopy_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCcopy_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCcopy_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZcopy_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZcopy_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSswap_v2(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSswap_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSswap_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDswap_v2(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDswap_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDswap_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCswap_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCswap_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCswap_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZswap_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZswap_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSwapEx(cublasHandle_t handle, int n, void *x, cudaDataType xType, int incx, void *y, cudaDataType yType, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSwapEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeofType(xType));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeofType(yType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSwapEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&xType, sizeof(xType));
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&yType, sizeof(yType));
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeofType(xType), true);
    mem2client(conn, (void *)y, n * sizeofType(yType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasIsamax_v2(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIsamax_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasIsamax_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIdamax_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasIdamax_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasIcamax_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIcamax_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasIcamax_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasIzamax_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIzamax_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasIzamax_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasIamaxEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIamaxEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeofType(xType));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasIamaxEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&xType, sizeof(xType));
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeofType(xType), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasIsamin_v2(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIsamin_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasIsamin_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasIdamin_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIdamin_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasIdamin_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasIcamin_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIcamin_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasIcamin_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasIzamin_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIzamin_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasIzamin_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasIaminEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIaminEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeofType(xType));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasIaminEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&xType, sizeof(xType));
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeofType(xType), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasAsumEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, void *result, cudaDataType resultType, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasAsumEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeofType(xType));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeofType(resultType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasAsumEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&xType, sizeof(xType));
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->write(&resultType, sizeof(resultType));
    conn->write(&executiontype, sizeof(executiontype));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeofType(xType), true);
    mem2client(conn, (void *)result, sizeofType(resultType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasSasum_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSasum_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDasum_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDasum_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasScasum_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasScasum_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasScasum_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDzasum_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDzasum_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0result;
    mem2server(conn, &_0result, (void *)result, sizeof(*result));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDzasum_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0result, sizeof(_0result));
    updateTmpPtr((void *)result, _0result);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)result, sizeof(*result), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSrot_v2(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *c, const float *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasSrot_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0c;
    mem2server(conn, &_0c, (void *)c, sizeof(*c));
    void *_0s;
    mem2server(conn, &_0s, (void *)s, sizeof(*s));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSrot_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0c, sizeof(_0c));
    updateTmpPtr((void *)c, _0c);
    conn->write(&_0s, sizeof(_0s));
    updateTmpPtr((void *)s, _0s);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)c, sizeof(*c), true);
    mem2client(conn, (void *)s, sizeof(*s), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDrot_v2(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *c, const double *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasDrot_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0c;
    mem2server(conn, &_0c, (void *)c, sizeof(*c));
    void *_0s;
    mem2server(conn, &_0s, (void *)s, sizeof(*s));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDrot_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0c, sizeof(_0c));
    updateTmpPtr((void *)c, _0c);
    conn->write(&_0s, sizeof(_0s));
    updateTmpPtr((void *)s, _0s);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)c, sizeof(*c), true);
    mem2client(conn, (void *)s, sizeof(*s), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCrot_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const cuComplex *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasCrot_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0c;
    mem2server(conn, &_0c, (void *)c, sizeof(*c));
    void *_0s;
    mem2server(conn, &_0s, (void *)s, sizeof(*s));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCrot_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0c, sizeof(_0c));
    updateTmpPtr((void *)c, _0c);
    conn->write(&_0s, sizeof(_0s));
    updateTmpPtr((void *)s, _0s);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)c, sizeof(*c), true);
    mem2client(conn, (void *)s, sizeof(*s), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCsrot_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const float *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsrot_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0c;
    mem2server(conn, &_0c, (void *)c, sizeof(*c));
    void *_0s;
    mem2server(conn, &_0s, (void *)s, sizeof(*s));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCsrot_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0c, sizeof(_0c));
    updateTmpPtr((void *)c, _0c);
    conn->write(&_0s, sizeof(_0s));
    updateTmpPtr((void *)s, _0s);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)c, sizeof(*c), true);
    mem2client(conn, (void *)s, sizeof(*s), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const cuDoubleComplex *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasZrot_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0c;
    mem2server(conn, &_0c, (void *)c, sizeof(*c));
    void *_0s;
    mem2server(conn, &_0s, (void *)s, sizeof(*s));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZrot_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0c, sizeof(_0c));
    updateTmpPtr((void *)c, _0c);
    conn->write(&_0s, sizeof(_0s));
    updateTmpPtr((void *)s, _0s);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)c, sizeof(*c), true);
    mem2client(conn, (void *)s, sizeof(*s), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZdrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const double *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdrot_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0c;
    mem2server(conn, &_0c, (void *)c, sizeof(*c));
    void *_0s;
    mem2server(conn, &_0s, (void *)s, sizeof(*s));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZdrot_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0c, sizeof(_0c));
    updateTmpPtr((void *)c, _0c);
    conn->write(&_0s, sizeof(_0s));
    updateTmpPtr((void *)s, _0s);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)c, sizeof(*c), true);
    mem2client(conn, (void *)s, sizeof(*s), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasRotEx(cublasHandle_t handle, int n, void *x, cudaDataType xType, int incx, void *y, cudaDataType yType, int incy, const void *c, const void *s, cudaDataType csType, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasRotEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeofType(xType));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeofType(yType));
    void *_0c;
    mem2server(conn, &_0c, (void *)c, sizeofType(csType));
    void *_0s;
    mem2server(conn, &_0s, (void *)s, sizeofType(csType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasRotEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&xType, sizeof(xType));
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&yType, sizeof(yType));
    conn->write(&incy, sizeof(incy));
    conn->write(&_0c, sizeof(_0c));
    updateTmpPtr((void *)c, _0c);
    conn->write(&_0s, sizeof(_0s));
    updateTmpPtr((void *)s, _0s);
    conn->write(&csType, sizeof(csType));
    conn->write(&executiontype, sizeof(executiontype));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeofType(xType), true);
    mem2client(conn, (void *)y, n * sizeofType(yType), true);
    mem2client(conn, (void *)c, sizeofType(csType), true);
    mem2client(conn, (void *)s, sizeofType(csType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSrotg_v2(cublasHandle_t handle, float *a, float *b, float *c, float *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasSrotg_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0a;
    mem2server(conn, &_0a, (void *)a, sizeof(*a));
    void *_0b;
    mem2server(conn, &_0b, (void *)b, sizeof(*b));
    void *_0c;
    mem2server(conn, &_0c, (void *)c, sizeof(*c));
    void *_0s;
    mem2server(conn, &_0s, (void *)s, sizeof(*s));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSrotg_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&_0a, sizeof(_0a));
    updateTmpPtr((void *)a, _0a);
    conn->write(&_0b, sizeof(_0b));
    updateTmpPtr((void *)b, _0b);
    conn->write(&_0c, sizeof(_0c));
    updateTmpPtr((void *)c, _0c);
    conn->write(&_0s, sizeof(_0s));
    updateTmpPtr((void *)s, _0s);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)a, sizeof(*a), true);
    mem2client(conn, (void *)b, sizeof(*b), true);
    mem2client(conn, (void *)c, sizeof(*c), true);
    mem2client(conn, (void *)s, sizeof(*s), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDrotg_v2(cublasHandle_t handle, double *a, double *b, double *c, double *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasDrotg_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0a;
    mem2server(conn, &_0a, (void *)a, sizeof(*a));
    void *_0b;
    mem2server(conn, &_0b, (void *)b, sizeof(*b));
    void *_0c;
    mem2server(conn, &_0c, (void *)c, sizeof(*c));
    void *_0s;
    mem2server(conn, &_0s, (void *)s, sizeof(*s));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDrotg_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&_0a, sizeof(_0a));
    updateTmpPtr((void *)a, _0a);
    conn->write(&_0b, sizeof(_0b));
    updateTmpPtr((void *)b, _0b);
    conn->write(&_0c, sizeof(_0c));
    updateTmpPtr((void *)c, _0c);
    conn->write(&_0s, sizeof(_0s));
    updateTmpPtr((void *)s, _0s);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)a, sizeof(*a), true);
    mem2client(conn, (void *)b, sizeof(*b), true);
    mem2client(conn, (void *)c, sizeof(*c), true);
    mem2client(conn, (void *)s, sizeof(*s), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCrotg_v2(cublasHandle_t handle, cuComplex *a, cuComplex *b, float *c, cuComplex *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasCrotg_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0a;
    mem2server(conn, &_0a, (void *)a, sizeof(*a));
    void *_0b;
    mem2server(conn, &_0b, (void *)b, sizeof(*b));
    void *_0c;
    mem2server(conn, &_0c, (void *)c, sizeof(*c));
    void *_0s;
    mem2server(conn, &_0s, (void *)s, sizeof(*s));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCrotg_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&_0a, sizeof(_0a));
    updateTmpPtr((void *)a, _0a);
    conn->write(&_0b, sizeof(_0b));
    updateTmpPtr((void *)b, _0b);
    conn->write(&_0c, sizeof(_0c));
    updateTmpPtr((void *)c, _0c);
    conn->write(&_0s, sizeof(_0s));
    updateTmpPtr((void *)s, _0s);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)a, sizeof(*a), true);
    mem2client(conn, (void *)b, sizeof(*b), true);
    mem2client(conn, (void *)c, sizeof(*c), true);
    mem2client(conn, (void *)s, sizeof(*s), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZrotg_v2(cublasHandle_t handle, cuDoubleComplex *a, cuDoubleComplex *b, double *c, cuDoubleComplex *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasZrotg_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0a;
    mem2server(conn, &_0a, (void *)a, sizeof(*a));
    void *_0b;
    mem2server(conn, &_0b, (void *)b, sizeof(*b));
    void *_0c;
    mem2server(conn, &_0c, (void *)c, sizeof(*c));
    void *_0s;
    mem2server(conn, &_0s, (void *)s, sizeof(*s));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZrotg_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&_0a, sizeof(_0a));
    updateTmpPtr((void *)a, _0a);
    conn->write(&_0b, sizeof(_0b));
    updateTmpPtr((void *)b, _0b);
    conn->write(&_0c, sizeof(_0c));
    updateTmpPtr((void *)c, _0c);
    conn->write(&_0s, sizeof(_0s));
    updateTmpPtr((void *)s, _0s);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)a, sizeof(*a), true);
    mem2client(conn, (void *)b, sizeof(*b), true);
    mem2client(conn, (void *)c, sizeof(*c), true);
    mem2client(conn, (void *)s, sizeof(*s), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasRotgEx(cublasHandle_t handle, void *a, void *b, cudaDataType abType, void *c, void *s, cudaDataType csType, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasRotgEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0a;
    mem2server(conn, &_0a, (void *)a, sizeofType(abType));
    void *_0b;
    mem2server(conn, &_0b, (void *)b, sizeofType(abType));
    void *_0c;
    mem2server(conn, &_0c, (void *)c, sizeofType(csType));
    void *_0s;
    mem2server(conn, &_0s, (void *)s, sizeofType(csType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasRotgEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&_0a, sizeof(_0a));
    updateTmpPtr((void *)a, _0a);
    conn->write(&_0b, sizeof(_0b));
    updateTmpPtr((void *)b, _0b);
    conn->write(&abType, sizeof(abType));
    conn->write(&_0c, sizeof(_0c));
    updateTmpPtr((void *)c, _0c);
    conn->write(&_0s, sizeof(_0s));
    updateTmpPtr((void *)s, _0s);
    conn->write(&csType, sizeof(csType));
    conn->write(&executiontype, sizeof(executiontype));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)a, sizeofType(abType), true);
    mem2client(conn, (void *)b, sizeofType(abType), true);
    mem2client(conn, (void *)c, sizeofType(csType), true);
    mem2client(conn, (void *)s, sizeofType(csType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSrotm_v2(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *param) {
#ifdef DEBUG
    std::cout << "Hook: cublasSrotm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0param;
    mem2server(conn, &_0param, (void *)param, 5 * sizeof(*param));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSrotm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0param, sizeof(_0param));
    updateTmpPtr((void *)param, _0param);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)param, 5 * sizeof(*param), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDrotm_v2(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *param) {
#ifdef DEBUG
    std::cout << "Hook: cublasDrotm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0param;
    mem2server(conn, &_0param, (void *)param, 5 * sizeof(*param));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDrotm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0param, sizeof(_0param));
    updateTmpPtr((void *)param, _0param);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)param, 5 * sizeof(*param), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasRotmEx(cublasHandle_t handle, int n, void *x, cudaDataType xType, int incx, void *y, cudaDataType yType, int incy, const void *param, cudaDataType paramType, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasRotmEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeofType(xType));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeofType(yType));
    void *_0param;
    mem2server(conn, &_0param, (void *)param, 5 * sizeofType(paramType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasRotmEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&xType, sizeof(xType));
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&yType, sizeof(yType));
    conn->write(&incy, sizeof(incy));
    conn->write(&_0param, sizeof(_0param));
    updateTmpPtr((void *)param, _0param);
    conn->write(&paramType, sizeof(paramType));
    conn->write(&executiontype, sizeof(executiontype));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)x, n * sizeofType(xType), true);
    mem2client(conn, (void *)y, n * sizeofType(yType), true);
    mem2client(conn, (void *)param, 5 * sizeofType(paramType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSrotmg_v2(cublasHandle_t handle, float *d1, float *d2, float *x1, const float *y1, float *param) {
#ifdef DEBUG
    std::cout << "Hook: cublasSrotmg_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0d1;
    mem2server(conn, &_0d1, (void *)d1, sizeof(*d1));
    void *_0d2;
    mem2server(conn, &_0d2, (void *)d2, sizeof(*d2));
    void *_0x1;
    mem2server(conn, &_0x1, (void *)x1, sizeof(*x1));
    void *_0y1;
    mem2server(conn, &_0y1, (void *)y1, sizeof(*y1));
    void *_0param;
    mem2server(conn, &_0param, (void *)param, 5 * sizeof(*param));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSrotmg_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&_0d1, sizeof(_0d1));
    updateTmpPtr((void *)d1, _0d1);
    conn->write(&_0d2, sizeof(_0d2));
    updateTmpPtr((void *)d2, _0d2);
    conn->write(&_0x1, sizeof(_0x1));
    updateTmpPtr((void *)x1, _0x1);
    conn->write(&_0y1, sizeof(_0y1));
    updateTmpPtr((void *)y1, _0y1);
    conn->write(&_0param, sizeof(_0param));
    updateTmpPtr((void *)param, _0param);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)d1, sizeof(*d1), true);
    mem2client(conn, (void *)d2, sizeof(*d2), true);
    mem2client(conn, (void *)x1, sizeof(*x1), true);
    mem2client(conn, (void *)y1, sizeof(*y1), true);
    mem2client(conn, (void *)param, 5 * sizeof(*param), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDrotmg_v2(cublasHandle_t handle, double *d1, double *d2, double *x1, const double *y1, double *param) {
#ifdef DEBUG
    std::cout << "Hook: cublasDrotmg_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0d1;
    mem2server(conn, &_0d1, (void *)d1, sizeof(*d1));
    void *_0d2;
    mem2server(conn, &_0d2, (void *)d2, sizeof(*d2));
    void *_0x1;
    mem2server(conn, &_0x1, (void *)x1, sizeof(*x1));
    void *_0y1;
    mem2server(conn, &_0y1, (void *)y1, sizeof(*y1));
    void *_0param;
    mem2server(conn, &_0param, (void *)param, 5 * sizeof(*param));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDrotmg_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&_0d1, sizeof(_0d1));
    updateTmpPtr((void *)d1, _0d1);
    conn->write(&_0d2, sizeof(_0d2));
    updateTmpPtr((void *)d2, _0d2);
    conn->write(&_0x1, sizeof(_0x1));
    updateTmpPtr((void *)x1, _0x1);
    conn->write(&_0y1, sizeof(_0y1));
    updateTmpPtr((void *)y1, _0y1);
    conn->write(&_0param, sizeof(_0param));
    updateTmpPtr((void *)param, _0param);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)d1, sizeof(*d1), true);
    mem2client(conn, (void *)d2, sizeof(*d2), true);
    mem2client(conn, (void *)x1, sizeof(*x1), true);
    mem2client(conn, (void *)y1, sizeof(*y1), true);
    mem2client(conn, (void *)param, 5 * sizeof(*param), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasRotmgEx(cublasHandle_t handle, void *d1, cudaDataType d1Type, void *d2, cudaDataType d2Type, void *x1, cudaDataType x1Type, const void *y1, cudaDataType y1Type, void *param, cudaDataType paramType, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasRotmgEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0d1;
    mem2server(conn, &_0d1, (void *)d1, sizeofType(d1Type));
    void *_0d2;
    mem2server(conn, &_0d2, (void *)d2, sizeofType(d2Type));
    void *_0x1;
    mem2server(conn, &_0x1, (void *)x1, sizeofType(x1Type));
    void *_0y1;
    mem2server(conn, &_0y1, (void *)y1, sizeofType(y1Type));
    void *_0param;
    mem2server(conn, &_0param, (void *)param, 5 * sizeofType(paramType));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasRotmgEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&_0d1, sizeof(_0d1));
    updateTmpPtr((void *)d1, _0d1);
    conn->write(&d1Type, sizeof(d1Type));
    conn->write(&_0d2, sizeof(_0d2));
    updateTmpPtr((void *)d2, _0d2);
    conn->write(&d2Type, sizeof(d2Type));
    conn->write(&_0x1, sizeof(_0x1));
    updateTmpPtr((void *)x1, _0x1);
    conn->write(&x1Type, sizeof(x1Type));
    conn->write(&_0y1, sizeof(_0y1));
    updateTmpPtr((void *)y1, _0y1);
    conn->write(&y1Type, sizeof(y1Type));
    conn->write(&_0param, sizeof(_0param));
    updateTmpPtr((void *)param, _0param);
    conn->write(&paramType, sizeof(paramType));
    conn->write(&executiontype, sizeof(executiontype));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)d1, sizeofType(d1Type), true);
    mem2client(conn, (void *)d2, sizeofType(d2Type), true);
    mem2client(conn, (void *)x1, sizeofType(x1Type), true);
    mem2client(conn, (void *)y1, sizeofType(y1Type), true);
    mem2client(conn, (void *)param, 5 * sizeofType(paramType), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, (trans == CUBLAS_OP_N ? n : m) * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, (trans == CUBLAS_OP_N ? m : n) * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSgemv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)x, (trans == CUBLAS_OP_N ? n : m) * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, (trans == CUBLAS_OP_N ? m : n) * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, (trans == CUBLAS_OP_N ? n : m) * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, (trans == CUBLAS_OP_N ? m : n) * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDgemv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)x, (trans == CUBLAS_OP_N ? n : m) * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, (trans == CUBLAS_OP_N ? m : n) * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, (trans == CUBLAS_OP_N ? n : m) * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, (trans == CUBLAS_OP_N ? m : n) * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgemv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)x, (trans == CUBLAS_OP_N ? n : m) * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, (trans == CUBLAS_OP_N ? m : n) * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, (trans == CUBLAS_OP_N ? n : m) * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, (trans == CUBLAS_OP_N ? m : n) * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgemv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)x, (trans == CUBLAS_OP_N ? n : m) * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, (trans == CUBLAS_OP_N ? m : n) * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgbmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, m * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSgbmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&kl, sizeof(kl));
    conn->write(&ku, sizeof(ku));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, m * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgbmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, m * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDgbmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&kl, sizeof(kl));
    conn->write(&ku, sizeof(ku));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, m * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgbmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, m * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgbmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&kl, sizeof(kl));
    conn->write(&ku, sizeof(ku));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, m * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgbmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, m * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgbmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&kl, sizeof(kl));
    conn->write(&ku, sizeof(ku));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, m * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasStrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasStrmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDtrmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCtrmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZtrmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasStbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStbmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * k * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasStbmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * k * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtbmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * k * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDtbmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * k * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtbmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * k * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCtbmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * k * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtbmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * k * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZtbmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * k * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasStpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *AP, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStpmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasStpmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *AP, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtpmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDtpmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtpmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCtpmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtpmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZtpmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasStrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrsv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasStrsv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrsv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDtrsv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrsv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCtrsv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrsv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZtrsv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasStpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *AP, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStpsv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasStpsv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *AP, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtpsv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDtpsv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtpsv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCtpsv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtpsv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZtpsv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasStbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStbsv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * k * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasStbsv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * k * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtbsv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * k * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDtbsv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * k * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtbsv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * k * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCtbsv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * k * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtbsv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * k * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZtbsv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * k * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsymv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSsymv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsymv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDsymv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsymv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCsymv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsymv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZsymv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasChemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasChemv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasChemv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhemv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZhemv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsbmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * k * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSsbmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, n * k * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsbmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * k * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDsbmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, n * k * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasChbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasChbmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * k * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasChbmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, n * k * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhbmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * k * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZhbmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, n * k * sizeof(*A), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *AP, const float *x, int incx, const float *beta, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSspmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSspmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *AP, const double *x, int incx, const double *beta, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDspmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDspmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasChpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *AP, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasChpmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasChpmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *AP, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhpmv_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZhpmv_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSger_v2(cublasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasSger_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, m * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSger_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, m * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDger_v2(cublasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasDger_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, m * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDger_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, m * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgeru_v2(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgeru_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, m * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgeru_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, m * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgerc_v2(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgerc_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, m * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgerc_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, m * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZgeru_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgeru_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, m * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgeru_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, m * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZgerc_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgerc_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, m * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgerc_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, m * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyr_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSsyr_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyr_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDsyr_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyr_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCsyr_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyr_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZsyr_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCher_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCher_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZher_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZher_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasSspr_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSspr_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasDspr_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDspr_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasChpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasChpr_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasChpr_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZhpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhpr_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZhpr_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyr2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSsyr2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyr2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDsyr2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyr2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCsyr2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyr2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZsyr2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCher2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCher2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZher2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZher2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasSspr2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSspr2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasDspr2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDspr2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasChpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasChpr2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasChpr2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZhpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhpr2_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, n * sizeof(*x));
    void *_0y;
    mem2server(conn, &_0y, (void *)y, n * sizeof(*y));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZhpr2_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0y, sizeof(_0y));
    updateTmpPtr((void *)y, _0y);
    conn->write(&incy, sizeof(incy));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)x, n * sizeof(*x), true);
    mem2client(conn, (void *)y, n * sizeof(*y), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSgemm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A), true);
    mem2client(conn, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDgemm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A), true);
    mem2client(conn, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgemm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A), true);
    mem2client(conn, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm3m called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgemm3m);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A), true);
    mem2client(conn, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm3mEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm3mEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeofType(Atype));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeofType(Btype));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeofType(Ctype));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgemm3mEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&Atype, sizeof(Atype));
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&Btype, sizeof(Btype));
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&Ctype, sizeof(Ctype));
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeofType(Atype), true);
    mem2client(conn, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeofType(Btype), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeofType(Ctype), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgemm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A), true);
    mem2client(conn, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemm3m called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgemm3m);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A), true);
    mem2client(conn, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, const __half *B, int ldb, const __half *beta, __half *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasHgemm called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasHgemm);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A), true);
    mem2client(conn, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const float *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemmEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeofType(Atype));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeofType(Btype));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeofType(Ctype));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSgemmEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&Atype, sizeof(Atype));
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&Btype, sizeof(Btype));
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&Ctype, sizeof(Ctype));
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeofType(Atype), true);
    mem2client(conn, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeofType(Btype), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeofType(Ctype), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemmEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeofType(Atype));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeofType(Btype));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeofType(Ctype));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgemmEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&Atype, sizeof(Atype));
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&Btype, sizeof(Btype));
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&Ctype, sizeof(Ctype));
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeofType(Atype), true);
    mem2client(conn, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeofType(Btype), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeofType(Ctype), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasUint8gemmBias(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, cublasOperation_t transc, int m, int n, int k, const unsigned char *A, int A_bias, int lda, const unsigned char *B, int B_bias, int ldb, unsigned char *C, int C_bias, int ldc, int C_mult, int C_shift) {
#ifdef DEBUG
    std::cout << "Hook: cublasUint8gemmBias called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasUint8gemmBias);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&transc, sizeof(transc));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&A_bias, sizeof(A_bias));
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&B_bias, sizeof(B_bias));
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&C_bias, sizeof(C_bias));
    conn->write(&ldc, sizeof(ldc));
    conn->write(&C_mult, sizeof(C_mult));
    conn->write(&C_shift, sizeof(C_shift));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A), true);
    mem2client(conn, (void *)B, transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *beta, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyrk_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSsyrk_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *beta, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyrk_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDsyrk_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyrk_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCsyrk_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyrk_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZsyrk_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyrkEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeofType(Atype));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeofType(Ctype));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCsyrkEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&Atype, sizeof(Atype));
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&Ctype, sizeof(Ctype));
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeofType(Atype), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeofType(Ctype), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCsyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyrk3mEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeofType(Atype));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeofType(Ctype));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCsyrk3mEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&Atype, sizeof(Atype));
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&Ctype, sizeof(Ctype));
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeofType(Atype), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeofType(Ctype), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const cuComplex *A, int lda, const float *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCherk_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCherk_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const cuDoubleComplex *A, int lda, const double *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZherk_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZherk_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const float *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCherkEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeofType(Atype));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeofType(Ctype));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCherkEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&Atype, sizeof(Atype));
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&Ctype, sizeof(Ctype));
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeofType(Atype), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeofType(Ctype), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const float *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCherk3mEx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeofType(Atype));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeofType(Ctype));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCherk3mEx);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&Atype, sizeof(Atype));
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&Ctype, sizeof(Ctype));
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeofType(Atype), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeofType(Ctype), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyr2k_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSsyr2k_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyr2k_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDsyr2k_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyr2k_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCsyr2k_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyr2k_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZsyr2k_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCher2k_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCher2k_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZher2k_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZher2k_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyrkx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSsyrkx);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyrkx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDsyrkx);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyrkx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCsyrkx);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyrkx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZsyrkx);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCherkx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCherkx);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A), true);
    mem2client(conn, (void *)B, trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZherkx called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * k * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, n * k * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, n * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZherkx);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, n * k * sizeof(*A), true);
    mem2client(conn, (void *)B, n * k * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, n * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsymm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSsymm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsymm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDsymm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsymm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCsymm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsymm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZsymm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasChemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasChemm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasChemm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZhemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhemm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZhemm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrsm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasStrsm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrsm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDtrsm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, cuComplex *B, int ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrsm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCtrsm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrsm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZtrsm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasStrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrmm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasStrmm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrmm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDtrmm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrmm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCtrmm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrmm_v2 called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZtrmm_v2);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, (side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *const Aarray[], int lda, const __half *const Barray[], int ldb, const __half *beta, __half *const Carray[], int ldc, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHgemmBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasHgemmBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(Aarray, sizeof(__half *) * batchCount, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Barray, sizeof(__half *) * batchCount, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(Carray, sizeof(__half *) * batchCount, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *const Aarray[], int lda, const float *const Barray[], int ldb, const float *beta, float *const Carray[], int ldc, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemmBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSgemmBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(Aarray, sizeof(float *) * batchCount, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Barray, sizeof(float *) * batchCount, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(Carray, sizeof(float *) * batchCount, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *const Aarray[], int lda, const double *const Barray[], int ldb, const double *beta, double *const Carray[], int ldc, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemmBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDgemmBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(Aarray, sizeof(double *) * batchCount, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Barray, sizeof(double *) * batchCount, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(Carray, sizeof(double *) * batchCount, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *const Aarray[], int lda, const cuComplex *const Barray[], int ldb, const cuComplex *beta, cuComplex *const Carray[], int ldc, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemmBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgemmBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(Aarray, sizeof(cuComplex *) * batchCount, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Barray, sizeof(cuComplex *) * batchCount, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(Carray, sizeof(cuComplex *) * batchCount, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm3mBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *const Aarray[], int lda, const cuComplex *const Barray[], int ldb, const cuComplex *beta, cuComplex *const Carray[], int ldc, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm3mBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgemm3mBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(Aarray, sizeof(cuComplex *) * batchCount, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Barray, sizeof(cuComplex *) * batchCount, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(Carray, sizeof(cuComplex *) * batchCount, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *const Aarray[], int lda, const cuDoubleComplex *const Barray[], int ldb, const cuDoubleComplex *beta, cuDoubleComplex *const Carray[], int ldc, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemmBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgemmBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(Aarray, sizeof(cuDoubleComplex *) * batchCount, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Barray, sizeof(cuDoubleComplex *) * batchCount, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(Carray, sizeof(cuDoubleComplex *) * batchCount, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemmStridedBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * k * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSgemmStridedBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&strideA, sizeof(strideA));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&strideB, sizeof(strideB));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&strideC, sizeof(strideC));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * k * sizeof(*A), true);
    mem2client(conn, (void *)B, k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, long long int strideA, const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemmStridedBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * k * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDgemmStridedBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&strideA, sizeof(strideA));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&strideB, sizeof(strideB));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&strideC, sizeof(strideC));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * k * sizeof(*A), true);
    mem2client(conn, (void *)B, k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA, const cuComplex *B, int ldb, long long int strideB, const cuComplex *beta, cuComplex *C, int ldc, long long int strideC, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemmStridedBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * k * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgemmStridedBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&strideA, sizeof(strideA));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&strideB, sizeof(strideB));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&strideC, sizeof(strideC));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * k * sizeof(*A), true);
    mem2client(conn, (void *)B, k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA, const cuComplex *B, int ldb, long long int strideB, const cuComplex *beta, cuComplex *C, int ldc, long long int strideC, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm3mStridedBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * k * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgemm3mStridedBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&strideA, sizeof(strideA));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&strideB, sizeof(strideB));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&strideC, sizeof(strideC));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * k * sizeof(*A), true);
    mem2client(conn, (void *)B, k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, long long int strideA, const cuDoubleComplex *B, int ldb, long long int strideB, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc, long long int strideC, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemmStridedBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * k * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgemmStridedBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&strideA, sizeof(strideA));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&strideB, sizeof(strideB));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&strideC, sizeof(strideC));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * k * sizeof(*A), true);
    mem2client(conn, (void *)B, k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, long long int strideA, const __half *B, int ldb, long long int strideB, const __half *beta, __half *C, int ldc, long long int strideC, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHgemmStridedBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * k * sizeof(*A));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, k * n * sizeof(*B));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasHgemmStridedBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&k, sizeof(k));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&strideA, sizeof(strideA));
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&strideB, sizeof(strideB));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&strideC, sizeof(strideC));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * k * sizeof(*A), true);
    mem2client(conn, (void *)B, k * n * sizeof(*B), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgeam called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSgeam);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double *alpha, const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgeam called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDgeam);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, const cuComplex *B, int ldb, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgeam called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgeam);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgeam called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0beta;
    mem2server(conn, &_0beta, (void *)beta, sizeof(*beta));
    void *_0B;
    mem2server(conn, &_0B, (void *)B, m * n * sizeof(*B));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgeam);
    conn->write(&handle, sizeof(handle));
    conn->write(&transa, sizeof(transa));
    conn->write(&transb, sizeof(transb));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0beta, sizeof(_0beta));
    updateTmpPtr((void *)beta, _0beta);
    conn->write(&_0B, sizeof(_0B));
    updateTmpPtr((void *)B, _0B);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)beta, sizeof(*beta), true);
    mem2client(conn, (void *)B, m * n * sizeof(*B), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float *const A[], int lda, int *P, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgetrfBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0P;
    mem2server(conn, &_0P, (void *)P, n * sizeof(*P));
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSgetrfBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(A, sizeof(float *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0P, sizeof(_0P));
    updateTmpPtr((void *)P, _0P);
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)P, n * sizeof(*P), true);
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

extern "C" cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double *const A[], int lda, int *P, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgetrfBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0P;
    mem2server(conn, &_0P, (void *)P, n * sizeof(*P));
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDgetrfBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(A, sizeof(double *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0P, sizeof(_0P));
    updateTmpPtr((void *)P, _0P);
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)P, n * sizeof(*P), true);
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

extern "C" cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex *const A[], int lda, int *P, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgetrfBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0P;
    mem2server(conn, &_0P, (void *)P, n * sizeof(*P));
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgetrfBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(A, sizeof(cuComplex *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0P, sizeof(_0P));
    updateTmpPtr((void *)P, _0P);
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)P, n * sizeof(*P), true);
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

extern "C" cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex *const A[], int lda, int *P, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgetrfBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0P;
    mem2server(conn, &_0P, (void *)P, n * sizeof(*P));
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgetrfBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(A, sizeof(cuDoubleComplex *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0P, sizeof(_0P));
    updateTmpPtr((void *)P, _0P);
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)P, n * sizeof(*P), true);
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

extern "C" cublasStatus_t cublasSgetriBatched(cublasHandle_t handle, int n, const float *const A[], int lda, const int *P, float *const C[], int ldc, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgetriBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0P;
    mem2server(conn, &_0P, (void *)P, n * sizeof(*P));
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSgetriBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(A, sizeof(float *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0P, sizeof(_0P));
    updateTmpPtr((void *)P, _0P);
    conn->write(C, sizeof(float *) * batchSize, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)P, n * sizeof(*P), true);
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

extern "C" cublasStatus_t cublasDgetriBatched(cublasHandle_t handle, int n, const double *const A[], int lda, const int *P, double *const C[], int ldc, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgetriBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0P;
    mem2server(conn, &_0P, (void *)P, n * sizeof(*P));
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDgetriBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(A, sizeof(double *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0P, sizeof(_0P));
    updateTmpPtr((void *)P, _0P);
    conn->write(C, sizeof(double *) * batchSize, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)P, n * sizeof(*P), true);
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

extern "C" cublasStatus_t cublasCgetriBatched(cublasHandle_t handle, int n, const cuComplex *const A[], int lda, const int *P, cuComplex *const C[], int ldc, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgetriBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0P;
    mem2server(conn, &_0P, (void *)P, n * sizeof(*P));
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgetriBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(A, sizeof(cuComplex *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0P, sizeof(_0P));
    updateTmpPtr((void *)P, _0P);
    conn->write(C, sizeof(cuComplex *) * batchSize, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)P, n * sizeof(*P), true);
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

extern "C" cublasStatus_t cublasZgetriBatched(cublasHandle_t handle, int n, const cuDoubleComplex *const A[], int lda, const int *P, cuDoubleComplex *const C[], int ldc, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgetriBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0P;
    mem2server(conn, &_0P, (void *)P, n * sizeof(*P));
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgetriBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(A, sizeof(cuDoubleComplex *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0P, sizeof(_0P));
    updateTmpPtr((void *)P, _0P);
    conn->write(C, sizeof(cuDoubleComplex *) * batchSize, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)P, n * sizeof(*P), true);
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

extern "C" cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *const Aarray[], int lda, const int *devIpiv, float *const Barray[], int ldb, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgetrsBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devIpiv;
    mem2server(conn, &_0devIpiv, (void *)devIpiv, sizeof(*devIpiv));
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSgetrsBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&nrhs, sizeof(nrhs));
    conn->write(Aarray, sizeof(float *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0devIpiv, sizeof(_0devIpiv));
    updateTmpPtr((void *)devIpiv, _0devIpiv);
    conn->write(Barray, sizeof(float *) * batchSize, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devIpiv, sizeof(*devIpiv), true);
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

extern "C" cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *const Aarray[], int lda, const int *devIpiv, double *const Barray[], int ldb, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgetrsBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devIpiv;
    mem2server(conn, &_0devIpiv, (void *)devIpiv, sizeof(*devIpiv));
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDgetrsBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&nrhs, sizeof(nrhs));
    conn->write(Aarray, sizeof(double *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0devIpiv, sizeof(_0devIpiv));
    updateTmpPtr((void *)devIpiv, _0devIpiv);
    conn->write(Barray, sizeof(double *) * batchSize, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devIpiv, sizeof(*devIpiv), true);
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

extern "C" cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex *const Aarray[], int lda, const int *devIpiv, cuComplex *const Barray[], int ldb, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgetrsBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devIpiv;
    mem2server(conn, &_0devIpiv, (void *)devIpiv, sizeof(*devIpiv));
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgetrsBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&nrhs, sizeof(nrhs));
    conn->write(Aarray, sizeof(cuComplex *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0devIpiv, sizeof(_0devIpiv));
    updateTmpPtr((void *)devIpiv, _0devIpiv);
    conn->write(Barray, sizeof(cuComplex *) * batchSize, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devIpiv, sizeof(*devIpiv), true);
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

extern "C" cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex *const Aarray[], int lda, const int *devIpiv, cuDoubleComplex *const Barray[], int ldb, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgetrsBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0devIpiv;
    mem2server(conn, &_0devIpiv, (void *)devIpiv, sizeof(*devIpiv));
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgetrsBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&n, sizeof(n));
    conn->write(&nrhs, sizeof(nrhs));
    conn->write(Aarray, sizeof(cuDoubleComplex *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0devIpiv, sizeof(_0devIpiv));
    updateTmpPtr((void *)devIpiv, _0devIpiv);
    conn->write(Barray, sizeof(cuDoubleComplex *) * batchSize, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)devIpiv, sizeof(*devIpiv), true);
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

extern "C" cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *const A[], int lda, float *const B[], int ldb, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrsmBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasStrsmBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(A, sizeof(float *) * batchCount, true);
    conn->write(&lda, sizeof(lda));
    conn->write(B, sizeof(float *) * batchCount, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *const A[], int lda, double *const B[], int ldb, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrsmBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDtrsmBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(A, sizeof(double *) * batchCount, true);
    conn->write(&lda, sizeof(lda));
    conn->write(B, sizeof(double *) * batchCount, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *const A[], int lda, cuComplex *const B[], int ldb, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrsmBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCtrsmBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(A, sizeof(cuComplex *) * batchCount, true);
    conn->write(&lda, sizeof(lda));
    conn->write(B, sizeof(cuComplex *) * batchCount, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *const A[], int lda, cuDoubleComplex *const B[], int ldb, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrsmBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0alpha;
    mem2server(conn, &_0alpha, (void *)alpha, sizeof(*alpha));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZtrsmBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&side, sizeof(side));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&trans, sizeof(trans));
    conn->write(&diag, sizeof(diag));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0alpha, sizeof(_0alpha));
    updateTmpPtr((void *)alpha, _0alpha);
    conn->write(A, sizeof(cuDoubleComplex *) * batchCount, true);
    conn->write(&lda, sizeof(lda));
    conn->write(B, sizeof(cuDoubleComplex *) * batchCount, true);
    conn->write(&ldb, sizeof(ldb));
    conn->write(&batchCount, sizeof(batchCount));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)alpha, sizeof(*alpha), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSmatinvBatched(cublasHandle_t handle, int n, const float *const A[], int lda, float *const Ainv[], int lda_inv, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasSmatinvBatched called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSmatinvBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(A, sizeof(float *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Ainv, sizeof(float *) * batchSize, true);
    conn->write(&lda_inv, sizeof(lda_inv));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasDmatinvBatched(cublasHandle_t handle, int n, const double *const A[], int lda, double *const Ainv[], int lda_inv, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasDmatinvBatched called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDmatinvBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(A, sizeof(double *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Ainv, sizeof(double *) * batchSize, true);
    conn->write(&lda_inv, sizeof(lda_inv));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasCmatinvBatched(cublasHandle_t handle, int n, const cuComplex *const A[], int lda, cuComplex *const Ainv[], int lda_inv, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasCmatinvBatched called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCmatinvBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(A, sizeof(cuComplex *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Ainv, sizeof(cuComplex *) * batchSize, true);
    conn->write(&lda_inv, sizeof(lda_inv));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasZmatinvBatched(cublasHandle_t handle, int n, const cuDoubleComplex *const A[], int lda, cuDoubleComplex *const Ainv[], int lda_inv, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasZmatinvBatched called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZmatinvBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&n, sizeof(n));
    conn->write(A, sizeof(cuDoubleComplex *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Ainv, sizeof(cuDoubleComplex *) * batchSize, true);
    conn->write(&lda_inv, sizeof(lda_inv));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float *const Aarray[], int lda, float *const TauArray[], int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgeqrfBatched called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSgeqrfBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(Aarray, sizeof(float *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(TauArray, sizeof(float *) * batchSize, true);
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double *const Aarray[], int lda, double *const TauArray[], int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgeqrfBatched called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDgeqrfBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(Aarray, sizeof(double *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(TauArray, sizeof(double *) * batchSize, true);
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex *const Aarray[], int lda, cuComplex *const TauArray[], int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgeqrfBatched called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgeqrfBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(Aarray, sizeof(cuComplex *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(TauArray, sizeof(cuComplex *) * batchSize, true);
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex *const Aarray[], int lda, cuDoubleComplex *const TauArray[], int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgeqrfBatched called" << std::endl;
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
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgeqrfBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(Aarray, sizeof(cuDoubleComplex *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(TauArray, sizeof(cuDoubleComplex *) * batchSize, true);
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float *const Aarray[], int lda, float *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgelsBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0info;
    mem2server(conn, &_0info, (void *)info, sizeof(*info));
    void *_0devInfoArray;
    mem2server(conn, &_0devInfoArray, (void *)devInfoArray, sizeof(*devInfoArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSgelsBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&nrhs, sizeof(nrhs));
    conn->write(Aarray, sizeof(float *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Carray, sizeof(float *) * batchSize, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&_0devInfoArray, sizeof(_0devInfoArray));
    updateTmpPtr((void *)devInfoArray, _0devInfoArray);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)info, sizeof(*info), true);
    mem2client(conn, (void *)devInfoArray, sizeof(*devInfoArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double *const Aarray[], int lda, double *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgelsBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0info;
    mem2server(conn, &_0info, (void *)info, sizeof(*info));
    void *_0devInfoArray;
    mem2server(conn, &_0devInfoArray, (void *)devInfoArray, sizeof(*devInfoArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDgelsBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&nrhs, sizeof(nrhs));
    conn->write(Aarray, sizeof(double *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Carray, sizeof(double *) * batchSize, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&_0devInfoArray, sizeof(_0devInfoArray));
    updateTmpPtr((void *)devInfoArray, _0devInfoArray);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)info, sizeof(*info), true);
    mem2client(conn, (void *)devInfoArray, sizeof(*devInfoArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuComplex *const Aarray[], int lda, cuComplex *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgelsBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0info;
    mem2server(conn, &_0info, (void *)info, sizeof(*info));
    void *_0devInfoArray;
    mem2server(conn, &_0devInfoArray, (void *)devInfoArray, sizeof(*devInfoArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCgelsBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&nrhs, sizeof(nrhs));
    conn->write(Aarray, sizeof(cuComplex *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Carray, sizeof(cuComplex *) * batchSize, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&_0devInfoArray, sizeof(_0devInfoArray));
    updateTmpPtr((void *)devInfoArray, _0devInfoArray);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)info, sizeof(*info), true);
    mem2client(conn, (void *)devInfoArray, sizeof(*devInfoArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex *const Aarray[], int lda, cuDoubleComplex *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgelsBatched called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0info;
    mem2server(conn, &_0info, (void *)info, sizeof(*info));
    void *_0devInfoArray;
    mem2server(conn, &_0devInfoArray, (void *)devInfoArray, sizeof(*devInfoArray));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZgelsBatched);
    conn->write(&handle, sizeof(handle));
    conn->write(&trans, sizeof(trans));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&nrhs, sizeof(nrhs));
    conn->write(Aarray, sizeof(cuDoubleComplex *) * batchSize, true);
    conn->write(&lda, sizeof(lda));
    conn->write(Carray, sizeof(cuDoubleComplex *) * batchSize, true);
    conn->write(&ldc, sizeof(ldc));
    conn->write(&_0info, sizeof(_0info));
    updateTmpPtr((void *)info, _0info);
    conn->write(&_0devInfoArray, sizeof(_0devInfoArray));
    updateTmpPtr((void *)devInfoArray, _0devInfoArray);
    conn->write(&batchSize, sizeof(batchSize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)info, sizeof(*info), true);
    mem2client(conn, (void *)devInfoArray, sizeof(*devInfoArray), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float *A, int lda, const float *x, int incx, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSdgmm called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, (mode == CUBLAS_SIDE_LEFT ? m : n) * sizeof(*x));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasSdgmm);
    conn->write(&handle, sizeof(handle));
    conn->write(&mode, sizeof(mode));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)x, (mode == CUBLAS_SIDE_LEFT ? m : n) * sizeof(*x), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double *A, int lda, const double *x, int incx, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDdgmm called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, (mode == CUBLAS_SIDE_LEFT ? m : n) * sizeof(*x));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDdgmm);
    conn->write(&handle, sizeof(handle));
    conn->write(&mode, sizeof(mode));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)x, (mode == CUBLAS_SIDE_LEFT ? m : n) * sizeof(*x), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCdgmm called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, (mode == CUBLAS_SIDE_LEFT ? m : n) * sizeof(*x));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCdgmm);
    conn->write(&handle, sizeof(handle));
    conn->write(&mode, sizeof(mode));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)x, (mode == CUBLAS_SIDE_LEFT ? m : n) * sizeof(*x), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdgmm called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, m * n * sizeof(*A));
    void *_0x;
    mem2server(conn, &_0x, (void *)x, (mode == CUBLAS_SIDE_LEFT ? m : n) * sizeof(*x));
    void *_0C;
    mem2server(conn, &_0C, (void *)C, m * n * sizeof(*C));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZdgmm);
    conn->write(&handle, sizeof(handle));
    conn->write(&mode, sizeof(mode));
    conn->write(&m, sizeof(m));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0x, sizeof(_0x));
    updateTmpPtr((void *)x, _0x);
    conn->write(&incx, sizeof(incx));
    conn->write(&_0C, sizeof(_0C));
    updateTmpPtr((void *)C, _0C);
    conn->write(&ldc, sizeof(ldc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, m * n * sizeof(*A), true);
    mem2client(conn, (void *)x, (mode == CUBLAS_SIDE_LEFT ? m : n) * sizeof(*x), true);
    mem2client(conn, (void *)C, m * n * sizeof(*C), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *AP, float *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasStpttr called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasStpttr);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *AP, double *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtpttr called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDtpttr);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *AP, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtpttr called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCtpttr);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *AP, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtpttr called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZtpttr);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasStrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *A, int lda, float *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrttp called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasStrttp);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasDtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *A, int lda, double *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrttp called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasDtrttp);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasCtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *A, int lda, cuComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrttp called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasCtrttp);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cublasStatus_t cublasZtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrttp called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    void *_0A;
    mem2server(conn, &_0A, (void *)A, n * n * sizeof(*A));
    void *_0AP;
    mem2server(conn, &_0AP, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP));
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    cublasStatus_t _result;
    conn->prepare_request(RPC_cublasZtrttp);
    conn->write(&handle, sizeof(handle));
    conn->write(&uplo, sizeof(uplo));
    conn->write(&n, sizeof(n));
    conn->write(&_0A, sizeof(_0A));
    updateTmpPtr((void *)A, _0A);
    conn->write(&lda, sizeof(lda));
    conn->write(&_0AP, sizeof(_0AP));
    updateTmpPtr((void *)AP, _0AP);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn, true);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    mem2client(conn, (void *)A, n * n * sizeof(*A), true);
    mem2client(conn, (void *)AP, (n * (n + 1)) / 2 * sizeof(*AP), true);
    if(conn->get_iov_read_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn, true);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}
