#include <iostream>
#include <unordered_map>
#include "hook_api.h"
#include "handle_server.h"
#include "../rpc.h"
#include "cublas_api.h"

int handle_cublasCreate_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCreate_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t *handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCreate_v2(handle);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDestroy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDestroy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDestroy_v2(handle);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasGetVersion_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetVersion_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int *version;
    rpc_read(client, &version, sizeof(version));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetVersion_v2(handle, version);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasGetProperty(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetProperty called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    libraryPropertyType type;
    rpc_read(client, &type, sizeof(type));
    int *value;
    rpc_read(client, &value, sizeof(value));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetProperty(type, value);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasGetCudartVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetCudartVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    size_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetCudartVersion();
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSetWorkspace_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetWorkspace_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    void *workspace;
    rpc_read(client, &workspace, sizeof(workspace));
    size_t workspaceSizeInBytes;
    rpc_read(client, &workspaceSizeInBytes, sizeof(workspaceSizeInBytes));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetWorkspace_v2(handle, workspace, workspaceSizeInBytes);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSetStream_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetStream_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cudaStream_t streamId;
    rpc_read(client, &streamId, sizeof(streamId));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetStream_v2(handle, streamId);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasGetStream_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetStream_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cudaStream_t *streamId;
    rpc_read(client, &streamId, sizeof(streamId));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetStream_v2(handle, streamId);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasGetPointerMode_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetPointerMode_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasPointerMode_t *mode;
    rpc_read(client, &mode, sizeof(mode));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetPointerMode_v2(handle, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSetPointerMode_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetPointerMode_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasPointerMode_t mode;
    rpc_read(client, &mode, sizeof(mode));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetPointerMode_v2(handle, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasGetAtomicsMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetAtomicsMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasAtomicsMode_t *mode;
    rpc_read(client, &mode, sizeof(mode));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetAtomicsMode(handle, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSetAtomicsMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetAtomicsMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasAtomicsMode_t mode;
    rpc_read(client, &mode, sizeof(mode));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetAtomicsMode(handle, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasGetMathMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetMathMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasMath_t *mode;
    rpc_read(client, &mode, sizeof(mode));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetMathMode(handle, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSetMathMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetMathMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasMath_t mode;
    rpc_read(client, &mode, sizeof(mode));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetMathMode(handle, mode);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasGetSmCountTarget(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetSmCountTarget called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int *smCountTarget;
    rpc_read(client, &smCountTarget, sizeof(smCountTarget));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetSmCountTarget(handle, smCountTarget);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSetSmCountTarget(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetSmCountTarget called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int smCountTarget;
    rpc_read(client, &smCountTarget, sizeof(smCountTarget));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetSmCountTarget(handle, smCountTarget);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasLoggerConfigure(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasLoggerConfigure called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int logIsOn;
    rpc_read(client, &logIsOn, sizeof(logIsOn));
    int logToStdOut;
    rpc_read(client, &logToStdOut, sizeof(logToStdOut));
    int logToStdErr;
    rpc_read(client, &logToStdErr, sizeof(logToStdErr));
    char *logFileName = nullptr;
    rpc_read(client, &logFileName, 0, true);
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(logFileName);
    _result = cublasLoggerConfigure(logIsOn, logToStdOut, logToStdErr, logFileName);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSetLoggerCallback(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetLoggerCallback called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasLogCallback userCallback;
    rpc_read(client, &userCallback, sizeof(userCallback));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetLoggerCallback(userCallback);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasGetLoggerCallback(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetLoggerCallback called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasLogCallback *userCallback;
    rpc_read(client, &userCallback, sizeof(userCallback));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetLoggerCallback(userCallback);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSetVector(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetVector called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int n;
    rpc_read(client, &n, sizeof(n));
    int elemSize;
    rpc_read(client, &elemSize, sizeof(elemSize));
    void *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    void *devicePtr;
    rpc_read(client, &devicePtr, sizeof(devicePtr));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetVector(n, elemSize, x, incx, devicePtr, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasGetVector(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetVector called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int n;
    rpc_read(client, &n, sizeof(n));
    int elemSize;
    rpc_read(client, &elemSize, sizeof(elemSize));
    void *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    void *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetVector(n, elemSize, x, incx, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSetMatrix(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetMatrix called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int rows;
    rpc_read(client, &rows, sizeof(rows));
    int cols;
    rpc_read(client, &cols, sizeof(cols));
    int elemSize;
    rpc_read(client, &elemSize, sizeof(elemSize));
    void *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    void *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasGetMatrix(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetMatrix called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int rows;
    rpc_read(client, &rows, sizeof(rows));
    int cols;
    rpc_read(client, &cols, sizeof(cols));
    int elemSize;
    rpc_read(client, &elemSize, sizeof(elemSize));
    void *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    void *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSetVectorAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetVectorAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int n;
    rpc_read(client, &n, sizeof(n));
    int elemSize;
    rpc_read(client, &elemSize, sizeof(elemSize));
    void *hostPtr;
    rpc_read(client, &hostPtr, sizeof(hostPtr));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    void *devicePtr;
    rpc_read(client, &devicePtr, sizeof(devicePtr));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasGetVectorAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetVectorAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int n;
    rpc_read(client, &n, sizeof(n));
    int elemSize;
    rpc_read(client, &elemSize, sizeof(elemSize));
    void *devicePtr;
    rpc_read(client, &devicePtr, sizeof(devicePtr));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    void *hostPtr;
    rpc_read(client, &hostPtr, sizeof(hostPtr));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSetMatrixAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetMatrixAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int rows;
    rpc_read(client, &rows, sizeof(rows));
    int cols;
    rpc_read(client, &cols, sizeof(cols));
    int elemSize;
    rpc_read(client, &elemSize, sizeof(elemSize));
    void *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    void *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasGetMatrixAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetMatrixAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    int rows;
    rpc_read(client, &rows, sizeof(rows));
    int cols;
    rpc_read(client, &cols, sizeof(cols));
    int elemSize;
    rpc_read(client, &elemSize, sizeof(elemSize));
    void *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    void *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cudaStream_t stream;
    rpc_read(client, &stream, sizeof(stream));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasXerbla(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasXerbla called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    char *srName = nullptr;
    rpc_read(client, &srName, 0, true);
    int info;
    rpc_read(client, &info, sizeof(info));
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(srName);
    cublasXerbla(srName, info);
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasNrm2Ex(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasNrm2Ex called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    void *x;
    rpc_read(client, &x, sizeof(x));
    cudaDataType xType;
    rpc_read(client, &xType, sizeof(xType));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    void *result;
    rpc_read(client, &result, sizeof(result));
    cudaDataType resultType;
    rpc_read(client, &resultType, sizeof(resultType));
    cudaDataType executionType;
    rpc_read(client, &executionType, sizeof(executionType));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasNrm2Ex(handle, n, x, xType, incx, result, resultType, executionType);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSnrm2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSnrm2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSnrm2_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDnrm2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDnrm2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDnrm2_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasScnrm2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasScnrm2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasScnrm2_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDznrm2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDznrm2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDznrm2_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDotEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDotEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    void *x;
    rpc_read(client, &x, sizeof(x));
    cudaDataType xType;
    rpc_read(client, &xType, sizeof(xType));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    void *y;
    rpc_read(client, &y, sizeof(y));
    cudaDataType yType;
    rpc_read(client, &yType, sizeof(yType));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    void *result;
    rpc_read(client, &result, sizeof(result));
    cudaDataType resultType;
    rpc_read(client, &resultType, sizeof(resultType));
    cudaDataType executionType;
    rpc_read(client, &executionType, sizeof(executionType));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDotEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDotcEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDotcEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    void *x;
    rpc_read(client, &x, sizeof(x));
    cudaDataType xType;
    rpc_read(client, &xType, sizeof(xType));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    void *y;
    rpc_read(client, &y, sizeof(y));
    cudaDataType yType;
    rpc_read(client, &yType, sizeof(yType));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    void *result;
    rpc_read(client, &result, sizeof(result));
    cudaDataType resultType;
    rpc_read(client, &resultType, sizeof(resultType));
    cudaDataType executionType;
    rpc_read(client, &executionType, sizeof(executionType));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDotcEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSdot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSdot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    float *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSdot_v2(handle, n, x, incx, y, incy, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDdot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDdot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    double *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDdot_v2(handle, n, x, incx, y, incy, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCdotu_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCdotu_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuComplex *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCdotu_v2(handle, n, x, incx, y, incy, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCdotc_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCdotc_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuComplex *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCdotc_v2(handle, n, x, incx, y, incy, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZdotu_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZdotu_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuDoubleComplex *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZdotu_v2(handle, n, x, incx, y, incy, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZdotc_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZdotc_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuDoubleComplex *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZdotc_v2(handle, n, x, incx, y, incy, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasScalEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasScalEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    void *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cudaDataType alphaType;
    rpc_read(client, &alphaType, sizeof(alphaType));
    void *x;
    rpc_read(client, &x, sizeof(x));
    cudaDataType xType;
    rpc_read(client, &xType, sizeof(xType));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cudaDataType executionType;
    rpc_read(client, &executionType, sizeof(executionType));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasScalEx(handle, n, alpha, alphaType, x, xType, incx, executionType);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSscal_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSscal_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSscal_v2(handle, n, alpha, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDscal_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDscal_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDscal_v2(handle, n, alpha, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCscal_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCscal_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCscal_v2(handle, n, alpha, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCsscal_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsscal_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsscal_v2(handle, n, alpha, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZscal_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZscal_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZscal_v2(handle, n, alpha, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZdscal_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZdscal_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZdscal_v2(handle, n, alpha, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasAxpyEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasAxpyEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    void *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cudaDataType alphaType;
    rpc_read(client, &alphaType, sizeof(alphaType));
    void *x;
    rpc_read(client, &x, sizeof(x));
    cudaDataType xType;
    rpc_read(client, &xType, sizeof(xType));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    void *y;
    rpc_read(client, &y, sizeof(y));
    cudaDataType yType;
    rpc_read(client, &yType, sizeof(yType));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cudaDataType executiontype;
    rpc_read(client, &executiontype, sizeof(executiontype));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasAxpyEx(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSaxpy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSaxpy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDaxpy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDaxpy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCaxpy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCaxpy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCaxpy_v2(handle, n, alpha, x, incx, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZaxpy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZaxpy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZaxpy_v2(handle, n, alpha, x, incx, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCopyEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCopyEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    void *x;
    rpc_read(client, &x, sizeof(x));
    cudaDataType xType;
    rpc_read(client, &xType, sizeof(xType));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    void *y;
    rpc_read(client, &y, sizeof(y));
    cudaDataType yType;
    rpc_read(client, &yType, sizeof(yType));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCopyEx(handle, n, x, xType, incx, y, yType, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasScopy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasScopy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasScopy_v2(handle, n, x, incx, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDcopy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDcopy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDcopy_v2(handle, n, x, incx, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCcopy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCcopy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCcopy_v2(handle, n, x, incx, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZcopy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZcopy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZcopy_v2(handle, n, x, incx, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSswap_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSswap_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSswap_v2(handle, n, x, incx, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDswap_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDswap_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDswap_v2(handle, n, x, incx, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCswap_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCswap_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCswap_v2(handle, n, x, incx, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZswap_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZswap_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZswap_v2(handle, n, x, incx, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSwapEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSwapEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    void *x;
    rpc_read(client, &x, sizeof(x));
    cudaDataType xType;
    rpc_read(client, &xType, sizeof(xType));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    void *y;
    rpc_read(client, &y, sizeof(y));
    cudaDataType yType;
    rpc_read(client, &yType, sizeof(yType));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSwapEx(handle, n, x, xType, incx, y, yType, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasIsamax_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIsamax_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    int *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIsamax_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasIdamax_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIdamax_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    int *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIdamax_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasIcamax_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIcamax_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    int *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIcamax_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasIzamax_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIzamax_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    int *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIzamax_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasIamaxEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIamaxEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    void *x;
    rpc_read(client, &x, sizeof(x));
    cudaDataType xType;
    rpc_read(client, &xType, sizeof(xType));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    int *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIamaxEx(handle, n, x, xType, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasIsamin_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIsamin_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    int *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIsamin_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasIdamin_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIdamin_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    int *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIdamin_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasIcamin_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIcamin_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    int *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIcamin_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasIzamin_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIzamin_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    int *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIzamin_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasIaminEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIaminEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    void *x;
    rpc_read(client, &x, sizeof(x));
    cudaDataType xType;
    rpc_read(client, &xType, sizeof(xType));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    int *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIaminEx(handle, n, x, xType, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasAsumEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasAsumEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    void *x;
    rpc_read(client, &x, sizeof(x));
    cudaDataType xType;
    rpc_read(client, &xType, sizeof(xType));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    void *result;
    rpc_read(client, &result, sizeof(result));
    cudaDataType resultType;
    rpc_read(client, &resultType, sizeof(resultType));
    cudaDataType executiontype;
    rpc_read(client, &executiontype, sizeof(executiontype));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasAsumEx(handle, n, x, xType, incx, result, resultType, executiontype);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSasum_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSasum_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSasum_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDasum_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDasum_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDasum_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasScasum_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasScasum_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasScasum_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDzasum_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDzasum_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *result;
    rpc_read(client, &result, sizeof(result));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDzasum_v2(handle, n, x, incx, result);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSrot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSrot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    float *c;
    rpc_read(client, &c, sizeof(c));
    float *s;
    rpc_read(client, &s, sizeof(s));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSrot_v2(handle, n, x, incx, y, incy, c, s);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDrot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDrot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    double *c;
    rpc_read(client, &c, sizeof(c));
    double *s;
    rpc_read(client, &s, sizeof(s));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDrot_v2(handle, n, x, incx, y, incy, c, s);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCrot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCrot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    float *c;
    rpc_read(client, &c, sizeof(c));
    cuComplex *s;
    rpc_read(client, &s, sizeof(s));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCrot_v2(handle, n, x, incx, y, incy, c, s);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCsrot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsrot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    float *c;
    rpc_read(client, &c, sizeof(c));
    float *s;
    rpc_read(client, &s, sizeof(s));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsrot_v2(handle, n, x, incx, y, incy, c, s);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZrot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZrot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    double *c;
    rpc_read(client, &c, sizeof(c));
    cuDoubleComplex *s;
    rpc_read(client, &s, sizeof(s));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZrot_v2(handle, n, x, incx, y, incy, c, s);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZdrot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZdrot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    double *c;
    rpc_read(client, &c, sizeof(c));
    double *s;
    rpc_read(client, &s, sizeof(s));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZdrot_v2(handle, n, x, incx, y, incy, c, s);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasRotEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasRotEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    void *x;
    rpc_read(client, &x, sizeof(x));
    cudaDataType xType;
    rpc_read(client, &xType, sizeof(xType));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    void *y;
    rpc_read(client, &y, sizeof(y));
    cudaDataType yType;
    rpc_read(client, &yType, sizeof(yType));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    void *c;
    rpc_read(client, &c, sizeof(c));
    void *s;
    rpc_read(client, &s, sizeof(s));
    cudaDataType csType;
    rpc_read(client, &csType, sizeof(csType));
    cudaDataType executiontype;
    rpc_read(client, &executiontype, sizeof(executiontype));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasRotEx(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSrotg_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSrotg_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    float *a;
    rpc_read(client, &a, sizeof(a));
    float *b;
    rpc_read(client, &b, sizeof(b));
    float *c;
    rpc_read(client, &c, sizeof(c));
    float *s;
    rpc_read(client, &s, sizeof(s));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSrotg_v2(handle, a, b, c, s);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDrotg_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDrotg_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    double *a;
    rpc_read(client, &a, sizeof(a));
    double *b;
    rpc_read(client, &b, sizeof(b));
    double *c;
    rpc_read(client, &c, sizeof(c));
    double *s;
    rpc_read(client, &s, sizeof(s));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDrotg_v2(handle, a, b, c, s);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCrotg_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCrotg_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cuComplex *a;
    rpc_read(client, &a, sizeof(a));
    cuComplex *b;
    rpc_read(client, &b, sizeof(b));
    float *c;
    rpc_read(client, &c, sizeof(c));
    cuComplex *s;
    rpc_read(client, &s, sizeof(s));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCrotg_v2(handle, a, b, c, s);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZrotg_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZrotg_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cuDoubleComplex *a;
    rpc_read(client, &a, sizeof(a));
    cuDoubleComplex *b;
    rpc_read(client, &b, sizeof(b));
    double *c;
    rpc_read(client, &c, sizeof(c));
    cuDoubleComplex *s;
    rpc_read(client, &s, sizeof(s));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZrotg_v2(handle, a, b, c, s);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasRotgEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasRotgEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    void *a;
    rpc_read(client, &a, sizeof(a));
    void *b;
    rpc_read(client, &b, sizeof(b));
    cudaDataType abType;
    rpc_read(client, &abType, sizeof(abType));
    void *c;
    rpc_read(client, &c, sizeof(c));
    void *s;
    rpc_read(client, &s, sizeof(s));
    cudaDataType csType;
    rpc_read(client, &csType, sizeof(csType));
    cudaDataType executiontype;
    rpc_read(client, &executiontype, sizeof(executiontype));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasRotgEx(handle, a, b, abType, c, s, csType, executiontype);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSrotm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSrotm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    float *param;
    rpc_read(client, &param, sizeof(param));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSrotm_v2(handle, n, x, incx, y, incy, param);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDrotm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDrotm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    double *param;
    rpc_read(client, &param, sizeof(param));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDrotm_v2(handle, n, x, incx, y, incy, param);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasRotmEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasRotmEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    void *x;
    rpc_read(client, &x, sizeof(x));
    cudaDataType xType;
    rpc_read(client, &xType, sizeof(xType));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    void *y;
    rpc_read(client, &y, sizeof(y));
    cudaDataType yType;
    rpc_read(client, &yType, sizeof(yType));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    void *param;
    rpc_read(client, &param, sizeof(param));
    cudaDataType paramType;
    rpc_read(client, &paramType, sizeof(paramType));
    cudaDataType executiontype;
    rpc_read(client, &executiontype, sizeof(executiontype));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasRotmEx(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSrotmg_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSrotmg_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    float *d1;
    rpc_read(client, &d1, sizeof(d1));
    float *d2;
    rpc_read(client, &d2, sizeof(d2));
    float *x1;
    rpc_read(client, &x1, sizeof(x1));
    float *y1;
    rpc_read(client, &y1, sizeof(y1));
    float *param;
    rpc_read(client, &param, sizeof(param));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSrotmg_v2(handle, d1, d2, x1, y1, param);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDrotmg_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDrotmg_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    double *d1;
    rpc_read(client, &d1, sizeof(d1));
    double *d2;
    rpc_read(client, &d2, sizeof(d2));
    double *x1;
    rpc_read(client, &x1, sizeof(x1));
    double *y1;
    rpc_read(client, &y1, sizeof(y1));
    double *param;
    rpc_read(client, &param, sizeof(param));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDrotmg_v2(handle, d1, d2, x1, y1, param);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasRotmgEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasRotmgEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    void *d1;
    rpc_read(client, &d1, sizeof(d1));
    cudaDataType d1Type;
    rpc_read(client, &d1Type, sizeof(d1Type));
    void *d2;
    rpc_read(client, &d2, sizeof(d2));
    cudaDataType d2Type;
    rpc_read(client, &d2Type, sizeof(d2Type));
    void *x1;
    rpc_read(client, &x1, sizeof(x1));
    cudaDataType x1Type;
    rpc_read(client, &x1Type, sizeof(x1Type));
    void *y1;
    rpc_read(client, &y1, sizeof(y1));
    cudaDataType y1Type;
    rpc_read(client, &y1Type, sizeof(y1Type));
    void *param;
    rpc_read(client, &param, sizeof(param));
    cudaDataType paramType;
    rpc_read(client, &paramType, sizeof(paramType));
    cudaDataType executiontype;
    rpc_read(client, &executiontype, sizeof(executiontype));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasRotmgEx(handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType, executiontype);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSgemv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgemv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDgemv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgemv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgemv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgemv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgemv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSgbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int kl;
    rpc_read(client, &kl, sizeof(kl));
    int ku;
    rpc_read(client, &ku, sizeof(ku));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDgbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int kl;
    rpc_read(client, &kl, sizeof(kl));
    int ku;
    rpc_read(client, &ku, sizeof(ku));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int kl;
    rpc_read(client, &kl, sizeof(kl));
    int ku;
    rpc_read(client, &ku, sizeof(ku));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int kl;
    rpc_read(client, &kl, sizeof(kl));
    int ku;
    rpc_read(client, &ku, sizeof(ku));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasStrmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStrmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDtrmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtrmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCtrmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtrmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZtrmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtrmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasStbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDtbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCtbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZtbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasStpmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStpmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *AP;
    rpc_read(client, &AP, sizeof(AP));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDtpmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtpmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *AP;
    rpc_read(client, &AP, sizeof(AP));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCtpmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtpmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZtpmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtpmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasStrsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStrsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDtrsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtrsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCtrsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtrsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZtrsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtrsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasStpsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStpsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *AP;
    rpc_read(client, &AP, sizeof(AP));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDtpsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtpsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *AP;
    rpc_read(client, &AP, sizeof(AP));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCtpsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtpsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZtpsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtpsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasStbsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStbsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDtbsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtbsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCtbsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtbsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZtbsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtbsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSsymv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsymv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDsymv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsymv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCsymv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsymv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZsymv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsymv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasChemv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasChemv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasChemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZhemv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZhemv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZhemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSsbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDsbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasChbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasChbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasChbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZhbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZhbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZhbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSspmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSspmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *AP;
    rpc_read(client, &AP, sizeof(AP));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDspmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDspmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *AP;
    rpc_read(client, &AP, sizeof(AP));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasChpmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasChpmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasChpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZhpmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZhpmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZhpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSger_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSger_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDger_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDger_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgeru_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgeru_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgerc_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgerc_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgeru_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgeru_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgerc_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgerc_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSsyr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsyr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDsyr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsyr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCsyr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZsyr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsyr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCher_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCher_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCher_v2(handle, uplo, n, alpha, x, incx, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZher_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZher_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZher_v2(handle, uplo, n, alpha, x, incx, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSspr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSspr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *AP;
    rpc_read(client, &AP, sizeof(AP));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSspr_v2(handle, uplo, n, alpha, x, incx, AP);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDspr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDspr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *AP;
    rpc_read(client, &AP, sizeof(AP));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDspr_v2(handle, uplo, n, alpha, x, incx, AP);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasChpr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasChpr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasChpr_v2(handle, uplo, n, alpha, x, incx, AP);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZhpr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZhpr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZhpr_v2(handle, uplo, n, alpha, x, incx, AP);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSsyr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsyr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDsyr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsyr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCsyr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZsyr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsyr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCher2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCher2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZher2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZher2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSspr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSspr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    float *AP;
    rpc_read(client, &AP, sizeof(AP));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDspr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDspr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    double *AP;
    rpc_read(client, &AP, sizeof(AP));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasChpr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasChpr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasChpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZhpr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZhpr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *y;
    rpc_read(client, &y, sizeof(y));
    int incy;
    rpc_read(client, &incy, sizeof(incy));
    cuDoubleComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZhpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSgemm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgemm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    float *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDgemm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgemm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    double *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgemm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgemm3m(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemm3m called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgemm3mEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemm3mEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    void *A;
    rpc_read(client, &A, sizeof(A));
    cudaDataType Atype;
    rpc_read(client, &Atype, sizeof(Atype));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    void *B;
    rpc_read(client, &B, sizeof(B));
    cudaDataType Btype;
    rpc_read(client, &Btype, sizeof(Btype));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    void *C;
    rpc_read(client, &C, sizeof(C));
    cudaDataType Ctype;
    rpc_read(client, &Ctype, sizeof(Ctype));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemm3mEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgemm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgemm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgemm3m(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgemm3m called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasHgemm(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasHgemm called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    __half *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    __half *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    __half *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    __half *beta;
    rpc_read(client, &beta, sizeof(beta));
    __half *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSgemmEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgemmEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    void *A;
    rpc_read(client, &A, sizeof(A));
    cudaDataType Atype;
    rpc_read(client, &Atype, sizeof(Atype));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    void *B;
    rpc_read(client, &B, sizeof(B));
    cudaDataType Btype;
    rpc_read(client, &Btype, sizeof(Btype));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    void *C;
    rpc_read(client, &C, sizeof(C));
    cudaDataType Ctype;
    rpc_read(client, &Ctype, sizeof(Ctype));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgemmEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemmEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    void *A;
    rpc_read(client, &A, sizeof(A));
    cudaDataType Atype;
    rpc_read(client, &Atype, sizeof(Atype));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    void *B;
    rpc_read(client, &B, sizeof(B));
    cudaDataType Btype;
    rpc_read(client, &Btype, sizeof(Btype));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    void *C;
    rpc_read(client, &C, sizeof(C));
    cudaDataType Ctype;
    rpc_read(client, &Ctype, sizeof(Ctype));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasUint8gemmBias(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasUint8gemmBias called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    cublasOperation_t transc;
    rpc_read(client, &transc, sizeof(transc));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    unsigned char *A;
    rpc_read(client, &A, sizeof(A));
    int A_bias;
    rpc_read(client, &A_bias, sizeof(A_bias));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    unsigned char *B;
    rpc_read(client, &B, sizeof(B));
    int B_bias;
    rpc_read(client, &B_bias, sizeof(B_bias));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    unsigned char *C;
    rpc_read(client, &C, sizeof(C));
    int C_bias;
    rpc_read(client, &C_bias, sizeof(C_bias));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int C_mult;
    rpc_read(client, &C_mult, sizeof(C_mult));
    int C_shift;
    rpc_read(client, &C_shift, sizeof(C_shift));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasUint8gemmBias(handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult, C_shift);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSsyrk_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsyrk_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    float *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDsyrk_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsyrk_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    double *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCsyrk_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyrk_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZsyrk_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsyrk_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCsyrkEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyrkEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    void *A;
    rpc_read(client, &A, sizeof(A));
    cudaDataType Atype;
    rpc_read(client, &Atype, sizeof(Atype));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    void *C;
    rpc_read(client, &C, sizeof(C));
    cudaDataType Ctype;
    rpc_read(client, &Ctype, sizeof(Ctype));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyrkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCsyrk3mEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyrk3mEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    void *A;
    rpc_read(client, &A, sizeof(A));
    cudaDataType Atype;
    rpc_read(client, &Atype, sizeof(Atype));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    void *C;
    rpc_read(client, &C, sizeof(C));
    cudaDataType Ctype;
    rpc_read(client, &Ctype, sizeof(Ctype));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyrk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCherk_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCherk_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZherk_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZherk_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCherkEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCherkEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    void *A;
    rpc_read(client, &A, sizeof(A));
    cudaDataType Atype;
    rpc_read(client, &Atype, sizeof(Atype));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    void *C;
    rpc_read(client, &C, sizeof(C));
    cudaDataType Ctype;
    rpc_read(client, &Ctype, sizeof(Ctype));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCherkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCherk3mEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCherk3mEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    void *A;
    rpc_read(client, &A, sizeof(A));
    cudaDataType Atype;
    rpc_read(client, &Atype, sizeof(Atype));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    void *C;
    rpc_read(client, &C, sizeof(C));
    cudaDataType Ctype;
    rpc_read(client, &Ctype, sizeof(Ctype));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCherk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSsyr2k_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsyr2k_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    float *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDsyr2k_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsyr2k_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    double *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCsyr2k_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyr2k_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZsyr2k_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsyr2k_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCher2k_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCher2k_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZher2k_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZher2k_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSsyrkx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsyrkx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    float *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDsyrkx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsyrkx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    double *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCsyrkx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyrkx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZsyrkx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsyrkx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCherkx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCherkx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZherkx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZherkx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSsymm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsymm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    float *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDsymm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsymm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    double *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCsymm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsymm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZsymm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsymm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasChemm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasChemm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasChemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZhemm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZhemm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZhemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasStrsm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStrsm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDtrsm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtrsm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCtrsm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtrsm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZtrsm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtrsm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasStrmm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStrmm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    float *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDtrmm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtrmm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    double *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCtrmm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtrmm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZtrmm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtrmm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasHgemmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasHgemmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    __half *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    __half *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    __half *Barray = nullptr;
    rpc_read(client, &Barray, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    __half *beta;
    rpc_read(client, &beta, sizeof(beta));
    __half *Carray = nullptr;
    rpc_read(client, &Carray, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, (const __half *const *)Aarray, lda, (const __half *const *)Barray, ldb, beta, (__half *const *)Carray, ldc, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSgemmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgemmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *Barray = nullptr;
    rpc_read(client, &Barray, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    float *Carray = nullptr;
    rpc_read(client, &Carray, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, (const float *const *)Aarray, lda, (const float *const *)Barray, ldb, beta, (float *const *)Carray, ldc, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDgemmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgemmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *Barray = nullptr;
    rpc_read(client, &Barray, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    double *Carray = nullptr;
    rpc_read(client, &Carray, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, (const double *const *)Aarray, lda, (const double *const *)Barray, ldb, beta, (double *const *)Carray, ldc, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgemmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *Barray = nullptr;
    rpc_read(client, &Barray, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *Carray = nullptr;
    rpc_read(client, &Carray, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemmBatched(handle, transa, transb, m, n, k, alpha, (const cuComplex *const *)Aarray, lda, (const cuComplex *const *)Barray, ldb, beta, (cuComplex *const *)Carray, ldc, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgemm3mBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemm3mBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *Barray = nullptr;
    rpc_read(client, &Barray, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *Carray = nullptr;
    rpc_read(client, &Carray, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemm3mBatched(handle, transa, transb, m, n, k, alpha, (const cuComplex *const *)Aarray, lda, (const cuComplex *const *)Barray, ldb, beta, (cuComplex *const *)Carray, ldc, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgemmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgemmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *Barray = nullptr;
    rpc_read(client, &Barray, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *Carray = nullptr;
    rpc_read(client, &Carray, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgemmBatched(handle, transa, transb, m, n, k, alpha, (const cuDoubleComplex *const *)Aarray, lda, (const cuDoubleComplex *const *)Barray, ldb, beta, (cuDoubleComplex *const *)Carray, ldc, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSgemmStridedBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgemmStridedBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    long long int strideA;
    rpc_read(client, &strideA, sizeof(strideA));
    float *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    long long int strideB;
    rpc_read(client, &strideB, sizeof(strideB));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    float *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    long long int strideC;
    rpc_read(client, &strideC, sizeof(strideC));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDgemmStridedBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgemmStridedBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    long long int strideA;
    rpc_read(client, &strideA, sizeof(strideA));
    double *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    long long int strideB;
    rpc_read(client, &strideB, sizeof(strideB));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    double *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    long long int strideC;
    rpc_read(client, &strideC, sizeof(strideC));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgemmStridedBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemmStridedBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    long long int strideA;
    rpc_read(client, &strideA, sizeof(strideA));
    cuComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    long long int strideB;
    rpc_read(client, &strideB, sizeof(strideB));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    long long int strideC;
    rpc_read(client, &strideC, sizeof(strideC));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgemm3mStridedBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemm3mStridedBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    long long int strideA;
    rpc_read(client, &strideA, sizeof(strideA));
    cuComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    long long int strideB;
    rpc_read(client, &strideB, sizeof(strideB));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    long long int strideC;
    rpc_read(client, &strideC, sizeof(strideC));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemm3mStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgemmStridedBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgemmStridedBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    long long int strideA;
    rpc_read(client, &strideA, sizeof(strideA));
    cuDoubleComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    long long int strideB;
    rpc_read(client, &strideB, sizeof(strideB));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    long long int strideC;
    rpc_read(client, &strideC, sizeof(strideC));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasHgemmStridedBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasHgemmStridedBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int k;
    rpc_read(client, &k, sizeof(k));
    __half *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    __half *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    long long int strideA;
    rpc_read(client, &strideA, sizeof(strideA));
    __half *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    long long int strideB;
    rpc_read(client, &strideB, sizeof(strideB));
    __half *beta;
    rpc_read(client, &beta, sizeof(beta));
    __half *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    long long int strideC;
    rpc_read(client, &strideC, sizeof(strideC));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSgeam(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgeam called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *beta;
    rpc_read(client, &beta, sizeof(beta));
    float *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    float *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDgeam(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgeam called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *beta;
    rpc_read(client, &beta, sizeof(beta));
    double *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    double *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgeam(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgeam called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgeam(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgeam called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t transa;
    rpc_read(client, &transa, sizeof(transa));
    cublasOperation_t transb;
    rpc_read(client, &transb, sizeof(transb));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *beta;
    rpc_read(client, &beta, sizeof(beta));
    cuDoubleComplex *B;
    rpc_read(client, &B, sizeof(B));
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSgetrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgetrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    int *P;
    rpc_read(client, &P, sizeof(P));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgetrfBatched(handle, n, (float *const *)A, lda, P, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDgetrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgetrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    int *P;
    rpc_read(client, &P, sizeof(P));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgetrfBatched(handle, n, (double *const *)A, lda, P, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgetrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgetrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    int *P;
    rpc_read(client, &P, sizeof(P));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgetrfBatched(handle, n, (cuComplex *const *)A, lda, P, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgetrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgetrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    int *P;
    rpc_read(client, &P, sizeof(P));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgetrfBatched(handle, n, (cuDoubleComplex *const *)A, lda, P, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSgetriBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgetriBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    int *P;
    rpc_read(client, &P, sizeof(P));
    float *C = nullptr;
    rpc_read(client, &C, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgetriBatched(handle, n, (const float *const *)A, lda, P, (float *const *)C, ldc, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDgetriBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgetriBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    int *P;
    rpc_read(client, &P, sizeof(P));
    double *C = nullptr;
    rpc_read(client, &C, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgetriBatched(handle, n, (const double *const *)A, lda, P, (double *const *)C, ldc, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgetriBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgetriBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    int *P;
    rpc_read(client, &P, sizeof(P));
    cuComplex *C = nullptr;
    rpc_read(client, &C, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgetriBatched(handle, n, (const cuComplex *const *)A, lda, P, (cuComplex *const *)C, ldc, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgetriBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgetriBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    int *P;
    rpc_read(client, &P, sizeof(P));
    cuDoubleComplex *C = nullptr;
    rpc_read(client, &C, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgetriBatched(handle, n, (const cuDoubleComplex *const *)A, lda, P, (cuDoubleComplex *const *)C, ldc, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSgetrsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgetrsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int nrhs;
    rpc_read(client, &nrhs, sizeof(nrhs));
    float *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    int *devIpiv;
    rpc_read(client, &devIpiv, sizeof(devIpiv));
    float *Barray = nullptr;
    rpc_read(client, &Barray, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgetrsBatched(handle, trans, n, nrhs, (const float *const *)Aarray, lda, devIpiv, (float *const *)Barray, ldb, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDgetrsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgetrsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int nrhs;
    rpc_read(client, &nrhs, sizeof(nrhs));
    double *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    int *devIpiv;
    rpc_read(client, &devIpiv, sizeof(devIpiv));
    double *Barray = nullptr;
    rpc_read(client, &Barray, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgetrsBatched(handle, trans, n, nrhs, (const double *const *)Aarray, lda, devIpiv, (double *const *)Barray, ldb, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgetrsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgetrsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int nrhs;
    rpc_read(client, &nrhs, sizeof(nrhs));
    cuComplex *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    int *devIpiv;
    rpc_read(client, &devIpiv, sizeof(devIpiv));
    cuComplex *Barray = nullptr;
    rpc_read(client, &Barray, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgetrsBatched(handle, trans, n, nrhs, (const cuComplex *const *)Aarray, lda, devIpiv, (cuComplex *const *)Barray, ldb, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgetrsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgetrsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int n;
    rpc_read(client, &n, sizeof(n));
    int nrhs;
    rpc_read(client, &nrhs, sizeof(nrhs));
    cuDoubleComplex *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    int *devIpiv;
    rpc_read(client, &devIpiv, sizeof(devIpiv));
    cuDoubleComplex *Barray = nullptr;
    rpc_read(client, &Barray, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgetrsBatched(handle, trans, n, nrhs, (const cuDoubleComplex *const *)Aarray, lda, devIpiv, (cuDoubleComplex *const *)Barray, ldb, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasStrsmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStrsmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    float *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *B = nullptr;
    rpc_read(client, &B, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, (const float *const *)A, lda, (float *const *)B, ldb, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDtrsmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtrsmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    double *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *B = nullptr;
    rpc_read(client, &B, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, (const double *const *)A, lda, (double *const *)B, ldb, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCtrsmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtrsmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *B = nullptr;
    rpc_read(client, &B, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, (const cuComplex *const *)A, lda, (cuComplex *const *)B, ldb, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZtrsmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtrsmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t side;
    rpc_read(client, &side, sizeof(side));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    cublasDiagType_t diag;
    rpc_read(client, &diag, sizeof(diag));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *alpha;
    rpc_read(client, &alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *B = nullptr;
    rpc_read(client, &B, 0, true);
    int ldb;
    rpc_read(client, &ldb, sizeof(ldb));
    int batchCount;
    rpc_read(client, &batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, (const cuDoubleComplex *const *)A, lda, (cuDoubleComplex *const *)B, ldb, batchCount);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSmatinvBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSmatinvBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *Ainv = nullptr;
    rpc_read(client, &Ainv, 0, true);
    int lda_inv;
    rpc_read(client, &lda_inv, sizeof(lda_inv));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSmatinvBatched(handle, n, (const float *const *)A, lda, (float *const *)Ainv, lda_inv, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDmatinvBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDmatinvBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *Ainv = nullptr;
    rpc_read(client, &Ainv, 0, true);
    int lda_inv;
    rpc_read(client, &lda_inv, sizeof(lda_inv));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDmatinvBatched(handle, n, (const double *const *)A, lda, (double *const *)Ainv, lda_inv, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCmatinvBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCmatinvBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *Ainv = nullptr;
    rpc_read(client, &Ainv, 0, true);
    int lda_inv;
    rpc_read(client, &lda_inv, sizeof(lda_inv));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCmatinvBatched(handle, n, (const cuComplex *const *)A, lda, (cuComplex *const *)Ainv, lda_inv, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZmatinvBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZmatinvBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *A = nullptr;
    rpc_read(client, &A, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *Ainv = nullptr;
    rpc_read(client, &Ainv, 0, true);
    int lda_inv;
    rpc_read(client, &lda_inv, sizeof(lda_inv));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZmatinvBatched(handle, n, (const cuDoubleComplex *const *)A, lda, (cuDoubleComplex *const *)Ainv, lda_inv, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSgeqrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgeqrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *TauArray = nullptr;
    rpc_read(client, &TauArray, 0, true);
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgeqrfBatched(handle, m, n, (float *const *)Aarray, lda, (float *const *)TauArray, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDgeqrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgeqrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *TauArray = nullptr;
    rpc_read(client, &TauArray, 0, true);
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgeqrfBatched(handle, m, n, (double *const *)Aarray, lda, (double *const *)TauArray, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgeqrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgeqrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *TauArray = nullptr;
    rpc_read(client, &TauArray, 0, true);
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgeqrfBatched(handle, m, n, (cuComplex *const *)Aarray, lda, (cuComplex *const *)TauArray, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgeqrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgeqrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *TauArray = nullptr;
    rpc_read(client, &TauArray, 0, true);
    int *info;
    rpc_read(client, &info, sizeof(info));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgeqrfBatched(handle, m, n, (cuDoubleComplex *const *)Aarray, lda, (cuDoubleComplex *const *)TauArray, info, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSgelsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgelsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int nrhs;
    rpc_read(client, &nrhs, sizeof(nrhs));
    float *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *Carray = nullptr;
    rpc_read(client, &Carray, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int *devInfoArray;
    rpc_read(client, &devInfoArray, sizeof(devInfoArray));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgelsBatched(handle, trans, m, n, nrhs, (float *const *)Aarray, lda, (float *const *)Carray, ldc, info, devInfoArray, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDgelsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgelsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int nrhs;
    rpc_read(client, &nrhs, sizeof(nrhs));
    double *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *Carray = nullptr;
    rpc_read(client, &Carray, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int *devInfoArray;
    rpc_read(client, &devInfoArray, sizeof(devInfoArray));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgelsBatched(handle, trans, m, n, nrhs, (double *const *)Aarray, lda, (double *const *)Carray, ldc, info, devInfoArray, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCgelsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgelsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int nrhs;
    rpc_read(client, &nrhs, sizeof(nrhs));
    cuComplex *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *Carray = nullptr;
    rpc_read(client, &Carray, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int *devInfoArray;
    rpc_read(client, &devInfoArray, sizeof(devInfoArray));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgelsBatched(handle, trans, m, n, nrhs, (cuComplex *const *)Aarray, lda, (cuComplex *const *)Carray, ldc, info, devInfoArray, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZgelsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgelsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasOperation_t trans;
    rpc_read(client, &trans, sizeof(trans));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    int nrhs;
    rpc_read(client, &nrhs, sizeof(nrhs));
    cuDoubleComplex *Aarray = nullptr;
    rpc_read(client, &Aarray, 0, true);
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *Carray = nullptr;
    rpc_read(client, &Carray, 0, true);
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    int *info;
    rpc_read(client, &info, sizeof(info));
    int *devInfoArray;
    rpc_read(client, &devInfoArray, sizeof(devInfoArray));
    int batchSize;
    rpc_read(client, &batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgelsBatched(handle, trans, m, n, nrhs, (cuDoubleComplex *const *)Aarray, lda, (cuDoubleComplex *const *)Carray, ldc, info, devInfoArray, batchSize);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasSdgmm(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSdgmm called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t mode;
    rpc_read(client, &mode, sizeof(mode));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    float *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDdgmm(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDdgmm called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t mode;
    rpc_read(client, &mode, sizeof(mode));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    double *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCdgmm(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCdgmm called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t mode;
    rpc_read(client, &mode, sizeof(mode));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZdgmm(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZdgmm called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasSideMode_t mode;
    rpc_read(client, &mode, sizeof(mode));
    int m;
    rpc_read(client, &m, sizeof(m));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *x;
    rpc_read(client, &x, sizeof(x));
    int incx;
    rpc_read(client, &incx, sizeof(incx));
    cuDoubleComplex *C;
    rpc_read(client, &C, sizeof(C));
    int ldc;
    rpc_read(client, &ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasStpttr(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStpttr called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *AP;
    rpc_read(client, &AP, sizeof(AP));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStpttr(handle, uplo, n, AP, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDtpttr(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtpttr called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *AP;
    rpc_read(client, &AP, sizeof(AP));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtpttr(handle, uplo, n, AP, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCtpttr(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtpttr called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtpttr(handle, uplo, n, AP, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZtpttr(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtpttr called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtpttr(handle, uplo, n, AP, A, lda);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasStrttp(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStrttp called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    float *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    float *AP;
    rpc_read(client, &AP, sizeof(AP));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStrttp(handle, uplo, n, A, lda, AP);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasDtrttp(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtrttp called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    double *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    double *AP;
    rpc_read(client, &AP, sizeof(AP));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtrttp(handle, uplo, n, A, lda, AP);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasCtrttp(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtrttp called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtrttp(handle, uplo, n, A, lda, AP);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

int handle_cublasZtrttp(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtrttp called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcClient *client = (RpcClient *)args0;
    cublasHandle_t handle;
    rpc_read(client, &handle, sizeof(handle));
    cublasFillMode_t uplo;
    rpc_read(client, &uplo, sizeof(uplo));
    int n;
    rpc_read(client, &n, sizeof(n));
    cuDoubleComplex *A;
    rpc_read(client, &A, sizeof(A));
    int lda;
    rpc_read(client, &lda, sizeof(lda));
    cuDoubleComplex *AP;
    rpc_read(client, &AP, sizeof(AP));
    cublasStatus_t _result;
    if(rpc_prepare_response(client) != 0) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtrttp(handle, uplo, n, A, lda, AP);
    rpc_write(client, &_result, sizeof(_result));
    if(rpc_submit_response(client) != 0) {
        std::cerr << "Failed to submit response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }

_RTN_:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        ::free(*it);
    }
    return rtn;
}

