#include <iostream>
#include <map>
#include <string.h>
#include "hook_api.h"
#include "handle_server.h"
#include "rpc/rpc_core.h"
#include "cublas_api.h"

using namespace rpc;
int handle_cublasCreate_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCreate_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t *handle;
    conn->read(&handle, sizeof(handle));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCreate_v2(handle);
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

int handle_cublasDestroy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDestroy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDestroy_v2(handle);
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

int handle_cublasGetVersion_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetVersion_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int *version;
    conn->read(&version, sizeof(version));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetVersion_v2(handle, version);
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

int handle_cublasGetProperty(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetProperty called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    libraryPropertyType type;
    conn->read(&type, sizeof(type));
    int *value;
    conn->read(&value, sizeof(value));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetProperty(type, value);
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

int handle_cublasGetCudartVersion(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetCudartVersion called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    size_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetCudartVersion();
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

int handle_cublasSetWorkspace_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetWorkspace_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    void *workspace;
    conn->read(&workspace, sizeof(workspace));
    size_t workspaceSizeInBytes;
    conn->read(&workspaceSizeInBytes, sizeof(workspaceSizeInBytes));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetWorkspace_v2(handle, workspace, workspaceSizeInBytes);
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

int handle_cublasSetStream_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetStream_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cudaStream_t streamId;
    conn->read(&streamId, sizeof(streamId));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetStream_v2(handle, streamId);
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

int handle_cublasGetStream_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetStream_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cudaStream_t *streamId;
    conn->read(&streamId, sizeof(streamId));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetStream_v2(handle, streamId);
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

int handle_cublasGetPointerMode_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetPointerMode_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasPointerMode_t *mode;
    conn->read(&mode, sizeof(mode));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetPointerMode_v2(handle, mode);
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

int handle_cublasSetPointerMode_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetPointerMode_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasPointerMode_t mode;
    conn->read(&mode, sizeof(mode));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetPointerMode_v2(handle, mode);
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

int handle_cublasGetAtomicsMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetAtomicsMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasAtomicsMode_t *mode;
    conn->read(&mode, sizeof(mode));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetAtomicsMode(handle, mode);
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

int handle_cublasSetAtomicsMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetAtomicsMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasAtomicsMode_t mode;
    conn->read(&mode, sizeof(mode));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetAtomicsMode(handle, mode);
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

int handle_cublasGetMathMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetMathMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasMath_t *mode;
    conn->read(&mode, sizeof(mode));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetMathMode(handle, mode);
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

int handle_cublasSetMathMode(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetMathMode called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasMath_t mode;
    conn->read(&mode, sizeof(mode));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetMathMode(handle, mode);
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

int handle_cublasGetSmCountTarget(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetSmCountTarget called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int *smCountTarget;
    conn->read(&smCountTarget, sizeof(smCountTarget));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetSmCountTarget(handle, smCountTarget);
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

int handle_cublasSetSmCountTarget(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetSmCountTarget called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int smCountTarget;
    conn->read(&smCountTarget, sizeof(smCountTarget));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetSmCountTarget(handle, smCountTarget);
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

int handle_cublasLoggerConfigure(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasLoggerConfigure called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int logIsOn;
    conn->read(&logIsOn, sizeof(logIsOn));
    int logToStdOut;
    conn->read(&logToStdOut, sizeof(logToStdOut));
    int logToStdErr;
    conn->read(&logToStdErr, sizeof(logToStdErr));
    char *logFileName = nullptr;
    conn->read(&logFileName, 0, true);
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(logFileName);
    _result = cublasLoggerConfigure(logIsOn, logToStdOut, logToStdErr, logFileName);
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

int handle_cublasSetLoggerCallback(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetLoggerCallback called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasLogCallback userCallback;
    conn->read(&userCallback, sizeof(userCallback));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetLoggerCallback(userCallback);
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

int handle_cublasGetLoggerCallback(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetLoggerCallback called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasLogCallback *userCallback;
    conn->read(&userCallback, sizeof(userCallback));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetLoggerCallback(userCallback);
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

int handle_cublasSetVector(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetVector called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int n;
    conn->read(&n, sizeof(n));
    int elemSize;
    conn->read(&elemSize, sizeof(elemSize));
    void *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    void *devicePtr;
    conn->read(&devicePtr, sizeof(devicePtr));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetVector(n, elemSize, x, incx, devicePtr, incy);
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

int handle_cublasGetVector(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetVector called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int n;
    conn->read(&n, sizeof(n));
    int elemSize;
    conn->read(&elemSize, sizeof(elemSize));
    void *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    void *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetVector(n, elemSize, x, incx, y, incy);
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

int handle_cublasSetMatrix(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetMatrix called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int rows;
    conn->read(&rows, sizeof(rows));
    int cols;
    conn->read(&cols, sizeof(cols));
    int elemSize;
    conn->read(&elemSize, sizeof(elemSize));
    void *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    void *B;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb);
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

int handle_cublasGetMatrix(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetMatrix called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int rows;
    conn->read(&rows, sizeof(rows));
    int cols;
    conn->read(&cols, sizeof(cols));
    int elemSize;
    conn->read(&elemSize, sizeof(elemSize));
    void *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    void *B;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb);
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

int handle_cublasSetVectorAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetVectorAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int n;
    conn->read(&n, sizeof(n));
    int elemSize;
    conn->read(&elemSize, sizeof(elemSize));
    void *hostPtr;
    conn->read(&hostPtr, sizeof(hostPtr));
    int incx;
    conn->read(&incx, sizeof(incx));
    void *devicePtr;
    conn->read(&devicePtr, sizeof(devicePtr));
    int incy;
    conn->read(&incy, sizeof(incy));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream);
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

int handle_cublasGetVectorAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetVectorAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int n;
    conn->read(&n, sizeof(n));
    int elemSize;
    conn->read(&elemSize, sizeof(elemSize));
    void *devicePtr;
    conn->read(&devicePtr, sizeof(devicePtr));
    int incx;
    conn->read(&incx, sizeof(incx));
    void *hostPtr;
    conn->read(&hostPtr, sizeof(hostPtr));
    int incy;
    conn->read(&incy, sizeof(incy));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream);
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

int handle_cublasSetMatrixAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSetMatrixAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int rows;
    conn->read(&rows, sizeof(rows));
    int cols;
    conn->read(&cols, sizeof(cols));
    int elemSize;
    conn->read(&elemSize, sizeof(elemSize));
    void *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    void *B;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
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

int handle_cublasGetMatrixAsync(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasGetMatrixAsync called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    int rows;
    conn->read(&rows, sizeof(rows));
    int cols;
    conn->read(&cols, sizeof(cols));
    int elemSize;
    conn->read(&elemSize, sizeof(elemSize));
    void *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    void *B;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cudaStream_t stream;
    conn->read(&stream, sizeof(stream));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
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

int handle_cublasXerbla(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasXerbla called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    char *srName = nullptr;
    conn->read(&srName, 0, true);
    int info;
    conn->read(&info, sizeof(info));
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    buffers.insert(srName);
    cublasXerbla(srName, info);
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

int handle_cublasNrm2Ex(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasNrm2Ex called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    void *x;
    conn->read(&x, sizeof(x));
    cudaDataType xType;
    conn->read(&xType, sizeof(xType));
    int incx;
    conn->read(&incx, sizeof(incx));
    void *result;
    conn->read(&result, sizeof(result));
    cudaDataType resultType;
    conn->read(&resultType, sizeof(resultType));
    cudaDataType executionType;
    conn->read(&executionType, sizeof(executionType));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasNrm2Ex(handle, n, x, xType, incx, result, resultType, executionType);
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

int handle_cublasSnrm2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSnrm2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSnrm2_v2(handle, n, x, incx, result);
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

int handle_cublasDnrm2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDnrm2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDnrm2_v2(handle, n, x, incx, result);
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

int handle_cublasScnrm2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasScnrm2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasScnrm2_v2(handle, n, x, incx, result);
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

int handle_cublasDznrm2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDznrm2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDznrm2_v2(handle, n, x, incx, result);
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

int handle_cublasDotEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDotEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    void *x;
    conn->read(&x, sizeof(x));
    cudaDataType xType;
    conn->read(&xType, sizeof(xType));
    int incx;
    conn->read(&incx, sizeof(incx));
    void *y;
    conn->read(&y, sizeof(y));
    cudaDataType yType;
    conn->read(&yType, sizeof(yType));
    int incy;
    conn->read(&incy, sizeof(incy));
    void *result;
    conn->read(&result, sizeof(result));
    cudaDataType resultType;
    conn->read(&resultType, sizeof(resultType));
    cudaDataType executionType;
    conn->read(&executionType, sizeof(executionType));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDotEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
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

int handle_cublasDotcEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDotcEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    void *x;
    conn->read(&x, sizeof(x));
    cudaDataType xType;
    conn->read(&xType, sizeof(xType));
    int incx;
    conn->read(&incx, sizeof(incx));
    void *y;
    conn->read(&y, sizeof(y));
    cudaDataType yType;
    conn->read(&yType, sizeof(yType));
    int incy;
    conn->read(&incy, sizeof(incy));
    void *result;
    conn->read(&result, sizeof(result));
    cudaDataType resultType;
    conn->read(&resultType, sizeof(resultType));
    cudaDataType executionType;
    conn->read(&executionType, sizeof(executionType));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDotcEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
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

int handle_cublasSdot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSdot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    float *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSdot_v2(handle, n, x, incx, y, incy, result);
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

int handle_cublasDdot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDdot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    double *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDdot_v2(handle, n, x, incx, y, incy, result);
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

int handle_cublasCdotu_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCdotu_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuComplex *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCdotu_v2(handle, n, x, incx, y, incy, result);
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

int handle_cublasCdotc_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCdotc_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuComplex *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCdotc_v2(handle, n, x, incx, y, incy, result);
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

int handle_cublasZdotu_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZdotu_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuDoubleComplex *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZdotu_v2(handle, n, x, incx, y, incy, result);
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

int handle_cublasZdotc_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZdotc_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuDoubleComplex *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZdotc_v2(handle, n, x, incx, y, incy, result);
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

int handle_cublasScalEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasScalEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    void *alpha;
    conn->read(&alpha, sizeof(alpha));
    cudaDataType alphaType;
    conn->read(&alphaType, sizeof(alphaType));
    void *x;
    conn->read(&x, sizeof(x));
    cudaDataType xType;
    conn->read(&xType, sizeof(xType));
    int incx;
    conn->read(&incx, sizeof(incx));
    cudaDataType executionType;
    conn->read(&executionType, sizeof(executionType));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasScalEx(handle, n, alpha, alphaType, x, xType, incx, executionType);
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

int handle_cublasSscal_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSscal_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSscal_v2(handle, n, alpha, x, incx);
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

int handle_cublasDscal_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDscal_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDscal_v2(handle, n, alpha, x, incx);
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

int handle_cublasCscal_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCscal_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCscal_v2(handle, n, alpha, x, incx);
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

int handle_cublasCsscal_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsscal_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsscal_v2(handle, n, alpha, x, incx);
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

int handle_cublasZscal_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZscal_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZscal_v2(handle, n, alpha, x, incx);
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

int handle_cublasZdscal_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZdscal_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZdscal_v2(handle, n, alpha, x, incx);
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

int handle_cublasAxpyEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasAxpyEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    void *alpha;
    conn->read(&alpha, sizeof(alpha));
    cudaDataType alphaType;
    conn->read(&alphaType, sizeof(alphaType));
    void *x;
    conn->read(&x, sizeof(x));
    cudaDataType xType;
    conn->read(&xType, sizeof(xType));
    int incx;
    conn->read(&incx, sizeof(incx));
    void *y;
    conn->read(&y, sizeof(y));
    cudaDataType yType;
    conn->read(&yType, sizeof(yType));
    int incy;
    conn->read(&incy, sizeof(incy));
    cudaDataType executiontype;
    conn->read(&executiontype, sizeof(executiontype));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasAxpyEx(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype);
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

int handle_cublasSaxpy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSaxpy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy);
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

int handle_cublasDaxpy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDaxpy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy);
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

int handle_cublasCaxpy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCaxpy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCaxpy_v2(handle, n, alpha, x, incx, y, incy);
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

int handle_cublasZaxpy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZaxpy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZaxpy_v2(handle, n, alpha, x, incx, y, incy);
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

int handle_cublasCopyEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCopyEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    void *x;
    conn->read(&x, sizeof(x));
    cudaDataType xType;
    conn->read(&xType, sizeof(xType));
    int incx;
    conn->read(&incx, sizeof(incx));
    void *y;
    conn->read(&y, sizeof(y));
    cudaDataType yType;
    conn->read(&yType, sizeof(yType));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCopyEx(handle, n, x, xType, incx, y, yType, incy);
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

int handle_cublasScopy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasScopy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasScopy_v2(handle, n, x, incx, y, incy);
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

int handle_cublasDcopy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDcopy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDcopy_v2(handle, n, x, incx, y, incy);
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

int handle_cublasCcopy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCcopy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCcopy_v2(handle, n, x, incx, y, incy);
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

int handle_cublasZcopy_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZcopy_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZcopy_v2(handle, n, x, incx, y, incy);
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

int handle_cublasSswap_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSswap_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSswap_v2(handle, n, x, incx, y, incy);
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

int handle_cublasDswap_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDswap_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDswap_v2(handle, n, x, incx, y, incy);
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

int handle_cublasCswap_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCswap_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCswap_v2(handle, n, x, incx, y, incy);
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

int handle_cublasZswap_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZswap_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZswap_v2(handle, n, x, incx, y, incy);
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

int handle_cublasSwapEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSwapEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    void *x;
    conn->read(&x, sizeof(x));
    cudaDataType xType;
    conn->read(&xType, sizeof(xType));
    int incx;
    conn->read(&incx, sizeof(incx));
    void *y;
    conn->read(&y, sizeof(y));
    cudaDataType yType;
    conn->read(&yType, sizeof(yType));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSwapEx(handle, n, x, xType, incx, y, yType, incy);
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

int handle_cublasIsamax_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIsamax_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    int *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIsamax_v2(handle, n, x, incx, result);
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

int handle_cublasIdamax_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIdamax_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    int *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIdamax_v2(handle, n, x, incx, result);
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

int handle_cublasIcamax_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIcamax_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    int *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIcamax_v2(handle, n, x, incx, result);
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

int handle_cublasIzamax_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIzamax_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    int *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIzamax_v2(handle, n, x, incx, result);
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

int handle_cublasIamaxEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIamaxEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    void *x;
    conn->read(&x, sizeof(x));
    cudaDataType xType;
    conn->read(&xType, sizeof(xType));
    int incx;
    conn->read(&incx, sizeof(incx));
    int *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIamaxEx(handle, n, x, xType, incx, result);
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

int handle_cublasIsamin_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIsamin_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    int *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIsamin_v2(handle, n, x, incx, result);
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

int handle_cublasIdamin_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIdamin_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    int *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIdamin_v2(handle, n, x, incx, result);
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

int handle_cublasIcamin_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIcamin_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    int *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIcamin_v2(handle, n, x, incx, result);
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

int handle_cublasIzamin_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIzamin_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    int *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIzamin_v2(handle, n, x, incx, result);
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

int handle_cublasIaminEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasIaminEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    void *x;
    conn->read(&x, sizeof(x));
    cudaDataType xType;
    conn->read(&xType, sizeof(xType));
    int incx;
    conn->read(&incx, sizeof(incx));
    int *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasIaminEx(handle, n, x, xType, incx, result);
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

int handle_cublasAsumEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasAsumEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    void *x;
    conn->read(&x, sizeof(x));
    cudaDataType xType;
    conn->read(&xType, sizeof(xType));
    int incx;
    conn->read(&incx, sizeof(incx));
    void *result;
    conn->read(&result, sizeof(result));
    cudaDataType resultType;
    conn->read(&resultType, sizeof(resultType));
    cudaDataType executiontype;
    conn->read(&executiontype, sizeof(executiontype));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasAsumEx(handle, n, x, xType, incx, result, resultType, executiontype);
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

int handle_cublasSasum_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSasum_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSasum_v2(handle, n, x, incx, result);
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

int handle_cublasDasum_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDasum_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDasum_v2(handle, n, x, incx, result);
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

int handle_cublasScasum_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasScasum_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasScasum_v2(handle, n, x, incx, result);
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

int handle_cublasDzasum_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDzasum_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *result;
    conn->read(&result, sizeof(result));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDzasum_v2(handle, n, x, incx, result);
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

int handle_cublasSrot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSrot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    float *c = nullptr;
    conn->read(&c, sizeof(c));
    float *s = nullptr;
    conn->read(&s, sizeof(s));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSrot_v2(handle, n, x, incx, y, incy, c, s);
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

int handle_cublasDrot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDrot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    double *c = nullptr;
    conn->read(&c, sizeof(c));
    double *s = nullptr;
    conn->read(&s, sizeof(s));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDrot_v2(handle, n, x, incx, y, incy, c, s);
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

int handle_cublasCrot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCrot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    float *c = nullptr;
    conn->read(&c, sizeof(c));
    cuComplex *s = nullptr;
    conn->read(&s, sizeof(s));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCrot_v2(handle, n, x, incx, y, incy, c, s);
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

int handle_cublasCsrot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsrot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    float *c = nullptr;
    conn->read(&c, sizeof(c));
    float *s = nullptr;
    conn->read(&s, sizeof(s));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsrot_v2(handle, n, x, incx, y, incy, c, s);
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

int handle_cublasZrot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZrot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    double *c = nullptr;
    conn->read(&c, sizeof(c));
    cuDoubleComplex *s = nullptr;
    conn->read(&s, sizeof(s));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZrot_v2(handle, n, x, incx, y, incy, c, s);
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

int handle_cublasZdrot_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZdrot_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    double *c = nullptr;
    conn->read(&c, sizeof(c));
    double *s = nullptr;
    conn->read(&s, sizeof(s));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZdrot_v2(handle, n, x, incx, y, incy, c, s);
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

int handle_cublasRotEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasRotEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    void *x;
    conn->read(&x, sizeof(x));
    cudaDataType xType;
    conn->read(&xType, sizeof(xType));
    int incx;
    conn->read(&incx, sizeof(incx));
    void *y;
    conn->read(&y, sizeof(y));
    cudaDataType yType;
    conn->read(&yType, sizeof(yType));
    int incy;
    conn->read(&incy, sizeof(incy));
    void *c;
    conn->read(&c, sizeof(c));
    void *s;
    conn->read(&s, sizeof(s));
    cudaDataType csType;
    conn->read(&csType, sizeof(csType));
    cudaDataType executiontype;
    conn->read(&executiontype, sizeof(executiontype));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasRotEx(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype);
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

int handle_cublasSrotg_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSrotg_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    float *a;
    conn->read(&a, sizeof(a));
    float *b;
    conn->read(&b, sizeof(b));
    float *c;
    conn->read(&c, sizeof(c));
    float *s;
    conn->read(&s, sizeof(s));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSrotg_v2(handle, a, b, c, s);
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

int handle_cublasDrotg_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDrotg_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    double *a;
    conn->read(&a, sizeof(a));
    double *b;
    conn->read(&b, sizeof(b));
    double *c;
    conn->read(&c, sizeof(c));
    double *s;
    conn->read(&s, sizeof(s));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDrotg_v2(handle, a, b, c, s);
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

int handle_cublasCrotg_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCrotg_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cuComplex *a;
    conn->read(&a, sizeof(a));
    cuComplex *b;
    conn->read(&b, sizeof(b));
    float *c;
    conn->read(&c, sizeof(c));
    cuComplex *s;
    conn->read(&s, sizeof(s));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCrotg_v2(handle, a, b, c, s);
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

int handle_cublasZrotg_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZrotg_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cuDoubleComplex *a;
    conn->read(&a, sizeof(a));
    cuDoubleComplex *b;
    conn->read(&b, sizeof(b));
    double *c;
    conn->read(&c, sizeof(c));
    cuDoubleComplex *s;
    conn->read(&s, sizeof(s));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZrotg_v2(handle, a, b, c, s);
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

int handle_cublasRotgEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasRotgEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    void *a;
    conn->read(&a, sizeof(a));
    void *b;
    conn->read(&b, sizeof(b));
    cudaDataType abType;
    conn->read(&abType, sizeof(abType));
    void *c;
    conn->read(&c, sizeof(c));
    void *s;
    conn->read(&s, sizeof(s));
    cudaDataType csType;
    conn->read(&csType, sizeof(csType));
    cudaDataType executiontype;
    conn->read(&executiontype, sizeof(executiontype));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasRotgEx(handle, a, b, abType, c, s, csType, executiontype);
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

int handle_cublasSrotm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSrotm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    float *param = nullptr;
    conn->read(&param, sizeof(param));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSrotm_v2(handle, n, x, incx, y, incy, param);
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

int handle_cublasDrotm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDrotm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    double *param = nullptr;
    conn->read(&param, sizeof(param));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDrotm_v2(handle, n, x, incx, y, incy, param);
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

int handle_cublasRotmEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasRotmEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    void *x;
    conn->read(&x, sizeof(x));
    cudaDataType xType;
    conn->read(&xType, sizeof(xType));
    int incx;
    conn->read(&incx, sizeof(incx));
    void *y;
    conn->read(&y, sizeof(y));
    cudaDataType yType;
    conn->read(&yType, sizeof(yType));
    int incy;
    conn->read(&incy, sizeof(incy));
    void *param;
    conn->read(&param, sizeof(param));
    cudaDataType paramType;
    conn->read(&paramType, sizeof(paramType));
    cudaDataType executiontype;
    conn->read(&executiontype, sizeof(executiontype));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasRotmEx(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype);
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

int handle_cublasSrotmg_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSrotmg_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    float *d1;
    conn->read(&d1, sizeof(d1));
    float *d2;
    conn->read(&d2, sizeof(d2));
    float *x1;
    conn->read(&x1, sizeof(x1));
    float *y1 = nullptr;
    conn->read(&y1, sizeof(y1));
    float *param;
    conn->read(&param, sizeof(param));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSrotmg_v2(handle, d1, d2, x1, y1, param);
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

int handle_cublasDrotmg_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDrotmg_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    double *d1;
    conn->read(&d1, sizeof(d1));
    double *d2;
    conn->read(&d2, sizeof(d2));
    double *x1;
    conn->read(&x1, sizeof(x1));
    double *y1 = nullptr;
    conn->read(&y1, sizeof(y1));
    double *param;
    conn->read(&param, sizeof(param));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDrotmg_v2(handle, d1, d2, x1, y1, param);
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

int handle_cublasRotmgEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasRotmgEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    void *d1;
    conn->read(&d1, sizeof(d1));
    cudaDataType d1Type;
    conn->read(&d1Type, sizeof(d1Type));
    void *d2;
    conn->read(&d2, sizeof(d2));
    cudaDataType d2Type;
    conn->read(&d2Type, sizeof(d2Type));
    void *x1;
    conn->read(&x1, sizeof(x1));
    cudaDataType x1Type;
    conn->read(&x1Type, sizeof(x1Type));
    void *y1;
    conn->read(&y1, sizeof(y1));
    cudaDataType y1Type;
    conn->read(&y1Type, sizeof(y1Type));
    void *param;
    conn->read(&param, sizeof(param));
    cudaDataType paramType;
    conn->read(&paramType, sizeof(paramType));
    cudaDataType executiontype;
    conn->read(&executiontype, sizeof(executiontype));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasRotmgEx(handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType, executiontype);
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

int handle_cublasSgemv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgemv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    float *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasDgemv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgemv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    double *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasCgemv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasZgemv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgemv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasSgbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int kl;
    conn->read(&kl, sizeof(kl));
    int ku;
    conn->read(&ku, sizeof(ku));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    float *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasDgbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int kl;
    conn->read(&kl, sizeof(kl));
    int ku;
    conn->read(&ku, sizeof(ku));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    double *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasCgbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int kl;
    conn->read(&kl, sizeof(kl));
    int ku;
    conn->read(&ku, sizeof(ku));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasZgbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int kl;
    conn->read(&kl, sizeof(kl));
    int ku;
    conn->read(&ku, sizeof(ku));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasStrmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStrmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
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

int handle_cublasDtrmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtrmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
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

int handle_cublasCtrmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtrmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
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

int handle_cublasZtrmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtrmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
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

int handle_cublasStbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
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

int handle_cublasDtbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
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

int handle_cublasCtbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
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

int handle_cublasZtbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
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

int handle_cublasStpmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStpmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    float *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    float *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
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

int handle_cublasDtpmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtpmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    double *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    double *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
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

int handle_cublasCtpmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtpmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    cuComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
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

int handle_cublasZtpmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtpmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    cuDoubleComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
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

int handle_cublasStrsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStrsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
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

int handle_cublasDtrsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtrsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
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

int handle_cublasCtrsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtrsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
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

int handle_cublasZtrsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtrsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
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

int handle_cublasStpsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStpsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    float *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    float *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
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

int handle_cublasDtpsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtpsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    double *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    double *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
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

int handle_cublasCtpsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtpsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    cuComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
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

int handle_cublasZtpsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtpsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    cuDoubleComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
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

int handle_cublasStbsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStbsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
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

int handle_cublasDtbsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtbsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
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

int handle_cublasCtbsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtbsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
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

int handle_cublasZtbsv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtbsv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *x;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
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

int handle_cublasSsymv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsymv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    float *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasDsymv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsymv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    double *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasCsymv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsymv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasZsymv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsymv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasChemv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasChemv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasChemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasZhemv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZhemv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZhemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasSsbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    float *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasDsbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    double *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasChbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasChbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasChbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasZhbmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZhbmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZhbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
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

int handle_cublasSspmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSspmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    float *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
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

int handle_cublasDspmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDspmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    double *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
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

int handle_cublasChpmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasChpmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasChpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
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

int handle_cublasZhpmv_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZhpmv_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *y;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZhpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
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

int handle_cublasSger_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSger_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    float *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
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

int handle_cublasDger_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDger_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    double *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
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

int handle_cublasCgeru_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgeru_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
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

int handle_cublasCgerc_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgerc_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
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

int handle_cublasZgeru_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgeru_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuDoubleComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
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

int handle_cublasZgerc_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgerc_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuDoubleComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
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

int handle_cublasSsyr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsyr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
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

int handle_cublasDsyr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsyr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
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

int handle_cublasCsyr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
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

int handle_cublasZsyr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsyr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
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

int handle_cublasCher_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCher_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCher_v2(handle, uplo, n, alpha, x, incx, A, lda);
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

int handle_cublasZher_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZher_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZher_v2(handle, uplo, n, alpha, x, incx, A, lda);
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

int handle_cublasSspr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSspr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *AP;
    conn->read(&AP, sizeof(AP));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSspr_v2(handle, uplo, n, alpha, x, incx, AP);
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

int handle_cublasDspr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDspr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *AP;
    conn->read(&AP, sizeof(AP));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDspr_v2(handle, uplo, n, alpha, x, incx, AP);
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

int handle_cublasChpr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasChpr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *AP;
    conn->read(&AP, sizeof(AP));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasChpr_v2(handle, uplo, n, alpha, x, incx, AP);
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

int handle_cublasZhpr_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZhpr_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *AP;
    conn->read(&AP, sizeof(AP));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZhpr_v2(handle, uplo, n, alpha, x, incx, AP);
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

int handle_cublasSsyr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsyr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    float *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
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

int handle_cublasDsyr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsyr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    double *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
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

int handle_cublasCsyr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
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

int handle_cublasZsyr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsyr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuDoubleComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
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

int handle_cublasCher2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCher2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
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

int handle_cublasZher2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZher2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuDoubleComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
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

int handle_cublasSspr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSspr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    float *AP;
    conn->read(&AP, sizeof(AP));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
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

int handle_cublasDspr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDspr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    double *AP;
    conn->read(&AP, sizeof(AP));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
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

int handle_cublasChpr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasChpr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuComplex *AP;
    conn->read(&AP, sizeof(AP));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasChpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
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

int handle_cublasZhpr2_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZhpr2_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *y = nullptr;
    conn->read(&y, sizeof(y));
    int incy;
    conn->read(&incy, sizeof(incy));
    cuDoubleComplex *AP;
    conn->read(&AP, sizeof(AP));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZhpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
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

int handle_cublasSgemm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgemm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    float *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasDgemm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgemm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    double *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasCgemm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasCgemm3m(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemm3m called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasCgemm3mEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemm3mEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    void *A;
    conn->read(&A, sizeof(A));
    cudaDataType Atype;
    conn->read(&Atype, sizeof(Atype));
    int lda;
    conn->read(&lda, sizeof(lda));
    void *B;
    conn->read(&B, sizeof(B));
    cudaDataType Btype;
    conn->read(&Btype, sizeof(Btype));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    void *C;
    conn->read(&C, sizeof(C));
    cudaDataType Ctype;
    conn->read(&Ctype, sizeof(Ctype));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemm3mEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
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

int handle_cublasZgemm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgemm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasZgemm3m(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgemm3m called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasHgemm(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasHgemm called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    __half *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    __half *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    __half *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    __half *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    __half *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasSgemmEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgemmEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    void *A;
    conn->read(&A, sizeof(A));
    cudaDataType Atype;
    conn->read(&Atype, sizeof(Atype));
    int lda;
    conn->read(&lda, sizeof(lda));
    void *B;
    conn->read(&B, sizeof(B));
    cudaDataType Btype;
    conn->read(&Btype, sizeof(Btype));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    void *C;
    conn->read(&C, sizeof(C));
    cudaDataType Ctype;
    conn->read(&Ctype, sizeof(Ctype));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
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

int handle_cublasCgemmEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemmEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    void *A;
    conn->read(&A, sizeof(A));
    cudaDataType Atype;
    conn->read(&Atype, sizeof(Atype));
    int lda;
    conn->read(&lda, sizeof(lda));
    void *B;
    conn->read(&B, sizeof(B));
    cudaDataType Btype;
    conn->read(&Btype, sizeof(Btype));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    void *C;
    conn->read(&C, sizeof(C));
    cudaDataType Ctype;
    conn->read(&Ctype, sizeof(Ctype));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
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

int handle_cublasUint8gemmBias(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasUint8gemmBias called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    cublasOperation_t transc;
    conn->read(&transc, sizeof(transc));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    unsigned char *A = nullptr;
    conn->read(&A, sizeof(A));
    int A_bias;
    conn->read(&A_bias, sizeof(A_bias));
    int lda;
    conn->read(&lda, sizeof(lda));
    unsigned char *B = nullptr;
    conn->read(&B, sizeof(B));
    int B_bias;
    conn->read(&B_bias, sizeof(B_bias));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    unsigned char *C;
    conn->read(&C, sizeof(C));
    int C_bias;
    conn->read(&C_bias, sizeof(C_bias));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int C_mult;
    conn->read(&C_mult, sizeof(C_mult));
    int C_shift;
    conn->read(&C_shift, sizeof(C_shift));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasUint8gemmBias(handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult, C_shift);
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

int handle_cublasSsyrk_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsyrk_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    float *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
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

int handle_cublasDsyrk_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsyrk_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    double *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
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

int handle_cublasCsyrk_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyrk_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
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

int handle_cublasZsyrk_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsyrk_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
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

int handle_cublasCsyrkEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyrkEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    void *A;
    conn->read(&A, sizeof(A));
    cudaDataType Atype;
    conn->read(&Atype, sizeof(Atype));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    void *C;
    conn->read(&C, sizeof(C));
    cudaDataType Ctype;
    conn->read(&Ctype, sizeof(Ctype));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyrkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
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

int handle_cublasCsyrk3mEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyrk3mEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    void *A;
    conn->read(&A, sizeof(A));
    cudaDataType Atype;
    conn->read(&Atype, sizeof(Atype));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    void *C;
    conn->read(&C, sizeof(C));
    cudaDataType Ctype;
    conn->read(&Ctype, sizeof(Ctype));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyrk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
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

int handle_cublasCherk_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCherk_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
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

int handle_cublasZherk_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZherk_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
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

int handle_cublasCherkEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCherkEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    void *A;
    conn->read(&A, sizeof(A));
    cudaDataType Atype;
    conn->read(&Atype, sizeof(Atype));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    void *C;
    conn->read(&C, sizeof(C));
    cudaDataType Ctype;
    conn->read(&Ctype, sizeof(Ctype));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCherkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
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

int handle_cublasCherk3mEx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCherk3mEx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    void *A;
    conn->read(&A, sizeof(A));
    cudaDataType Atype;
    conn->read(&Atype, sizeof(Atype));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    void *C;
    conn->read(&C, sizeof(C));
    cudaDataType Ctype;
    conn->read(&Ctype, sizeof(Ctype));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCherk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
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

int handle_cublasSsyr2k_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsyr2k_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    float *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasDsyr2k_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsyr2k_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    double *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasCsyr2k_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyr2k_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasZsyr2k_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsyr2k_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasCher2k_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCher2k_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasZher2k_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZher2k_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasSsyrkx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsyrkx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    float *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasDsyrkx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsyrkx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    double *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasCsyrkx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsyrkx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasZsyrkx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsyrkx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasCherkx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCherkx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasZherkx(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZherkx called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasSsymm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSsymm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    float *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasDsymm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDsymm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    double *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasCsymm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCsymm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasZsymm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZsymm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasChemm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasChemm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasChemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasZhemm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZhemm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZhemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
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

int handle_cublasStrsm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStrsm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *B;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
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

int handle_cublasDtrsm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtrsm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *B;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
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

int handle_cublasCtrsm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtrsm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *B;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
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

int handle_cublasZtrsm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtrsm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *B;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
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

int handle_cublasStrmm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStrmm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    float *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
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

int handle_cublasDtrmm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtrmm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    double *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
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

int handle_cublasCtrmm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtrmm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
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

int handle_cublasZtrmm_v2(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtrmm_v2 called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
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

int handle_cublasHgemmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasHgemmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    __half *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    __half *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    __half *Barray = nullptr;
    conn->read(&Barray, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    __half *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    __half *Carray = nullptr;
    conn->read(&Carray, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, (const __half *const *)Aarray, lda, (const __half *const *)Barray, ldb, beta, (__half *const *)Carray, ldc, batchCount);
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

int handle_cublasSgemmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgemmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    float *Barray = nullptr;
    conn->read(&Barray, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    float *Carray = nullptr;
    conn->read(&Carray, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, (const float *const *)Aarray, lda, (const float *const *)Barray, ldb, beta, (float *const *)Carray, ldc, batchCount);
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

int handle_cublasDgemmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgemmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    double *Barray = nullptr;
    conn->read(&Barray, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    double *Carray = nullptr;
    conn->read(&Carray, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, (const double *const *)Aarray, lda, (const double *const *)Barray, ldb, beta, (double *const *)Carray, ldc, batchCount);
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

int handle_cublasCgemmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *Barray = nullptr;
    conn->read(&Barray, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *Carray = nullptr;
    conn->read(&Carray, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemmBatched(handle, transa, transb, m, n, k, alpha, (const cuComplex *const *)Aarray, lda, (const cuComplex *const *)Barray, ldb, beta, (cuComplex *const *)Carray, ldc, batchCount);
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

int handle_cublasCgemm3mBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemm3mBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *Barray = nullptr;
    conn->read(&Barray, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *Carray = nullptr;
    conn->read(&Carray, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemm3mBatched(handle, transa, transb, m, n, k, alpha, (const cuComplex *const *)Aarray, lda, (const cuComplex *const *)Barray, ldb, beta, (cuComplex *const *)Carray, ldc, batchCount);
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

int handle_cublasZgemmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgemmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *Barray = nullptr;
    conn->read(&Barray, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *Carray = nullptr;
    conn->read(&Carray, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgemmBatched(handle, transa, transb, m, n, k, alpha, (const cuDoubleComplex *const *)Aarray, lda, (const cuDoubleComplex *const *)Barray, ldb, beta, (cuDoubleComplex *const *)Carray, ldc, batchCount);
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

int handle_cublasSgemmStridedBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgemmStridedBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    long long int strideA;
    conn->read(&strideA, sizeof(strideA));
    float *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    long long int strideB;
    conn->read(&strideB, sizeof(strideB));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    float *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    long long int strideC;
    conn->read(&strideC, sizeof(strideC));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
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

int handle_cublasDgemmStridedBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgemmStridedBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    long long int strideA;
    conn->read(&strideA, sizeof(strideA));
    double *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    long long int strideB;
    conn->read(&strideB, sizeof(strideB));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    double *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    long long int strideC;
    conn->read(&strideC, sizeof(strideC));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
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

int handle_cublasCgemmStridedBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemmStridedBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    long long int strideA;
    conn->read(&strideA, sizeof(strideA));
    cuComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    long long int strideB;
    conn->read(&strideB, sizeof(strideB));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    long long int strideC;
    conn->read(&strideC, sizeof(strideC));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
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

int handle_cublasCgemm3mStridedBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgemm3mStridedBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    long long int strideA;
    conn->read(&strideA, sizeof(strideA));
    cuComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    long long int strideB;
    conn->read(&strideB, sizeof(strideB));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    long long int strideC;
    conn->read(&strideC, sizeof(strideC));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgemm3mStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
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

int handle_cublasZgemmStridedBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgemmStridedBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    long long int strideA;
    conn->read(&strideA, sizeof(strideA));
    cuDoubleComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    long long int strideB;
    conn->read(&strideB, sizeof(strideB));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    long long int strideC;
    conn->read(&strideC, sizeof(strideC));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
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

int handle_cublasHgemmStridedBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasHgemmStridedBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int k;
    conn->read(&k, sizeof(k));
    __half *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    __half *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    long long int strideA;
    conn->read(&strideA, sizeof(strideA));
    __half *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    long long int strideB;
    conn->read(&strideB, sizeof(strideB));
    __half *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    __half *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    long long int strideC;
    conn->read(&strideC, sizeof(strideC));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
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

int handle_cublasSgeam(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgeam called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    float *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    float *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
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

int handle_cublasDgeam(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgeam called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    double *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    double *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
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

int handle_cublasCgeam(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgeam called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
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

int handle_cublasZgeam(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgeam called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t transa;
    conn->read(&transa, sizeof(transa));
    cublasOperation_t transb;
    conn->read(&transb, sizeof(transb));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *beta = nullptr;
    conn->read(&beta, sizeof(beta));
    cuDoubleComplex *B = nullptr;
    conn->read(&B, sizeof(B));
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
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

int handle_cublasSgetrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgetrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    int *P;
    conn->read(&P, sizeof(P));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgetrfBatched(handle, n, (float *const *)A, lda, P, info, batchSize);
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

int handle_cublasDgetrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgetrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    int *P;
    conn->read(&P, sizeof(P));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgetrfBatched(handle, n, (double *const *)A, lda, P, info, batchSize);
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

int handle_cublasCgetrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgetrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    int *P;
    conn->read(&P, sizeof(P));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgetrfBatched(handle, n, (cuComplex *const *)A, lda, P, info, batchSize);
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

int handle_cublasZgetrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgetrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    int *P;
    conn->read(&P, sizeof(P));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgetrfBatched(handle, n, (cuDoubleComplex *const *)A, lda, P, info, batchSize);
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

int handle_cublasSgetriBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgetriBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    int *P = nullptr;
    conn->read(&P, sizeof(P));
    float *C = nullptr;
    conn->read(&C, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgetriBatched(handle, n, (const float *const *)A, lda, P, (float *const *)C, ldc, info, batchSize);
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

int handle_cublasDgetriBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgetriBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    int *P = nullptr;
    conn->read(&P, sizeof(P));
    double *C = nullptr;
    conn->read(&C, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgetriBatched(handle, n, (const double *const *)A, lda, P, (double *const *)C, ldc, info, batchSize);
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

int handle_cublasCgetriBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgetriBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    int *P = nullptr;
    conn->read(&P, sizeof(P));
    cuComplex *C = nullptr;
    conn->read(&C, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgetriBatched(handle, n, (const cuComplex *const *)A, lda, P, (cuComplex *const *)C, ldc, info, batchSize);
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

int handle_cublasZgetriBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgetriBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    int *P = nullptr;
    conn->read(&P, sizeof(P));
    cuDoubleComplex *C = nullptr;
    conn->read(&C, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgetriBatched(handle, n, (const cuDoubleComplex *const *)A, lda, P, (cuDoubleComplex *const *)C, ldc, info, batchSize);
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

int handle_cublasSgetrsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgetrsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int nrhs;
    conn->read(&nrhs, sizeof(nrhs));
    float *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    int *devIpiv = nullptr;
    conn->read(&devIpiv, sizeof(devIpiv));
    float *Barray = nullptr;
    conn->read(&Barray, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgetrsBatched(handle, trans, n, nrhs, (const float *const *)Aarray, lda, devIpiv, (float *const *)Barray, ldb, info, batchSize);
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

int handle_cublasDgetrsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgetrsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int nrhs;
    conn->read(&nrhs, sizeof(nrhs));
    double *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    int *devIpiv = nullptr;
    conn->read(&devIpiv, sizeof(devIpiv));
    double *Barray = nullptr;
    conn->read(&Barray, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgetrsBatched(handle, trans, n, nrhs, (const double *const *)Aarray, lda, devIpiv, (double *const *)Barray, ldb, info, batchSize);
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

int handle_cublasCgetrsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgetrsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int nrhs;
    conn->read(&nrhs, sizeof(nrhs));
    cuComplex *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    int *devIpiv = nullptr;
    conn->read(&devIpiv, sizeof(devIpiv));
    cuComplex *Barray = nullptr;
    conn->read(&Barray, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgetrsBatched(handle, trans, n, nrhs, (const cuComplex *const *)Aarray, lda, devIpiv, (cuComplex *const *)Barray, ldb, info, batchSize);
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

int handle_cublasZgetrsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgetrsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int n;
    conn->read(&n, sizeof(n));
    int nrhs;
    conn->read(&nrhs, sizeof(nrhs));
    cuDoubleComplex *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    int *devIpiv = nullptr;
    conn->read(&devIpiv, sizeof(devIpiv));
    cuDoubleComplex *Barray = nullptr;
    conn->read(&Barray, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgetrsBatched(handle, trans, n, nrhs, (const cuDoubleComplex *const *)Aarray, lda, devIpiv, (cuDoubleComplex *const *)Barray, ldb, info, batchSize);
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

int handle_cublasStrsmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStrsmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    float *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    float *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    float *B = nullptr;
    conn->read(&B, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, (const float *const *)A, lda, (float *const *)B, ldb, batchCount);
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

int handle_cublasDtrsmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtrsmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    double *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    double *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    double *B = nullptr;
    conn->read(&B, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, (const double *const *)A, lda, (double *const *)B, ldb, batchCount);
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

int handle_cublasCtrsmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtrsmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuComplex *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *B = nullptr;
    conn->read(&B, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, (const cuComplex *const *)A, lda, (cuComplex *const *)B, ldb, batchCount);
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

int handle_cublasZtrsmBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtrsmBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t side;
    conn->read(&side, sizeof(side));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    cublasDiagType_t diag;
    conn->read(&diag, sizeof(diag));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *alpha = nullptr;
    conn->read(&alpha, sizeof(alpha));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *B = nullptr;
    conn->read(&B, 0, true);
    int ldb;
    conn->read(&ldb, sizeof(ldb));
    int batchCount;
    conn->read(&batchCount, sizeof(batchCount));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, (const cuDoubleComplex *const *)A, lda, (cuDoubleComplex *const *)B, ldb, batchCount);
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

int handle_cublasSmatinvBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSmatinvBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    float *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    float *Ainv = nullptr;
    conn->read(&Ainv, 0, true);
    int lda_inv;
    conn->read(&lda_inv, sizeof(lda_inv));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSmatinvBatched(handle, n, (const float *const *)A, lda, (float *const *)Ainv, lda_inv, info, batchSize);
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

int handle_cublasDmatinvBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDmatinvBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    double *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    double *Ainv = nullptr;
    conn->read(&Ainv, 0, true);
    int lda_inv;
    conn->read(&lda_inv, sizeof(lda_inv));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDmatinvBatched(handle, n, (const double *const *)A, lda, (double *const *)Ainv, lda_inv, info, batchSize);
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

int handle_cublasCmatinvBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCmatinvBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *Ainv = nullptr;
    conn->read(&Ainv, 0, true);
    int lda_inv;
    conn->read(&lda_inv, sizeof(lda_inv));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCmatinvBatched(handle, n, (const cuComplex *const *)A, lda, (cuComplex *const *)Ainv, lda_inv, info, batchSize);
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

int handle_cublasZmatinvBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZmatinvBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *Ainv = nullptr;
    conn->read(&Ainv, 0, true);
    int lda_inv;
    conn->read(&lda_inv, sizeof(lda_inv));
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZmatinvBatched(handle, n, (const cuDoubleComplex *const *)A, lda, (cuDoubleComplex *const *)Ainv, lda_inv, info, batchSize);
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

int handle_cublasSgeqrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgeqrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    float *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    float *TauArray = nullptr;
    conn->read(&TauArray, 0, true);
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgeqrfBatched(handle, m, n, (float *const *)Aarray, lda, (float *const *)TauArray, info, batchSize);
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

int handle_cublasDgeqrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgeqrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    double *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    double *TauArray = nullptr;
    conn->read(&TauArray, 0, true);
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgeqrfBatched(handle, m, n, (double *const *)Aarray, lda, (double *const *)TauArray, info, batchSize);
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

int handle_cublasCgeqrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgeqrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *TauArray = nullptr;
    conn->read(&TauArray, 0, true);
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgeqrfBatched(handle, m, n, (cuComplex *const *)Aarray, lda, (cuComplex *const *)TauArray, info, batchSize);
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

int handle_cublasZgeqrfBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgeqrfBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *TauArray = nullptr;
    conn->read(&TauArray, 0, true);
    int *info;
    conn->read(&info, sizeof(info));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgeqrfBatched(handle, m, n, (cuDoubleComplex *const *)Aarray, lda, (cuDoubleComplex *const *)TauArray, info, batchSize);
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

int handle_cublasSgelsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSgelsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int nrhs;
    conn->read(&nrhs, sizeof(nrhs));
    float *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    float *Carray = nullptr;
    conn->read(&Carray, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int *info;
    conn->read(&info, sizeof(info));
    int *devInfoArray;
    conn->read(&devInfoArray, sizeof(devInfoArray));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSgelsBatched(handle, trans, m, n, nrhs, (float *const *)Aarray, lda, (float *const *)Carray, ldc, info, devInfoArray, batchSize);
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

int handle_cublasDgelsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDgelsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int nrhs;
    conn->read(&nrhs, sizeof(nrhs));
    double *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    double *Carray = nullptr;
    conn->read(&Carray, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int *info;
    conn->read(&info, sizeof(info));
    int *devInfoArray;
    conn->read(&devInfoArray, sizeof(devInfoArray));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDgelsBatched(handle, trans, m, n, nrhs, (double *const *)Aarray, lda, (double *const *)Carray, ldc, info, devInfoArray, batchSize);
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

int handle_cublasCgelsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCgelsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int nrhs;
    conn->read(&nrhs, sizeof(nrhs));
    cuComplex *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *Carray = nullptr;
    conn->read(&Carray, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int *info;
    conn->read(&info, sizeof(info));
    int *devInfoArray;
    conn->read(&devInfoArray, sizeof(devInfoArray));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCgelsBatched(handle, trans, m, n, nrhs, (cuComplex *const *)Aarray, lda, (cuComplex *const *)Carray, ldc, info, devInfoArray, batchSize);
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

int handle_cublasZgelsBatched(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZgelsBatched called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasOperation_t trans;
    conn->read(&trans, sizeof(trans));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    int nrhs;
    conn->read(&nrhs, sizeof(nrhs));
    cuDoubleComplex *Aarray = nullptr;
    conn->read(&Aarray, 0, true);
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *Carray = nullptr;
    conn->read(&Carray, 0, true);
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    int *info;
    conn->read(&info, sizeof(info));
    int *devInfoArray;
    conn->read(&devInfoArray, sizeof(devInfoArray));
    int batchSize;
    conn->read(&batchSize, sizeof(batchSize));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZgelsBatched(handle, trans, m, n, nrhs, (cuDoubleComplex *const *)Aarray, lda, (cuDoubleComplex *const *)Carray, ldc, info, devInfoArray, batchSize);
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

int handle_cublasSdgmm(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasSdgmm called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t mode;
    conn->read(&mode, sizeof(mode));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    float *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
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

int handle_cublasDdgmm(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDdgmm called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t mode;
    conn->read(&mode, sizeof(mode));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    double *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
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

int handle_cublasCdgmm(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCdgmm called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t mode;
    conn->read(&mode, sizeof(mode));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
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

int handle_cublasZdgmm(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZdgmm called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasSideMode_t mode;
    conn->read(&mode, sizeof(mode));
    int m;
    conn->read(&m, sizeof(m));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *x = nullptr;
    conn->read(&x, sizeof(x));
    int incx;
    conn->read(&incx, sizeof(incx));
    cuDoubleComplex *C;
    conn->read(&C, sizeof(C));
    int ldc;
    conn->read(&ldc, sizeof(ldc));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
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

int handle_cublasStpttr(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStpttr called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    float *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    float *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStpttr(handle, uplo, n, AP, A, lda);
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

int handle_cublasDtpttr(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtpttr called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    double *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    double *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtpttr(handle, uplo, n, AP, A, lda);
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

int handle_cublasCtpttr(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtpttr called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    cuComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtpttr(handle, uplo, n, AP, A, lda);
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

int handle_cublasZtpttr(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtpttr called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *AP = nullptr;
    conn->read(&AP, sizeof(AP));
    cuDoubleComplex *A;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtpttr(handle, uplo, n, AP, A, lda);
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

int handle_cublasStrttp(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasStrttp called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    float *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    float *AP;
    conn->read(&AP, sizeof(AP));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasStrttp(handle, uplo, n, A, lda, AP);
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

int handle_cublasDtrttp(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasDtrttp called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    double *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    double *AP;
    conn->read(&AP, sizeof(AP));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasDtrttp(handle, uplo, n, A, lda, AP);
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

int handle_cublasCtrttp(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasCtrttp called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuComplex *AP;
    conn->read(&AP, sizeof(AP));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasCtrttp(handle, uplo, n, A, lda, AP);
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

int handle_cublasZtrttp(void *args0) {
#ifdef DEBUG
    std::cout << "Handle function cublasZtrttp called" << std::endl;
#endif
    int rtn = 0;
    std::set<void *> buffers;
    RpcConn *conn = (RpcConn *)args0;
    cublasHandle_t handle;
    conn->read(&handle, sizeof(handle));
    cublasFillMode_t uplo;
    conn->read(&uplo, sizeof(uplo));
    int n;
    conn->read(&n, sizeof(n));
    cuDoubleComplex *A = nullptr;
    conn->read(&A, sizeof(A));
    int lda;
    conn->read(&lda, sizeof(lda));
    cuDoubleComplex *AP;
    conn->read(&AP, sizeof(AP));
    cublasStatus_t _result;
    if(conn->prepare_response() != RpcError::OK) {
        std::cerr << "Failed to prepare response" << std::endl;
        rtn = 1;
        goto _RTN_;
    }
    _result = cublasZtrttp(handle, uplo, n, A, lda, AP);
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
